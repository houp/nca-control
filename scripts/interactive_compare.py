from __future__ import annotations

import tkinter as tk
from pathlib import Path

import typer

from nca_control.actions import Action
from nca_control.grid import GridState, step_grid
from nca_control.inference import predict_next_state
from nca_control.interactive import action_from_keysym, prediction_to_grid_state

app = typer.Typer(add_completion=False)


class InteractiveCompareApp:
    def __init__(
        self,
        root: tk.Tk,
        checkpoint: Path,
        initial_state: GridState,
        *,
        cell_size: int,
        device: str,
    ) -> None:
        self.root = root
        self.checkpoint = checkpoint
        self.reference_state = initial_state
        self.model_state = initial_state
        self.initial_state = initial_state
        self.cell_size = cell_size
        self.device = device
        self.last_action = Action.NONE

        self.status_var = tk.StringVar()
        self.reference_canvas = tk.Canvas(
            root,
            width=initial_state.width * cell_size,
            height=initial_state.height * cell_size,
            highlightthickness=0,
        )
        self.model_canvas = tk.Canvas(
            root,
            width=initial_state.width * cell_size,
            height=initial_state.height * cell_size,
            highlightthickness=0,
        )
        self._build_layout()
        self._render()
        self.root.bind("<KeyPress>", self._on_keypress)

    def _build_layout(self) -> None:
        self.root.title("Controllable NCA: Reference vs Model")
        container = tk.Frame(self.root, padx=16, pady=16)
        container.pack(fill="both", expand=True)

        instructions = tk.Label(
            container,
            text="Arrows move. Space = no-op. R resets. Esc quits.",
            anchor="w",
        )
        instructions.pack(fill="x")

        canvas_row = tk.Frame(container, pady=12)
        canvas_row.pack()

        ref_frame = tk.Frame(canvas_row)
        ref_frame.pack(side="left", padx=8)
        tk.Label(ref_frame, text="Reference").pack()
        self.reference_canvas.pack(in_=ref_frame)

        model_frame = tk.Frame(canvas_row)
        model_frame.pack(side="left", padx=8)
        tk.Label(model_frame, text="Model").pack()
        self.model_canvas.pack(in_=model_frame)

        status_label = tk.Label(container, textvariable=self.status_var, anchor="w", justify="left")
        status_label.pack(fill="x")

    def _on_keypress(self, event: tk.Event[tk.Misc]) -> None:
        if event.keysym.lower() == "escape":
            self.root.destroy()
            return
        if event.keysym.lower() == "r":
            self.reference_state = self.initial_state
            self.model_state = self.initial_state
            self.last_action = Action.NONE
            self._render()
            return

        action = action_from_keysym(event.keysym)
        if action is None:
            return

        self.last_action = action
        self.reference_state = step_grid(self.reference_state, action)
        prediction = predict_next_state(
            self.checkpoint,
            self.model_state,
            action,
            device=self.device,
            hard_decode=True,
        )
        self.model_state = prediction_to_grid_state(prediction, value=self.model_state.value)
        self._render()

    def _render(self) -> None:
        mismatch = (self.reference_state.row, self.reference_state.col) != (
            self.model_state.row,
            self.model_state.col,
        )
        self._draw_state(self.reference_canvas, self.reference_state)
        self._draw_state(self.model_canvas, self.model_state, mismatch=mismatch)
        self.status_var.set(
            "\n".join(
                [
                    f"last_action={self.last_action.value}",
                    f"reference=({self.reference_state.row}, {self.reference_state.col})",
                    f"model=({self.model_state.row}, {self.model_state.col})",
                    f"match={'yes' if not mismatch else 'no'}",
                ]
            )
        )

    def _draw_state(self, canvas: tk.Canvas, state: GridState, mismatch: bool = False) -> None:
        canvas.delete("all")
        empty = "#f4f1ea"
        active = "#0a7f3f"
        outline = "#d3c7b8"
        mismatch_outline = "#b42318"
        for row in range(state.height):
            for col in range(state.width):
                x0 = col * self.cell_size
                y0 = row * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                is_active = (row, col) == (state.row, state.col)
                canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=active if is_active else empty,
                    outline=mismatch_outline if mismatch and is_active else outline,
                    width=3 if mismatch and is_active else 1,
                )


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    height: int = typer.Option(6, min=1),
    width: int = typer.Option(6, min=1),
    row: int = typer.Option(0, min=0),
    col: int = typer.Option(0, min=0),
    value: float = typer.Option(1.0),
    cell_size: int = typer.Option(48, min=8),
    device: str = typer.Option("auto"),
) -> None:
    root = tk.Tk()
    app = InteractiveCompareApp(
        root,
        checkpoint,
        GridState(height=height, width=width, row=row, col=col, value=value),
        cell_size=cell_size,
        device=device,
    )
    _ = app
    root.mainloop()


if __name__ == "__main__":
    app()
