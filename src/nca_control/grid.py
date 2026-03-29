from __future__ import annotations

from dataclasses import dataclass

from .actions import Action


@dataclass(frozen=True, slots=True)
class GridState:
    height: int
    width: int
    row: int
    col: int
    value: float = 1.0

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("grid dimensions must be positive")
        if not (0 <= self.row < self.height):
            raise ValueError("row must be within grid bounds")
        if not (0 <= self.col < self.width):
            raise ValueError("col must be within grid bounds")
        if self.value == 0:
            raise ValueError("active cell value must be non-zero")

    def as_text(self) -> str:
        lines: list[str] = []
        for r in range(self.height):
            symbols: list[str] = []
            for c in range(self.width):
                symbols.append("X" if (r, c) == (self.row, self.col) else ".")
            lines.append("".join(symbols))
        return "\n".join(lines)


def step_grid(state: GridState, action: Action) -> GridState:
    row, col = state.row, state.col

    if action == Action.NONE:
        next_row, next_col = row, col
    elif action == Action.UP:
        next_row, next_col = (row - 1) % state.height, col
    elif action == Action.DOWN:
        next_row, next_col = (row + 1) % state.height, col
    elif action == Action.LEFT:
        next_row, next_col = row, (col - 1) % state.width
    elif action == Action.RIGHT:
        next_row, next_col = row, (col + 1) % state.width
    else:
        raise ValueError(f"unsupported action: {action}")

    return GridState(
        height=state.height,
        width=state.width,
        row=next_row,
        col=next_col,
        value=state.value,
    )

