from __future__ import annotations

"""Deterministic grid semantics used as the reference system for training and tests."""

from dataclasses import dataclass, field

from .actions import Action


@dataclass(frozen=True, slots=True)
class GridState:
    """Full deterministic game state for the current maze/control task."""

    height: int
    width: int
    row: int
    col: int
    value: float = 1.0
    blocked: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    exit_cell: tuple[int, int] | None = None
    exit_fill: frozenset[tuple[int, int]] | None = None
    terminated: bool = False

    def __post_init__(self) -> None:
        # Keep every state instance valid so the same object can be reused by
        # the dataset builders, the visualizer, and the rollout evaluators.
        if self.height <= 0 or self.width <= 0:
            raise ValueError("grid dimensions must be positive")
        if not (0 <= self.row < self.height):
            raise ValueError("row must be within grid bounds")
        if not (0 <= self.col < self.width):
            raise ValueError("col must be within grid bounds")
        if self.value == 0:
            raise ValueError("active cell value must be non-zero")
        for blocked_row, blocked_col in self.blocked:
            if not (0 <= blocked_row < self.height and 0 <= blocked_col < self.width):
                raise ValueError("blocked cells must be within grid bounds")
        if (self.row, self.col) in self.blocked:
            raise ValueError("active cell cannot overlap a blocked cell")
        if self.exit_cell is not None:
            exit_row, exit_col = self.exit_cell
            if not (0 <= exit_row < self.height and 0 <= exit_col < self.width):
                raise ValueError("exit cell must be within grid bounds")
            if self.exit_cell in self.blocked:
                raise ValueError("exit cell cannot overlap a blocked cell")

        exit_fill = self.exit_fill
        if exit_fill is None:
            exit_fill = frozenset({self.exit_cell}) if self.exit_cell is not None else frozenset()
            object.__setattr__(self, "exit_fill", exit_fill)

        for exit_row, exit_col in exit_fill:
            if not (0 <= exit_row < self.height and 0 <= exit_col < self.width):
                raise ValueError("exit-state cells must be within grid bounds")
        if self.exit_cell is not None and self.exit_cell not in exit_fill:
            object.__setattr__(self, "exit_fill", frozenset(set(exit_fill) | {self.exit_cell}))
        if self.terminated and self.exit_cell is None:
            raise ValueError("terminated state requires an exit cell")

    def is_blocked(self, row: int, col: int) -> bool:
        return (row, col) in self.blocked

    def is_exit(self, row: int, col: int) -> bool:
        return (row, col) in (self.exit_fill or frozenset())

    def is_active(self, row: int, col: int) -> bool:
        return not self.terminated and (row, col) == (self.row, self.col)

    def as_text(self) -> str:
        lines: list[str] = []
        for r in range(self.height):
            symbols: list[str] = []
            for c in range(self.width):
                if self.is_active(r, c):
                    symbols.append("X")
                elif self.is_blocked(r, c):
                    symbols.append("#")
                elif self.is_exit(r, c):
                    symbols.append("E")
                else:
                    symbols.append(".")
            lines.append("".join(symbols))
        return "\n".join(lines)


def step_grid(state: GridState, action: Action) -> GridState:
    """Apply one exact environment step, including terminal fill propagation."""

    if state.terminated:
        # After termination, actions are ignored and only the exit color spreads.
        return GridState(
            height=state.height,
            width=state.width,
            row=state.row,
            col=state.col,
            value=state.value,
            blocked=state.blocked,
            exit_cell=state.exit_cell,
            exit_fill=_expand_exit_fill(state),
            terminated=True,
        )

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

    if state.is_blocked(next_row, next_col):
        next_row, next_col = row, col

    terminated = state.exit_cell is not None and (next_row, next_col) == state.exit_cell

    return GridState(
        height=state.height,
        width=state.width,
        row=next_row,
        col=next_col,
        value=state.value,
        blocked=state.blocked,
        exit_cell=state.exit_cell,
        exit_fill=state.exit_fill,
        terminated=terminated,
    )


def _expand_exit_fill(state: GridState) -> frozenset[tuple[int, int]]:
    """Grow the terminal fill over the non-periodic maze neighborhood."""

    filled = set(state.exit_fill or frozenset())
    for row, col in list(filled):
        for next_row, next_col in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if 0 <= next_row < state.height and 0 <= next_col < state.width:
                filled.add((next_row, next_col))
    return frozenset(filled)
