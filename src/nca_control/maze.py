from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .grid import GridState


@dataclass(frozen=True, slots=True)
class MazeLayout:
    height: int
    width: int
    blocked: frozenset[tuple[int, int]]

    def __post_init__(self) -> None:
        if self.height < 3 or self.width < 3:
            raise ValueError("maze dimensions must be at least 3x3")
        for row, col in self.blocked:
            if not (0 <= row < self.height and 0 <= col < self.width):
                raise ValueError("blocked cells must stay within maze bounds")

    def open_cells(self) -> list[tuple[int, int]]:
        return [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if (row, col) not in self.blocked
        ]

    def to_grid_state(self, row: int, col: int, value: float = 1.0) -> GridState:
        return GridState(
            height=self.height,
            width=self.width,
            row=row,
            col=col,
            value=value,
            blocked=self.blocked,
        )


def generate_maze(height: int, width: int, seed: int = 0) -> MazeLayout:
    logical_height = max(1, (height - 1) // 2)
    logical_width = max(1, (width - 1) // 2)
    blocked = {(row, col) for row in range(height) for col in range(width)}
    rng = Random(seed)

    visited: set[tuple[int, int]] = set()
    stack = [(0, 0)]
    visited.add((0, 0))

    def carve(cell_row: int, cell_col: int) -> None:
        blocked.discard((2 * cell_row + 1, 2 * cell_col + 1))

    carve(0, 0)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        current_row, current_col = stack[-1]
        neighbors: list[tuple[int, int, int, int]] = []
        for delta_row, delta_col in directions:
            next_row = current_row + delta_row
            next_col = current_col + delta_col
            if not (0 <= next_row < logical_height and 0 <= next_col < logical_width):
                continue
            if (next_row, next_col) in visited:
                continue
            neighbors.append((next_row, next_col, delta_row, delta_col))

        if not neighbors:
            stack.pop()
            continue

        next_row, next_col, delta_row, delta_col = rng.choice(neighbors)
        carve(next_row, next_col)
        wall_row = 2 * current_row + 1 + delta_row
        wall_col = 2 * current_col + 1 + delta_col
        blocked.discard((wall_row, wall_col))
        visited.add((next_row, next_col))
        stack.append((next_row, next_col))

    return MazeLayout(height=height, width=width, blocked=frozenset(blocked))
