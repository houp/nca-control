from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from random import Random

from .grid import GridState


@dataclass(frozen=True, slots=True)
class MazeLayout:
    height: int
    width: int
    blocked: frozenset[tuple[int, int]]
    start_cell: tuple[int, int]
    exit_cell: tuple[int, int]

    def __post_init__(self) -> None:
        if self.height < 3 or self.width < 3:
            raise ValueError("maze dimensions must be at least 3x3")
        for row, col in self.blocked:
            if not (0 <= row < self.height and 0 <= col < self.width):
                raise ValueError("blocked cells must stay within maze bounds")
        for cell in [self.start_cell, self.exit_cell]:
            row, col = cell
            if not (0 <= row < self.height and 0 <= col < self.width):
                raise ValueError("maze endpoints must stay within bounds")
            if cell in self.blocked:
                raise ValueError("maze endpoints must be open cells")

    def open_cells(self) -> list[tuple[int, int]]:
        return [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if (row, col) not in self.blocked
        ]

    def to_grid_state(self, row: int | None = None, col: int | None = None, value: float = 1.0) -> GridState:
        active_row, active_col = (self.start_cell if row is None or col is None else (row, col))
        return GridState(
            height=self.height,
            width=self.width,
            row=active_row,
            col=active_col,
            value=value,
            blocked=self.blocked,
            exit_cell=self.exit_cell,
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

    start_cell = (1, 1)
    exit_cell = _farthest_open_cell(height, width, frozenset(blocked), start_cell)
    return MazeLayout(
        height=height,
        width=width,
        blocked=frozenset(blocked),
        start_cell=start_cell,
        exit_cell=exit_cell,
    )


def _farthest_open_cell(
    height: int,
    width: int,
    blocked: frozenset[tuple[int, int]],
    start_cell: tuple[int, int],
) -> tuple[int, int]:
    visited = {start_cell}
    queue = deque([(start_cell, 0)])
    farthest_cell = start_cell
    farthest_distance = 0

    while queue:
        (row, col), distance = queue.popleft()
        if distance > farthest_distance:
            farthest_cell = (row, col)
            farthest_distance = distance
        for next_row, next_col in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if (next_row, next_col) in blocked:
                continue
            if (next_row, next_col) in visited:
                continue
            visited.add((next_row, next_col))
            queue.append(((next_row, next_col), distance + 1))

    return farthest_cell
