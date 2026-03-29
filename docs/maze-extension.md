# Maze Extension

## Goal

Extend the controllable-cell experiment into a simple maze-control task.

## Cell Types

- empty: traversable background
- blocked: static wall cell that cannot be entered
- active: the controllable cell

## Transition Rule

- `up`, `down`, `left`, `right`, `none` remain the control actions
- if the requested target cell is empty, the active cell moves into it
- if the requested target cell is blocked, the active cell stays in place
- blocked cells never move
- the active cell remains unique and keeps its value

## Implementation Strategy

### Step 1

- extend the deterministic grid state to include blocked cells
- add a pure-Python maze generator
- verify wall collisions and maze validity

### Step 2

- add a wall channel to model inputs
- create supervised training data from generated mazes

### Step 3

- train and evaluate the NCA on maze transitions
- verify both one-step and rollout behavior

### Step 4

- update the browser visualizer to draw walls and player state correctly
