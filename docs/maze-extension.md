# Maze Extension

## Goal

Extend the controllable-cell experiment into a simple maze-control task.

## Cell Types

- empty: traversable background
- blocked: static wall cell that cannot be entered
- active: the controllable cell
- exit: a goal cell that ends the game when reached

## Transition Rule

- `up`, `down`, `left`, `right`, `none` remain the control actions
- if the requested target cell is empty, the active cell moves into it
- if the requested target cell is blocked, the active cell stays in place
- if the active cell reaches the exit, the game enters a terminal state
- once terminal, future input actions have no effect
- after terminal, the exit color spreads across the grid over time
- blocked cells never move
- the active cell remains unique and keeps its value

## Implementation Strategy

### Step 1

- extend the deterministic grid state to include blocked cells
- add a pure-Python maze generator
- verify wall collisions and maze validity

### Step 1b

- add explicit start and exit cells
- verify that the exit is reachable
- add deterministic terminal-state spread semantics

### Step 2

- add a wall channel to model inputs
- create supervised training data from generated mazes

### Step 3

- train and evaluate the NCA on maze transitions
- verify both one-step and rollout behavior

### Step 4

- update the browser visualizer to draw walls and player state correctly
- regenerate a fresh solvable maze on every visualizer reset
- surface terminal-state rendering and exit-fill spread in the browser comparison app

### Step 5

- extend the learned state with exit dynamics
- train and evaluate the NCA on terminal lockout and end-state spread

## Exit-Aware Inference Note

- the learned checkpoint models wall-aware control and the transition into the terminal state
- once terminal, decoded rollout and browser playback use deterministic exit-fill expansion to keep the game semantics exact
