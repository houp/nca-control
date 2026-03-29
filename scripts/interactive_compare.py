from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import typer

from nca_control.actions import Action
from nca_control.grid import GridState
from nca_control.inference import load_checkpoint
from nca_control.interactive import InteractiveCompareSession
from nca_control.maze import generate_maze

app = typer.Typer(add_completion=False)

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Controllable NCA Visualizer</title>
  <style>
    :root {
      --bg: #f6f1e7;
      --panel: #fffaf1;
      --grid: #d5c7b2;
      --active: #0c8a47;
      --exit: #d8a106;
      --mismatch: #bf1e2e;
      --text: #1f2933;
    }
    body {
      margin: 0;
      font-family: Georgia, "Iowan Old Style", serif;
      background: radial-gradient(circle at top, #fffaf1, var(--bg));
      color: var(--text);
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      margin-bottom: 18px;
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: 32px;
    }
    .hero p {
      margin: 0;
      line-height: 1.45;
    }
    .status {
      margin: 16px 0 20px;
      padding: 14px 16px;
      border: 1px solid #d9ccb8;
      background: var(--panel);
      border-radius: 14px;
    }
    .boards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }
    .board {
      background: var(--panel);
      border: 1px solid #dfd3c1;
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 14px 30px rgba(120, 89, 41, 0.08);
    }
    .board h2 {
      margin: 0 0 12px;
      font-size: 22px;
    }
    .grid {
      display: grid;
      gap: 4px;
    }
    .cell {
      width: 100%;
      aspect-ratio: 1;
      border: 1px solid var(--grid);
      background: #f3ece1;
      border-radius: 6px;
    }
    .cell.active {
      background: var(--active);
    }
    .cell.mismatch {
      outline: 3px solid var(--mismatch);
      outline-offset: -2px;
    }
    .help {
      margin-top: 20px;
      font-size: 15px;
      line-height: 1.5;
    }
    kbd {
      border: 1px solid #c6b79f;
      border-bottom-width: 3px;
      border-radius: 6px;
      padding: 1px 7px;
      background: #fff;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Controllable NCA Visualizer</h1>
      <p>Arrow keys queue moves. Space queues a no-op. The simulation advances on a fixed clock, so terminal exit-fill spread continues even when you stop pressing keys. R resets. The right panel is highlighted red if the learned model diverges from the deterministic reference.</p>
    </div>
    <div id="status" class="status">Loading...</div>
    <div class="boards">
      <section class="board">
        <h2>Reference</h2>
        <div id="reference-grid" class="grid"></div>
      </section>
      <section class="board">
        <h2>Model</h2>
        <div id="model-grid" class="grid"></div>
      </section>
    </div>
    <div class="help">
      Controls:
      <kbd>↑</kbd> <kbd>↓</kbd> <kbd>←</kbd> <kbd>→</kbd> move,
      <kbd>Space</kbd> no-op,
      <kbd>R</kbd> reset.
      Fixed tick: <span id="tick-ms"></span> ms.
    </div>
  </div>
  <script>
    const statusEl = document.getElementById("status");
    const refGrid = document.getElementById("reference-grid");
    const modelGrid = document.getElementById("model-grid");
    const tickMsEl = document.getElementById("tick-ms");
    let latestVersion = -1;
    let requestQueue = Promise.resolve();
    const pendingActions = [];
    const tickMs = __TICK_MS__;
    let clockStarted = false;

    function drawGrid(target, state, mismatch) {
      const blocked = new Set((state.blocked || []).map(([row, col]) => `${row},${col}`));
      const exitFill = new Set((state.exit_fill || []).map(([row, col]) => `${row},${col}`));
      target.style.gridTemplateColumns = `repeat(${state.width}, minmax(0, 1fr))`;
      target.innerHTML = "";
      for (let row = 0; row < state.height; row += 1) {
        for (let col = 0; col < state.width; col += 1) {
          const cell = document.createElement("div");
          cell.className = "cell";
          if (blocked.has(`${row},${col}`)) {
            cell.style.background = "#2f3a45";
            cell.style.borderColor = "#20262d";
          } else if (exitFill.has(`${row},${col}`)) {
            cell.style.background = "var(--exit)";
            cell.style.borderColor = "#8c6608";
          }
          if (!state.terminated && row === state.row && col === state.col) {
            cell.classList.add("active");
            if (mismatch) {
              cell.classList.add("mismatch");
            }
          }
          target.appendChild(cell);
        }
      }
    }

    function render(data) {
      if (typeof data.version === "number" && data.version < latestVersion) {
        return;
      }
      latestVersion = typeof data.version === "number" ? data.version : latestVersion;
      drawGrid(refGrid, data.reference, false);
      drawGrid(modelGrid, data.model, !data.match);
      statusEl.textContent =
        `tick_ms=${tickMs} | version=${data.version} | last_action=${data.last_action} | queued_actions=${pendingActions.length} | reference=(${data.reference.row}, ${data.reference.col}, terminated=${data.reference.terminated}) | model=(${data.model.row}, ${data.model.col}, terminated=${data.model.terminated}) | match=${data.match ? "yes" : "no"}`;
    }

    function enqueueRequest(path, options = {}) {
      requestQueue = requestQueue
        .catch(() => {})
        .then(async () => {
          const response = await fetch(path, options);
          const payload = await response.json();
          render(payload);
        })
        .catch((error) => {
          statusEl.textContent = `request_failed=${error}`;
        });
      return requestQueue;
    }

    function sleep(ms) {
      return new Promise((resolve) => window.setTimeout(resolve, ms));
    }

    function nextAction() {
      return pendingActions.length > 0 ? pendingActions.shift() : "none";
    }

    async function runClock() {
      if (clockStarted) {
        return;
      }
      clockStarted = true;
      while (true) {
        const startedAt = performance.now();
        await enqueueRequest(`/step?action=${nextAction()}`, { method: "POST" });
        const elapsedMs = performance.now() - startedAt;
        await sleep(Math.max(0, tickMs - elapsedMs));
      }
    }

    const keyToAction = {
      ArrowUp: "up",
      ArrowDown: "down",
      ArrowLeft: "left",
      ArrowRight: "right",
      " ": "none",
    };

    window.addEventListener("keydown", async (event) => {
      if (event.key === "r" || event.key === "R") {
        event.preventDefault();
        pendingActions.length = 0;
        await enqueueRequest("/reset", { method: "POST" });
        return;
      }
      const action = keyToAction[event.key];
      if (!action) {
        return;
      }
      event.preventDefault();
      pendingActions.push(action);
    });

    tickMsEl.textContent = String(tickMs);
    enqueueRequest("/state").then(() => runClock());
  </script>
</body>
</html>
"""


def build_html_page(tick_ms: int) -> str:
    return HTML_TEMPLATE.replace("__TICK_MS__", str(tick_ms))


def make_handler(session: InteractiveCompareSession, *, tick_ms: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(build_html_page(tick_ms))
                return
            if parsed.path == "/state":
                self._send_json(session.snapshot())
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/reset":
                self._send_json(session.reset())
                return
            if parsed.path == "/step":
                params = parse_qs(parsed.query)
                raw_action = params.get("action", [""])[0]
                try:
                    action = Action(raw_action)
                except ValueError:
                    self.send_error(HTTPStatus.BAD_REQUEST, "invalid action")
                    return
                self._send_json(session.apply_action(action))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _send_html(self, payload: str) -> None:
            encoded = payload.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    height: int = typer.Option(6, min=1),
    width: int = typer.Option(6, min=1),
    row: int | None = typer.Option(None, min=0),
    col: int | None = typer.Option(None, min=0),
    value: float = typer.Option(1.0),
    maze_seed: int | None = typer.Option(None, help="Generate and visualize a maze using this seed."),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000, min=1, max=65535),
    tick_ms: int = typer.Option(120, min=1, help="Fixed simulation tick interval in milliseconds."),
    device: str = typer.Option("auto"),
) -> None:
    _model, config, _resolved = load_checkpoint(checkpoint, device="cpu")
    effective_maze_seed = maze_seed
    if effective_maze_seed is None and str(config.get("task", "plain")) in {"maze", "maze_exit"}:
        effective_maze_seed = int(config.get("maze_seed", 0))

    if effective_maze_seed is not None:
        reset_counter = {"value": 0}

        def build_maze_state() -> GridState:
            layout = generate_maze(height=height, width=width, seed=effective_maze_seed + reset_counter["value"])
            reset_counter["value"] += 1
            if row is not None and col is not None and (row, col) not in layout.blocked:
                start_row, start_col = row, col
                return layout.to_grid_state(row=start_row, col=start_col, value=value)
            return layout.to_grid_state(value=value)

        initial_state = build_maze_state()
        reset_factory = build_maze_state
    else:
        start_row = row if row is not None else 0
        start_col = col if col is not None else 0
        initial_state = GridState(height=height, width=width, row=start_row, col=start_col, value=value)
        reset_factory = None

    session = InteractiveCompareSession(
        checkpoint_path=str(checkpoint),
        initial_state=initial_state,
        device=device,
        reset_factory=reset_factory,
    )
    server = HTTPServer((host, port), make_handler(session, tick_ms=tick_ms))
    typer.echo(f"Visualizer running at http://{host}:{port}")
    typer.echo(f"Open that URL in your browser. Fixed tick={tick_ms} ms. Use arrow keys, Space, and R in the page.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nShutting down visualizer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    app()
