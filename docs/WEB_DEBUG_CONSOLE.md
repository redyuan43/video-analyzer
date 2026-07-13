# Web Debug Console

`web_debug_console` is a reusable Flask module that adds two project-scoped
development surfaces:

- a browser terminal backed by a real PTY;
- a Debug conversation backed by `codex app-server --stdio`.

The browser terminal can start the login shell, Codex CLI, or Claude CLI when
the corresponding executable is installed. The Debug surface creates an
ephemeral Codex thread and injects the current page context only on the first
turn.

## Integration

```python
from pathlib import Path
from flask import Flask, render_template
from web_debug_console import WebDebugConsole

app = Flask(__name__)
console = WebDebugConsole(
    app,
    Path.cwd(),
    context_provider=lambda page_id: {
        "cwd": str(Path.cwd()),
        "page_id": page_id,
        "status": "failed",
        "error": "optional error summary",
        "log_tail": "optional selected log tail",
    },
)

@app.get("/")
def index():
    return render_template("index.html", debug_console_token=console.token)
```

Load the module assets from the registered blueprint:

```html
<meta name="web-debug-token" content="{{ debug_console_token }}">
<link rel="stylesheet"
      href="{{ url_for('web_debug_console.static', filename='vendor/xterm/xterm.css') }}">
<link rel="stylesheet"
      href="{{ url_for('web_debug_console.static', filename='debug-console.css') }}">
<script type="module"
        src="{{ url_for('web_debug_console.static', filename='debug-console.js') }}"></script>
```

The current Video Analyzer integration maps `job` to the job `run_dir`, failed
stage, error, selected stage log path, and the last 160 log lines.

## Security Boundary

- Every API request requires the process-local capability token embedded in the
  same-origin page.
- Requests are accepted only from loopback or the Tailscale CGNAT range.
- A mismatched `Origin` is rejected.
- Requested working directories must stay inside the configured project root.
- Codex Debug supports only `read-only` and `workspace-write`; it uses
  `approvalPolicy=never` and does not expose `danger-full-access`.
- Server-side approval requests are declined because the current web module
  does not implement an approval review UI.
- Page unload requests close active sessions, and the server reaps sessions
  idle for `WEB_DEBUG_SESSION_TTL_SECONDS` (default 3600 seconds).

The host page still needs its own user authentication when it is shared with
multiple people. The capability token is a CSRF boundary, not user identity.

## Vendored Open Source

- `@xterm/xterm` 6.0.0, MIT
- `@xterm/addon-fit` 0.11.0, MIT

Their license files are stored under
`web_debug_console/static/vendor/xterm/`.

## Verification

```bash
.venv/bin/python -m unittest \
  tests.test_web_debug_console \
  tests.test_video_analyzer_ui
```

The module test suite starts a real PTY, runs a shell command, checks the
working directory, verifies access controls, and covers the Debug API
lifecycle.
