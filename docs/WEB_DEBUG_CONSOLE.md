# Web Debug Console

`web_debug_console` is a reusable Flask module that adds two project-scoped
development surfaces:

- a browser terminal backed by a real PTY;
- a Debug conversation backed by `codex app-server --stdio`.

The browser terminal can start the login shell, Codex CLI, or Claude CLI when
the corresponding executable is installed. The Debug surface creates a
persisted Codex thread and injects the current page context only on the first
turn.

Debug history is partitioned by the current `job` query parameter. User
messages, Markdown assistant replies, and key command/file/error events are
written atomically to the user state directory. Reloading the page restores the
visible history, and the next message resumes the saved Codex `thread_id` so
the model context is preserved as well. The `新会话` action explicitly clears
the current job history and starts a fresh thread on the next message.

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
- Page unload requests close active processes without deleting persisted Debug
  history. The server reaps live sessions idle for
  `WEB_DEBUG_SESSION_TTL_SECONDS` (default 3600 seconds).

## Persistence

The default history location is:

```text
$XDG_STATE_HOME/web-debug-console/<project-hash>/
```

When `XDG_STATE_HOME` is unset, the module uses
`~/.local/state/web-debug-console/<project-hash>/`. Override it with
`WEB_DEBUG_HISTORY_DIR`. Each JSON file is mode `0600`; the directory is mode
`0700`. `WEB_DEBUG_HISTORY_LIMIT` controls the retained event count per job and
defaults to `500`.

The host page still needs its own user authentication when it is shared with
multiple people. The capability token is a CSRF boundary, not user identity.

## Vendored Open Source

- `@xterm/xterm` 6.0.0, MIT
- `@xterm/addon-fit` 0.11.0, MIT
- `markdown-it` 14.1.0, MIT
- `DOMPurify`, Apache-2.0 or MPL-2.0

Their license files are stored under `web_debug_console/static/vendor/`.

## Verification

```bash
.venv/bin/python -m unittest \
  tests.test_web_debug_console \
  tests.test_video_analyzer_ui
```

The module test suite starts a real PTY, runs a shell command, checks the
working directory, verifies access controls, and covers the Debug API
lifecycle, persistence, thread resume, and history clearing.
