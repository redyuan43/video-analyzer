# Video Analyzer for iOS

A native iOS shell around the analyzer web console, plus a share extension that
turns a shared video link into an analysis job.

The app reaches the analyzer over Tailscale, so the phone and the analyzer host
only need to be on the same tailnet — nothing is exposed publicly.

## What it is

| Piece | What it does |
| --- | --- |
| `VideoAnalyzer` | SwiftUI app hosting the full web console in a `WKWebView`. Inline video/audio playback, pull-to-refresh, back/forward swipes, downloads to the share sheet, and an in-app server-address setting. |
| `ShareExtension` | Appears in the iOS share sheet for web links. Confirms the URL, expands Bilibili multi-part videos through the batch API, and creates ordinary links directly — no app switch. |
| `Shared/` | `AppConfig` (server address, App Group) and `VideoAnalyzerClient` (health + job creation), used by both targets. |

Because the console runs as the real web app, every feature the desktop UI has
is present — console, Q&A, learning resources and settings. The narrow-screen
layout comes from `video-analyzer-ui/video_analyzer_ui/static/css/mobile.css`
and `static/js/mobile.js`, which serve the browser too.

## Build and install

Everything below runs on the Mac.

### 1. Configure

Edit [`Config/App.xcconfig`](Config/App.xcconfig):

```
DEVELOPMENT_TEAM = ABCDE12345
PRODUCT_BUNDLE_PREFIX = com.yourname
DEFAULT_SERVER_SCHEME = http
DEFAULT_SERVER_HOST = 100.91.42.28:5000
```

- `DEVELOPMENT_TEAM` — Xcode → Settings → Accounts → your team, or the Team ID
  at <https://developer.apple.com/account> under Membership Details.
- `PRODUCT_BUNDLE_PREFIX` — any reverse-DNS prefix you control. The bundle IDs
  and the App Group all derive from it.
- `DEFAULT_SERVER_SCHEME` and `DEFAULT_SERVER_HOST` — the analyzer's Tailscale
  address. They are only the defaults; the app lets you change the address at
  runtime.

### 2. Generate the project

```bash
cd ios && ./bootstrap.sh
```

This installs XcodeGen via Homebrew if needed and writes
`VideoAnalyzer.xcodeproj` from [`project.yml`](project.yml). The `.xcodeproj` is
generated and gitignored — edit `project.yml`, not the project file. Re-run
after changing `project.yml` or adding/removing a Swift file.

### 3. Register the App Group

Once, at <https://developer.apple.com/account/resources/identifiers>: create an
App Group named `group.<your-prefix>.videoanalyzer`. Xcode's automatic signing
handles the rest.

The App Group is what lets the share extension read the server address you set
in the app. Without it both still work, but the extension falls back to
`DEFAULT_SERVER_SCHEME` and `DEFAULT_SERVER_HOST`.

### 4. Run

```bash
open VideoAnalyzer.xcodeproj
```

Plug in the iPhone, pick it as the run destination, press Run. On first launch
the phone may ask you to trust the developer certificate under
Settings → General → VPN & Device Management.

### Command line alternative

```bash
xcodebuild -project ios/VideoAnalyzer.xcodeproj -scheme VideoAnalyzer -destination 'generic/platform=iOS' build
```

## Using it

**Requirement:** the iPhone must be connected to the tailnet — the Tailscale app
installed, logged in, and toggled on. Everything else fails with "连接不上分析服务".

- **Server address** — tap the server icon in the top-right corner. It accepts a
  bare `100.91.42.28:5000` or a MagicDNS name like `ai-x10drg:5000`; `http://`
  is filled in automatically. "测试连接" probes `/api/video-link/health` before
  you commit to it.
- **Share a video** — in Safari, YouTube or anywhere with a share sheet, pick
  *Video Analyzer*. Confirm the link, choose whether to start analysis
  immediately, tap 创建任务.
- **Pull down** on the console to reload.
- **Downloads** of report artifacts open the iOS share sheet, so they can be
  saved to Files.

## Notes

### Plain HTTP

Both targets set `NSAllowsArbitraryLoads`. The traffic runs inside Tailscale's
WireGuard tunnel, so it is already end-to-end encrypted; ATS would be
re-encrypting an encrypted link. The exception is broad rather than
per-domain because the server address is user-configurable.

To drop it, put the UI behind `tailscale serve` for a real ts.net certificate
and set `DEFAULT_SERVER_SCHEME = https` plus the corresponding
`DEFAULT_SERVER_HOST`.

### Signing expiry

With a paid Apple Developer account the build stays valid for a year. On a free
Apple ID it expires after 7 days and needs a re-run from Xcode.

### App icon

Generated, not hand-drawn:

```bash
.venv/bin/python tools/generate_app_icons.py
```

That script writes both the iOS `AppIcon-1024.png` and the web
`apple-touch-icon.png`, so the phone icon and the browser icon cannot drift.
