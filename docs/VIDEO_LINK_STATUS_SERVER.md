# Video Link Status Server

本地状态服务用于从浏览器启动 `video-link` 后台任务，并在状态页查看阶段进度、日志尾部、失败原因和最终产物。

## 启动

```bash
tools/run_video_link_status_server.sh restart
```

打开：

```text
http://127.0.0.1:18120/video-link
```

运行状态、PID 和日志保存在：

```text
tmp/video-link-status/
```

## 页面参数

默认显示常用参数：

- `video_url`：必填视频链接。
- `analysis_mode`：`auto` / `fast` / `balanced` / `deep` / `long-talk-fast`。
- `profile`：从 `config/config.json` 和 `video_analyzer/config/default_config.json` 的 `runtime_profiles` 读取；默认优先 `deepseek_v4_flash`。
- `run_name`：输出目录名，默认 `operation-manual`。
- `cookies_from_browser`：`chrome` / `none` / `edge` / `firefox` / `chromium` / `brave`。
- `skip_images`：跳过最后的配图提示词阶段。

采集选项在折叠区：

- `keep_existing`：复用已有下载，默认开启。
- `include_subtitles`：下载并纳入字幕，默认开启。
- `prefer_subtitle_transcript`：有字幕时优先跳过 ASR。
- `include_comments`：下载并纳入评论，默认开启。
- `max_comments`：最多纳入评论数，默认 `30`。
- `subtitle_langs`：字幕语言优先级，默认 `zh-CN,zh-Hans,zh,en`。
- `refresh_context`：刷新描述、字幕和评论上下文。

模型、endpoint、OCR/VL/ASR 后端不在页面直接配置；需要改这些内容时，修改 runtime profile。

## API

```bash
curl -fsS http://127.0.0.1:18120/api/video-link/options | python3 -m json.tool
```

创建并启动任务：

```bash
curl -fsS -X POST http://127.0.0.1:18120/api/video-link/jobs \
  -H 'Content-Type: application/json' \
  -d '{"video_url":"https://example.com/video","analysis_mode":"auto","profile":"deepseek_v4_flash","auto_start":true}'
```

查看任务：

```bash
curl -fsS http://127.0.0.1:18120/api/video-link/jobs/<job_id> | python3 -m json.tool
```

查看状态页：

```text
http://127.0.0.1:18120/video-link/jobs/<job_id>
```

续跑后台任务：

```bash
curl -fsS -X POST http://127.0.0.1:18120/api/video-link/jobs/<job_id>/run
```

查看日志尾部：

```bash
curl -fsS "http://127.0.0.1:18120/api/video-link/jobs/<job_id>/logs/analyze-core?tail=80"
```

## 验证

修改状态服务后运行：

```bash
python3 -m py_compile tools/video_link_status_server.py tests/test_video_link_status_server.py
python3 -m unittest tests.test_video_link_status_server
```
