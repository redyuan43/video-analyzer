# Video Link Status Server

本地状态服务用于从浏览器启动 `video-link` 后台任务，并在统一首页查看阶段进度、核心分析子项、日志、失败原因、队列状态和最终产物。

## 启动

```bash
tools/run_video_link_status_server.sh restart
```

打开：

```text
http://127.0.0.1:5000/
```

运行状态、PID 和日志保存在：

```text
tmp/video-link-status/
```

## 页面参数

首页左侧固定显示常用参数：

- `video_url`：必填视频链接。
- `analysis_mode`：`auto` / `fast` / `balanced` / `deep` / `long-talk-fast`。
- `profile`：从 `config/config.json` 和 `video_analyzer/config/default_config.json` 的 `runtime_profiles` 读取；默认优先 `deepseek_v4_flash`。
- `run_name`：输出目录名，默认 `operation-manual`。
- `cookies_from_browser`：`chrome` / `none` / `edge` / `firefox` / `chromium` / `brave`。
- `skip_images`：跳过配图提示词和最终图片生成。

采集选项在折叠区：

- `keep_existing`：复用已有下载，默认开启。
- `include_subtitles`：下载并纳入字幕，默认开启。
- `prefer_subtitle_transcript`：有字幕时优先跳过 ASR。
- `include_comments`：下载并纳入评论，默认开启。
- `max_comments`：最多纳入评论数，默认 `30`。
- `subtitle_langs`：字幕语言优先级，默认 `zh-CN,zh-Hans,zh,en`。
- `refresh_context`：刷新描述、字幕和评论上下文。

模型、endpoint、OCR/VL/ASR 后端不在页面直接配置；需要改这些内容时，修改 runtime profile。任务创建后不会跳转到长尾 URL，首页保持 `/`，并可通过 `/?job=<job_id>` 直接选中任务。

## 队列与恢复

- `probe` / `prepare` 最多并行 2 个。
- `analyze-core` 一次只跑 1 个，避免抢占 ASR、Ray、GPU、OCR/VL 资源。
- `multidoc`、`deep-v2`、`image-prompts`、`final-publish` 各自一次只跑 1 个。
- 资源忙时阶段会显示 `queued`，而不是返回锁冲突失败。
- 服务重启后如果发现旧任务停在 `running`/`queued`，会检查记录的阶段进程：进程仍在则保持 `running`，进程已退出且产物不完整则标记 `failed`，最终 PDF 已完整则标记 `succeeded`。
- 最终发布阶段默认生成 Baoyu 最终图片并插入 Markdown，然后生成手机优先 PDF；不再默认生成长图 PNG。PDF 由 `tools/md_to_mobile_pdf.py` 使用 WeasyPrint 渲染。线性 `flowchart TD` 会转成手机友好的原生 HTML 流程图，其他 Mermaid 图可降级为高分辨率 PNG。
- 如需连续阅读长图，使用 `tools/run_video_doc_final_publish.sh RUN_DIR --finalize-only --long-png`，会额外生成 `<name>.long.png`，并裁掉 PDF 页间大块空白后纵向拼接。

## API

```bash
curl -fsS http://127.0.0.1:5000/api/video-link/options | python3 -m json.tool
```

创建并启动任务：

```bash
curl -fsS -X POST http://127.0.0.1:5000/api/video-link/jobs \
  -H 'Content-Type: application/json' \
  -d '{"video_url":"https://example.com/video","analysis_mode":"auto","profile":"deepseek_v4_flash","auto_start":true}'
```

列出最近任务：

```bash
curl -fsS http://127.0.0.1:5000/api/video-link/jobs | python3 -m json.tool
```

查看任务：

```bash
curl -fsS http://127.0.0.1:5000/api/video-link/jobs/<job_id> | python3 -m json.tool
```

查看首页并选中任务：

```text
http://127.0.0.1:5000/?job=<job_id>
```

续跑后台任务：

```bash
curl -fsS -X POST http://127.0.0.1:5000/api/video-link/jobs/<job_id>/run
```

查看日志尾部：

```bash
curl -fsS "http://127.0.0.1:5000/api/video-link/jobs/<job_id>/logs/analyze-core?tail=80"
```

复制完整日志使用同一个日志 API：

```bash
curl -fsS "http://127.0.0.1:5000/api/video-link/jobs/<job_id>/logs/analyze-core?full=1"
```

## 验证

修改状态服务后运行：

```bash
.venv/bin/python -m py_compile tools/video_link_status_server.py video-analyzer-ui/video_analyzer_ui/server.py tests/test_video_link_status_server.py tests/test_video_analyzer_ui.py
.venv/bin/python -m unittest tests.test_video_link_status_server tests.test_video_analyzer_ui
```
