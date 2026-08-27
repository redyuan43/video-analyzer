# Phase 4 巨型模块拆分变更说明

日期：2026-08-27
范围：`video_analyzer/jobengine/`（video-link 状态引擎）与 `video_analyzer/cli.py`（分析 CLI）
原则：行为不变、对外 API 不变、兼容 re-export 保留到下一阶段收敛。

## 1. 背景

Phase 3a/3b 完成目录迁移后，两处巨型模块仍是主要维护痛点：

- `video_analyzer/jobengine/video_link_status_server.py`：约 11,600 行单文件，`VideoLinkStatusServer` 单类承载设置、移动音频作业、后台循环、阶段执行全部职责。
- `video_analyzer/cli.py`：1,983 行，`main()` 约 1,300 行，参数定义、checkpoint、client 构建、三阶段流水线编排全部内联。

## 2. jobengine 拆分（主模块 11,600 → 10,323 行，新模块 1,972 行）

采用 mixin 切片方案：主类 `VideoLinkStatusServer` 继承各功能 mixin，对外 API 完全不变。

### 新增模块

| 模块 | 行数 | 职责 |
|---|---|---|
| `jobengine/errors.py` | 13 | `BridgeError` 异常 |
| `jobengine/_shared.py` | 236 | 跨切片共享常量与纯函数：重试消息（`ORPHANED_PROCESS_*`、`TRANSIENT_*`、`YOUTUBE_FORMAT_*`、`MANUAL_RERUN_*`、`AUTO_RETRY_REASONS`）、阶段别名（`STAGE_ALIASES`）、阶段资源映射（`STAGE_RESOURCES`、`RESOURCE_LIMITS`、`job_stage_resource`）、`parse_schedule_datetime`、`is_youtube_url`、`process_alive`、`iso_from_timestamp` 等 |
| `jobengine/settings.py` | 131 | `SettingsMixin`：模型与 runtime profile 设置 |
| `jobengine/mobile_audio.py` | 316 | `MobileAudioMixin`：移动音频作业（12 个方法） |
| `jobengine/background_loops.py` | 420 | `BackgroundLoopsMixin`：auto-retry 循环、定时调度循环、后台 TTS 循环 |
| `jobengine/stage_runner.py` | 869 | `StageRunnerMixin`：阶段执行、资源等待、失败分类与重试、阶段命令构建 |

### 主模块变化

- `VideoLinkStatusServer(SettingsMixin, MobileAudioMixin, BackgroundLoopsMixin, StageRunnerMixin)` 组合继承。
- 已迁移的常量/函数定义从主模块删除，统一改为 `from ._shared import ...`。
- 未迁移部分（job 持久化、HTTP 路由、stage_probe 等）仍留在主模块，等待后续按需再切。

### 测试同步

- `tests/test_video_link_status_server.py`：
  - `missing_tencent_credentials` 的 patch 目标从 `server_mod` 改为 `stage_runner_mod`（方法随切片迁移）。
  - `media_has_video_stream` 的 `subprocess.run` patch 目标从 `cli_mod` 改为 `cli_helpers_mod`。

## 3. cli.py 拆分（1,983 → 1,283 行）

cli 无子命令，是单一三阶段流水线（audio/frames→ocr→vl→manual→write）。`main()` 内约 25 个跨阶段共享可变状态变量，整段搬移风险高，因此采用低风险两步拆分：

### 新增模块

| 模块 | 行数 | 职责 |
|---|---|---|
| `cli_helpers.py` | 645 | 24 个模块级 helper：参数解析（`parse_auto_int_arg` 等）、transcript 读取、OCR/VL checkpoint 读写、签名 payload、VL 帧分析 `analyze_frames_for_vl`、client 构建（`create_client`、`create_operation_manual_text_client` 等）、`media_has_video_stream` |
| `cli_parser.py` | 145 | `build_arg_parser()`：argparse 参数面，65 个参数定义不变 |

### 兼容性

- `cli.py` 顶部回导全部 helper 名字（`from .cli_helpers import ...`），`from video_analyzer.cli import X` 的既有调用方（如 `tools/pipelines/resume_operation_manual_from_frames.py`、tests）无需改动。
- `python -m video_analyzer.cli` 入口行为不变。

### 未做（明确推迟）

- `main()` 三阶段流水线本体未再切片：共享状态多、回归面大，收益低于风险，留待有单测覆盖后再评估。
- 原计划中 `run_audio_template_analysis.py` 下沉 `video_analyzer/pipeline/audio/` 一项未实施。

## 4. tools/ 旧路径与 shim 同步检查结果

- `tools/`、`video-analyzer-ui/` 中无对被移动符号的旧路径引用。
- `tools/run_video_link_status_server.sh` 旧路径 shim 正常转发到 `tools/video_link/`。
- `tools/pipelines/resume_operation_manual_from_frames.py` 引用的 4 个 cli 符号（`analyze_frames_for_vl`、`create_client`、`create_operation_manual_text_client`、`read_page_context_metadata`）经回导验证 import 通过。
- `tools/pipelines/run_s36ri23_fast_full.sh` 使用 `-m video_analyzer.cli`，模块入口不变，无需改动。

## 5. 验证结果

| 验证项 | 结果 |
|---|---|
| `py_compile` 全部新模块 + 主模块 | 通过 |
| `tests.test_video_link_status_server` + `tests.test_video_analyzer_ui`（267 测试） | 全过 |
| cli 相关测试（`test_video_link_status_server` + `test_vl_strategy` + `test_manual_quality` + `test_analysis_progress`，251 测试） | 全过 |
| UI 服务真实重启 `tools/video_link/run_video_link_status_server.sh restart` | 成功，首页 HTTP 200 |
| `python -m video_analyzer.cli --help` | 正常 |
| `build_arg_parser()` 冒烟（65 个 action，参数解析） | 通过 |

### 已知非回归问题（改动前即存在）

- `tests.test_openai_client` 2 个失败：本地 `config/config.json` 覆盖导致（`git stash` 后原代码同样失败），与本次拆分无关。
- `tests.test_operation_manual` 单跑会等待真实 DotsMOCR 服务冷启动（环境依赖），未在本次回归中完成。

## 6. 文件清单

新增：
- `video_analyzer/jobengine/errors.py`、`_shared.py`、`settings.py`、`mobile_audio.py`、`background_loops.py`、`stage_runner.py`
- `video_analyzer/cli_helpers.py`、`video_analyzer/cli_parser.py`
- `docs/PHASE_4_MIGRATION_REPORT.md`（本文档）

修改：
- `video_analyzer/jobengine/video_link_status_server.py`（删迁移项、挂 mixin、扩 `_shared` 导入）
- `video_analyzer/cli.py`（helper/parser 外移、回导、`build_arg_parser()` 调用）
- `tests/test_video_link_status_server.py`（两处 patch 目标同步）
- `docs/CODE_ARCHITECTURE_PLAN.md`（Phase 4 状态更新）
