# Phase 3b 变更说明：tools/ 分目录迁移与兼容 shim

> 状态：已完成 ✅（真实重启验证通过）
> 时间：2026-08-27
> 范围：`tools/` 按目标结构（[CODE_ARCHITECTURE_PLAN.md](CODE_ARCHITECTURE_PLAN.md) §3.2）分目录，保留根目录向后兼容 shim
> 原则：**行为零改动、小步提交、可测试、可回滚**；模型服务类脚本仅移动不重构

---

## 1. 背景与目标

Phase 3a 已把可复用 job 引擎 `video_link_status_server.py` 迁入正式包 `video_analyzer/jobengine/`，消除了 UI 对 `tools` 的反向依赖。Phase 3b 承接 Phase 3 的剩余部分：把 `tools/` 内约 60 个脚本按职责分入子目录，使 `tools/` 收敛为纯粹的「入口/运维层」。

目标结构：

```
tools/
  asr_servers/    # vibevoice / firered / qwen3 / nx2 ASR 服务
  ocr_servers/    # minicpm / nx2_ocr 服务
  pipelines/      # run_*.sh / run_*.py（流水线入口）
  publish/        # export / pdf / 配图 相关
  benchmarks/     # bench_*.py / benchmark_*.py
  ops/            # 系统服务 / 维护脚本
  video_link/     # 引擎入口与 supervisor（引擎本体在正式包）
```

**关键约束**：迁移后必须在旧路径保留可运行且可导入的 shim，保证仓库内外部（shell、systemd、`.codex/skills/`、docs、CI、存量调用方）在收敛前不受破坏。

---

## 2. 提交清单（6 个批次，每个独立可回滚）

| 提交 | 批次 | 迁移内容 |
|---|---|---|
| `0c983bc` | ① | benchmarks/publish |
| `964e021` | ② | ASR servers |
| `106c1cc` | ③ | OCR servers |
| `55be899` | ④ | pipelines |
| `46f8fa5` | ⑤ | ops |
| `63a3955` | ⑥ | video_link |

### 批次① `0c983bc` — benchmarks/publish
- `git mv` 6 个 benchmark 脚本 → `tools/benchmarks/`；9 个 publish 脚本 → `tools/publish/`
- 修正迁移后脚本的 `REPO_ROOT`/`ROOT_DIR` 深度计算（`parents[2]`、`/../..`）
- 同步更新引擎 `video_link_status_server` 与 `speaker_diarization` 的引用路径
- 在 `tools/` 根目录创建 2 个 shell + 7 个 Python 向后兼容 shim（可运行 + 可导入）
- 更新测试与文档（AGENTS.md / readme / docs）指向新路径

### 批次② `964e021` — ASR servers
- 9 个 ASR 服务/工具脚本 → `tools/asr_servers/`
- 更新测试中 `from tools.xxx import` 引用
- 旧路径保留 shim

### 批次③ `106c1cc` — OCR servers
- `minicpm_p40_proxy.py`、`nx2_easyocr_openai_server.py` → `tools/ocr_servers/`
- 同步 `start_minicpm_p40_service.sh` 与相关测试

### 批次④ `55be899` — pipelines
- 14 个流水线脚本（`run_*.sh`/`run_*.py`、`generate_audio_narration.*`、`regenerate_*`、`resume_*` 等）→ `tools/pipelines/`
- `run_audio_template_analysis.py`（3,196 行）作为大型独立流水线迁入 `tools/pipelines/`
- 更新 AGENTS.md / docs / readme / `start_example.sh` / skills 中的路径引用

### 批次⑤ `46f8fa5` — ops
- 运维脚本（`prepare_ai_local_model_stage.sh`、`run_local_model_stage.py`、`check_jetson_frame_workers.sh`、`start_jetson_frame_ray.sh`、`start_*_service.sh`、`ytdlp_runtime_maintenance.sh`、`bonsai_local_pool.py`、`operation_manual_no_proxy_env.sh` 等）→ `tools/ops/`
- 含 `tools/ops/systemd/` 下的 ytdlp 维护 service/timer
- 旧路径保留 shim

### 批次⑥ `63a3955` — video_link
- `run_video_link_status_server.sh`、`video_link_status_supervisor.py` → `tools/video_link/`
- 修正 launcher 的 `ROOT_DIR` 深度
- 更新 AGENTS.md / docs / tests 中的引擎引用

---

## 3. 兼容 shim 设计

所有 shim 同时满足「可直接执行」与「可被 Python 导入」两种用法：

- **Shell shim**（`.sh`）：`exec "$(dirname "$0")/子目录/原脚本" "$@"` 透传参数
- **Python shim**（`.py`）：`from 子目录.模块 import *` 并 re-export 模块级符号（含 `__all__`），保证 `from tools.xxx import Y` 与 `python tools/xxx.py` 两种形态均可用

迁移后 `tools/` 根目录共 **53 个 shim**，逐一校验目标实体文件存在，无悬空引用。

---

## 4. 验证结果

### 4.1 单元测试
- 每批次提交后运行受影响单测，全部通过
- 迁移过程中同步更新了测试中的路径引用（如 `tools.asr_servers`、`tools.pipelines` 等 import）

### 4.2 引用与路径完整性审计（全仓库扫描）
对源码、文档、测试、skills、shell 脚本做全量路径引用扫描，**96 个去重引用路径全部命中真实文件**（除 3 处旧引擎路径，均已在 Phase 3a/3b 收尾修正）。`docs/` 下全部路径引用已指向新目录；`AGENTS.md` 18 处工具引用全部为新路径。

### 4.3 真实重启验证（关键路径）
- `tools/video_link/run_video_link_status_server.sh restart` 真实重启 UI 服务成功，首页正常渲染
- 确认系统级调用方（systemd、launcher）走新路径正常拉起

### 4.4 遗留清理
- 保留 2 个历史备份文件（未跟踪、零引用）：`tools/bonsai_local_pool.py.backup-20260820-223012`、`tools/video_link_status_server.py.before-audio-tenants-20260820-113943`
- 无基准脚本外部引用：`benchmarks/` 6 个文件全仓零引用，无需 shim

---

## 5. 已知注意事项

- **备份文件**：`tools/` 根目录两个 `.backup-*` / `.before-*` 文件为历史遗留备份，不属于迁移产物，删除前需人工确认
- **文档 docstring**：`video_analyzer/jobengine/__init__.py` 与 `docs/CODE_ARCHITECTURE_PLAN.md` 中保留的历史路径描述为说明性内容，不影响运行时
- **收敛目标**：按 Phase 5 计划，待调用方全部切到新路径后，可删除根目录 shim 完成最终收敛

---

## 6. 下一步（Phase 3c 候选）

Phase 3b 仅完成 `tools/` 分目录。剩余迁移项见 Phase 3c 规划（主包内部按目标结构重组、`run_audio_template_analysis.py` 逻辑下沉主包、re-export 包装器去重、`web_debug_console` 归属收敛），另文档同步更新于 `docs/CODE_ARCHITECTURE_PLAN.md`。
