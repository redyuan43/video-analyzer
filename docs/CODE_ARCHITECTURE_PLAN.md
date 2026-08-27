# 代码架构优化方案

> 状态：方案（待评审后按阶段实施）
> 范围：`video_analyzer` 主包、`tools/`、`video-analyzer-ui`、`video-analyzer-tune`、`web_debug_console`、`tests/`、根目录散落文件
> 原则：**每阶段保持功能一致，先固化基线，再逐步重构；小步提交、可测试、可回滚**

---

## 1. 现状分析

### 1.1 仓库总体结构

| 目录/文件 | 规模 | 定位 |
|---|---|---|
| `video_analyzer/` | 27,439 行（32 个模块） | 核心 Python 包，`setup.py` 发布 |
| `tools/` | 29,126 行（约 60 个文件） | 运维/流水线/模型服务脚本集合，同时承载“可复用 job 引擎” |
| `video-analyzer-ui/` | `server.py` 1,489 行 + 静态资源 | 独立 Flask 前端（pyproject） |
| `video-analyzer-tune/` | 独立包（setup.py） | 提示词/指标调优工具 |
| `web_debug_console/` | 根级包 | 调试控制台（WebSocket） |
| `tests/` | 604 个测试方法 | 主测试集（无 `__init__.py`） |
| 根目录 | `test_operation_manual.py`(82 个测试)、`test_prompt_loading.py` | 游离测试文件 |
| `docs/`、`ios/`、`.codex/skills/`、`.github/` | — | 文档 / iOS 客户端 / 技能 / CI |

### 1.2 `video_analyzer/` 模块职责与体量

| 模块 | 行数 | 职责 | 备注 |
|---|---|---|---|
| `cli.py` | 1,982 | 主入口 + 全流程编排 | 入口与业务混在一起，含大量辅助函数 |
| `model_settings.py` | 2,589 | 模型/运行时 profile 管理 | 巨型 |
| `skill_distillation.py` | 3,046 | 技能蒸馏 | 巨型 |
| `jetson_frames.py` | 2,298 | Jetson 分布式抽帧 | 巨型 |
| `study_guide.py` | 1,579 | 学习指南生成 | 巨型 |
| `url_context.py` | 1,385 | URL 上下文/下载 | 巨型 |
| `speaker_diarization.py` | 1,249 | 说话人分离 | 大 |
| `multidoc.py` | 1,165 | 多文档分析 | 大 |
| `ocr.py` | 1,062 | OCR 引擎 | 大 |
| `manual.py` | 1,037 | 操作手册生成 | 大 |
| `asr_providers.py` | 940 | ASR 提供商 | 大 |
| `skill_projects.py` | 943 | 技能项目存储 | 大 |
| 其余（config/analyzer/frame_selection 等） | — | 支撑模块 | 中/小 |

### 1.3 `tools/` 脚本分类

- **薄 CLI 包装器**（纯 re-export）：`run_multidoc_analysis.py`、`run_study_guide.py`、`run_operation_manual_from_url.py`
- **大型独立流水线**：`run_audio_template_analysis.py`（3,196 行，未下沉主包）
- **可复用 job 引擎**：`video_link_status_server.py`（12,091 行，被 UI `from tools.video_link_status_server import ...` 反向依赖）
- **本地模型服务/代理**：`vibevoice_asr_http_server.py`、`minicpm_p40_proxy.py`、`firered_asr2_p40_proxy.py`、`qwen3_asr_p40_proxy.py`、`nx2_*_http_server.py`、`bonsai_local_pool.py` 等
- **基准/诊断**：`bench_*.py`、`benchmark_*.py`、`run_diarization_benchmark.py`
- **shell 运维脚本**：`start_*.sh`、`run_*.sh`、`prepare_*.sh`、`check_*.sh`

### 1.4 代码规范现状

- 根目录**无** `pyproject.toml` / `setup.cfg` / `.flake8` / `ruff` / `mypy` / `pre-commit` 等任何规范配置
- 仅依赖 `setup.py`（`find_packages()`）
- 类型注解风格不统一：部分 `Optional[X]`，部分 `X | None`
- 命名风格不一：模块内混用 snake_case/camelCase 变量、重复常量（如 `AUTO`、`OCR_AUTO`）

---

## 2. 核心问题清单

### P1 巨型单文件（可维护性差）
- `tools/video_link_status_server.py` 12,091 行：job 生命周期 + 移动音频任务 + 设置管理 + 资源队列 + 阶段编排全部混在一个类
- `tools/run_audio_template_analysis.py` 3,196 行：独立流水线未进入主包
- `video_analyzer/cli.py` 1,982 行：入口函数 `main()` 单函数承载全流程

### P2 目录组织与职责边界混乱
- `tools/` 同时承担“运维脚本目录”与“可复用引擎包”两种角色
- UI 反向依赖 `tools.video_link_status_server`（依赖方向倒置）
- 根目录散落 `web_debug_console/` 包、`test_operation_manual.py`、`test_prompt_loading.py`
- `tests/` 与根目录测试分离，标准 discover 不收集根目录测试

### P3 无统一代码规范与命名约定
- 无 lint/format/typing 工具链与配置
- 无统一 import 排序、常量命名、类型注解约定

### P4 代码冗余与重复逻辑
- `tools/run_multidoc_analysis.py`、`run_study_guide.py`、`run_operation_manual_from_url.py` 为 re-export 包装器，与 `video_analyzer.multidoc`/`study_guide`/`url_context` 重复
- 大量脚本重复“注入仓库根目录到 sys.path”样板
- 根目录测试与 `tests/` 测试职责重叠

### P5 测试组织不统一
- 无 `tests/__init__.py`，`unittest discover -s tests` 与 AGENTS.md 指定的 `tests.test_xxx` 模块导入方式不一致
- 根目录 82 个测试方法游离于标准收集之外

---

## 3. 目标架构

### 3.1 分层与依赖规则

```
┌─────────────────────────────────────────────┐
│ 入口层  cli.py / tools/*.sh / UI            │  ← 只负责参数解析与编排
├─────────────────────────────────────────────┤
│ 应用层  pipeline.*  (manual/multidoc/...)   │  ← 业务流水线，可互相调用
├─────────────────────────────────────────────┤
│ 能力层  asr/ocr/frame/model/runtime/*       │  ← 领域能力，无流水线依赖
├─────────────────────────────────────────────┤
│ 基础层  clients/ config/ artifacts/...      │  ← 无内部业务依赖
└─────────────────────────────────────────────┘
```

依赖规则：
1. 禁止反向依赖：`tools/` 是**入口/运维层**，不得作为业务包被 `video_analyzer`/UI 依赖
2. 能力层不依赖应用层
3. 一个模块只做一件事；公共逻辑下沉基础层

### 3.2 目标目录结构

```
video_analyzer/
  __init__.py
  cli.py                      # 瘦身为参数解析 + 编排
  config.py                   # Config / runtime profile
  clients/                    # llm_client / generic_openai_api / ollama
  pipeline/
    audio/                    # asr_providers, transcription_pipeline,
                              # speaker_diarization, audio_processor, tencent_hy_asr
    frames/                   # frame, frame_selection, candidate_frame_strategies,
                              # frame_manifest, jetson_frames, ocr, ocr_keyframes
    analysis/                 # analyzer, analysis_progress
    documents/                # manual, multidoc, study_guide, review_artifacts
    context/                  # url_context, doc_chat, qa_index
  runtime/                    # local_model_runtime, resource_locks
  model_settings.py           # （拆分后保留）
  skill_distillation.py       # （拆分后保留，或下沉 skills/）
  artifacts.py / failures.py  # 基础层
  prompt.py / prompts/        # 提示词

tools/
  asr_servers/                # vibevoice / firered / qwen3 / nx2 服务
  ocr_servers/                # minicpm / nx2_ocr
  pipelines/                  # run_*.sh / run_*.py（仅入口）
  publish/                    # export/ /pdf 相关
  benchmarks/                 # bench_*.py
  ops/                        # 系统服务 / 维护脚本
  video_link/                 # 引擎入口与 supervisor（引擎本体移入正式包）

web_debug_console/            # 并入 video_analyzer_ui 或独立发布包（待定）
```

### 3.3 模块划分矩阵（现有 → 目标）

| 现有 | 目标归属 | 动作 |
|---|---|---|
| `tools/video_link_status_server.py`（引擎本体） | 新包 `video_analyzer_ui/video_link/` 或 `video_analyzer/jobengine/` | 迁移 + 修复 UI 依赖方向 |
| `tools/video_link_status_supervisor.py` | `tools/video_link/` 入口 | 迁移 |
| `tools/run_audio_template_analysis.py`（逻辑） | `video_analyzer/pipeline/audio/template_analysis.py` | 下沉主包 |
| `tools/run_multidoc_analysis.py` 等 re-export | 删除包装器，入口改用主包 | 去重 |
| 根目录 `test_operation_manual.py` / `test_prompt_loading.py` | `tests/` | 迁移统一 |
| `web_debug_console/` | 并入 UI 包 | 迁移 |

---

## 4. 代码规范与命名约定（Phase 1 落地）

1. **工具链**：新增 `pyproject.toml`，引入 `ruff`（lint + format）作为统一规范，`python_requires = ">=3.11"`，`unittest` 作为测试后端，兼容现有 `setup.py`
2. **lint 规则（渐进式）**：`E,F,I` 全开；`UP`（pyupgrade）开启但分批修复；先 `ruff check` 无新增告警，存量告警逐步清零
3. **命名约定**
   - 模块：`snake_case`；类：`PascalCase`；函数/变量：`snake_case`；常量：`UPPER_SNAKE_CASE`；私有：`_` 前缀
   - 模块级 `AUTO` 等通用常量统一归入所在领域常量区，避免跨模块重名歧义
4. **类型注解**：统一 `X | None`（Python ≥3.11），公共函数必须完整注解参数与返回值
5. **import 顺序**：stdlib → 第三方 → 本地（由 ruff isort 强制）
6. **文档字符串**：模块级 docstring 说明职责；公共函数 docstring 说明入参/返回
7. **测试规范**：`tests/` 内每个被测模块对应 `test_<module>.py`；新增行为必须有对应单测

---

## 5. 分阶段实施计划

> 每阶段结束都跑：`py_compile` 全量 + 相关单测；P0 之后每次提交可回滚。

### 实施进度

- ✅ **Phase 0-2 已完成**（2026-08-27）
  - 基线固化：迁移后全量测试 688 个方法全部通过（原 tests/ 604 + 操作手册 82 + prompt 2）
  - 规范工具链：新增 [pyproject.toml](../pyproject.toml)（ruff `E4/E7/E9/F/I` + per-file-ignores），全仓 `ruff check` 0 错误
  - CI：新增 [.github/workflows/ci.yml](../.github/workflows/ci.yml)（ruff + 确定性单测门禁）
  - 测试统一：新增 `tests/__init__.py`；`test_operation_manual.py`、`test_prompt_loading.py` 迁入 `tests/`
  - 顺带修复：`tools/video_link_status_server.py` 缺失 `requests` 导入的潜在 NameError；清除 60+ 处未用导入/导入排序问题
- ⏳ **Phase 3-6 待实施**（每阶段独立可回滚）

### Phase 0 基线固化（低风险）
- 运行全量测试并记录基线：`tests/` 604 + 根目录 82，当前全部通过
- 固化验证命令：`py_compile` + `unittest discover`

### Phase 1 代码规范与工具链（低风险，先行）
- 新增 `pyproject.toml`（ruff 配置、测试发现配置、`[tool.setuptools]` 元数据）
- 配置 CI（`.github/workflows/`）增加 ruff check + 全量测试 job
- 新增 `tests/__init__.py`，统一测试模块导入方式
- 收益：建立“统一的代码规范与命名约定”，为后续重构提供自动化护栏

### Phase 2 测试组织统一（低风险）
- 将根目录 `test_operation_manual.py`、`test_prompt_loading.py` 迁入 `tests/`
- 修正 `test_operation_manual.py` 中基于 `__file__` 的路径解析（`parent` → `parents[1]`）
- 统一：一条命令跑全部测试；消除游离测试

### Phase 3 目录与依赖方向修正（中风险）
- 将 `video_link_status_server.py` 引擎本体移出 `tools/` 移入正式包（如 `video_analyzer_ui/video_link/`），修复 UI `from tools.xxx import` 的反向依赖
- 将 `tools/` 按 3.2 目标结构分目录，同步更新 `run_*.sh`、systemd、skills 中的路径引用
- 保留薄包装器到最终收敛，先保证行为不变

### Phase 4 巨型模块拆分（高风险，分多次提交）
- `cli.py`：把 `main()` 拆为 `pipeline/operation_manual.py` 等子流水线 + 参数解析留在 `cli.py`
- `video_link_status_server.py`：按职责拆出 `jobs/`（生命周期）、`mobile_audio/`、`settings/`、`stages/`
- `run_audio_template_analysis.py`：逻辑下沉 `video_analyzer/pipeline/audio/`
- 每个拆分子任务先抽公共函数 → 单测覆盖 → 再移动

### Phase 5 冗余消除（中风险）
- 删除/收敛 re-export 包装器，入口脚本改为直接调用主包
- 抽取统一 `repo_root/sys.path` 引导样板为公共 util，删除 40+ 处重复插入
- 常量去重、重复辅助函数下沉基础层

### Phase 6 收尾（低风险）
- 全量回归：`py_compile` + 全部单测 + 手工冒烟（CLI 帮助/UI 启动）
- 更新 `readme.md` 项目结构说明与 `docs/` 相关章节
- 提交并推送到 Git（用户确认后）

---

## 6. 验证与回归策略

- **每次提交**：`python -m py_compile <变更文件>` + 受影响单测
- **每阶段结束**：`.venv/bin/python -m unittest discover -s tests -p "test_*.py"`（含根目录迁移后全部测试）全绿
- **关键路径冒烟**：`video-analyzer --help`、UI `server.py` 可导入、`tools/run_*` 包装器可导入
- **回归护栏**：Phase 1 引入 ruff + CI 后，任何破坏 lint/测试的提交在合并前被拦截

## 7. 风险与回滚

| 风险 | 缓解 |
|---|---|
| 巨型模块拆分破坏 import 图 | 每个拆分点先抽函数+单测再移动；保留兼容 re-export 到下一阶段 |
| `tools/` 路径被 shell/systemd/skills 引用 | Phase 3 改动前先全仓库 grep 引用清单，同步更新 |
| 全量重构一次性引入大量回归 | 严格分阶段，每阶段独立可回滚提交 |
| 本地模型服务脚本改动影响线上运行 | 模型服务类脚本（`vibevoice/minicpm/...`）仅移动不重构，行为零改动 |
