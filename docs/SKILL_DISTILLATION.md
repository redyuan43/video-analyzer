# Skills Distillation

The video-link UI can distill a completed analysis run into a reviewable set of
Codex skills. The workflow is a clean-room, RIA-TV++ compatible implementation
inspired by `kangarooking/cangjie-skill`; it does not copy that project's
AGPL-3.0 prompts, templates, or code.

## Runtime

- Default profile: `deepseek_v4_pro`
- Generation model: the selected profile's `text_model`
- Review model: `review_model`, falling back to `text_model`
- Output directory: `<run_dir>/skills/cangjie_pack`
- Enable target: `<repo>/.codex/skills/<skill-name>`

The main video-analysis profile is unchanged. Selecting another distillation
profile affects only this optional post-processing workflow.

## Evidence

The source bundle reads raw or directly indexed evidence:

- `orin/transcript.json`
- `orin/ocr_events.json`
- `orin/frame_analyses.json` or `orin/visual_events.json`
- `orin/page_context.md`
- `orin/comments.md` as low-confidence context

Generated manuals and study documents may help humans navigate the run but do
not independently satisfy evidence checks. `RUN_MANIFEST.md` is an index only
and cannot make a run eligible for distillation.

Transcript, OCR, and visual records from the same 30-second video window are
assigned to one `video-event`. Different modalities corroborating the same
moment do not count as independent V1 contexts.

Events from the same source video are also assigned to one primary `case_id`.
V1 counts independent cases, not timestamps: different steps of one project
remain one case even when they occur minutes apart.

Terms are delivered through `GLOSSARY.md`; they are not counted as failed skill
candidates. Non-term candidates are classified as:

- `verified`: V1, V2, and V3 pass with at least two independent evidence events.
- `single_case`: V2 and V3 pass, multimodal evidence does not contradict the
  claim, but only one independent evidence event is available.
- `rejected`: transfer, novelty, evidence, or multimodal grounding failed.

The default `deepseek_v4_pro` profile uses DeepSeek V4 Pro for text distillation
and review, then uses its configured MiniCPM vision model to re-open only the
frames associated with each non-term candidate. The vision audit receives at
most three frames plus the matching transcript and OCR evidence. Loopback vision
calls use the shared `local-model-runtime` lock and the normal `vl` stage switch.

## Stages

1. Source normalization and fingerprinting.
2. Chunk-level evidence understanding and `BOOK_OVERVIEW.md`.
3. Human overview confirmation.
4. Five isolated candidate extractors.
5. Evidence-event grouping, multimodal frame audit, triple verification, and
   human candidate selection.
6. Skill construction, relation linking, blind trigger tests, and bounded repair.
7. `INDEX.md`, `GLOSSARY.md`, `DIGEST.md`, and the delivery manifest.

State is stored in `PIPELINE_STATE.json` and mirrored in
`PIPELINE_STATE.md`. A server restart marks an active run interrupted; it does
not automatically resume model calls.

## Workspace

Open `Learning/Resources -> Skills Workspace` for the selected job. The
workspace has two responsibilities:

- `Current task` shows live stage progress, overview/candidate review, active
  generated skills, evidence records, key frames, multimodal audits, test
  prompts, test results, and raw JSON.
- `Enabled`, `Disabled`, and `Trash` manage the project skill library.

On desktop, drag the separators to resize the left list, center content, right
inspector, and main content height. Current-task and library layouts are stored
independently in browser local storage. Narrow screens fall back to the stacked
layout without horizontal overflow.

The Skills page merges resource navigation, library scope, search, and focus
mode into one compact toolbar. Focus mode hides the global application header
while keeping the Skills toolbar available; press `Esc` or use the focus button
to restore it. The preference persists across page reloads and applies only
while the Skills workspace is active.

While distillation is running, the workspace shows an animated progress sweep,
an activity signal, the current model substep, elapsed time, last state update,
and an eight-stage rail. Waiting-for-review states remain static and clearly
request human action. Motion is disabled when the browser requests reduced
motion.

Generated skills under the current task are read-only. In the project library,
only `SKILL.md` is editable. The frontmatter `name` is immutable and auxiliary
files are read-only. Every successful save snapshots the previous file under
`.codex/skills-history/<name>/<timestamp>/SKILL.md`.

Disabling a skill moves it to `.codex/skills-disabled`. Deleting an enabled or
disabled skill moves it to `.codex/skills-trash`; permanent deletion is
available only from Trash and requires typing the exact skill name.

## API

- `GET /api/video-link/jobs/<job_id>/skill-distillation`
- `GET /api/video-link/jobs/<job_id>/skill-distillation/workspace`
- `GET /api/video-link/jobs/<job_id>/skill-distillation/items/<item_id>`
- `POST /api/video-link/jobs/<job_id>/skill-distillation/start`
- `POST /api/video-link/jobs/<job_id>/skill-distillation/review-overview`
- `POST /api/video-link/jobs/<job_id>/skill-distillation/review-candidates`
- `POST /api/video-link/jobs/<job_id>/skill-distillation/resume`
- `POST /api/video-link/jobs/<job_id>/skill-distillation/cancel`
- `POST /api/video-link/jobs/<job_id>/skill-distillation/enable`
- `GET /api/skills?state=enabled|disabled|trash&query=...`
- `GET|PUT|DELETE /api/skills/<state>/<skill-id>`
- `POST /api/skills/enabled/<name>/disable`
- `POST /api/skills/disabled/<name>/restore`
- `POST /api/skills/trash/<skill-id>/restore`
- `GET /api/skills/<state>/<skill-id>/versions`
- `POST /api/skills/<state>/<skill-id>/versions/<version-id>/restore`

Existing `skill-candidate` routes remain compatibility aliases.

Enabling defaults to `overwrite=false`. If any target skill already exists,
the server returns `409` before installation. The caller must explicitly retry
with `{"overwrite": true}`.
