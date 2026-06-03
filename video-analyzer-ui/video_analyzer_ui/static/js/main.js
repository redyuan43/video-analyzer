const stageNames = {
    probe: '探测时长',
    prepare: '下载/上下文',
    'analyze-core': '核心分析',
    'verify-core': '校验产物',
    multidoc: '多文档分析',
    'deep-v2': '章节深度报告',
    'study-guide': '学习证据账本',
    'evidence-review': '证据复核/发布门禁',
    'image-prompts': '生成配图提示词',
    'final-publish': '最终定稿/发布'
};

const initialParams = new URLSearchParams(window.location.search);
let selectedJobId = initialParams.get('job') || document.querySelector('.app-shell')?.dataset.initialJob || null;
let selectedLogStage = null;
let refreshTimer = null;
let scanAnimationFrame = null;
let currentJob = null;
let latestJobs = [];
let currentView = ['preview', 'vscode'].includes(initialParams.get('view'))
    ? initialParams.get('view')
    : 'console';
let pendingUrls = [];
let vscodeStarting = false;
let selectedDocPath = '';
let renderedDocListKey = '';
let loadedDocPreviewKey = '';
let loadedStudyKey = '';
const videoLoadErrors = {};
const imageViewer = {
    node: null,
    image: null,
    scaleLabel: null,
    scale: 1
};

const nodes = {
    consoleTab: document.getElementById('consoleTab'),
    previewTab: document.getElementById('previewTab'),
    vscodeTab: document.getElementById('vscodeTab'),
    consoleView: document.getElementById('consoleView'),
    previewView: document.getElementById('previewView'),
    vscodeView: document.getElementById('vscodeView'),
    jobForm: document.getElementById('jobForm'),
    formError: document.getElementById('formError'),
    createButton: document.getElementById('createButton'),
    batchResult: document.getElementById('batchResult'),
    videoUrlInput: document.getElementById('videoUrlInput'),
    addUrlButton: document.getElementById('addUrlButton'),
    videoUrls: document.getElementById('videoUrls'),
    urlList: document.getElementById('urlList'),
    focusPrompt: document.getElementById('focusPrompt'),
    globalSummary: document.getElementById('globalSummary'),
    resourceLanes: document.getElementById('resourceLanes'),
    refreshJobsButton: document.getElementById('refreshJobsButton'),
    jobList: document.getElementById('jobList'),
    runButton: document.getElementById('runButton'),
    selectedTitle: document.getElementById('selectedTitle'),
    selectedSubtitle: document.getElementById('selectedSubtitle'),
    statusValue: document.getElementById('statusValue'),
    currentStageValue: document.getElementById('currentStageValue'),
    nextStageValue: document.getElementById('nextStageValue'),
    queueValue: document.getElementById('queueValue'),
    progressText: document.getElementById('progressText'),
    progressBar: document.getElementById('progressBar'),
    errorPanel: document.getElementById('errorPanel'),
    errorTitle: document.getElementById('errorTitle'),
    errorMessage: document.getElementById('errorMessage'),
    detailUrl: document.getElementById('detailUrl'),
    detailRunDir: document.getElementById('detailRunDir'),
    detailMode: document.getElementById('detailMode'),
    detailUpdated: document.getElementById('detailUpdated'),
    stageRows: document.getElementById('stageRows'),
    stageDurationSummary: document.getElementById('stageDurationSummary'),
    corePanel: document.getElementById('corePanel'),
    corePanelTitle: document.getElementById('corePanelTitle'),
    coreRows: document.getElementById('coreRows'),
    coreDiagnosticsPanel: document.getElementById('coreDiagnosticsPanel'),
    coreDiagnosticsSummary: document.getElementById('coreDiagnosticsSummary'),
    coreDiagnosticsStatus: document.getElementById('coreDiagnosticsStatus'),
    coreDiagnosticsMetrics: document.getElementById('coreDiagnosticsMetrics'),
    coreDiagnosticsIssues: document.getElementById('coreDiagnosticsIssues'),
    artifactSummary: document.getElementById('artifactSummary'),
    logHint: document.getElementById('logHint'),
    logText: document.getElementById('logText'),
    copyLogButton: document.getElementById('copyLogButton'),
    copyMessage: document.getElementById('copyMessage'),
    refreshPreviewButton: document.getElementById('refreshPreviewButton'),
    previewSummary: document.getElementById('previewSummary'),
    previewGrid: document.getElementById('previewGrid'),
    vscodeSummary: document.getElementById('vscodeSummary'),
    startVscodeButton: document.getElementById('startVscodeButton'),
    openVscodeLink: document.getElementById('openVscodeLink'),
    restartVscodeButton: document.getElementById('restartVscodeButton'),
    stopVscodeButton: document.getElementById('stopVscodeButton'),
    vscodePlaceholder: document.getElementById('vscodePlaceholder'),
    vscodeFrame: document.getElementById('vscodeFrame'),
    docPreviewSummary: document.getElementById('docPreviewSummary'),
    docList: document.getElementById('docList'),
    docPreviewPanel: document.querySelector('.doc-preview-panel'),
    docPreviewTitle: document.getElementById('docPreviewTitle'),
    docOpenLink: document.getElementById('docOpenLink'),
    docPreviewClose: document.getElementById('docPreviewClose'),
    docPreviewBody: document.getElementById('docPreviewBody'),
    studySummary: document.getElementById('studySummary'),
    studyDecision: document.getElementById('studyDecision'),
    studyBody: document.getElementById('studyBody')
};

const jobStatusPriority = {
    running: 0,
    queued: 0,
    failed: 1,
    created: 2,
    pending: 2,
    succeeded: 3,
    skipped: 3
};

function setText(node, value) {
    node.textContent = value || '-';
}

function duration(value) {
    return value == null ? '-' : `${value}s`;
}

function durationMinutes(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds <= 0) return '-';
    const minutes = seconds / 60;
    if (minutes < 1) return `${seconds.toFixed(1)} 秒`;
    return `${minutes.toFixed(1)} 分钟`;
}

function totalStageDuration(job) {
    return (job.stage_order || []).reduce((total, stage) => {
        const value = Number(job.stages?.[stage]?.duration_seconds);
        return Number.isFinite(value) && value > 0 ? total + value : total;
    }, 0);
}

function stageDurationSummary(job) {
    const videoSeconds = Number(job.preview?.duration_seconds);
    const videoText = `原视频长度：${durationMinutes(videoSeconds)}`;
    const stageText = `阶段总耗时：${durationMinutes(totalStageDuration(job))}`;
    return `${videoText} · ${stageText}`;
}

function formatClock(value) {
    const seconds = Math.max(0, Number.isFinite(value) ? Math.floor(value) : 0);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remaining = seconds % 60;
    if (hours) return `${hours}:${String(minutes).padStart(2, '0')}:${String(remaining).padStart(2, '0')}`;
    return `${minutes}:${String(remaining).padStart(2, '0')}`;
}

function clampPercent(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(100, value));
}

function statusBadge(status) {
    const value = status || 'pending';
    const spinner = value === 'running' ? '<span class="status-spinner" aria-hidden="true"></span>' : '';
    return `<span class="status ${value}">${spinner}${escapeHtml(value)}</span>`;
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[char]));
}

async function getJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
}

function fillSelect(id, values, selected) {
    const node = document.getElementById(id);
    node.innerHTML = (values || []).map(value => `<option value="${value}">${value}</option>`).join('');
    node.value = selected || '';
}

function splitUrlInput(value) {
    return String(value || '').split(/[\s,]+/).map(item => item.trim()).filter(Boolean);
}

function syncUrlField() {
    nodes.videoUrls.value = pendingUrls.map(item => item.url).join('\n');
}

function focusPromptMap() {
    return Object.fromEntries(
        pendingUrls
            .filter(item => String(item.focus_prompt || '').trim())
            .map(item => [item.url, item.focus_prompt])
    );
}

function renderUrlList() {
    syncUrlField();
    nodes.urlList.innerHTML = pendingUrls.length ? pendingUrls.map((item, index) => `
        <div class="url-item">
            <span title="${escapeHtml(item.url)}">${escapeHtml(item.url)}</span>
            <button class="icon-button light remove-url" type="button" data-index="${index}" title="移除链接" aria-label="移除链接">×</button>
            <label class="url-focus">
                <span>关注重点</span>
                <textarea class="url-focus-input" data-index="${index}" rows="3" placeholder="这条视频分析时要特别关注的内容"></textarea>
            </label>
        </div>
    `).join('') : '<div class="muted">暂无链接</div>';
    document.querySelectorAll('.url-focus-input').forEach(textarea => {
        const item = pendingUrls[Number(textarea.dataset.index)];
        textarea.value = item?.focus_prompt || '';
        textarea.addEventListener('input', () => {
            const current = pendingUrls[Number(textarea.dataset.index)];
            if (current) current.focus_prompt = textarea.value;
        });
    });
    document.querySelectorAll('.remove-url').forEach(button => {
        button.addEventListener('click', () => {
            pendingUrls.splice(Number(button.dataset.index), 1);
            renderUrlList();
        });
    });
}

function addPendingUrls() {
    const urls = splitUrlInput(nodes.videoUrlInput.value);
    if (!urls.length) return;
    const known = new Set(pendingUrls.map(item => item.url));
    urls.forEach(url => {
        if (!known.has(url)) {
            pendingUrls.push({ url, focus_prompt: '' });
            known.add(url);
        }
    });
    nodes.videoUrlInput.value = '';
    renderUrlList();
}

async function loadOptions() {
    const options = await getJson('/api/video-link/options');
    const defaults = options.defaults || {};
    const choices = options.choices || {};
    fillSelect('analysisMode', choices.analysis_modes, defaults.analysis_mode);
    fillSelect('profile', choices.profiles, defaults.profile);
    fillSelect('cookieBrowser', choices.cookie_browsers, defaults.cookies_from_browser);
    fillSelect('downloadDevice', choices.download_devices, defaults.download_device);
    document.getElementById('runName').value = defaults.run_name || 'operation-manual';
    document.getElementById('skipImages').checked = Boolean(defaults.skip_images);
    document.getElementById('keepExisting').checked = Boolean(defaults.keep_existing);
    document.getElementById('includeSubtitles').checked = Boolean(defaults.include_subtitles);
    document.getElementById('preferSubtitleTranscript').checked = Boolean(defaults.prefer_subtitle_transcript);
    document.getElementById('includeComments').checked = Boolean(defaults.include_comments);
    document.getElementById('refreshContext').checked = Boolean(defaults.refresh_context);
    nodes.focusPrompt.value = defaults.focus_prompt || '';
    document.getElementById('maxComments').value = defaults.max_comments ?? 3000;
    document.getElementById('subtitleLangs').value = defaults.subtitle_langs || '';
}

function jobPayload() {
    addPendingUrls();
    return {
        video_urls_text: pendingUrls.map(item => item.url).join('\n'),
        focus_prompt: nodes.focusPrompt.value.trim(),
        focus_prompts: focusPromptMap(),
        analysis_mode: document.getElementById('analysisMode').value,
        profile: document.getElementById('profile').value,
        run_name: document.getElementById('runName').value.trim(),
        cookies_from_browser: document.getElementById('cookieBrowser').value,
        download_device: document.getElementById('downloadDevice').value,
        skip_images: document.getElementById('skipImages').checked,
        keep_existing: document.getElementById('keepExisting').checked,
        include_subtitles: document.getElementById('includeSubtitles').checked,
        prefer_subtitle_transcript: document.getElementById('preferSubtitleTranscript').checked,
        include_comments: document.getElementById('includeComments').checked,
        refresh_context: document.getElementById('refreshContext').checked,
        max_comments: Number(document.getElementById('maxComments').value || 0),
        subtitle_langs: document.getElementById('subtitleLangs').value.trim(),
        auto_start: true
    };
}

async function createJob(event) {
    event.preventDefault();
    nodes.formError.textContent = '';
    nodes.batchResult.textContent = '';
    nodes.createButton.disabled = true;
    try {
        const result = await getJson('/api/video-link/jobs/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(jobPayload())
        });
        const jobs = result.jobs || [];
        const focusJob = preferredJob(jobs);
        if (focusJob) selectJob(focusJob.job_id, true);
        nodes.batchResult.textContent = `已创建 ${result.created || 0}/${result.total || 0} 个任务${result.failed ? `，失败 ${result.failed} 个` : ''}`;
        if (result.errors?.length) {
            nodes.formError.textContent = result.errors.map(item => `${item.index || '-'}: ${item.error}`).join('；');
        }
        nodes.jobForm.reset();
        pendingUrls = [];
        renderUrlList();
        await loadOptions();
        await refreshJobs();
    } catch (error) {
        nodes.formError.textContent = error.message;
    } finally {
        nodes.createButton.disabled = false;
    }
}

function selectJob(jobId, updateUrl = true) {
    selectedJobId = jobId;
    selectedLogStage = null;
    if (updateUrl) {
        const url = new URL(window.location.href);
        url.searchParams.set('job', jobId);
        if (currentView !== 'console') url.searchParams.set('view', currentView);
        window.history.replaceState({}, '', url);
    }
    refreshSelectedJob();
}

function setView(view, updateUrl = true) {
    currentView = ['preview', 'vscode'].includes(view) ? view : 'console';
    nodes.consoleView.hidden = currentView !== 'console';
    nodes.previewView.hidden = currentView !== 'preview';
    nodes.vscodeView.hidden = currentView !== 'vscode';
    nodes.consoleView.classList.toggle('active', currentView === 'console');
    nodes.previewView.classList.toggle('active', currentView === 'preview');
    nodes.vscodeView.classList.toggle('active', currentView === 'vscode');
    nodes.consoleTab.classList.toggle('active', currentView === 'console');
    nodes.previewTab.classList.toggle('active', currentView === 'preview');
    nodes.vscodeTab.classList.toggle('active', currentView === 'vscode');
    if (updateUrl) {
        const url = new URL(window.location.href);
        if (currentView !== 'console') {
            url.searchParams.set('view', currentView);
        } else {
            url.searchParams.delete('view');
        }
        window.history.replaceState({}, '', url);
    }
    renderPreviewGrid(latestJobs);
    if (currentView === 'preview') startScanAnimation();
    renderVscodePanel(currentJob);
}

function jobTimeValue(job) {
    const value = job.updated_at || job.created_at || '';
    const parsed = Date.parse(value);
    return Number.isNaN(parsed) ? 0 : parsed;
}

function sortJobsForAttention(jobs) {
    return [...jobs].sort((left, right) => {
        const leftPriority = jobStatusPriority[left.status || 'created'] ?? 2;
        const rightPriority = jobStatusPriority[right.status || 'created'] ?? 2;
        if (leftPriority !== rightPriority) return leftPriority - rightPriority;
        return jobTimeValue(right) - jobTimeValue(left);
    });
}

function preferredJob(jobs) {
    return sortJobsForAttention(jobs)[0] || null;
}

function jobDisplayTitle(job) {
    return job.display_title || job.title || job.summary?.study?.title || job.video_url || job.job_id || '-';
}

function renderJobList(jobs) {
    const orderedJobs = sortJobsForAttention(jobs);
    nodes.jobList.innerHTML = orderedJobs.length ? orderedJobs.map(job => {
        const selected = job.job_id === selectedJobId ? ' selected' : '';
        const statusClass = job.status ? ` ${job.status}` : '';
        const title = escapeHtml(jobDisplayTitle(job));
        const url = escapeHtml(job.video_url || '-');
        const stage = job.current_stage || job.error_summary?.stage || job.next_stage || '-';
        return `<div class="job-item${selected}${statusClass}" data-job-id="${escapeHtml(job.job_id)}">
            <button class="job-select" type="button" data-job-id="${escapeHtml(job.job_id)}">
                <strong title="${title}">${title}</strong>
                <span class="job-url" title="${url}">${url}</span>
                <span class="job-status-line">${escapeHtml(job.status || '-')} · ${escapeHtml(stageNames[stage] || stage)}</span>
            </button>
            <button class="icon-button light delete-job" type="button" data-job-id="${escapeHtml(job.job_id)}" title="删除任务" aria-label="删除任务">×</button>
        </div>`;
    }).join('') : '<div class="empty">暂无任务</div>';
    bindJobButtons();
}

function mergeSelectedJobSnapshot(snapshot) {
    if (!snapshot || currentJob?.job_id !== snapshot.job_id) return snapshot;
    const merged = { ...currentJob, ...snapshot };
    [
        'summary',
        'core_diagnostics',
        'preview',
        'vscode_preview',
        'warnings',
        'queue',
        'core_progress',
        'stage_progress'
    ].forEach(key => {
        if (snapshot[key] === undefined && currentJob[key] !== undefined) {
            merged[key] = currentJob[key];
        }
    });
    return merged;
}

function renderSelectedJobSnapshot(jobs) {
    const snapshot = selectedJobId ? jobs.find(job => job.job_id === selectedJobId) : null;
    if (!snapshot) return;
    renderJob(mergeSelectedJobSnapshot(snapshot));
}

async function refreshJobs() {
    const data = await getJson('/api/video-link/jobs?limit=50');
    renderGlobal(data);
    const jobs = data.jobs || [];
    latestJobs = jobs;
    renderJobList(jobs);
    renderSelectedJobSnapshot(jobs);
    renderPreviewGrid(jobs);
    const first = preferredJob(jobs);
    if (!selectedJobId && first) selectJob(first.job_id, true);
}

async function refreshSelectedJob() {
    if (!selectedJobId) {
        renderEmpty();
        return;
    }
    try {
        const job = await getJson(`/api/video-link/jobs/${selectedJobId}`);
        renderJob(job);
        await refreshJobsNoSelect();
    } catch (error) {
        renderServiceOffline(error);
    }
}

async function refreshJobsNoSelect() {
    const data = await getJson('/api/video-link/jobs?limit=50');
    renderGlobal(data);
    const jobs = data.jobs || [];
    latestJobs = jobs;
    renderJobList(jobs);
    renderSelectedJobSnapshot(jobs);
    renderPreviewGrid(jobs);
}

function renderGlobal(data) {
    const summary = data.summary || {};
    const counts = summary.counts || {};
    const cells = [
        ['总任务', summary.total ?? data.total ?? 0],
        ['运行中', counts.running || 0],
        ['排队中', counts.queued || 0],
        ['成功', counts.succeeded || 0],
        ['失败', counts.failed || 0],
        ['平均进度', `${summary.average_progress || 0}%`]
    ];
    nodes.globalSummary.innerHTML = cells.map(([label, value]) => `
        <div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>
    `).join('');

    const resources = data.resources || {};
    const names = Object.keys(resources);
    nodes.resourceLanes.innerHTML = names.length ? names.map(name => {
        const info = resources[name] || {};
        const running = info.running || [];
        const queued = info.queued || [];
        const runningText = running.length ? running.map(item => resourceItem(item)).join('') : '<div class="lane-empty">空闲</div>';
        const queuedText = queued.length ? queued.map(item => resourceItem(item, true)).join('') : '<div class="lane-empty">无排队</div>';
        return `<section class="resource-lane">
            <header>
                <strong>${escapeHtml(name)}</strong>
                <span>${running.length}/${info.limit || 0} 运行 · ${queued.length} 排队</span>
            </header>
            <div class="lane-columns">
                <div><span>运行</span>${runningText}</div>
                <div><span>排队</span>${queuedText}</div>
            </div>
        </section>`;
    }).join('') : '<div class="empty">暂无资源状态</div>';
}

function resourceItem(item, queued = false) {
    const prefix = queued && item.position ? `#${item.position} · ` : '';
    const title = item.video_url || item.job_id || '-';
    return `<button class="lane-item" type="button" data-job-id="${escapeHtml(item.job_id || '')}">
        <strong>${escapeHtml(prefix + (item.stage_label || item.stage || '-'))}</strong>
        <span>${escapeHtml(title)} · ${escapeHtml(item.progress_percent ?? 0)}%</span>
    </button>`;
}

function bindJobButtons() {
    document.querySelectorAll('.job-select, .lane-item').forEach(button => {
        button.addEventListener('click', () => {
            if (button.dataset.jobId) selectJob(button.dataset.jobId);
        });
    });
    document.querySelectorAll('.delete-job').forEach(button => {
        button.addEventListener('click', async event => {
            event.stopPropagation();
            const jobId = button.dataset.jobId;
            const job = latestJobs.find(item => item.job_id === jobId);
            const title = jobDisplayTitle(job || { job_id: jobId });
            if (!jobId || !window.confirm(`删除最近任务？\n\n${title}\n\n只移除任务记录，不删除分析产物。`)) return;
            button.disabled = true;
            try {
                await getJson(`/api/video-link/jobs/${jobId}`, { method: 'DELETE' });
                if (selectedJobId === jobId) {
                    selectedJobId = null;
                    const url = new URL(window.location.href);
                    url.searchParams.delete('job');
                    window.history.replaceState({}, '', url);
                    renderEmpty();
                }
                await refreshJobs();
            } catch (error) {
                nodes.formError.textContent = error.message;
                button.disabled = false;
            }
        });
    });
}

function renderEmpty() {
    currentJob = null;
    nodes.runButton.disabled = true;
    nodes.runButton.classList.remove('success-action', 'play-action', 'stop-action');
    nodes.runButton.dataset.action = 'run';
    nodes.runButton.title = '';
    nodes.runButton.textContent = '继续运行';
    nodes.selectedTitle.textContent = '未选择任务';
    nodes.selectedSubtitle.textContent = '创建或选择一个任务后查看进度。';
    nodes.stageDurationSummary.textContent = '原视频长度：- · 阶段总耗时：-';
}

function renderServiceOffline(error) {
    currentJob = null;
    nodes.runButton.disabled = true;
    nodes.runButton.classList.remove('success-action', 'play-action', 'stop-action');
    nodes.runButton.dataset.action = 'run';
    nodes.runButton.title = '';
    nodes.runButton.textContent = '继续运行';
    nodes.selectedTitle.textContent = '服务未连接';
    nodes.selectedSubtitle.textContent = error.message;
    setText(nodes.statusValue, 'offline');
    setText(nodes.currentStageValue, '-');
    setText(nodes.nextStageValue, '-');
    setText(nodes.queueValue, '-');
    nodes.stageDurationSummary.textContent = '原视频长度：- · 阶段总耗时：-';
    nodes.progressText.textContent = '0/0 · 0%';
    nodes.progressBar.style.width = '0%';
}

function activeProcess(job) {
    const stage = job.current_stage || job.next_stage;
    return job.process || job.stages?.[stage]?.process || null;
}

function runDisabledReason(job) {
    const process = activeProcess(job);
    if (process?.alive || job.runner?.status === 'running' || job.runner?.status === 'queued' || job.status === 'running' || job.status === 'queued') return '';
    return '';
}

function jobIsActive(job) {
    const process = activeProcess(job);
    return Boolean(process?.alive || job.runner?.status === 'running' || job.runner?.status === 'queued' || job.status === 'running' || job.status === 'queued');
}

function renderJob(job) {
    currentJob = job;
    const progress = job.progress || {};
    const stageProgress = job.stage_progress || job.core_progress;
    const queue = job.queue || {};
    const reason = runDisabledReason(job);
    const process = activeProcess(job);
    const runDir = job.summary?.run_dir || job.run_dir;
    const isSucceeded = job.status === 'succeeded';
    const isActive = jobIsActive(job);
    const missingRunDir = isSucceeded && !runDir;
    nodes.selectedTitle.textContent = jobDisplayTitle(job);
    const subtitleReason = missingRunDir ? '资源目录不可用' : reason;
    const subtitleBase = `任务 ID: ${job.job_id} · ${job.video_url || '-'}`;
    nodes.selectedSubtitle.textContent = subtitleReason ? `${subtitleBase} · ${subtitleReason}` : subtitleBase;
    nodes.runButton.disabled = Boolean(missingRunDir);
    nodes.runButton.dataset.action = isActive ? 'stop' : (isSucceeded ? 'open-run-dir' : 'run');
    nodes.runButton.classList.toggle('success-action', isSucceeded && !isActive);
    nodes.runButton.classList.toggle('play-action', !isSucceeded && !isActive);
    nodes.runButton.classList.toggle('stop-action', isActive);
    nodes.runButton.textContent = isActive ? '停止' : (isSucceeded ? '成功' : (job.status === 'failed' ? '重试失败阶段' : '继续运行'));
    nodes.runButton.title = isActive ? '停止当前运行任务' : (isSucceeded && runDir ? `打开资源目录：${runDir}` : '继续运行任务');
    setText(nodes.statusValue, job.status);
    setText(nodes.currentStageValue, stageNames[job.current_stage] || job.current_stage);
    setText(nodes.nextStageValue, stageNames[job.next_stage] || job.next_stage);
    const queueText = queue.resource ? `${queue.resource} #${queue.position || '-'}/${queue.size || '-'}` : '-';
    setText(nodes.queueValue, process?.alive ? `${queueText} · PID ${process.pid}` : queueText);
    const subProgress = stageProgress?.percent != null && job.current_stage === stageProgress.stage
        ? ` · ${stageProgress.stage_label || stageNames[stageProgress.stage] || stageProgress.stage}约 ${stageProgress.percent}%`
        : '';
    nodes.progressText.textContent = `${progress.completed || 0}/${progress.total || 0} · ${progress.percent || 0}%${subProgress}`;
    nodes.progressBar.style.width = `${progress.percent || 0}%`;
    setText(nodes.detailUrl, job.video_url);
    setText(nodes.detailRunDir, job.summary?.run_dir || job.run_dir);
    setText(nodes.detailMode, `${job.options?.analysis_mode || '-'} -> ${job.resolved_mode || '-'}`);
    setText(nodes.detailUpdated, job.updated_at);

    if (job.error_summary) {
        nodes.errorPanel.hidden = false;
        nodes.errorTitle.textContent = `流程失败：${job.error_summary.stage_label || job.error_summary.stage || '未知阶段'}`;
        nodes.errorMessage.textContent = job.error_summary.message || '未提供错误信息';
    } else if ((job.warnings || []).length) {
        nodes.errorPanel.hidden = false;
        nodes.errorTitle.textContent = '已生成，部分环节有警告';
        nodes.errorMessage.textContent = job.warnings.map(item => {
            const stage = stageNames[item.stage] || item.stage || '流程';
            return `${stage}: ${item.message || '-'}`;
        }).join('；');
    } else {
        nodes.errorPanel.hidden = true;
    }
    renderStages(job);
    renderStageProgress(stageProgress);
    renderCoreDiagnostics(job.core_diagnostics);
    renderArtifacts(job.summary || {});
    renderVscodePanel(job);
    loadSelectedLog(job);
}

function renderStages(job) {
    nodes.stageRows.innerHTML = (job.stage_order || []).map(stage => {
        const info = job.stages?.[stage] || {};
        const status = info.status || 'pending';
        const queue = info.queue_position ? `${info.queued_for || ''} #${info.queue_position}` : (info.queued_for || '-');
        const error = info.error ? `<div class="row-error">${escapeHtml(info.error)}</div>` : '';
        const warning = info.warning ? `<div class="row-warning">${escapeHtml(info.warning)}</div>` : '';
        const retry = info.retry_reason ? `<div class="muted">${escapeHtml(info.retry_reason)}</div>` : '';
        const log = info.log_path ? `<button class="log-link" type="button" data-stage="${stage}">查看日志</button>` : '-';
        return `<tr class="stage-row ${escapeHtml(status)}">
            <td>${escapeHtml(stageNames[stage] || stage)}${error}${warning}${retry}</td>
            <td>${statusBadge(status)}</td>
            <td>${escapeHtml(duration(info.duration_seconds))}</td>
            <td>${escapeHtml(queue)}</td>
            <td>${log}</td>
        </tr>`;
    }).join('');
    document.querySelectorAll('.log-link').forEach(button => {
        button.addEventListener('click', () => {
            selectedLogStage = button.dataset.stage;
            loadSelectedLog(job);
        });
    });
    nodes.stageDurationSummary.textContent = stageDurationSummary(job);
}

function renderStageProgress(progress) {
    const hasVisibleStep = progress && (progress.steps || []).some(step => step.status !== 'pending');
    nodes.corePanel.hidden = !hasVisibleStep;
    if (!hasVisibleStep) return;
    const percentText = progress.percent != null ? ` · 约 ${progress.percent}%` : '';
    nodes.corePanelTitle.textContent = progress.stage_label ? `${progress.stage_label}子项${percentText}` : `阶段子项${percentText}`;
    const summary = progress.summary || progress.current_label || progress.last_signal_label || '';
    const summaryRow = `<tr class="stage-progress-meta ${progress.live ? 'live' : ''} ${progress.stale ? 'stale' : ''}">
        <td colspan="4">
            <div class="stage-progress-head">
                <span>${escapeHtml(summary)}</span>
                <strong>${escapeHtml(progress.percent ?? 0)}%</strong>
            </div>
            <div class="bar stage-progress-bar"><div style="width:${escapeHtml(progress.percent ?? 0)}%"></div></div>
        </td>
    </tr>`;
    nodes.coreRows.innerHTML = summaryRow + progress.steps.map(step => `<tr class="substep-row ${escapeHtml(step.status || 'pending')} ${progress.stale && step.status !== 'pending' ? 'stale' : ''}">
        <td>${escapeHtml(step.label)}</td>
        <td>${statusBadge(step.status)}</td>
        <td>${escapeHtml(duration(step.duration_seconds))}</td>
        <td>${escapeHtml(step.message || '-')}</td>
    </tr>`).join('');
}

function renderCoreDiagnostics(diagnostics) {
    if (!nodes.coreDiagnosticsPanel) return;
    const hasDiagnostics = diagnostics && (diagnostics.summary || diagnostics.efficiency || (diagnostics.issues || []).length);
    nodes.coreDiagnosticsPanel.hidden = !hasDiagnostics;
    if (!hasDiagnostics) return;

    const status = diagnostics.status || 'ok';
    nodes.coreDiagnosticsPanel.dataset.status = status;
    nodes.coreDiagnosticsSummary.textContent = diagnostics.summary || '暂无异常信号';
    nodes.coreDiagnosticsStatus.textContent = diagnosticStatusLabel(status);
    nodes.coreDiagnosticsStatus.className = `diagnostic-status ${status}`;

    const efficiency = diagnostics.efficiency || {};
    const bottleneck = efficiency.bottleneck;
    const metrics = [
        ['总耗时', duration(efficiency.total_seconds)],
        ['视频时长', duration(efficiency.video_duration_seconds)],
        ['耗时比', efficiency.runtime_ratio == null ? '-' : `${efficiency.runtime_ratio}x`],
        ['瓶颈', bottleneck ? `${bottleneck.label || bottleneck.key}: ${duration(bottleneck.seconds)}` : '-']
    ];
    nodes.coreDiagnosticsMetrics.innerHTML = metrics.map(([label, value]) => `
        <div class="diagnostic-metric">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
        </div>
    `).join('') + renderGpuDiagnostics(diagnostics.gpu);

    const issues = diagnostics.issues || [];
    nodes.coreDiagnosticsIssues.innerHTML = issues.length ? issues.map(item => `
        <div class="diagnostic-issue ${escapeHtml(item.severity || 'watch')}">
            <div class="diagnostic-issue-head">
                <span>${escapeHtml(diagnosticStatusLabel(item.severity || 'watch'))}</span>
                <strong>${escapeHtml(item.title || item.code || '诊断项')}</strong>
            </div>
            <p>${escapeHtml(item.detail || '-')}</p>
            <p class="muted">${escapeHtml(item.recommendation || '')}</p>
            ${item.evidence ? `<code>${escapeHtml(item.evidence)}</code>` : ''}
        </div>
    `).join('') : '<div class="diagnostic-empty">没有发现效率或失败异常。</div>';
}

function renderGpuDiagnostics(gpu) {
    if (!gpu) return '';
    if (gpu.status !== 'ok') {
        return `<div class="diagnostic-gpu-summary"><span>GPU</span><strong>不可用</strong></div>`;
    }
    const devices = gpu.devices || [];
    if (!devices.length) {
        return `<div class="diagnostic-gpu-summary"><span>GPU</span><strong>未发现设备</strong></div>`;
    }
    return `<div class="diagnostic-gpu-table">
        <div class="diagnostic-gpu-title">GPU 实时快照 · ${escapeHtml(gpu.sampled_at || '-')}</div>
        ${devices.map(device => {
            const used = device.memory_used_mib ?? 0;
            const total = device.memory_total_mib ?? 0;
            const util = device.utilization_gpu_percent ?? 0;
            const powerDraw = device.power_draw_w == null ? '-' : `${device.power_draw_w}W`;
            const powerLimit = device.power_limit_w == null ? '-' : `${device.power_limit_w}W`;
            const processText = (device.processes || []).slice(0, 2).map(proc => {
                const name = String(proc.process_name || '').split('/').pop();
                return `${proc.pid || '-'} ${name || '-'} ${proc.used_memory_mib || 0}MiB`;
            }).join(' · ') || '无计算进程';
            return `<div class="diagnostic-gpu-row">
                <span>GPU ${escapeHtml(device.index ?? '-')} · ${escapeHtml(device.name || '-')}</span>
                <strong>${escapeHtml(used)}/${escapeHtml(total)} MiB · ${escapeHtml(util)}% · ${escapeHtml(powerDraw)}/${escapeHtml(powerLimit)}</strong>
                <em>${escapeHtml(processText)}</em>
            </div>`;
        }).join('')}
    </div>`;
}

function diagnosticStatusLabel(status) {
    return {
        ok: '正常',
        watch: '观察',
        warning: '警告',
        error: '错误'
    }[status] || status || '-';
}

function renderArtifacts(summary) {
    const counts = summary.core_counts || {};
    const countRows = [
        ['扫描帧', counts.scan_frames],
        ['OCR候选帧', counts.ocr_candidate_frames],
        ['实际OCR帧', counts.ocr_keyframes],
        ['OCR文本事件', counts.ocr_text_events],
        ['VL帧', counts.vl_frames]
    ].filter(([, value]) => value !== undefined && value !== null);
    nodes.artifactSummary.innerHTML = [
        ...countRows.map(([label, value]) => `${label}: ${escapeHtml(value)}`),
        `Markdown: ${(summary.markdown_files || []).length}`,
        `导出文件: ${(summary.export_files || []).length}`,
        `配图提示词: ${(summary.prompt_files || []).length}`,
        `最终图片: ${(summary.final_images || []).length}`
    ].join('<br>');
}

function renderVscodePanel(job) {
    if (!nodes.vscodeView) return;
    const runDir = job?.summary?.run_dir || job?.run_dir || job?.vscode_preview?.run_dir || '';
    const preview = job?.vscode_preview || {};
    const ready = Boolean(preview.ready && preview.url);
    nodes.startVscodeButton.disabled = !job || !runDir || vscodeStarting;
    nodes.restartVscodeButton.disabled = !job || !runDir || vscodeStarting;
    nodes.stopVscodeButton.disabled = !job || !ready || vscodeStarting;
    nodes.openVscodeLink.hidden = !ready;
    if (ready) {
        nodes.openVscodeLink.href = preview.url;
    } else {
        nodes.openVscodeLink.removeAttribute('href');
    }
    if (!job) {
        nodes.vscodeSummary.textContent = '选择一个已生成资源包的任务';
        resetVscodeShell();
        renderStudyPanel(job);
        renderDocPreviewPanel(job);
        return;
    }
    if (!runDir) {
        nodes.vscodeSummary.textContent = '资源包目录尚未生成';
        resetVscodeShell();
        renderStudyPanel(job);
        renderDocPreviewPanel(job);
        return;
    }
    nodes.vscodeSummary.textContent = ready
        ? `WebIDE 已就绪 · PID ${preview.pid} · ${runDir}`
        : `${vscodeStarting ? '正在启动 WebIDE' : '资源包目录'} · ${runDir}`;
    resetVscodeShell();
    renderStudyPanel(job);
    renderDocPreviewPanel(job);
}

function resetVscodeShell() {
    if (nodes.vscodeFrame) {
        nodes.vscodeFrame.hidden = true;
        nodes.vscodeFrame.src = 'about:blank';
    }
    if (nodes.vscodePlaceholder) {
        nodes.vscodePlaceholder.hidden = false;
        nodes.vscodePlaceholder.textContent = '';
    }
}

function showVscodePlaceholder(message) {
    nodes.vscodeFrame.hidden = true;
    nodes.vscodePlaceholder.hidden = false;
    nodes.vscodePlaceholder.textContent = message;
}

function renderStudyPanel(job) {
    if (!nodes.studyBody) return;
    const study = job?.summary?.study || {};
    const decision = study.publish_decision || {};
    const durationLabel = durationMinutes(job?.preview?.duration_seconds || job?.duration_seconds);
    nodes.studySummary.textContent = study.available
        ? `${study.chapter_count || 0} 章${durationLabel === '-' ? '' : ` · 约 ${durationLabel}`}`
        : '等待 study guide';
    const status = decision.status || '-';
    nodes.studyDecision.textContent = status;
    nodes.studyDecision.className = `study-decision ${escapeHtml(status)}`;
    if (!job?.job_id || !study.available) {
        loadedStudyKey = '';
        nodes.studyBody.innerHTML = '<div class="doc-empty">暂无结构化学习数据</div>';
        return;
    }
    loadStudyGuide(job.job_id);
}

async function loadStudyGuide(jobId) {
    if (loadedStudyKey === jobId) return;
    nodes.studyBody.innerHTML = '<div class="doc-empty">加载学习视图...</div>';
    try {
        const guide = await getJson(`/api/video-link/jobs/${jobId}/study-guide`);
        if (currentJob?.job_id !== jobId) return;
        nodes.studyBody.innerHTML = renderStudyGuide(guide, jobId);
        wireStudyGuideInteractions(guide, jobId);
        loadedStudyKey = jobId;
    } catch (error) {
        nodes.studyBody.innerHTML = `<div class="doc-empty">学习视图加载失败：${escapeHtml(error.message)}</div>`;
        loadedStudyKey = '';
    }
}

function renderStudyGuide(guide, jobId) {
    const chapters = guide.chapters || [];
    const overview = guide.overview || {};
    if (!chapters.length) {
        return `<section class="study-learning-shell">
            <div class="study-content-summary">
                <span>内容总结</span>
                <p>${escapeHtml(overview.summary || '暂无学习总览')}</p>
            </div>
            <div class="doc-empty">暂无可展示的学习步骤</div>
        </section>`;
    }
    const firstChapter = chapters[0];
    const html = [
        `<section class="study-learning-shell">
            <div class="study-content-summary">
                <span>内容总结</span>
                <p>${escapeHtml(overview.summary || '暂无学习总览')}</p>
                <div class="study-progress-meta">
                    <strong data-study-progress-text>当前步骤：1 / ${escapeHtml(chapters.length)}</strong>
                    <div class="bar study-progress-bar"><div data-study-progress-bar style="width:${escapeHtml(progressPercent(1, chapters.length))}%"></div></div>
                </div>
            </div>
            <div class="study-workflow-section">
                <div class="study-section-title">工作流</div>
                <div class="study-workflow" role="list" aria-label="视频内容学习工作流">
                    ${chapters.map((chapter, index) => renderStudyNode(chapter, index === 0, index)).join('')}
                </div>
            </div>
            <div class="study-detail-shell">
                ${renderStudyDetail(firstChapter, jobId, 0, chapters.length)}
            </div>
        </section>`
    ];
    return html.join('');
}

function renderStudyNode(chapter, selected, index) {
    return `<button class="study-node${selected ? ' active' : ''}" type="button" role="listitem" data-study-index="${escapeHtml(index)}" aria-pressed="${selected ? 'true' : 'false'}">
        <span>${escapeHtml(String(chapter.index || index + 1).padStart(2, '0'))}</span>
        <strong>${escapeHtml(chapter.title || '未命名章节')}</strong>
        <em>${escapeHtml(studyTimeRange(chapter))}</em>
    </button>`;
}

function renderStudyDetail(chapter, jobId, index, total) {
    const title = `${String(chapter.index || index + 1).padStart(2, '0')}. ${chapter.title || '未命名章节'}`;
    const points = Array.isArray(chapter.key_points) ? chapter.key_points.filter(Boolean).slice(0, 5) : [];
    return `<article class="study-detail" data-study-detail-index="${escapeHtml(index)}">
        <div class="study-frame">
            ${renderStudyFrame(chapter, jobId)}
        </div>
        <div class="study-detail-content">
            <div class="study-detail-head">
                <div>
                    <span>当前学习步骤</span>
                    <h4>${escapeHtml(title)}</h4>
                    <p>${escapeHtml(studyTimeRange(chapter))}</p>
                </div>
                <button class="study-play" type="button" data-job-id="${escapeHtml(jobId)}" data-seconds="${escapeHtml(chapter.start_sec || 0)}">从这里播放</button>
            </div>
            <section>
                <h5>这一段在讲什么</h5>
                <p>${escapeHtml(chapter.summary || '暂无章节总结')}</p>
            </section>
            <section>
                <h5>需要理解的要点</h5>
                ${points.length
                    ? `<ol class="study-key-points">${points.map(point => `<li>${escapeHtml(point)}</li>`).join('')}</ol>`
                    : '<div class="study-empty-note">暂无单独提取的理解要点</div>'}
            </section>
            <div class="study-step-actions">
                <button class="secondary study-step-prev" type="button" data-study-target="${escapeHtml(index - 1)}" ${index <= 0 ? 'disabled' : ''}>← 上一步</button>
                <span>${escapeHtml(index + 1)} / ${escapeHtml(total)}</span>
                <button class="secondary study-step-next" type="button" data-study-target="${escapeHtml(index + 1)}" ${index >= total - 1 ? 'disabled' : ''}>下一步 →</button>
            </div>
        </div>
    </article>`;
}

function renderStudyFrame(chapter, jobId) {
    const frame = chapter.representative_frame || {};
    const path = frame.path || frame.frame_path || '';
    if (!path || !jobId) {
        return '<div class="study-frame-empty">暂无章节截图</div>';
    }
    const src = resourceUrl(jobId, path);
    const alt = chapter.title || '章节代表截图';
    return `<img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="lazy" data-image-viewer-src="${escapeHtml(src)}" data-image-viewer-alt="${escapeHtml(alt)}">`;
}

function wireStudyGuideInteractions(guide, jobId) {
    const chapters = guide.chapters || [];
    if (!chapters.length || !nodes.studyBody) return;
    const workflow = nodes.studyBody.querySelector('.study-workflow');
    const detailShell = nodes.studyBody.querySelector('.study-detail-shell');
    const progressText = nodes.studyBody.querySelector('[data-study-progress-text]');
    const progressBar = nodes.studyBody.querySelector('[data-study-progress-bar]');
    const selectChapter = index => {
        if (!Number.isInteger(index) || index < 0 || index >= chapters.length || !detailShell) return;
        detailShell.innerHTML = renderStudyDetail(chapters[index], jobId, index, chapters.length);
        nodes.studyBody.querySelectorAll('.study-node').forEach(node => {
            const active = Number(node.dataset.studyIndex || -1) === index;
            node.classList.toggle('active', active);
            node.setAttribute('aria-pressed', active ? 'true' : 'false');
            if (active) node.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        });
        if (progressText) progressText.textContent = `当前步骤：${index + 1} / ${chapters.length}`;
        if (progressBar) progressBar.style.width = `${progressPercent(index + 1, chapters.length)}%`;
        wireStudyDetailActions(selectChapter);
    };
    workflow?.querySelectorAll('.study-node').forEach(button => {
        button.addEventListener('click', () => selectChapter(Number(button.dataset.studyIndex || 0)));
    });
    wireStudyDetailActions(selectChapter);
}

function wireStudyDetailActions(selectChapter) {
    nodes.studyBody.querySelectorAll('.study-step-prev, .study-step-next').forEach(button => {
        button.addEventListener('click', () => selectChapter(Number(button.dataset.studyTarget)));
    });
    nodes.studyBody.querySelectorAll('.study-play').forEach(button => {
        button.addEventListener('click', () => jumpToVideoTime(button.dataset.jobId, Number(button.dataset.seconds || 0)));
    });
}

function studyTimeRange(chapter) {
    const start = chapter.start || formatTimestampFromSeconds(chapter.start_sec);
    const end = chapter.end || formatTimestampFromSeconds(chapter.end_sec);
    return [start, end].filter(Boolean).join(' - ') || '-';
}

function progressPercent(current, total) {
    const value = Number(total) > 0 ? (Number(current) / Number(total)) * 100 : 0;
    return Math.max(0, Math.min(100, value)).toFixed(1);
}

function formatTimestampFromSeconds(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0) return '';
    const whole = Math.floor(seconds);
    const hours = Math.floor(whole / 3600);
    const minutes = Math.floor((whole % 3600) / 60);
    const rest = whole % 60;
    return hours > 0
        ? `${hours}:${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}`
        : `${minutes}:${String(rest).padStart(2, '0')}`;
}

async function jumpToVideoTime(jobId, seconds) {
    if (!jobId || !Number.isFinite(seconds)) return;
    setView('preview');
    await refreshJobs();
    const video = previewVideo(jobId);
    if (!video) return;
    const seek = () => {
        video.currentTime = Math.max(0, seconds);
        updateVideoControls(video);
        video.scrollIntoView({ behavior: 'smooth', block: 'center' });
    };
    if (Number.isFinite(video.duration) && video.duration > 0) {
        seek();
    } else {
        video.addEventListener('loadedmetadata', seek, { once: true });
        video.load();
    }
}

function resourceUrl(jobId, path) {
    return `/api/video-link/jobs/${jobId}/resource?path=${encodeURIComponent(path)}`;
}

function previewableDocs(job) {
    const summary = job?.summary || {};
    const markdownFiles = (summary.markdown_files || []).map(path => ({ path, kind: 'markdown' }));
    const pdfFiles = (summary.export_files || [])
        .filter(path => path.toLowerCase().endsWith('.pdf'))
        .map(path => ({ path, kind: 'pdf' }));
    const preferred = [
        'study_overview.md',
        'study_cards.md',
        'evidence_index.md',
        'review_notes.md',
        'exports/operation_manual.pdf',
        'operation_manual.md',
        'exports/knowledge_notes_v2.pdf',
        'docs_analysis_chapters/knowledge_notes_v2.md',
        'exports/deep_report_v2.pdf',
        'docs_analysis_chapters/deep_report_v2.md',
        'exports/manual_evidence.pdf',
        'manual_evidence.md'
    ];
    return [...markdownFiles, ...pdfFiles].sort((left, right) => {
        const leftIndex = preferred.indexOf(left.path);
        const rightIndex = preferred.indexOf(right.path);
        const leftRank = leftIndex === -1 ? 1000 : leftIndex;
        const rightRank = rightIndex === -1 ? 1000 : rightIndex;
        if (leftRank !== rightRank) return leftRank - rightRank;
        if (left.kind !== right.kind) return left.kind === 'pdf' ? -1 : 1;
        return left.path.localeCompare(right.path);
    });
}

function renderDocPreviewPanel(job) {
    if (!nodes.docList) return;
    const docs = previewableDocs(job);
    nodes.docPreviewSummary.textContent = docs.length ? `${docs.length} 个文档` : '无文档';
    if (!job || !docs.length) {
        selectedDocPath = '';
        renderedDocListKey = '';
        loadedDocPreviewKey = '';
        nodes.docList.innerHTML = '<div class="doc-empty">暂无 Markdown/PDF</div>';
        hideDocPreview();
        return;
    }
    if (selectedDocPath && !docs.some(doc => doc.path === selectedDocPath)) selectedDocPath = '';
    const listKey = `${job.job_id}|${selectedDocPath}|${docs.map(doc => `${doc.kind}:${doc.path}`).join('|')}`;
    if (renderedDocListKey !== listKey) {
        renderedDocListKey = listKey;
        nodes.docList.innerHTML = docs.map(doc => {
            const active = doc.path === selectedDocPath ? ' active' : '';
            const label = doc.kind === 'pdf' ? 'PDF' : 'MD';
            return `<button class="doc-item${active}" type="button" data-doc-path="${escapeHtml(doc.path)}">
                <span>${escapeHtml(label)}</span>
                <strong>${escapeHtml(doc.path)}</strong>
            </button>`;
        }).join('');
        nodes.docList.querySelectorAll('.doc-item').forEach(button => {
            button.addEventListener('click', () => {
                selectedDocPath = button.dataset.docPath || '';
                loadedDocPreviewKey = '';
                renderedDocListKey = '';
                renderDocPreviewPanel(currentJob);
            });
        });
    }
    if (!selectedDocPath) {
        hideDocPreview();
        return;
    }
    showDocPreviewPanel();
    loadDocPreview(job, selectedDocPath);
}

function showDocPreview(title, contentHtml) {
    showDocPreviewPanel();
    nodes.docPreviewTitle.textContent = title;
    nodes.docOpenLink.hidden = true;
    nodes.docOpenLink.removeAttribute('href');
    nodes.docPreviewBody.className = 'doc-preview-body';
    nodes.docPreviewBody.innerHTML = typeof contentHtml === 'string' ? contentHtml : '';
}

function showDocPreviewPanel() {
    if (nodes.docPreviewPanel) nodes.docPreviewPanel.hidden = false;
    nodes.vscodeView?.querySelector('.vscode-docs')?.classList.add('preview-open');
}

function hideDocPreview() {
    if (nodes.docPreviewPanel) nodes.docPreviewPanel.hidden = true;
    nodes.vscodeView?.querySelector('.vscode-docs')?.classList.remove('preview-open');
    nodes.docPreviewTitle.textContent = '选择文档';
    nodes.docOpenLink.hidden = true;
    nodes.docOpenLink.removeAttribute('href');
    nodes.docPreviewBody.className = 'doc-preview-body';
    nodes.docPreviewBody.innerHTML = '';
}

function closeDocPreview() {
    selectedDocPath = '';
    loadedDocPreviewKey = '';
    renderedDocListKey = '';
    renderDocPreviewPanel(currentJob);
}

async function loadDocPreview(job, path) {
    if (!job?.job_id || !path) return;
    const previewKey = `${job.job_id}|${path}`;
    if (loadedDocPreviewKey === previewKey) return;
    const url = resourceUrl(job.job_id, path);
    nodes.docPreviewTitle.textContent = path;
    nodes.docOpenLink.href = url;
    nodes.docOpenLink.hidden = false;
    if (path.toLowerCase().endsWith('.pdf')) {
        nodes.docPreviewBody.className = 'doc-preview-body pdf';
        nodes.docPreviewBody.innerHTML = `<iframe title="${escapeHtml(path)}" src="${escapeHtml(url)}"></iframe>`;
        loadedDocPreviewKey = previewKey;
        return;
    }
    nodes.docPreviewBody.className = 'doc-preview-body markdown loading';
    nodes.docPreviewBody.textContent = '加载中...';
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        const markdown = await response.text();
        if (selectedDocPath !== path) return;
        nodes.docPreviewBody.className = 'doc-preview-body markdown';
        nodes.docPreviewBody.innerHTML = renderMarkdown(markdown, job.job_id, path);
        renderMarkdownMath(nodes.docPreviewBody);
        loadedDocPreviewKey = previewKey;
    } catch (error) {
        nodes.docPreviewBody.className = 'doc-preview-body markdown';
        nodes.docPreviewBody.textContent = `加载失败：${error.message}`;
        loadedDocPreviewKey = '';
    }
}

function renderMarkdown(markdown, jobId = '', docPath = '') {
    if (!window.markdownit || !window.DOMPurify) {
        return renderLegacyMarkdown(markdown, jobId, docPath);
    }
    const renderer = createMarkdownRenderer(jobId, docPath);
    return window.DOMPurify.sanitize(renderer.render(normalizeMarkdownForPreview(markdown)), {
        ADD_TAGS: ['figure', 'figcaption'],
        ADD_ATTR: ['target', 'rel', 'loading']
    });
}

function normalizeMarkdownForPreview(markdown) {
    const lines = String(markdown || '').split(/\r?\n/);
    const normalized = [];
    for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index];
        const inlineTableLines = splitInlineMarkdownTableLine(line);
        if (inlineTableLines.length > 1) {
            normalized.push(...inlineTableLines);
            continue;
        }
        if (isPotentialMarkdownTableRow(line) && !String(lines[index + 1] || '').trim() && isMarkdownTableSeparator(lines[index + 2] || '')) {
            normalized.push(line, lines[index + 2]);
            index += 2;
            continue;
        }
        normalized.push(line);
    }
    return normalized.join('\n');
}

function splitInlineMarkdownTableLine(line) {
    const value = String(line || '');
    const match = value.match(/^(.+?[：:])\s+(\|.+)$/);
    if (!match) return [line];
    const rows = match[2]
        .trim()
        .split(/(?<=\|)\s+(?=\|)/)
        .map(row => row.trim())
        .filter(Boolean);
    if (rows.length < 2 || !isMarkdownTableSeparator(rows[1])) return [line];
    return [match[1].trim(), '', ...rows];
}

function isPotentialMarkdownTableRow(line) {
    const value = String(line || '').trim();
    return value.includes('|') && !isMarkdownTableSeparator(value);
}

function createMarkdownRenderer(jobId = '', docPath = '') {
    const md = window.markdownit({
        html: false,
        linkify: true,
        typographer: false,
        breaks: false
    });
    const defaultImage = md.renderer.rules.image || ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
    md.renderer.rules.image = (tokens, idx, options, env, self) => {
        const token = tokens[idx];
        const srcIndex = token.attrIndex('src');
        if (srcIndex >= 0) {
            const src = token.attrs[srcIndex][1];
            const assetPath = markdownAssetPath(docPath, src);
            const resolvedSrc = assetPath && jobId ? resourceUrl(jobId, assetPath) : assetPath;
            token.attrs[srcIndex][1] = resolvedSrc;
            token.attrSet('data-image-viewer-src', resolvedSrc);
            token.attrSet('data-image-viewer-alt', token.content || '');
        }
        token.attrSet('loading', 'lazy');
        token.attrSet('class', 'markdown-image');
        return defaultImage(tokens, idx, options, env, self);
    };
    const defaultFence = md.renderer.rules.fence || ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
    md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx];
        const language = String(token.info || '').trim().split(/\s+/)[0].toLowerCase();
        if (language === 'mermaid') {
            const flowHtml = renderSimpleMermaidFlowchart(token.content);
            if (flowHtml) return flowHtml;
        }
        return defaultFence(tokens, idx, options, env, self);
    };
    const defaultLinkOpen = md.renderer.rules.link_open || ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
    md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
        const token = tokens[idx];
        const href = token.attrGet('href') || '';
        if (/^https?:\/\//i.test(href)) {
            token.attrSet('target', '_blank');
            token.attrSet('rel', 'noreferrer');
        }
        return defaultLinkOpen(tokens, idx, options, env, self);
    };
    return md;
}

function renderMarkdownMath(container) {
    if (!container || typeof window.renderMathInElement !== 'function') return;
    window.renderMathInElement(container, {
        delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\[', right: '\\]', display: true },
            { left: '\\(', right: '\\)', display: false }
        ],
        throwOnError: false,
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    });
}

function renderLegacyMarkdown(markdown, jobId = '', docPath = '') {
    const lines = String(markdown || '').split(/\r?\n/);
    const html = [];
    let paragraph = [];
    let listOpen = false;
    let codeOpen = false;
    let codeLang = '';
    let codeLines = [];
    const flushParagraph = () => {
        if (paragraph.length) {
            html.push(`<p>${inlineMarkdown(paragraph.join(' '), jobId, docPath)}</p>`);
            paragraph = [];
        }
    };
    const closeList = () => {
        if (listOpen) {
            html.push('</ul>');
            listOpen = false;
        }
    };
    for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index];
        if (line.trim().startsWith('```')) {
            if (codeOpen) {
                const codeText = codeLines.join('\n');
                const flowHtml = codeLang === 'mermaid' ? renderSimpleMermaidFlowchart(codeText) : '';
                html.push(flowHtml || `<pre><code>${escapeHtml(codeText)}</code></pre>`);
                codeOpen = false;
                codeLang = '';
                codeLines = [];
            } else {
                flushParagraph();
                closeList();
                codeOpen = true;
                codeLang = line.trim().slice(3).trim().toLowerCase();
                codeLines = [];
            }
            continue;
        }
        if (codeOpen) {
            codeLines.push(line);
            continue;
        }
        const tableSeparatorIndex = markdownTableSeparatorIndex(lines, index);
        if (tableSeparatorIndex !== -1) {
            flushParagraph();
            closeList();
            const tableLines = [line, lines[tableSeparatorIndex]];
            index = tableSeparatorIndex + 1;
            while (index < lines.length && isMarkdownTableRow(lines[index])) {
                tableLines.push(lines[index]);
                index += 1;
            }
            index -= 1;
            html.push(renderMarkdownTable(tableLines, jobId, docPath));
            continue;
        }
        const heading = line.match(/^(#{1,4})\s+(.+)$/);
        if (heading) {
            flushParagraph();
            closeList();
            html.push(`<h${heading[1].length}>${inlineMarkdown(heading[2], jobId, docPath)}</h${heading[1].length}>`);
            continue;
        }
        if (isMarkdownHorizontalRule(line)) {
            flushParagraph();
            closeList();
            html.push('<hr>');
            continue;
        }
        const quote = line.match(/^\s*>\s?(.*)$/);
        if (quote) {
            flushParagraph();
            closeList();
            const quoteLines = [quote[1]];
            while (index + 1 < lines.length) {
                const nextQuote = lines[index + 1].match(/^\s*>\s?(.*)$/);
                if (!nextQuote) break;
                quoteLines.push(nextQuote[1]);
                index += 1;
            }
            html.push(`<blockquote><p>${inlineMarkdown(quoteLines.join(' '), jobId, docPath)}</p></blockquote>`);
            continue;
        }
        const bullet = line.match(/^\s*[-*]\s+(.+)$/);
        if (bullet) {
            flushParagraph();
            if (!listOpen) {
                html.push('<ul>');
                listOpen = true;
            }
            html.push(`<li>${inlineMarkdown(bullet[1], jobId, docPath)}</li>`);
            continue;
        }
        if (!line.trim()) {
            flushParagraph();
            closeList();
            continue;
        }
        paragraph.push(line.trim());
    }
    flushParagraph();
    closeList();
    if (codeOpen) html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
    return html.join('\n');
}

function isMarkdownHorizontalRule(line) {
    return /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(String(line || ''));
}

function markdownTableSeparatorIndex(lines, index) {
    if (!isMarkdownTableRow(lines[index])) return -1;
    let cursor = index + 1;
    while (cursor < lines.length && !String(lines[cursor] || '').trim()) {
        cursor += 1;
    }
    return isMarkdownTableSeparator(lines[cursor] || '') ? cursor : -1;
}

function isMarkdownTableRow(line) {
    const value = String(line || '').trim();
    return value.includes('|') && !isMarkdownTableSeparator(value);
}

function isMarkdownTableSeparator(line) {
    const value = String(line || '').trim();
    if (!value.includes('|')) return false;
    const cells = parseMarkdownTableRow(value);
    return cells.length > 0 && cells.every(cell => /^:?-{3,}:?$/.test(cell.replace(/\s+/g, '')));
}

function parseMarkdownTableRow(line) {
    let value = String(line || '').trim();
    if (value.startsWith('|')) value = value.slice(1);
    if (value.endsWith('|')) value = value.slice(0, -1);
    return value.split('|').map(cell => cell.trim());
}

function renderMarkdownTable(tableLines, jobId = '', docPath = '') {
    const header = parseMarkdownTableRow(tableLines[0]);
    const align = parseMarkdownTableRow(tableLines[1]).map(cell => {
        const value = cell.replace(/\s+/g, '');
        if (value.startsWith(':') && value.endsWith(':')) return 'center';
        if (value.endsWith(':')) return 'right';
        return 'left';
    });
    const rows = tableLines.slice(2).map(parseMarkdownTableRow).filter(row => row.some(Boolean));
    const cellStyle = index => ` style="text-align:${escapeHtml(align[index] || 'left')}"`;
    const headHtml = header.map((cell, index) => `<th${cellStyle(index)}>${inlineMarkdown(cell, jobId, docPath)}</th>`).join('');
    const bodyHtml = rows.map(row => `<tr>${header.map((_, index) => `<td${cellStyle(index)}>${inlineMarkdown(row[index] || '', jobId, docPath)}</td>`).join('')}</tr>`).join('');
    return `<div class="markdown-table-wrap"><table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`;
}

function inlineMarkdown(text, jobId = '', docPath = '') {
    const value = String(text ?? '');
    let html = '';
    let lastIndex = 0;
    const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g;
    for (const match of value.matchAll(imagePattern)) {
        html += escapeHtml(value.slice(lastIndex, match.index));
        const alt = match[1] || '';
        const imagePath = markdownAssetPath(docPath, match[2] || '');
        const src = imagePath && jobId ? resourceUrl(jobId, imagePath) : imagePath;
        html += src
            ? `<figure class="markdown-image"><img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="lazy" data-image-viewer-src="${escapeHtml(src)}" data-image-viewer-alt="${escapeHtml(alt)}"><figcaption>${escapeHtml(alt)}</figcaption></figure>`
            : escapeHtml(match[0]);
        lastIndex = (match.index || 0) + match[0].length;
    }
    html += escapeHtml(value.slice(lastIndex));
    return html
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
}

function markdownAssetPath(docPath, rawPath) {
    let assetPath = String(rawPath || '').trim();
    if (!assetPath) return '';
    assetPath = assetPath.replace(/^<|>$/g, '');
    const titleMatch = assetPath.match(/^([^"'\s]+)(?:\s+["'][^"']*["'])?$/);
    if (titleMatch) assetPath = titleMatch[1];
    if (/^(https?:|data:|blob:)/i.test(assetPath)) return assetPath;
    if (assetPath.startsWith('/')) return assetPath.replace(/^\/+/, '');
    const baseParts = String(docPath || '').split('/').slice(0, -1);
    const parts = [...baseParts, ...assetPath.split('/')];
    const normalized = [];
    parts.forEach(part => {
        if (!part || part === '.') return;
        if (part === '..') {
            normalized.pop();
            return;
        }
        normalized.push(part);
    });
    return normalized.join('/');
}

function parseMermaidNode(rawNode) {
    const cleaned = String(rawNode || '').trim().replace(/;$/, '');
    const match = cleaned.match(/^([A-Za-z][A-Za-z0-9_]*)(?:([\[\{\(])(.*)([\]\}\)]))?$/);
    if (!match) return { id: cleaned, label: '', shape: 'box' };
    let label = match[3] || '';
    label = label.trim();
    if (label.length >= 2 && label[0] === '"' && label[label.length - 1] === '"') {
        label = label.slice(1, -1);
    }
    return {
        id: match[1],
        label,
        shape: match[2] === '{' ? 'decision' : 'box'
    };
}

function mermaidLabelHtml(label) {
    return escapeHtml(label).replace(/&lt;br\s*\/?&gt;/gi, '<br>');
}

function renderSimpleMermaidFlowchart(diagram) {
    const lines = String(diagram || '')
        .split(/\r?\n/)
        .map(line => line.trim())
        .filter(line => line && !line.startsWith('%%'));
    if (!lines.length) return '';
    if (!/^(flowchart\s+(TB|TD|LR|RL)|graph\s+(TB|TD|LR|RL))\b/i.test(lines[0])) return '';

    const labels = new Map();
    const shapes = new Map();
    const edges = [];
    const nodeOrder = [];
    const standaloneNodes = [];
    const rememberNode = node => {
        if (!node.id) return false;
        if (/[{}()[\]]/.test(node.id)) return false;
        if (!nodeOrder.includes(node.id)) nodeOrder.push(node.id);
        if (node.label) labels.set(node.id, node.label);
        if (!shapes.has(node.id)) shapes.set(node.id, node.shape);
        return true;
    };
    for (const line of lines.slice(1)) {
        if (/^(subgraph|end\b|direction\b|style\b|classDef\b|class\b)/i.test(line)) continue;
        if (/^[};\s]+$/.test(line)) continue;
        const edgeMatch = line.match(/^(.+?)\s*-->\s*(?:\|.*?\|\s*)?(.+?)\s*;?$/);
        if (!edgeMatch) {
            const node = parseMermaidNode(line);
            if (!rememberNode(node)) return '';
            if (!standaloneNodes.includes(node.id)) standaloneNodes.push(node.id);
            continue;
        }
        const left = parseMermaidNode(edgeMatch[1]);
        const right = parseMermaidNode(edgeMatch[2]);
        if (/[{}()[\]]/.test(left.id) || /[{}()[\]]/.test(right.id)) continue;
        rememberNode(left);
        rememberNode(right);
        edges.push([left.id, right.id]);
    }

    const nodes = new Set();
    const outgoing = new Map();
    const incoming = new Map();
    edges.forEach(([left, right]) => {
        nodes.add(left);
        nodes.add(right);
        if (!outgoing.has(left)) outgoing.set(left, []);
        outgoing.get(left).push(right);
        incoming.set(right, (incoming.get(right) || 0) + 1);
        if (!incoming.has(left)) incoming.set(left, incoming.get(left) || 0);
    });
    const standaloneOnly = standaloneNodes.filter(node => !nodes.has(node));
    if (!edges.length && !standaloneOnly.length) return '';
    const orderIndex = new Map(nodeOrder.map((node, index) => [node, index]));
    const starts = [...nodes]
        .filter(node => (incoming.get(node) || 0) === 0)
        .sort((left, right) => (orderIndex.get(left) ?? 1e9) - (orderIndex.get(right) ?? 1e9));
    if (edges.length && !starts.length) return '';

    const remainingIncoming = new Map(incoming);
    const queue = [...starts];
    const topological = [];
    while (queue.length) {
        const node = queue.shift();
        topological.push(node);
        (outgoing.get(node) || []).forEach(next => {
            remainingIncoming.set(next, (remainingIncoming.get(next) || 0) - 1);
            if (remainingIncoming.get(next) === 0) {
                queue.push(next);
                queue.sort((left, right) => (orderIndex.get(left) ?? 1e9) - (orderIndex.get(right) ?? 1e9));
            }
        });
    }
    if (edges.length && topological.length !== nodes.size) return '';

    const levels = new Map(starts.map(node => [node, 0]));
    topological.forEach(node => {
        const baseLevel = levels.get(node) || 0;
        (outgoing.get(node) || []).forEach(next => {
            levels.set(next, Math.max(levels.get(next) || 0, baseLevel + 1));
        });
    });
    const maxLevel = edges.length ? Math.max(...[...levels.values()]) : -1;
    const grouped = maxLevel >= 0 ? Array.from({ length: maxLevel + 1 }, () => []) : [];
    topological.forEach(node => grouped[levels.get(node) || 0].push(node));

    let index = 1;
    const parts = ['<div class="mobile-flowchart" aria-label="流程图">'];
    const appendRow = row => {
        parts.push('<div class="flow-row">');
        row.forEach(node => {
            const shape = shapes.get(node) === 'decision' ? ' decision' : '';
            const label = mermaidLabelHtml(labels.get(node) || node);
            parts.push(`<div class="flow-node${shape}"><span class="flow-index">${index}</span>${label}</div>`);
            index += 1;
        });
        parts.push('</div>');
    };
    grouped.forEach((row, level) => {
        if (!row.length) return;
        appendRow(row);
        if (level < maxLevel) parts.push('<div class="flow-arrow">↓</div>');
    });
    if (standaloneOnly.length) {
        if (grouped.length) parts.push('<div class="flow-arrow">↓</div>');
        appendRow(standaloneOnly);
    }
    parts.push('</div>');
    return parts.join('\n');
}

async function ensureVscodeSession(restart = false) {
    if (!selectedJobId || vscodeStarting) return;
    const runDir = currentJob?.summary?.run_dir || currentJob?.run_dir || currentJob?.vscode_preview?.run_dir;
    if (!runDir) {
        renderVscodePanel(currentJob);
        return;
    }
    vscodeStarting = true;
    renderVscodePanel(currentJob);
    try {
        const session = await getJson(`/api/video-link/jobs/${selectedJobId}/vscode-session`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ restart })
        });
        currentJob = {
            ...(currentJob || {}),
            vscode_preview: session
        };
        renderVscodePanel(currentJob);
    } catch (error) {
        nodes.vscodeSummary.textContent = `WebIDE 启动失败：${error.message}`;
    } finally {
        vscodeStarting = false;
        renderVscodePanel(currentJob);
    }
}

async function stopVscodeSession() {
    if (!selectedJobId) return;
    nodes.stopVscodeButton.disabled = true;
    try {
        await getJson(`/api/video-link/jobs/${selectedJobId}/vscode-session`, { method: 'DELETE' });
        if (currentJob?.vscode_preview) currentJob.vscode_preview.ready = false;
        resetVscodeShell();
        renderVscodePanel(currentJob);
    } catch (error) {
        showVscodePlaceholder(`停止失败：${error.message}`);
    }
}

function previewStage(job) {
    const stage = job.current_stage || job.error_summary?.stage || job.next_stage || '-';
    return stageNames[stage] || stage;
}

function scanStartTime(job) {
    const stage = job.current_stage || job.next_stage;
    const stageInfo = job.stages?.[stage] || {};
    return stageInfo.started_at || job.runner?.started_at || job.updated_at || job.created_at || '';
}

function scanPercent(job, now = Date.now()) {
    const status = job.status || 'created';
    const baseProgress = clampPercent(job.progress?.percent || 0);
    if (videoLoadErrors[job.job_id]) return baseProgress;
    if (status === 'succeeded') return 100;
    if (status !== 'running') return baseProgress;
    const durationSeconds = Number(job.preview?.duration_seconds || 0) || 600;
    const startedAt = Date.parse(scanStartTime(job));
    if (Number.isNaN(startedAt)) return Math.min(99, baseProgress);
    const elapsedSeconds = Math.max(0, (now - startedAt) / 1000);
    return Math.min(99, Math.max(baseProgress, (elapsedSeconds / durationSeconds) * 100));
}

function renderPreviewGrid(jobs) {
    if (!nodes.previewGrid) return;
    const visibleJobs = jobs || [];
    nodes.previewSummary.textContent = visibleJobs.length ? `最近 ${visibleJobs.length} 个任务` : '暂无任务';
    nodes.previewGrid.innerHTML = visibleJobs.length ? visibleJobs.map(job => renderPreviewCard(job)).join('') : '<div class="empty preview-empty">暂无任务</div>';
    bindPreviewControls();
    updateScanFrames();
}

function previewStatusControl(job) {
    if (job.status === 'succeeded') {
        return `<button class="status succeeded preview-success-link" type="button" data-job-id="${escapeHtml(job.job_id)}" title="打开资源目录">succeeded</button>`;
    }
    return statusBadge(videoLoadErrors[job.job_id] ? 'failed' : job.status);
}

function renderPreviewCard(job) {
    const preview = job.preview || {};
    const ready = Boolean(preview.video_ready && preview.video_url && !videoLoadErrors[job.job_id]);
    const failed = job.status === 'failed' || Boolean(videoLoadErrors[job.job_id]);
    const scanning = job.status === 'running' && !failed;
    const percent = scanPercent(job);
    const durationSeconds = Number(preview.duration_seconds || 0);
    const video = ready
        ? `<video class="preview-video" preload="none" playsinline data-src="${escapeHtml(preview.video_url)}"></video>`
        : `<div class="preview-placeholder">${failed ? '加载停止' : '等待视频'}</div>`;
    const buttonLabel = ready ? '播放' : (failed ? '失败' : '等待');
    const sourceLink = job.video_url
        ? `<a class="preview-source-link" href="${escapeHtml(job.video_url)}" target="_blank" rel="noreferrer">原站</a>`
        : '';
    return `<article class="preview-card ${escapeHtml(job.status || 'created')} ${scanning ? 'scanning' : ''} ${failed ? 'preview-failed' : ''}"
        data-job-id="${escapeHtml(job.job_id)}"
        data-status="${escapeHtml(job.status || 'created')}"
        data-duration="${escapeHtml(durationSeconds || '')}"
        data-started-at="${escapeHtml(scanStartTime(job))}"
        data-progress="${escapeHtml(job.progress?.percent || 0)}">
        <div class="preview-frame">
            ${video}
            <div class="scan-line" aria-hidden="true"></div>
            <div class="preview-status-row">
                ${previewStatusControl(job)}
                <span>${escapeHtml(previewStage(job))}</span>
            </div>
        </div>
        <div class="preview-body">
            <strong title="${escapeHtml(jobDisplayTitle(job))}">${escapeHtml(jobDisplayTitle(job))}</strong>
            <div class="video-controls">
                <button class="preview-play" type="button" data-job-id="${escapeHtml(job.job_id)}" ${ready ? '' : 'disabled'}>${buttonLabel}</button>
                <span class="preview-time" data-job-id="${escapeHtml(job.job_id)}">0:00 / ${escapeHtml(formatClock(durationSeconds))}</span>
                ${sourceLink}
            </div>
            <input class="video-seek" data-job-id="${escapeHtml(job.job_id)}" type="range" min="0" max="1000" value="0" ${ready ? '' : 'disabled'} aria-label="视频进度">
            <div class="scan-meter">
                <div class="scan-meter-head">
                    <span>扫描</span>
                    <strong class="scan-percent">${Math.round(percent)}%</strong>
                </div>
                <div class="bar scan-bar"><div class="preview-scan-progress" style="width:${percent}%"></div></div>
            </div>
        </div>
    </article>`;
}

function bindPreviewControls() {
    document.querySelectorAll('.preview-video').forEach(video => {
        video.addEventListener('loadedmetadata', () => updateVideoControls(video));
        video.addEventListener('timeupdate', () => updateVideoControls(video));
        video.addEventListener('play', () => setPreviewButton(video, '暂停'));
        video.addEventListener('pause', () => setPreviewButton(video, '播放'));
        video.addEventListener('error', () => markPreviewVideoError(video));
    });
    document.querySelectorAll('.preview-play').forEach(button => {
        button.addEventListener('click', () => togglePreviewPlayback(button.dataset.jobId));
    });
    document.querySelectorAll('.preview-success-link').forEach(button => {
        button.addEventListener('click', () => openPreviewRunDir(button.dataset.jobId));
    });
    document.querySelectorAll('.video-seek').forEach(input => {
        input.addEventListener('input', () => seekPreviewVideo(input));
    });
}

function previewCard(jobId) {
    return document.querySelector(`.preview-card[data-job-id="${CSS.escape(jobId)}"]`);
}

function previewVideo(jobId) {
    return previewCard(jobId)?.querySelector('video');
}

function setPreviewButton(video, label) {
    const jobId = video.closest('.preview-card')?.dataset.jobId;
    const button = jobId ? document.querySelector(`.preview-play[data-job-id="${CSS.escape(jobId)}"]`) : null;
    if (button && !button.disabled) button.textContent = label;
}

async function togglePreviewPlayback(jobId) {
    const video = previewVideo(jobId);
    if (!video) return;
    ensurePreviewVideoSource(video);
    if (video.paused) {
        await video.play().catch(() => markPreviewVideoError(video));
    } else {
        video.pause();
    }
}

function ensurePreviewVideoSource(video) {
    if (!video || video.src) return;
    const source = video.dataset.src || '';
    if (!source) return;
    video.src = source;
    video.load();
}

async function openPreviewRunDir(jobId) {
    if (!jobId) return;
    const button = document.querySelector(`.preview-success-link[data-job-id="${CSS.escape(jobId)}"]`);
    if (button) button.disabled = true;
    try {
        await getJson(`/api/video-link/jobs/${jobId}/open-run-dir`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}'
        });
    } catch (error) {
        if (button) {
            button.textContent = '打开失败';
            button.title = error.message;
        }
    } finally {
        if (button) button.disabled = false;
    }
}

function seekPreviewVideo(input) {
    const video = previewVideo(input.dataset.jobId);
    if (!video || !Number.isFinite(video.duration) || video.duration <= 0) return;
    video.currentTime = (Number(input.value) / 1000) * video.duration;
    updateVideoControls(video);
}

function updateVideoControls(video) {
    const card = video.closest('.preview-card');
    const jobId = card?.dataset.jobId;
    if (!jobId) return;
    const durationSeconds = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : Number(card.dataset.duration || 0);
    const input = document.querySelector(`.video-seek[data-job-id="${CSS.escape(jobId)}"]`);
    const time = document.querySelector(`.preview-time[data-job-id="${CSS.escape(jobId)}"]`);
    if (input && durationSeconds > 0) input.value = String(Math.round((video.currentTime / durationSeconds) * 1000));
    if (time) time.textContent = `${formatClock(video.currentTime)} / ${formatClock(durationSeconds)}`;
}

function markPreviewVideoError(video) {
    const jobId = video.closest('.preview-card')?.dataset.jobId;
    if (!jobId) return;
    videoLoadErrors[jobId] = true;
    const card = previewCard(jobId);
    if (card) {
        card.classList.add('preview-failed');
        card.classList.remove('scanning');
        const button = card.querySelector('.preview-play');
        if (button) {
            button.disabled = true;
            button.textContent = '失败';
        }
    }
}

function startScanAnimation() {
    if (scanAnimationFrame) return;
    const tick = () => {
        updateScanFrames();
        scanAnimationFrame = requestAnimationFrame(tick);
    };
    scanAnimationFrame = requestAnimationFrame(tick);
}

function updateScanFrames(now = Date.now()) {
    if (!nodes.previewGrid) return;
    document.querySelectorAll('.preview-card').forEach(card => {
        const status = card.dataset.status || 'created';
        const failed = card.classList.contains('preview-failed');
        const baseProgress = clampPercent(Number(card.dataset.progress || 0));
        let percent = baseProgress;
        if (status === 'succeeded') {
            percent = 100;
        } else if (status === 'running' && !failed) {
            const durationSeconds = Number(card.dataset.duration || 0) || 600;
            const startedAt = Date.parse(card.dataset.startedAt || '');
            const elapsed = Number.isNaN(startedAt) ? 0 : Math.max(0, (now - startedAt) / 1000);
            percent = Math.min(99, Math.max(baseProgress, (elapsed / durationSeconds) * 100));
        }
        const bar = card.querySelector('.preview-scan-progress');
        const label = card.querySelector('.scan-percent');
        if (bar) bar.style.width = `${percent}%`;
        if (label) label.textContent = `${Math.round(percent)}%`;
    });
}

function chooseLogStage(job) {
    if (selectedLogStage) return selectedLogStage;
    return job.current_stage || job.error_summary?.stage || job.next_stage || [...(job.stage_order || [])].reverse().find(stage => job.stages?.[stage]?.log_path);
}

async function loadSelectedLog(job) {
    const stage = chooseLogStage(job);
    nodes.copyLogButton.disabled = !stage;
    nodes.copyMessage.textContent = '';
    if (!stage) {
        nodes.logHint.textContent = '暂无日志';
        nodes.logText.textContent = '-';
        return;
    }
    nodes.logHint.textContent = `显示：${stageNames[stage] || stage} 的日志尾部`;
    const log = await getJson(`/api/video-link/jobs/${job.job_id}/logs/${stage}?tail=80`).catch(() => ({ lines: [] }));
    nodes.logText.textContent = (log.lines || []).join('\n') || '-';
}

async function copySelectedLog() {
    if (!selectedJobId) return;
    const job = await getJson(`/api/video-link/jobs/${selectedJobId}`);
    const stage = chooseLogStage(job);
    if (!stage) return;
    try {
        const log = await getJson(`/api/video-link/jobs/${selectedJobId}/logs/${stage}?full=1`);
        await copyText(log.text || (log.lines || []).join('\n'));
        nodes.copyMessage.textContent = '已复制';
    } catch (error) {
        nodes.copyMessage.textContent = `复制失败：${error.message}`;
    }
}

async function copyText(text) {
    const value = String(text || '');
    if (navigator.clipboard?.writeText && window.isSecureContext) {
        try {
            await navigator.clipboard.writeText(value);
            return;
        } catch (_error) {
            // Fall through to the textarea path for HTTP/Tailscale dashboards.
        }
    }
    const textarea = document.createElement('textarea');
    textarea.value = value;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.top = '-1000px';
    textarea.style.left = '-1000px';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    try {
        if (!document.execCommand('copy')) {
            throw new Error('浏览器拒绝复制');
        }
    } finally {
        textarea.remove();
    }
}

function ensureImageViewer() {
    if (imageViewer.node) return imageViewer.node;
    const viewer = document.createElement('div');
    viewer.className = 'image-viewer';
    viewer.hidden = true;
    viewer.innerHTML = `
        <div class="image-viewer-backdrop" data-image-viewer-close></div>
        <div class="image-viewer-panel" role="dialog" aria-modal="true" aria-label="图片预览">
            <div class="image-viewer-toolbar">
                <strong class="image-viewer-title">图片预览</strong>
                <div class="image-viewer-actions">
                    <button type="button" data-image-viewer-zoom="out" aria-label="缩小">−</button>
                    <span data-image-viewer-scale>100%</span>
                    <button type="button" data-image-viewer-zoom="in" aria-label="放大">+</button>
                    <button type="button" data-image-viewer-reset>1:1</button>
                    <button type="button" data-image-viewer-close aria-label="关闭">×</button>
                </div>
            </div>
            <div class="image-viewer-stage">
                <img alt="">
            </div>
        </div>`;
    document.body.appendChild(viewer);
    imageViewer.node = viewer;
    imageViewer.image = viewer.querySelector('img');
    imageViewer.scaleLabel = viewer.querySelector('[data-image-viewer-scale]');
    viewer.addEventListener('click', event => {
        if (event.target.closest('[data-image-viewer-close]')) {
            closeImageViewer();
            return;
        }
        const zoomButton = event.target.closest('[data-image-viewer-zoom]');
        if (zoomButton) {
            zoomImageViewer(zoomButton.dataset.imageViewerZoom === 'in' ? 0.2 : -0.2);
            return;
        }
        if (event.target.closest('[data-image-viewer-reset]')) {
            setImageViewerScale(1);
        }
    });
    viewer.querySelector('.image-viewer-stage')?.addEventListener('wheel', event => {
        event.preventDefault();
        zoomImageViewer(event.deltaY < 0 ? 0.1 : -0.1);
    }, { passive: false });
    return viewer;
}

function openImageViewer(src, alt = '') {
    if (!src) return;
    const viewer = ensureImageViewer();
    imageViewer.image.src = src;
    imageViewer.image.alt = alt || '图片预览';
    viewer.hidden = false;
    document.body.classList.add('image-viewer-open');
    setImageViewerScale(1);
}

function closeImageViewer() {
    if (!imageViewer.node) return;
    imageViewer.node.hidden = true;
    document.body.classList.remove('image-viewer-open');
}

function setImageViewerScale(scale) {
    imageViewer.scale = Math.max(0.25, Math.min(5, scale));
    if (imageViewer.image) {
        imageViewer.image.style.transform = `scale(${imageViewer.scale})`;
    }
    if (imageViewer.scaleLabel) {
        imageViewer.scaleLabel.textContent = `${Math.round(imageViewer.scale * 100)}%`;
    }
}

function zoomImageViewer(delta) {
    setImageViewerScale(imageViewer.scale + delta);
}

function bindImageViewer() {
    document.addEventListener('dblclick', event => {
        const image = event.target.closest('img[data-image-viewer-src]');
        if (!image) return;
        event.preventDefault();
        openImageViewer(image.dataset.imageViewerSrc || image.currentSrc || image.src, image.dataset.imageViewerAlt || image.alt || '');
    });
    document.addEventListener('keydown', event => {
        if (!imageViewer.node || imageViewer.node.hidden) return;
        if (event.key === 'Escape') closeImageViewer();
        if (event.key === '+' || event.key === '=') zoomImageViewer(0.2);
        if (event.key === '-' || event.key === '_') zoomImageViewer(-0.2);
        if (event.key === '0') setImageViewerScale(1);
    });
}

async function runSelectedJob() {
    if (!selectedJobId) return;
    nodes.runButton.disabled = true;
    try {
        const action = nodes.runButton.dataset.action || (currentJob?.status === 'succeeded' ? 'open-run-dir' : 'run');
        await getJson(`/api/video-link/jobs/${selectedJobId}/${action}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}'
        });
        if (action === 'run' || action === 'stop') {
            await refreshSelectedJob();
        } else {
            nodes.runButton.disabled = false;
        }
    } catch (error) {
        nodes.selectedSubtitle.textContent = error.message;
        nodes.runButton.disabled = false;
    }
}

async function boot() {
    nodes.consoleTab.addEventListener('click', () => setView('console'));
    nodes.previewTab.addEventListener('click', () => setView('preview'));
    nodes.vscodeTab.addEventListener('click', () => setView('vscode'));
    nodes.jobForm.addEventListener('submit', createJob);
    nodes.addUrlButton.addEventListener('click', addPendingUrls);
    nodes.videoUrlInput.addEventListener('keydown', event => {
        if (event.key === 'Enter') {
            event.preventDefault();
            addPendingUrls();
        }
    });
    renderUrlList();
    nodes.refreshJobsButton.addEventListener('click', refreshJobs);
    nodes.refreshPreviewButton.addEventListener('click', refreshJobs);
    nodes.startVscodeButton.addEventListener('click', () => ensureVscodeSession(false));
    nodes.restartVscodeButton.addEventListener('click', () => ensureVscodeSession(true));
    nodes.stopVscodeButton.addEventListener('click', stopVscodeSession);
    nodes.docPreviewClose.addEventListener('click', closeDocPreview);
    nodes.runButton.addEventListener('click', runSelectedJob);
    nodes.copyLogButton.addEventListener('click', copySelectedLog);
    bindImageViewer();
    await loadOptions();
    setView(currentView, false);
    await refreshJobs();
    if (selectedJobId) await refreshSelectedJob();
    startScanAnimation();
    refreshTimer = setInterval(() => {
        if (selectedJobId) {
            refreshSelectedJob();
        } else {
            refreshJobs();
        }
    }, 5000);
}

boot().catch(error => {
    nodes.formError.textContent = error.message;
});
