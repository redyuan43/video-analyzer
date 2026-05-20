const stageNames = {
    probe: '探测时长',
    prepare: '下载/上下文',
    'analyze-core': '核心分析',
    'verify-core': '校验产物',
    multidoc: '多文档分析',
    'deep-v2': '章节深度报告',
    'image-prompts': '生成配图提示词',
    'final-publish': '最终定稿/发布'
};

let selectedJobId = new URLSearchParams(window.location.search).get('job');
let selectedLogStage = null;
let refreshTimer = null;

const nodes = {
    jobForm: document.getElementById('jobForm'),
    formError: document.getElementById('formError'),
    createButton: document.getElementById('createButton'),
    batchResult: document.getElementById('batchResult'),
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
    corePanel: document.getElementById('corePanel'),
    coreRows: document.getElementById('coreRows'),
    artifactSummary: document.getElementById('artifactSummary'),
    logHint: document.getElementById('logHint'),
    logText: document.getElementById('logText'),
    copyLogButton: document.getElementById('copyLogButton'),
    copyMessage: document.getElementById('copyMessage')
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

function statusBadge(status) {
    const value = status || 'pending';
    return `<span class="status ${value}">${value}</span>`;
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

async function loadOptions() {
    const options = await getJson('/api/video-link/options');
    const defaults = options.defaults || {};
    const choices = options.choices || {};
    fillSelect('analysisMode', choices.analysis_modes, defaults.analysis_mode);
    fillSelect('profile', choices.profiles, defaults.profile);
    fillSelect('cookieBrowser', choices.cookie_browsers, defaults.cookies_from_browser);
    document.getElementById('runName').value = defaults.run_name || 'operation-manual';
    document.getElementById('skipImages').checked = Boolean(defaults.skip_images);
    document.getElementById('keepExisting').checked = Boolean(defaults.keep_existing);
    document.getElementById('includeSubtitles').checked = Boolean(defaults.include_subtitles);
    document.getElementById('preferSubtitleTranscript').checked = Boolean(defaults.prefer_subtitle_transcript);
    document.getElementById('includeComments').checked = Boolean(defaults.include_comments);
    document.getElementById('refreshContext').checked = Boolean(defaults.refresh_context);
    document.getElementById('maxComments').value = defaults.max_comments ?? 30;
    document.getElementById('subtitleLangs').value = defaults.subtitle_langs || '';
}

function jobPayload() {
    return {
        video_urls_text: document.getElementById('videoUrls').value.trim(),
        analysis_mode: document.getElementById('analysisMode').value,
        profile: document.getElementById('profile').value,
        run_name: document.getElementById('runName').value.trim(),
        cookies_from_browser: document.getElementById('cookieBrowser').value,
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
        window.history.replaceState({}, '', url);
    }
    refreshSelectedJob();
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

function renderJobList(jobs) {
    const orderedJobs = sortJobsForAttention(jobs);
    nodes.jobList.innerHTML = orderedJobs.length ? orderedJobs.map(job => {
        const selected = job.job_id === selectedJobId ? ' selected' : '';
        const statusClass = job.status ? ` ${job.status}` : '';
        const title = escapeHtml(job.video_url || job.job_id);
        const stage = job.current_stage || job.error_summary?.stage || job.next_stage || '-';
        return `<button class="job-item${selected}${statusClass}" type="button" data-job-id="${job.job_id}">
            <strong>${title}</strong>
            <span>${escapeHtml(job.status || '-')} · ${escapeHtml(stageNames[stage] || stage)}</span>
        </button>`;
    }).join('') : '<div class="empty">暂无任务</div>';
    bindJobButtons();
}

async function refreshJobs() {
    const data = await getJson('/api/video-link/jobs?limit=50');
    renderGlobal(data);
    const jobs = data.jobs || [];
    renderJobList(jobs);
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
    renderJobList(jobs);
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
    document.querySelectorAll('.job-item, .lane-item').forEach(button => {
        button.addEventListener('click', () => {
            if (button.dataset.jobId) selectJob(button.dataset.jobId);
        });
    });
}

function renderEmpty() {
    nodes.runButton.disabled = true;
    nodes.selectedTitle.textContent = '未选择任务';
    nodes.selectedSubtitle.textContent = '创建或选择一个任务后查看进度。';
}

function renderServiceOffline(error) {
    nodes.runButton.disabled = true;
    nodes.selectedTitle.textContent = '服务未连接';
    nodes.selectedSubtitle.textContent = error.message;
    setText(nodes.statusValue, 'offline');
    setText(nodes.currentStageValue, '-');
    setText(nodes.nextStageValue, '-');
    setText(nodes.queueValue, '-');
    nodes.progressText.textContent = '0/0 · 0%';
    nodes.progressBar.style.width = '0%';
}

function activeProcess(job) {
    const stage = job.current_stage || job.next_stage;
    return job.process || job.stages?.[stage]?.process || null;
}

function runDisabledReason(job) {
    const process = activeProcess(job);
    if (process?.alive) return `已有外部进程 PID ${process.pid} 仍在运行`;
    if (job.runner?.status === 'running') return '阶段仍在运行，等待完成';
    if (job.runner?.status === 'queued') return `等待资源: ${job.queue?.resource || job.runner?.queued_for || '-'}`;
    return '';
}

function renderJob(job) {
    const progress = job.progress || {};
    const queue = job.queue || {};
    const reason = runDisabledReason(job);
    const process = activeProcess(job);
    nodes.selectedTitle.textContent = job.video_url || job.job_id;
    nodes.selectedSubtitle.textContent = reason ? `任务 ID: ${job.job_id} · ${reason}` : `任务 ID: ${job.job_id}`;
    nodes.runButton.disabled = Boolean(reason);
    nodes.runButton.textContent = job.status === 'failed' ? '重试失败阶段' : '继续运行';
    setText(nodes.statusValue, job.status);
    setText(nodes.currentStageValue, stageNames[job.current_stage] || job.current_stage);
    setText(nodes.nextStageValue, stageNames[job.next_stage] || job.next_stage);
    const queueText = queue.resource ? `${queue.resource} #${queue.position || '-'}/${queue.size || '-'}` : '-';
    setText(nodes.queueValue, process?.alive ? `${queueText} · PID ${process.pid}` : queueText);
    nodes.progressText.textContent = `${progress.completed || 0}/${progress.total || 0} · ${progress.percent || 0}%`;
    nodes.progressBar.style.width = `${progress.percent || 0}%`;
    setText(nodes.detailUrl, job.video_url);
    setText(nodes.detailRunDir, job.summary?.run_dir || job.run_dir);
    setText(nodes.detailMode, `${job.options?.analysis_mode || '-'} -> ${job.resolved_mode || '-'}`);
    setText(nodes.detailUpdated, job.updated_at);

    if (job.error_summary) {
        nodes.errorPanel.hidden = false;
        nodes.errorTitle.textContent = `流程失败：${job.error_summary.stage_label || job.error_summary.stage || '未知阶段'}`;
        nodes.errorMessage.textContent = job.error_summary.message || '未提供错误信息';
    } else {
        nodes.errorPanel.hidden = true;
    }
    renderStages(job);
    renderCore(job.core_progress);
    renderArtifacts(job.summary || {});
    loadSelectedLog(job);
}

function renderStages(job) {
    nodes.stageRows.innerHTML = (job.stage_order || []).map(stage => {
        const info = job.stages?.[stage] || {};
        const queue = info.queue_position ? `${info.queued_for || ''} #${info.queue_position}` : (info.queued_for || '-');
        const error = info.error ? `<div class="row-error">${escapeHtml(info.error)}</div>` : '';
        const retry = info.retry_reason ? `<div class="muted">${escapeHtml(info.retry_reason)}</div>` : '';
        const log = info.log_path ? `<button class="log-link" type="button" data-stage="${stage}">查看日志</button>` : '-';
        return `<tr>
            <td>${escapeHtml(stageNames[stage] || stage)}${error}${retry}</td>
            <td>${statusBadge(info.status)}</td>
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
}

function renderCore(core) {
    const hasVisibleStep = core && (core.steps || []).some(step => step.status !== 'pending');
    nodes.corePanel.hidden = !hasVisibleStep;
    if (!hasVisibleStep) return;
    nodes.coreRows.innerHTML = core.steps.map(step => `<tr>
        <td>${escapeHtml(step.label)}</td>
        <td>${statusBadge(step.status)}</td>
        <td>${escapeHtml(duration(step.duration_seconds))}</td>
        <td>${escapeHtml(step.message || '-')}</td>
    </tr>`).join('');
}

function renderArtifacts(summary) {
    nodes.artifactSummary.innerHTML = [
        `Markdown: ${(summary.markdown_files || []).length}`,
        `导出文件: ${(summary.export_files || []).length}`,
        `配图提示词: ${(summary.prompt_files || []).length}`,
        `最终图片: ${(summary.final_images || []).length}`
    ].join('<br>');
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
        await navigator.clipboard.writeText(log.text || (log.lines || []).join('\n'));
        nodes.copyMessage.textContent = '已复制';
    } catch (error) {
        nodes.copyMessage.textContent = `复制失败：${error.message}`;
    }
}

async function runSelectedJob() {
    if (!selectedJobId) return;
    nodes.runButton.disabled = true;
    try {
        await getJson(`/api/video-link/jobs/${selectedJobId}/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}'
        });
        await refreshSelectedJob();
    } catch (error) {
        nodes.selectedSubtitle.textContent = error.message;
        nodes.runButton.disabled = false;
    }
}

async function boot() {
    nodes.jobForm.addEventListener('submit', createJob);
    nodes.refreshJobsButton.addEventListener('click', refreshJobs);
    nodes.runButton.addEventListener('click', runSelectedJob);
    nodes.copyLogButton.addEventListener('click', copySelectedLog);
    await loadOptions();
    await refreshJobs();
    if (selectedJobId) await refreshSelectedJob();
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
