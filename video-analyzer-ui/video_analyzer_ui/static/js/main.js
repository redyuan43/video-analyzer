const stageNames = {
    probe: '探测时长',
    prepare: '下载/上下文',
    'analyze-core': '核心分析',
    'verify-core': '校验产物',
    multidoc: '多文档分析',
    'deep-v2': '章节深度报告',
    'study-guide': '学习证据账本',
    'evidence-review': '证据复核/发布门禁',
    'qa-index': '问答证据索引',
    'image-prompts': '生成配图提示词',
    'final-publish': '最终定稿/发布'
};

const skillStageNames = {
    source: '整理原始证据',
    overview: '整体理解',
    extract: '五路提取',
    verify: '三重验证',
    build: '构造 Skills',
    link: '关联整理',
    test: '压力测试',
    deliver: '生成交付包'
};

const skillTestPhaseNames = {
    preparing: '准备测试',
    generating_tests: '生成压力测试题',
    blind_judging: '执行盲测',
    repairing: '修订 Skill',
    skill_completed: '单个 Skill 测试完成',
    completed: '压力测试完成'
};

const initialParams = new URLSearchParams(window.location.search);
let selectedJobId = initialParams.get('job') || document.querySelector('.app-shell')?.dataset.initialJob || null;
let selectedLogStage = null;
let selectedConsoleStage = '';
let selectedConsoleNodeId = '';
let consoleFlowJobId = '';
let consoleFlowCenteredKey = '';
let consoleFlowRenderKey = '';
let consoleFlowRenderSequence = 0;
let consoleFlowScale = 1;
let refreshTimer = null;
let currentJob = null;
let latestJobs = [];
let showNonRerunFailures = false;
let currentView = ['qa', 'vscode', 'settings'].includes(initialParams.get('view'))
    ? initialParams.get('view')
    : 'console';
let currentResourceView = initialParams.get('resource') === 'skills' ? 'skills' : 'docs';
let currentSkillsScope = ['projects', 'enabled', 'disabled', 'trash'].includes(initialParams.get('scope'))
    ? initialParams.get('scope')
    : 'current';
let pendingUrls = [];
let sourceMode = 'url';
let activeIntent = 'smart';
let promptTemplates = [];
let selectedTemplate = null;
let vscodeStarting = false;
let selectedDocPath = '';
let renderedDocListKey = '';
let loadedDocPreviewKey = '';
let loadedStudyKey = '';
let loadedQaHistoryKey = '';
let loadedSkillsWorkspaceKey = '';
let selectedCurrentSkillItemId = '';
let currentSkillWorkspace = null;
let currentSkillItemDetail = null;
let currentSkillDetailTab = 'evidence';
let selectedLibrarySkillId = '';
let currentLibrarySkill = null;
let currentSkillEditorTab = 'edit';
let skillLibrarySearchTimer = null;
let skillLibraryProjectSkillNames = null;
let selectedSkillProjectId = initialParams.get('project') || '';
let currentSkillProject = null;
let currentSkillProjectWorkspace = null;
let currentSkillProjectFlow = null;
let skillProjectPollTimer = null;
let skillProjectPollInFlight = false;
let skillProjectWorkbenchLoading = false;
let skillProjectListSignature = '';
let skillProjectDetailSignature = '';
let selectedSkillProjectFlowNodeId = '';
let skillProjectPackagePreview = null;
let skillProjectPackageBusy = '';
let skillProjectFlowActionBusy = '';
let settingsData = null;
let settingsSection = 'models';
let selectedSettingsModelId = '';
let selectedSettingsProfileName = '';
let selectedProfileFlowNodeId = 'asr';
let profileModelSelections = {};
let profileFlowDrawFrame = null;
let profileFlowResizeObserver = null;
let profileTestReport = null;
let profileTestRunning = false;
let modelTestRunning = false;
const skillProjectCandidateDrafts = new Map();
let selectedSkillProjectLogStage = '';
let skillProjectLogRequestId = 0;
const skillCandidateDrafts = new Map();
const frameTimeMaps = {};
const sourcePlayerState = {
    jobId: '',
    seconds: null,
    loaded: false
};
const imageViewer = {
    node: null,
    image: null,
    playButton: null,
    scaleLabel: null,
    scale: 1,
    jobId: '',
    seconds: null
};
const paneResizeState = {
    active: null,
    pointerId: null
};
const paneLayoutStorageKey = 'videoAnalyzerPaneLayout';
const panelVisibilityStorageKey = 'videoAnalyzerLearningPanels';
const skillsFocusStorageKey = 'videoAnalyzerSkillsFocusMode';
let skillsFocusEnabled = false;
const learningPanelVisibility = {
    docList: true,
    study: true,
    sourcePlayer: true,
    docContent: true
};

const nodes = {
    appShell: document.querySelector('.app-shell'),
    appTopbar: document.getElementById('appTopbar'),
    consoleTab: document.getElementById('consoleTab'),
    qaTab: document.getElementById('qaTab'),
    vscodeTab: document.getElementById('vscodeTab'),
    settingsTab: document.getElementById('settingsTab'),
    consoleView: document.getElementById('consoleView'),
    qaView: document.getElementById('qaView'),
    vscodeView: document.getElementById('vscodeView'),
    settingsView: document.getElementById('settingsView'),
    settingsSummary: document.getElementById('settingsSummary'),
    settingsModelsTab: document.getElementById('settingsModelsTab'),
    settingsProfilesTab: document.getElementById('settingsProfilesTab'),
    settingsModelsView: document.getElementById('settingsModelsView'),
    settingsProfilesView: document.getElementById('settingsProfilesView'),
    settingsModelKindFilter: document.getElementById('settingsModelKindFilter'),
    settingsModelSearch: document.getElementById('settingsModelSearch'),
    settingsModelList: document.getElementById('settingsModelList'),
    newModelButton: document.getElementById('newModelButton'),
    modelSettingsForm: document.getElementById('modelSettingsForm'),
    modelEditorTitle: document.getElementById('modelEditorTitle'),
    modelEditorMeta: document.getElementById('modelEditorMeta'),
    modelId: document.getElementById('modelId'),
    modelName: document.getElementById('modelName'),
    modelKind: document.getElementById('modelKind'),
    modelProtocol: document.getElementById('modelProtocol'),
    modelNameValue: document.getElementById('modelNameValue'),
    modelEndpoints: document.getElementById('modelEndpoints'),
    modelDeployment: document.getElementById('modelDeployment'),
    modelWorkerCount: document.getElementById('modelWorkerCount'),
    modelConcurrency: document.getElementById('modelConcurrency'),
    modelApiKeyEnv: document.getElementById('modelApiKeyEnv'),
    modelHealthUrl: document.getElementById('modelHealthUrl'),
    modelOptions: document.getElementById('modelOptions'),
    saveModelButton: document.getElementById('saveModelButton'),
    testModelButton: document.getElementById('testModelButton'),
    deleteModelButton: document.getElementById('deleteModelButton'),
    modelSettingsMessage: document.getElementById('modelSettingsMessage'),
    settingsProfileList: document.getElementById('settingsProfileList'),
    newProfileButton: document.getElementById('newProfileButton'),
    profileSettingsForm: document.getElementById('profileSettingsForm'),
    profileEditorTitle: document.getElementById('profileEditorTitle'),
    profileEditorMeta: document.getElementById('profileEditorMeta'),
    profileId: document.getElementById('profileId'),
    profileLabel: document.getElementById('profileLabel'),
    profileWorkflow: document.getElementById('profileWorkflow'),
    profileDescription: document.getElementById('profileDescription'),
    profileFlowLegend: document.getElementById('profileFlowLegend'),
    profileFlowViewport: document.getElementById('profileFlowViewport'),
    profileFlowCanvas: document.getElementById('profileFlowCanvas'),
    profileFlowEdges: document.getElementById('profileFlowEdges'),
    profileFlowNodes: document.getElementById('profileFlowNodes'),
    profileFlowInspector: document.getElementById('profileFlowInspector'),
    profileFlowInspectorTitle: document.getElementById('profileFlowInspectorTitle'),
    profileFlowInspectorMeta: document.getElementById('profileFlowInspectorMeta'),
    profileFlowInspectorBody: document.getElementById('profileFlowInspectorBody'),
    profileFlowInspectorActions: document.getElementById('profileFlowInspectorActions'),
    profileTestMode: document.getElementById('profileTestMode'),
    testProfileButton: document.getElementById('testProfileButton'),
    profileTestAvailability: document.getElementById('profileTestAvailability'),
    profileTestSummary: document.getElementById('profileTestSummary'),
    profileAsrModel: document.getElementById('profileAsrModel'),
    profileDiarizationModel: document.getElementById('profileDiarizationModel'),
    profileOcrModel: document.getElementById('profileOcrModel'),
    profileVisionModel: document.getElementById('profileVisionModel'),
    profileTextModel: document.getElementById('profileTextModel'),
    profileReviewModel: document.getElementById('profileReviewModel'),
    profileStudyModel: document.getElementById('profileStudyModel'),
    profileTriageModel: document.getElementById('profileTriageModel'),
    profileImageModel: document.getElementById('profileImageModel'),
    profilePipelineMode: document.getElementById('profilePipelineMode'),
    profileVlConcurrency: document.getElementById('profileVlConcurrency'),
    profileOcrConcurrency: document.getElementById('profileOcrConcurrency'),
    profileTextTimeout: document.getElementById('profileTextTimeout'),
    profileSettingsJson: document.getElementById('profileSettingsJson'),
    duplicateProfileButton: document.getElementById('duplicateProfileButton'),
    activateProfileButton: document.getElementById('activateProfileButton'),
    deleteProfileButton: document.getElementById('deleteProfileButton'),
    saveProfileButton: document.getElementById('saveProfileButton'),
    profileSettingsMessage: document.getElementById('profileSettingsMessage'),
    jobForm: document.getElementById('jobForm'),
    formError: document.getElementById('formError'),
    createButton: document.getElementById('createButton'),
    batchResult: document.getElementById('batchResult'),
    urlSourceTab: document.getElementById('urlSourceTab'),
    fileSourceTab: document.getElementById('fileSourceTab'),
    urlEntry: document.getElementById('urlEntry'),
    mediaEntry: document.getElementById('mediaEntry'),
    mediaFile: document.getElementById('mediaFile'),
    videoUrlInput: document.getElementById('videoUrlInput'),
    addUrlButton: document.getElementById('addUrlButton'),
    videoUrls: document.getElementById('videoUrls'),
    urlList: document.getElementById('urlList'),
    intentCards: Array.from(document.querySelectorAll('.intent-card')),
    templatePanel: document.getElementById('templatePanel'),
    templateSearch: document.getElementById('templateSearch'),
    templateCategory: document.getElementById('templateCategory'),
    templateStatus: document.getElementById('templateStatus'),
    templateList: document.getElementById('templateList'),
    selectedTemplatePanel: document.getElementById('selectedTemplatePanel'),
    selectedTemplateName: document.getElementById('selectedTemplateName'),
    clearTemplateButton: document.getElementById('clearTemplateButton'),
    focusPrompt: document.getElementById('focusPrompt'),
    globalSummary: document.getElementById('globalSummary'),
    resourceLanes: document.getElementById('resourceLanes'),
    refreshJobsButton: document.getElementById('refreshJobsButton'),
    showNonRerunFailures: document.getElementById('showNonRerunFailures'),
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
    consoleFlowPanel: document.getElementById('consoleFlowPanel'),
    consoleFlowSummary: document.getElementById('consoleFlowSummary'),
    consoleFlowPrevious: document.getElementById('consoleFlowPrevious'),
    consoleFlowZoomOut: document.getElementById('consoleFlowZoomOut'),
    consoleFlowFit: document.getElementById('consoleFlowFit'),
    consoleFlowZoomIn: document.getElementById('consoleFlowZoomIn'),
    consoleFlowCurrent: document.getElementById('consoleFlowCurrent'),
    consoleFlowNext: document.getElementById('consoleFlowNext'),
    consoleStageFlowViewport: document.getElementById('consoleStageFlowViewport'),
    consoleStageFlow: document.getElementById('consoleStageFlow'),
    consoleStageInspector: document.getElementById('consoleStageInspector'),
    consoleStageInspectorKicker: document.getElementById('consoleStageInspectorKicker'),
    consoleStageInspectorTitle: document.getElementById('consoleStageInspectorTitle'),
    consoleStageInspectorStatus: document.getElementById('consoleStageInspectorStatus'),
    consoleStageInspectorMetrics: document.getElementById('consoleStageInspectorMetrics'),
    consoleStageInspectorMessage: document.getElementById('consoleStageInspectorMessage'),
    consoleStageInspectorArtifacts: document.getElementById('consoleStageInspectorArtifacts'),
    consoleStageLogButton: document.getElementById('consoleStageLogButton'),
    detailUrl: document.getElementById('detailUrl'),
    detailRunDir: document.getElementById('detailRunDir'),
    detailMode: document.getElementById('detailMode'),
    detailProfile: document.getElementById('detailProfile'),
    detailModels: document.getElementById('detailModels'),
    detailUpdated: document.getElementById('detailUpdated'),
    stageRows: document.getElementById('stageRows'),
    stageDurationSummary: document.getElementById('stageDurationSummary'),
    corePanel: document.getElementById('corePanel'),
    corePanelTitle: document.getElementById('corePanelTitle'),
    coreRows: document.getElementById('coreRows'),
    consoleSummaryPanel: document.getElementById('consoleSummaryPanel'),
    consoleSummaryHeadline: document.getElementById('consoleSummaryHeadline'),
    consoleSummaryGrid: document.getElementById('consoleSummaryGrid'),
    consoleResultSummary: document.getElementById('consoleResultSummary'),
    coreDiagnosticsPanel: document.getElementById('coreDiagnosticsPanel'),
    coreDiagnosticsSummary: document.getElementById('coreDiagnosticsSummary'),
    coreDiagnosticsStatus: document.getElementById('coreDiagnosticsStatus'),
    coreDiagnosticsMetrics: document.getElementById('coreDiagnosticsMetrics'),
    coreDiagnosticsIssues: document.getElementById('coreDiagnosticsIssues'),
    artifactSummary: document.getElementById('artifactSummary'),
    consoleSkillSummary: document.getElementById('consoleSkillSummary'),
    openSkillsWorkspaceButton: document.getElementById('openSkillsWorkspaceButton'),
    logHint: document.getElementById('logHint'),
    logText: document.getElementById('logText'),
    copyLogButton: document.getElementById('copyLogButton'),
    copyMessage: document.getElementById('copyMessage'),
    qaSummary: document.getElementById('qaSummary'),
    qaWarnings: document.getElementById('qaWarnings'),
    qaMessages: document.getElementById('qaMessages'),
    qaForm: document.getElementById('qaForm'),
    qaQuestion: document.getElementById('qaQuestion'),
    qaAskButton: document.getElementById('qaAskButton'),
    qaSourcePlayerPanel: document.querySelector('.qa-source-player'),
    qaLayout: document.querySelector('.qa-layout'),
    qaSidePanel: document.querySelector('.qa-side-panel'),
    qaSourcePlayerSummary: document.getElementById('qaSourcePlayerSummary'),
    qaSourcePlayerOpenLink: document.getElementById('qaSourcePlayerOpenLink'),
    qaSourcePlayerStopButton: document.getElementById('qaSourcePlayerStopButton'),
    qaSourcePlayerBody: document.getElementById('qaSourcePlayerBody'),
    qaSourceResizer: document.querySelector('.qa-source-resizer'),
    qaSourceHeightResizer: document.querySelector('.qa-source-height-resizer'),
    skillSummary: document.getElementById('skillSummary'),
    skillWarnings: document.getElementById('skillWarnings'),
    skillProfile: document.getElementById('skillProfile'),
    skillProgress: document.getElementById('skillProgress'),
    skillProgressBar: document.getElementById('skillProgressBar'),
    skillLiveActivity: document.getElementById('skillLiveActivity'),
    skillStageRail: document.getElementById('skillStageRail'),
    skillOverviewReview: document.getElementById('skillOverviewReview'),
    skillOverviewPreview: document.getElementById('skillOverviewPreview'),
    skillOverviewFeedback: document.getElementById('skillOverviewFeedback'),
    regenerateSkillOverviewButton: document.getElementById('regenerateSkillOverviewButton'),
    confirmSkillOverviewButton: document.getElementById('confirmSkillOverviewButton'),
    skillCandidateReview: document.getElementById('skillCandidateReview'),
    skillCandidateList: document.getElementById('skillCandidateList'),
    confirmSkillCandidatesButton: document.getElementById('confirmSkillCandidatesButton'),
    generateSkillButton: document.getElementById('generateSkillButton'),
    createTargetedSkillProjectButton: document.getElementById('createTargetedSkillProjectButton'),
    resumeSkillButton: document.getElementById('resumeSkillButton'),
    cancelSkillButton: document.getElementById('cancelSkillButton'),
    enableSkillButton: document.getElementById('enableSkillButton'),
    vscodeSummary: document.getElementById('vscodeSummary'),
    resourceToolbar: document.getElementById('resourceToolbar'),
    resourceDocsTab: document.getElementById('resourceDocsTab'),
    resourceSkillsTab: document.getElementById('resourceSkillsTab'),
    skillsResourceDocsTab: document.getElementById('skillsResourceDocsTab'),
    skillsResourceSkillsTab: document.getElementById('skillsResourceSkillsTab'),
    toggleSkillsFocusButton: document.getElementById('toggleSkillsFocusButton'),
    docsToolbarActions: document.getElementById('docsToolbarActions'),
    resourceDocsView: document.getElementById('resourceDocsView'),
    resourceSkillsView: document.getElementById('resourceSkillsView'),
    startVscodeButton: document.getElementById('startVscodeButton'),
    openVscodeLink: document.getElementById('openVscodeLink'),
    restartVscodeButton: document.getElementById('restartVscodeButton'),
    stopVscodeButton: document.getElementById('stopVscodeButton'),
    vscodeDocs: document.querySelector('.vscode-docs'),
    toggleDocListPanel: document.getElementById('toggleDocListPanel'),
    toggleStudyPanel: document.getElementById('toggleStudyPanel'),
    toggleSourcePlayerPanel: document.getElementById('toggleSourcePlayerPanel'),
    toggleDocContentPanel: document.getElementById('toggleDocContentPanel'),
    vscodePlaceholder: document.getElementById('vscodePlaceholder'),
    vscodeFrame: document.getElementById('vscodeFrame'),
    docPreviewSummary: document.getElementById('docPreviewSummary'),
    docListPanel: document.querySelector('.doc-list-panel'),
    docList: document.getElementById('docList'),
    docListResizer: document.querySelector('.doc-list-resizer'),
    studyPanel: document.getElementById('studyPanel'),
    studyResizer: document.querySelector('[data-resize-pane="study"]'),
    sourcePlayerPanel: document.getElementById('sourcePlayerPanel'),
    sourcePlayerResizer: document.querySelector('[data-resize-pane="source-player"]'),
    sourcePlayerSummary: document.getElementById('sourcePlayerSummary'),
    sourcePlayerOpenLink: document.getElementById('sourcePlayerOpenLink'),
    sourcePlayerStopButton: document.getElementById('sourcePlayerStopButton'),
    sourcePlayerBody: document.getElementById('sourcePlayerBody'),
    docPreviewPanel: document.querySelector('.doc-preview-panel'),
    docPreviewTitle: document.getElementById('docPreviewTitle'),
    docOpenLink: document.getElementById('docOpenLink'),
    docPreviewClose: document.getElementById('docPreviewClose'),
    docPreviewBody: document.getElementById('docPreviewBody'),
    studySummary: document.getElementById('studySummary'),
    studyDecision: document.getElementById('studyDecision'),
    studyBody: document.getElementById('studyBody'),
    skillsScopeTabs: Array.from(document.querySelectorAll('.skills-scope-tab')),
    skillLibrarySearchLabel: document.getElementById('skillLibrarySearchLabel'),
    skillLibrarySearch: document.getElementById('skillLibrarySearch'),
    currentSkillsWorkspace: document.getElementById('currentSkillsWorkspace'),
    currentSkillsGrid: document.querySelector('.skills-current-grid'),
    skillProjectsWorkspace: document.getElementById('skillProjectsWorkspace'),
    skillProjectsCount: document.getElementById('skillProjectsCount'),
    skillProjectsList: document.getElementById('skillProjectsList'),
    skillProjectForm: document.getElementById('skillProjectForm'),
    skillProjectTitle: document.getElementById('skillProjectTitle'),
    skillProjectGoal: document.getElementById('skillProjectGoal'),
    createSkillProjectButton: document.getElementById('createSkillProjectButton'),
    skillProjectFlowSummary: document.getElementById('skillProjectFlowSummary'),
    skillProjectFlowStatus: document.getElementById('skillProjectFlowStatus'),
    skillProjectFlow: document.getElementById('skillProjectFlow'),
    skillProjectInspectorTitle: document.getElementById('skillProjectInspectorTitle'),
    skillProjectInspectorMeta: document.getElementById('skillProjectInspectorMeta'),
    skillProjectDetail: document.getElementById('skillProjectDetail'),
    skillProjectSources: document.getElementById('skillProjectSources'),
    skillProjectPackageId: document.getElementById('skillProjectPackageId'),
    previewSkillProjectPackageButton: document.getElementById('previewSkillProjectPackageButton'),
    skillProjectPackageStatus: document.getElementById('skillProjectPackageStatus'),
    skillProjectPackagePreview: document.getElementById('skillProjectPackagePreview'),
    importSkillProjectPackageButton: document.getElementById('importSkillProjectPackageButton'),
    skillProjectLogSummary: document.getElementById('skillProjectLogSummary'),
    skillProjectLogTabs: document.getElementById('skillProjectLogTabs'),
    copySkillProjectLogButton: document.getElementById('copySkillProjectLogButton'),
    skillProjectLogText: document.getElementById('skillProjectLogText'),
    skillLibraryWorkspace: document.getElementById('skillLibraryWorkspace'),
    skillLibraryGrid: document.getElementById('skillLibraryGrid'),
    skillsCurrentCount: document.getElementById('skillsCurrentCount'),
    skillsCurrentList: document.getElementById('skillsCurrentList'),
    skillsDetailTitle: document.getElementById('skillsDetailTitle'),
    skillsDetailMeta: document.getElementById('skillsDetailMeta'),
    skillsDetailPreview: document.getElementById('skillsDetailPreview'),
    skillsDetailTabs: Array.from(document.querySelectorAll('.skills-detail-tab')),
    skillsInspectorBody: document.getElementById('skillsInspectorBody'),
    skillLibraryTitle: document.getElementById('skillLibraryTitle'),
    skillLibraryCount: document.getElementById('skillLibraryCount'),
    skillLibraryList: document.getElementById('skillLibraryList'),
    skillEditorTitle: document.getElementById('skillEditorTitle'),
    skillEditorMeta: document.getElementById('skillEditorMeta'),
    saveSkillButton: document.getElementById('saveSkillButton'),
    disableSkillButton: document.getElementById('disableSkillButton'),
    restoreSkillButton: document.getElementById('restoreSkillButton'),
    deleteSkillButton: document.getElementById('deleteSkillButton'),
    permanentDeleteSkillButton: document.getElementById('permanentDeleteSkillButton'),
    skillEditorTabs: Array.from(document.querySelectorAll('.skill-editor-tab')),
    skillEditor: document.getElementById('skillEditor'),
    skillRenderedPreview: document.getElementById('skillRenderedPreview'),
    skillEditorMessage: document.getElementById('skillEditorMessage'),
    skillAuxiliaryFiles: document.getElementById('skillAuxiliaryFiles'),
    skillVersionList: document.getElementById('skillVersionList')
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

function ensureSkillActivityNodes() {
    if (!nodes.skillProgress && nodes.skillProgressBar?.parentElement) {
        nodes.skillProgress = nodes.skillProgressBar.parentElement;
        nodes.skillProgress.id = 'skillProgress';
    }
    if (!nodes.skillProgress) return;
    if (!nodes.skillLiveActivity) {
        nodes.skillLiveActivity = document.createElement('div');
        nodes.skillLiveActivity.id = 'skillLiveActivity';
        nodes.skillLiveActivity.className = 'skill-live-activity';
        nodes.skillLiveActivity.hidden = true;
        nodes.skillProgress.insertAdjacentElement('afterend', nodes.skillLiveActivity);
    }
    if (!nodes.skillStageRail) {
        nodes.skillStageRail = document.createElement('div');
        nodes.skillStageRail.id = 'skillStageRail';
        nodes.skillStageRail.className = 'skill-stage-rail';
        nodes.skillStageRail.setAttribute('aria-label', 'Skills 蒸馏阶段');
        nodes.skillStageRail.hidden = true;
        nodes.skillLiveActivity.insertAdjacentElement('afterend', nodes.skillStageRail);
    }
}

function setText(node, value) {
    node.textContent = value || '-';
}

function loadSkillsFocusMode() {
    try {
        skillsFocusEnabled = localStorage.getItem(skillsFocusStorageKey) === 'true';
    } catch (_error) {
        skillsFocusEnabled = false;
    }
    applySkillsFocusMode();
}

function applySkillsFocusMode() {
    const active = Boolean(
        skillsFocusEnabled
        && currentView === 'vscode'
        && currentResourceView === 'skills'
    );
    nodes.appShell?.classList.toggle('skills-focus-mode', active);
    if (!nodes.toggleSkillsFocusButton) return;
    nodes.toggleSkillsFocusButton.setAttribute('aria-pressed', String(active));
    nodes.toggleSkillsFocusButton.setAttribute(
        'aria-label',
        active ? '退出 Skills 专注模式' : '进入 Skills 专注模式'
    );
    nodes.toggleSkillsFocusButton.title = active ? '退出 Skills 专注模式（Esc）' : '进入 Skills 专注模式';
    const icon = nodes.toggleSkillsFocusButton.querySelector('span');
    if (icon) icon.textContent = active ? '↙' : '⛶';
}

function toggleSkillsFocusMode(force) {
    skillsFocusEnabled = typeof force === 'boolean' ? force : !skillsFocusEnabled;
    localStorage.setItem(skillsFocusStorageKey, String(skillsFocusEnabled));
    applySkillsFocusMode();
    window.requestAnimationFrame(constrainSkillsLayouts);
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
        const info = job.stages?.[stage] || {};
        const value = Number(info.duration_seconds);
        if (Number.isFinite(value) && value > 0) return total + value;
        const live = info.status === 'running' ? elapsedSeconds(info.started_at) : null;
        return live != null ? total + live : total;
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

function timestampMilliseconds(value) {
    if (!value) return null;
    const normalized = String(value).replace(/([+-]\d{2})(\d{2})$/, '$1:$2');
    const parsed = Date.parse(normalized);
    return Number.isFinite(parsed) ? parsed : null;
}

function elapsedSeconds(startedAt, finishedAt = null) {
    const started = timestampMilliseconds(startedAt);
    const finished = timestampMilliseconds(finishedAt) ?? Date.now();
    if (started == null || finished < started) return null;
    return Math.max(0, (finished - started) / 1000);
}

function consoleElapsedMarkup(startedAt, finishedAt, prefix) {
    const seconds = elapsedSeconds(startedAt, finishedAt);
    if (seconds == null) return '-';
    const live = !finishedAt;
    const attributes = live
        ? ` data-console-elapsed-start="${escapeHtml(startedAt)}" data-console-elapsed-prefix="${escapeHtml(prefix)}"`
        : '';
    return `<span class="console-live-elapsed"${attributes}>${escapeHtml(prefix)}${escapeHtml(formatClock(seconds))}</span>`;
}

function updateConsoleElapsedClock() {
    document.querySelectorAll('[data-console-elapsed-start]').forEach(node => {
        const seconds = elapsedSeconds(node.dataset.consoleElapsedStart);
        if (seconds == null) return;
        node.textContent = `${node.dataset.consoleElapsedPrefix || ''}${formatClock(seconds)}`;
    });
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

const templateCategoryLabels = {
    all: '全部模板',
    building: '建筑工程',
    call: '通话',
    education: '教育',
    finance: '金融',
    functionality: '功能工具',
    genera: '通用',
    interview: '访谈',
    it: 'IT',
    law: '法律',
    medical: '医疗',
    meeting: '会议',
    sales: '销售',
    speech: '演讲'
};

const intentDefaults = {
    smart: {
        analysisMode: 'auto',
        prompt: '请自动判断内容类型，优先生成结构清晰、可直接阅读的总结。保留关键结论、行动项、风险点和后续问题；如果内容更像会议、讲座、访谈、通话或演讲，请自动采用最合适的结构。'
    },
    transcribe: {
        analysisMode: 'fast',
        prompt: '请优先保证转写内容完整和可读，只做轻量整理。输出重点包括：完整转写、关键段落、明显的专有名词修正，以及非常简短的摘要。'
    }
};

const templateBlockRe = /【模板指令开始】[\s\S]*?【模板指令结束】\n*/;
const userSupplementMarker = '【用户补充】';
const maxFocusPromptChars = 3900;

async function getJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
}

function setSelectValue(id, value) {
    const node = document.getElementById(id);
    if (node && Array.from(node.options).some(option => option.value === value)) {
        node.value = value;
    }
}

function stripTemplateBlock(value) {
    return String(value || '').replace(templateBlockRe, '').replace(userSupplementMarker, '').trim();
}

function compactTemplatePrompt(template, userSupplement = '') {
    const source = String(template.prompt_original || '').trim();
    const header = [
        '【模板指令开始】',
        `模板：${template.title_zh || template.title}`,
        `分类：${template.first_category_zh || template.first_category} / ${template.second_category_zh || template.second_category}`,
        '',
        '请按以下模板分析输入的音频、视频、字幕或转写文本：',
        source,
        '【模板指令结束】'
    ].join('\n');
    const suffix = userSupplement ? `\n\n${userSupplementMarker}\n${userSupplement.trim()}` : '';
    const budget = maxFocusPromptChars - suffix.length;
    const body = header.length > budget
        ? `${header.slice(0, Math.max(0, budget - 80)).trim()}\n\n[模板内容已压缩到长度限制内]\n【模板指令结束】`
        : header;
    return `${body}${suffix}`.slice(0, maxFocusPromptChars).trim();
}

function applyFocusPromptTemplate(template) {
    selectedTemplate = template;
    const supplement = stripTemplateBlock(nodes.focusPrompt.value);
    nodes.focusPrompt.value = compactTemplatePrompt(template, supplement);
    renderSelectedTemplate();
}

function renderSelectedTemplate() {
    const name = selectedTemplate?.title_zh || selectedTemplate?.title || '';
    if (nodes.selectedTemplatePanel) nodes.selectedTemplatePanel.hidden = !name;
    if (nodes.selectedTemplateName) nodes.selectedTemplateName.textContent = name || '-';
}

function clearTemplateSelection() {
    selectedTemplate = null;
    nodes.focusPrompt.value = stripTemplateBlock(nodes.focusPrompt.value);
    renderSelectedTemplate();
    renderTemplateList();
}

function applyIntent(intent) {
    activeIntent = intent;
    nodes.intentCards.forEach(button => button.classList.toggle('active', button.dataset.intent === intent));
    if (nodes.templatePanel) nodes.templatePanel.hidden = !['scene', 'tools'].includes(intent);
    if (intent === 'smart' || intent === 'transcribe') {
        selectedTemplate = null;
        const config = intentDefaults[intent];
        if (config) {
            setSelectValue('analysisMode', config.analysisMode);
            const supplement = stripTemplateBlock(nodes.focusPrompt.value);
            nodes.focusPrompt.value = supplement || config.prompt;
        }
        renderSelectedTemplate();
    }
    if (intent === 'scene') {
        setSelectValue('analysisMode', 'auto');
        if (nodes.templateCategory && nodes.templateCategory.value === 'functionality') {
            nodes.templateCategory.value = 'all';
        }
    }
    if (intent === 'tools') {
        setSelectValue('analysisMode', 'auto');
        if (nodes.templateCategory) nodes.templateCategory.value = 'functionality';
    }
    renderTemplateList();
}

function templateMatchesIntent(template) {
    if (activeIntent === 'tools') return template.first_category === 'functionality';
    if (activeIntent === 'scene') return template.first_category !== 'functionality';
    return true;
}

function renderTemplateCategories() {
    if (!nodes.templateCategory) return;
    const categories = Array.from(new Set(promptTemplates.map(item => item.first_category).filter(Boolean))).sort();
    nodes.templateCategory.innerHTML = [
        '<option value="all">全部模板</option>',
        ...categories.map(category => `<option value="${escapeHtml(category)}">${escapeHtml(templateCategoryLabels[category] || category)}</option>`)
    ].join('');
}

function templateSearchText(template) {
    return [
        template.title,
        template.title_zh,
        template.first_category,
        template.first_category_zh,
        template.second_category,
        template.second_category_zh,
        ...(template.tags || [])
    ].join(' ').toLowerCase();
}

function renderTemplateList() {
    if (!nodes.templateList || !nodes.templateStatus) return;
    if (!['scene', 'tools'].includes(activeIntent)) return;
    const query = String(nodes.templateSearch?.value || '').trim().toLowerCase();
    const category = nodes.templateCategory?.value || 'all';
    const filtered = promptTemplates
        .filter(templateMatchesIntent)
        .filter(template => category === 'all' || template.first_category === category)
        .filter(template => !query || templateSearchText(template).includes(query))
        .slice(0, 80);
    nodes.templateStatus.textContent = promptTemplates.length
        ? `显示 ${filtered.length} 个模板 / 共 ${promptTemplates.length} 个`
        : '模板暂不可用，可手动填写关注重点';
    nodes.templateList.innerHTML = filtered.map(template => {
        const active = selectedTemplate?.id === template.id ? ' active' : '';
        const categoryText = `${template.first_category_zh || template.first_category} / ${template.second_category_zh || template.second_category}`;
        return `
            <button class="template-item${active}" type="button" data-template-id="${escapeHtml(template.id)}">
                <strong>${escapeHtml(template.title_zh || template.title)}</strong>
                <span>${escapeHtml(categoryText)}</span>
            </button>
        `;
    }).join('') || '<div class="muted">没有匹配的模板</div>';
    nodes.templateList.querySelectorAll('.template-item').forEach(button => {
        button.addEventListener('click', () => {
            const template = promptTemplates.find(item => item.id === button.dataset.templateId);
            if (template) {
                applyFocusPromptTemplate(template);
                renderTemplateList();
            }
        });
    });
}

async function loadPromptTemplates() {
    if (!nodes.templateList) return;
    try {
        const response = await fetch('/static/data/audio_prompt_templates.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        promptTemplates = Array.isArray(data)
            ? data.filter(item => item?.id && item?.prompt_original)
            : [];
        renderTemplateCategories();
        renderTemplateList();
    } catch (error) {
        promptTemplates = [];
        nodes.templateStatus.textContent = `模板加载失败：${error.message}`;
    }
}

function fillSelect(id, values, selected) {
    const node = document.getElementById(id);
    node.innerHTML = (values || []).map(value => `<option value="${value}">${value}</option>`).join('');
    node.value = selected || '';
}

function fillObjectSelect(node, values, selected) {
    if (!node) return;
    node.innerHTML = (values || []).map(item => (
        `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label || item.value)}</option>`
    )).join('');
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

function setSourceMode(mode) {
    sourceMode = mode === 'file' ? 'file' : 'url';
    nodes.urlSourceTab?.classList.toggle('active', sourceMode === 'url');
    nodes.fileSourceTab?.classList.toggle('active', sourceMode === 'file');
    if (nodes.urlEntry) nodes.urlEntry.hidden = sourceMode !== 'url';
    if (nodes.urlList) nodes.urlList.hidden = sourceMode !== 'url';
    if (nodes.mediaEntry) nodes.mediaEntry.hidden = sourceMode !== 'file';
    const urlOnlyDisabled = sourceMode === 'file';
    ['cookieBrowser', 'downloadDevice', 'includeSubtitles', 'preferSubtitleTranscript', 'includeComments', 'refreshContext', 'maxComments', 'subtitleLangs']
        .forEach(id => {
            const node = document.getElementById(id);
            if (node) node.disabled = urlOnlyDisabled;
        });
    nodes.createButton.textContent = sourceMode === 'file' ? '上传并启动' : '启动任务';
    nodes.formError.textContent = '';
    nodes.batchResult.textContent = '';
}

async function loadOptions() {
    const options = await getJson('/api/video-link/options');
    const defaults = options.defaults || {};
    const choices = options.choices || {};
    fillSelect('analysisMode', choices.analysis_modes, defaults.analysis_mode);
    fillSelect('profile', choices.profiles, defaults.profile);
    fillSelect('cookieBrowser', choices.cookie_browsers, defaults.cookies_from_browser);
    fillSelect('downloadDevice', choices.download_devices, defaults.download_device);
    fillObjectSelect(
        nodes.skillProfile,
        choices.skill_distillation_profiles,
        defaults.skill_distillation_profile || 'deepseek_v4_pro'
    );
    document.getElementById('runName').value = defaults.run_name || 'operation-manual';
    document.getElementById('skipImages').checked = Boolean(defaults.skip_images);
    document.getElementById('keepExisting').checked = Boolean(defaults.keep_existing);
    document.getElementById('includeSubtitles').checked = Boolean(defaults.include_subtitles);
    document.getElementById('preferSubtitleTranscript').checked = Boolean(defaults.prefer_subtitle_transcript);
    document.getElementById('includeComments').checked = Boolean(defaults.include_comments);
    document.getElementById('refreshContext').checked = Boolean(defaults.refresh_context);
    nodes.focusPrompt.value = defaults.focus_prompt || '';
    selectedTemplate = null;
    renderSelectedTemplate();
    document.getElementById('maxComments').value = defaults.max_comments ?? 3000;
    document.getElementById('subtitleLangs').value = defaults.subtitle_langs || '';
}

const profileModelNodes = {
    asr: () => nodes.profileAsrModel,
    diarization: () => nodes.profileDiarizationModel,
    ocr: () => nodes.profileOcrModel,
    vision: () => nodes.profileVisionModel,
    text: () => nodes.profileTextModel,
    review: () => nodes.profileReviewModel,
    study: () => nodes.profileStudyModel,
    triage: () => nodes.profileTriageModel,
    image: () => nodes.profileImageModel
};

function setSettingsSection(section) {
    settingsSection = section === 'profiles' ? 'profiles' : 'models';
    nodes.settingsModelsView.hidden = settingsSection !== 'models';
    nodes.settingsProfilesView.hidden = settingsSection !== 'profiles';
    nodes.settingsModelsTab.classList.toggle('active', settingsSection === 'models');
    nodes.settingsProfilesTab.classList.toggle('active', settingsSection === 'profiles');
    if (settingsSection === 'profiles') scheduleProfileFlowEdges();
}

function parseJsonField(node, label) {
    try {
        const value = JSON.parse(node.value || '{}');
        if (!value || Array.isArray(value) || typeof value !== 'object') {
            throw new Error(`${label}必须是 JSON 对象`);
        }
        return value;
    } catch (error) {
        if (error.message.includes('必须是')) throw error;
        throw new Error(`${label} JSON 格式错误：${error.message}`);
    }
}

function settingsModels(kind = '') {
    return (settingsData?.models || []).filter(item => !kind || item.kind === kind);
}

function syncModelProtocolOptions(selected = '') {
    const protocols = settingsData?.schema?.kinds?.[nodes.modelKind.value] || [];
    fillSelect('modelProtocol', protocols, selected || protocols[0] || '');
}

function renderSettingsModelList() {
    const kind = nodes.settingsModelKindFilter.value;
    const query = nodes.settingsModelSearch.value.trim().toLowerCase();
    const items = settingsModels(kind).filter(item => (
        !query
        || `${item.id} ${item.name} ${item.model || ''} ${item.protocol}`.toLowerCase().includes(query)
    ));
    nodes.settingsModelList.innerHTML = items.length ? items.map(item => {
        const selected = item.id === selectedSettingsModelId ? ' selected' : '';
        const flags = [
            item.built_in ? '内置' : '自定义',
            item.overridden ? '已覆盖' : ''
        ].filter(Boolean).join(' · ');
        return `<button class="settings-item${selected}" type="button" data-settings-model="${escapeHtml(item.id)}">
            <strong>${escapeHtml(item.name)}</strong>
            <span>${escapeHtml(item.kind)} · ${escapeHtml(item.protocol)}</span>
            <small>${escapeHtml(item.model || flags || item.id)}</small>
        </button>`;
    }).join('') : '<div class="empty">没有匹配的模型</div>';
    nodes.settingsModelList.querySelectorAll('[data-settings-model]').forEach(button => {
        button.addEventListener('click', () => selectSettingsModel(button.dataset.settingsModel));
    });
}

function resetModelEditor(kind = '') {
    selectedSettingsModelId = '';
    const selectedKind = kind || nodes.settingsModelKindFilter.value || Object.keys(settingsData?.schema?.kinds || {})[0] || 'asr';
    nodes.modelSettingsForm.reset();
    nodes.modelId.disabled = false;
    nodes.modelKind.disabled = false;
    nodes.modelId.value = '';
    nodes.modelName.value = '';
    nodes.modelKind.value = selectedKind;
    syncModelProtocolOptions();
    nodes.modelDeployment.value = '';
    nodes.modelWorkerCount.value = '';
    nodes.modelConcurrency.value = '';
    nodes.modelOptions.value = '{}';
    nodes.modelEditorTitle.textContent = '新增模型';
    nodes.modelEditorMeta.textContent = '选择类型和协议';
    nodes.testModelButton.disabled = true;
    nodes.deleteModelButton.disabled = true;
    nodes.modelSettingsMessage.textContent = '';
    renderSettingsModelList();
}

function selectSettingsModel(modelId) {
    const item = (settingsData?.models || []).find(model => model.id === modelId);
    if (!item) return;
    selectedSettingsModelId = modelId;
    nodes.modelId.disabled = true;
    nodes.modelKind.disabled = true;
    nodes.modelId.value = item.id;
    nodes.modelName.value = item.name || item.id;
    nodes.modelKind.value = item.kind;
    syncModelProtocolOptions(item.protocol);
    nodes.modelNameValue.value = item.model || '';
    nodes.modelEndpoints.value = (item.endpoints || []).join('\n');
    nodes.modelDeployment.value = item.options?.deployment || '';
    nodes.modelWorkerCount.value = item.options?.worker_count ?? '';
    nodes.modelConcurrency.value = item.options?.concurrency ?? '';
    nodes.modelApiKeyEnv.value = item.api_key_env || '';
    nodes.modelHealthUrl.value = item.health_url || '';
    nodes.modelOptions.value = JSON.stringify(item.options || {}, null, 2);
    nodes.modelEditorTitle.textContent = item.name || item.id;
    nodes.modelEditorMeta.textContent = `${item.kind} · ${item.protocol}${item.built_in ? ' · 内置资源' : ''}`;
    nodes.testModelButton.disabled = activeProfileTestJobs().length > 0;
    nodes.deleteModelButton.disabled = false;
    nodes.modelSettingsMessage.textContent = '';
    renderSettingsModelList();
}

async function saveModelSettings(event) {
    event.preventDefault();
    const modelId = nodes.modelId.value.trim();
    const options = parseJsonField(nodes.modelOptions, '高级参数');
    const deployment = nodes.modelDeployment.value;
    const workerCount = Number(nodes.modelWorkerCount.value || 0);
    const concurrency = Number(nodes.modelConcurrency.value || 0);
    if (deployment) options.deployment = deployment;
    else delete options.deployment;
    if (workerCount > 0) options.worker_count = workerCount;
    else delete options.worker_count;
    if (concurrency > 0) options.concurrency = concurrency;
    else delete options.concurrency;
    const payload = {
        id: modelId,
        name: nodes.modelName.value.trim(),
        kind: nodes.modelKind.value,
        protocol: nodes.modelProtocol.value,
        model: nodes.modelNameValue.value.trim(),
        endpoints: nodes.modelEndpoints.value.split(/\n+/).map(item => item.trim()).filter(Boolean),
        api_key_env: nodes.modelApiKeyEnv.value.trim(),
        health_url: nodes.modelHealthUrl.value.trim(),
        options
    };
    nodes.saveModelButton.disabled = true;
    try {
        await getJson(
            selectedSettingsModelId
                ? `/api/settings/models/${encodeURIComponent(selectedSettingsModelId)}`
                : '/api/settings/models',
            {
                method: selectedSettingsModelId ? 'PUT' : 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }
        );
        selectedSettingsModelId = modelId;
        await loadSettings({ modelId });
        nodes.modelSettingsMessage.textContent = '模型配置已保存';
    } finally {
        nodes.saveModelButton.disabled = false;
    }
}

async function testSelectedSettingsModel() {
    if (!selectedSettingsModelId || modelTestRunning || activeProfileTestJobs().length) {
        updateProfileTestAvailability();
        return;
    }
    modelTestRunning = true;
    nodes.testModelButton.disabled = true;
    nodes.modelSettingsMessage.textContent = '正在测试连接...';
    try {
        const result = await getJson(`/api/settings/models/${encodeURIComponent(selectedSettingsModelId)}/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}'
        });
        nodes.modelSettingsMessage.textContent = `${result.ok ? '可用' : '不可用'}：${result.detail || result.status}`;
        nodes.modelSettingsMessage.classList.toggle('error-text', !result.ok);
    } finally {
        modelTestRunning = false;
        updateProfileTestAvailability();
    }
}

async function deleteSelectedSettingsModel() {
    if (!selectedSettingsModelId) return;
    const item = (settingsData?.models || []).find(model => model.id === selectedSettingsModelId);
    if (!window.confirm(`删除模型配置？\n\n${item?.name || selectedSettingsModelId}`)) return;
    await getJson(`/api/settings/models/${encodeURIComponent(selectedSettingsModelId)}`, { method: 'DELETE' });
    selectedSettingsModelId = '';
    await loadSettings();
}

function profileAdvancedSettings(profile) {
    const workflowFields = Object.values(settingsData?.schema?.workflows || {})
        .flatMap(workflow => Object.values(workflow.model_fields || {}).map(spec => spec.field));
    const omitted = new Set([
        'name', 'label', 'description', 'workflow_id', 'built_in', 'overridden',
        ...Object.values(settingsData?.schema?.profile_model_fields || {}),
        ...workflowFields,
        'asr_provider', 'vibevoice_url', 'vibevoice_urls', 'remote_asr_url', 'remote_asr_urls',
        'firered_3dspeaker_url', 'openai_audio_url', 'openai_audio_model', 'asr_api_key_env',
        'speaker_diarization', 'ocr_provider', 'ocr_base_url', 'ocr_base_urls', 'ocr_model',
        'vision_base_url', 'vision_model', 'vision_api_key_env',
        'llm_base_url', 'text_base_url', 'text_model', 'text_api_key_env',
        'review_base_url', 'review_model', 'review_api_key_env',
        'study_card_llm_base_url', 'study_card_model', 'study_card_api_key_env',
        'triage_llm_base_url', 'triage_model', 'triage_api_key_env',
        'image_provider', 'image_model'
    ]);
    return Object.fromEntries(Object.entries(profile || {}).filter(([key]) => !omitted.has(key)));
}

function fillProfileModelSelect(kind, selected = '') {
    const node = profileModelNodes[kind]();
    const items = settingsModels(kind);
    node.innerHTML = items.map(item => (
        `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)} · ${escapeHtml(item.protocol)}</option>`
    )).join('');
    node.value = selected || items[0]?.id || '';
}

function profileFlowSchema() {
    return selectedWorkflowSchema()?.flow
        || settingsData?.schema?.profile_flow
        || { lanes: [], nodes: [], edges: [] };
}

function selectedWorkflowId() {
    return nodes.profileWorkflow?.value || 'video_operation_manual';
}

function selectedWorkflowSchema() {
    return settingsData?.schema?.workflows?.[selectedWorkflowId()] || null;
}

function workflowModelFields() {
    return selectedWorkflowSchema()?.model_fields || {};
}

function setProfileModelSelections(profile = null) {
    const next = {};
    Object.entries(workflowModelFields()).forEach(([slot, spec]) => {
        const items = settingsModels(spec.kind);
        const requested = profile?.[spec.field] || profileModelSelections[slot] || '';
        const selected = items.find(item => item.id === requested)
            || (!spec.required ? items.find(item => item.protocol === 'none') : null)
            || items.find(item => item.protocol !== 'none')
            || null;
        next[slot] = selected?.id || '';
    });
    profileModelSelections = next;
}

function profileFlowNode(nodeId) {
    return profileFlowSchema().nodes.find(item => item.id === nodeId);
}

function selectedProfileModel(flowNode) {
    const slot = flowNode?.model_slot || flowNode?.model_kind || '';
    const kind = flowNode?.model_kind || workflowModelFields()[slot]?.kind || '';
    const modelId = profileModelSelections[slot] || '';
    return settingsModels(kind).find(item => item.id === modelId) || null;
}

function profileFlowSource(model) {
    if (!model) return '-';
    const deployment = model.options?.deployment || '';
    if (deployment === 'local') return '本机服务';
    if (deployment === 'cloud') return '云端服务';
    if (deployment === 'remote') return '远程设备';
    if (model.protocol === 'none') return '已禁用';
    if (model.protocol.startsWith('inherit_')) return '继承方案模型';
    if (model.protocol === 'asr_embedded') return 'ASR 内置能力';
    if (model.protocol === 'codex_imagegen') return '本机 Codex';
    const endpoint = (model.endpoints || [])[0] || '';
    if (!endpoint) return '本地运行时';
    try {
        const host = new URL(endpoint).hostname;
        return ['127.0.0.1', 'localhost', '0.0.0.0'].includes(host) ? '本机服务' : `远程服务 · ${host}`;
    } catch (_error) {
        return endpoint;
    }
}

function profileFlowDisabled(model) {
    return model?.protocol === 'none';
}

function profileNodeTestResult(nodeId) {
    return profileTestReport?.results?.[nodeId] || null;
}

function profileTestStatusLabel(result) {
    const labels = {
        reachable: '可达',
        sleeping: '休眠',
        configured: '已配置',
        passed: '通过',
        inherited: '继承',
        disabled: '禁用',
        auth_only: '鉴权通过',
        blocked: '阻塞',
        failed: '失败',
        unreachable: '不可达',
        missing: '缺失',
        missing_credentials: '缺少密钥',
        model_missing: '模型不存在',
        invalid: '配置错误'
    };
    return labels[result?.status] || result?.status || '';
}

function profileTestElapsed(milliseconds) {
    const value = Number(milliseconds);
    if (!Number.isFinite(value) || value < 0) return '-';
    if (value < 1000) return `${Math.round(value)}ms`;
    if (value < 60000) return `${(value / 1000).toFixed(1)}秒`;
    return durationMinutes(value / 1000);
}

function renderProfileTestSummary() {
    if (!nodes.profileTestSummary) return;
    if (!profileTestReport) {
        nodes.profileTestSummary.hidden = true;
        nodes.profileTestSummary.innerHTML = '';
        return;
    }
    const summary = profileTestReport.summary || {};
    const modeLabels = { quick: '快速检查', inference: '最小推理', pathway: '全链路冒烟' };
    nodes.profileTestSummary.hidden = false;
    nodes.profileTestSummary.classList.toggle('failed', !profileTestReport.ok);
    nodes.profileTestSummary.innerHTML = `
        <div>
            <strong>${escapeHtml(modeLabels[profileTestReport.mode] || profileTestReport.mode)}</strong>
            <span>${escapeHtml(summary.detail || '')}</span>
        </div>
        <dl>
            <div><dt>通过</dt><dd>${escapeHtml(String(summary.passed ?? 0))}</dd></div>
            <div><dt>失败</dt><dd>${escapeHtml(String(summary.failed ?? 0))}</dd></div>
            <div><dt>阻塞</dt><dd>${escapeHtml(String(summary.blocked ?? 0))}</dd></div>
            <div><dt>耗时</dt><dd>${escapeHtml(profileTestElapsed(profileTestReport.elapsed_ms))}</dd></div>
        </dl>`;
}

function currentProfileModelRefs() {
    return { ...profileModelSelections };
}

function activeProfileTestJobs() {
    return latestJobs.filter(job => jobIsActive(job));
}

function updateProfileTestAvailability() {
    const activeJobs = activeProfileTestJobs();
    const busy = activeJobs.length > 0;
    if (nodes.testModelButton) {
        nodes.testModelButton.disabled = !selectedSettingsModelId || modelTestRunning || busy;
        nodes.testModelButton.title = busy
            ? '后台任务运行期间不能测试模型连接'
            : '测试当前模型连接';
    }
    if (!nodes.testProfileButton || !nodes.profileTestAvailability) return;
    nodes.testProfileButton.disabled = profileTestRunning || busy;
    if (profileTestRunning) {
        nodes.profileTestAvailability.textContent = '通路测试正在运行';
        nodes.testProfileButton.title = '通路测试正在运行';
        return;
    }
    if (busy) {
        const first = activeJobs[0];
        const stage = stageNames[first.current_stage] || first.current_stage || first.runner?.current_stage || '后台处理';
        nodes.profileTestAvailability.textContent = `后台有 ${activeJobs.length} 个任务正在运行或排队，测试暂不可用`;
        nodes.testProfileButton.title = `${jobDisplayTitle(first)} · ${stage}`;
        return;
    }
    nodes.profileTestAvailability.textContent = '后台空闲，可以执行通路测试';
    nodes.testProfileButton.title = '测试当前运行方案的模型通路';
}

async function testCurrentProfile() {
    if (profileTestRunning || activeProfileTestJobs().length) {
        updateProfileTestAvailability();
        return;
    }
    profileTestRunning = true;
    profileTestReport = null;
    updateProfileTestAvailability();
    nodes.testProfileButton.textContent = nodes.profileTestMode.value === 'quick' ? '正在检查...' : '正在运行...';
    nodes.profileTestSummary.hidden = false;
    nodes.profileTestSummary.classList.remove('failed');
    nodes.profileTestSummary.innerHTML = '<div><strong>通路测试进行中</strong><span>本地模型冷启动时可能需要等待。</span></div>';
    try {
        profileTestReport = await getJson('/api/settings/profile-test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                profile_name: nodes.profileId.value.trim(),
                workflow_id: selectedWorkflowId(),
                mode: nodes.profileTestMode.value,
                models: currentProfileModelRefs()
            })
        });
        renderProfileFlow();
        renderProfileTestSummary();
    } finally {
        profileTestRunning = false;
        nodes.testProfileButton.textContent = '测试通路';
        updateProfileTestAvailability();
    }
}

function scheduleProfileFlowEdges() {
    if (profileFlowDrawFrame) window.cancelAnimationFrame(profileFlowDrawFrame);
    profileFlowDrawFrame = window.requestAnimationFrame(() => {
        profileFlowDrawFrame = null;
        drawProfileFlowEdges();
    });
}

function drawProfileFlowEdges() {
    if (!nodes.profileFlowCanvas || !nodes.profileFlowEdges || window.innerWidth <= 760) return;
    const canvasRect = nodes.profileFlowCanvas.getBoundingClientRect();
    if (!canvasRect.width || !canvasRect.height) return;
    nodes.profileFlowEdges.setAttribute('viewBox', `0 0 ${canvasRect.width} ${canvasRect.height}`);
    nodes.profileFlowEdges.setAttribute('width', canvasRect.width);
    nodes.profileFlowEdges.setAttribute('height', canvasRect.height);
    const paths = profileFlowSchema().edges.map(edge => {
        const source = nodes.profileFlowNodes.querySelector(`[data-flow-node="${edge.from}"]`);
        const target = nodes.profileFlowNodes.querySelector(`[data-flow-node="${edge.to}"]`);
        if (!source || !target) return '';
        const from = source.getBoundingClientRect();
        const to = target.getBoundingClientRect();
        const x1 = from.right - canvasRect.left;
        const y1 = from.top + from.height / 2 - canvasRect.top;
        const x2 = to.left - canvasRect.left;
        const y2 = to.top + to.height / 2 - canvasRect.top;
        const bend = Math.max(28, Math.abs(x2 - x1) * 0.42);
        return `<path class="profile-flow-edge lane-${escapeHtml(edge.lane || 'main')}" d="M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}"></path>`;
    }).join('');
    nodes.profileFlowEdges.innerHTML = paths;
}

function profileFlowParameter(node) {
    if (node.model_kind === 'ocr') {
        return { label: 'OCR 并发', target: nodes.profileOcrConcurrency, type: 'text', min: '' };
    }
    if (node.model_kind === 'vision') {
        return { label: 'VL 并发', target: nodes.profileVlConcurrency, type: 'number', min: '1' };
    }
    if (node.model_kind === 'text' && node.model_slot !== 'text_fallback') {
        return { label: '文本超时秒数', target: nodes.profileTextTimeout, type: 'number', min: '1' };
    }
    return null;
}

function renderProfileFlowInspector(nodeId = selectedProfileFlowNodeId) {
    const flowNode = profileFlowNode(nodeId) || profileFlowSchema().nodes[0];
    if (!flowNode) return;
    selectedProfileFlowNodeId = flowNode.id;
    nodes.profileFlowInspectorTitle.textContent = flowNode.title;
    nodes.profileFlowInspectorMeta.textContent = flowNode.subtitle || '';
    if (!flowNode.model_kind) {
        nodes.profileFlowInspectorBody.innerHTML = `
            <dl class="profile-flow-details">
                <div><dt>步骤</dt><dd>${escapeHtml(String(flowNode.step))}</dd></div>
                <div><dt>阶段</dt><dd>${escapeHtml(stageNames[flowNode.stage] || flowNode.stage || '固定流程')}</dd></div>
                <div><dt>类型</dt><dd>系统流程节点</dd></div>
            </dl>`;
        nodes.profileFlowInspectorActions.innerHTML = '';
        return;
    }
    const model = selectedProfileModel(flowNode);
    const endpoint = (model?.endpoints || [])[0] || '';
    const workerCount = model?.options?.worker_count;
    const concurrency = model?.options?.concurrency;
    const parameter = profileFlowParameter(flowNode);
    const testResult = profileNodeTestResult(flowNode.id);
    nodes.profileFlowInspectorBody.innerHTML = `
        <dl class="profile-flow-details">
            <div><dt>当前模型</dt><dd>${escapeHtml(model?.name || '未选择')}</dd></div>
            <div><dt>协议</dt><dd>${escapeHtml(model?.protocol || '-')}</dd></div>
            <div><dt>运行位置</dt><dd>${escapeHtml(profileFlowSource(model))}</dd></div>
            <div><dt>模型名</dt><dd>${escapeHtml(model?.model || '-')}</dd></div>
            <div><dt>Worker 数</dt><dd>${escapeHtml(workerCount == null ? '-' : String(workerCount))}</dd></div>
            <div><dt>并发数</dt><dd>${escapeHtml(concurrency == null ? '-' : String(concurrency))}</dd></div>
            <div class="wide"><dt>端点</dt><dd>${escapeHtml(endpoint || '-')}</dd></div>
        </dl>
        ${parameter ? `<label class="profile-flow-parameter">
            <span>${escapeHtml(parameter.label)}</span>
            <input data-flow-parameter="${escapeHtml(flowNode.id)}" type="${parameter.type}" min="${parameter.min}" value="${escapeHtml(parameter.target.value)}">
        </label>` : ''}
        ${profileFlowDisabled(model) ? '<p class="profile-flow-warning">此节点已禁用，运行时会跳过对应的模型能力。</p>' : ''}
        ${testResult ? `<p class="profile-flow-node-result ${testResult.ok ? 'passed' : 'failed'}">
            ${escapeHtml(profileTestStatusLabel(testResult))}：${escapeHtml(testResult.detail || '')}
            ${testResult.elapsed_ms != null ? ` · ${escapeHtml(String(testResult.elapsed_ms))}ms` : ''}
        </p>` : ''}`;
    const controlOnly = model?.source === 'control';
    nodes.profileFlowInspectorActions.innerHTML = `
        <button class="secondary" type="button" data-flow-test ${profileFlowDisabled(model) ? 'disabled' : ''}>测试连接</button>
        <button class="secondary" type="button" data-flow-edit ${controlOnly ? 'disabled' : ''}>编辑模型资源</button>
        <span class="profile-flow-test-result" data-flow-test-result></span>`;
    nodes.profileFlowInspectorBody.querySelector('[data-flow-parameter]')?.addEventListener('input', event => {
        parameter.target.value = event.target.value;
    });
    nodes.profileFlowInspectorActions.querySelector('[data-flow-edit]')?.addEventListener('click', () => {
        if (!model) return;
        setSettingsSection('models');
        selectSettingsModel(model.id);
    });
    nodes.profileFlowInspectorActions.querySelector('[data-flow-test]')?.addEventListener('click', async event => {
        if (!model) return;
        const button = event.currentTarget;
        const resultNode = nodes.profileFlowInspectorActions.querySelector('[data-flow-test-result]');
        button.disabled = true;
        resultNode.textContent = '正在测试...';
        try {
            const result = await getJson(`/api/settings/models/${encodeURIComponent(model.id)}/test`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: nodes.profileTestMode.value })
            });
            profileTestReport = profileTestReport || { mode: nodes.profileTestMode.value, results: {}, summary: {} };
            profileTestReport.results[flowNode.id] = result;
            resultNode.textContent = `${result.ok ? '可用' : '不可用'}：${result.detail || result.status}`;
            resultNode.classList.toggle('error-text', !result.ok);
            renderProfileFlow();
        } catch (error) {
            resultNode.textContent = error.message;
            resultNode.classList.add('error-text');
        } finally {
            button.disabled = false;
        }
    });
}

function renderProfileFlow() {
    if (!nodes.profileFlowNodes) return;
    const schema = profileFlowSchema();
    const maxColumn = Math.max(1, ...(schema.nodes || []).map(node => Number(node.column) || 1));
    const maxRow = Math.max(1, ...(schema.nodes || []).map(node => Number(node.row) || 1));
    nodes.profileFlowCanvas.style.setProperty('--profile-flow-columns', String(maxColumn));
    nodes.profileFlowCanvas.style.setProperty('--profile-flow-rows', String(maxRow));
    nodes.profileFlowCanvas.style.setProperty(
        '--profile-flow-height',
        `${72 + (maxRow * 142) + (Math.max(0, maxRow - 1) * 54)}px`
    );
    const lanes = Object.fromEntries((schema.lanes || []).map(item => [item.id, item.label]));
    nodes.profileFlowLegend.innerHTML = (schema.lanes || []).map(lane => (
        `<span class="profile-flow-legend-item lane-${escapeHtml(lane.id)}">${escapeHtml(lane.label)}</span>`
    )).join('');
    const flowNodes = [...(schema.nodes || [])].sort((left, right) => left.mobile_order - right.mobile_order);
    nodes.profileFlowNodes.innerHTML = flowNodes.map(flowNode => {
        const model = flowNode.model_kind ? selectedProfileModel(flowNode) : null;
        const disabled = profileFlowDisabled(model);
        const selected = flowNode.id === selectedProfileFlowNodeId;
        const testResult = profileNodeTestResult(flowNode.id);
        const testClass = testResult ? ` test-${escapeHtml(testResult.status || (testResult.ok ? 'passed' : 'failed'))}` : '';
        const options = flowNode.model_kind ? settingsModels(flowNode.model_kind)
            .filter(item => !flowNode.required || item.protocol !== 'none')
            .map(item => `<option value="${escapeHtml(item.id)}" ${item.id === model?.id ? 'selected' : ''}>${escapeHtml(item.name)}</option>`)
            .join('') : '';
        return `<article
            class="profile-flow-node lane-${escapeHtml(flowNode.lane)}${selected ? ' selected' : ''}${disabled ? ' disabled' : ''}${testClass}"
            data-flow-node="${escapeHtml(flowNode.id)}"
            style="--flow-column:${flowNode.column};--flow-row:${flowNode.row};"
            tabindex="0"
        >
            <div class="profile-flow-node-head">
                <span class="profile-flow-step">${escapeHtml(String(flowNode.step))}</span>
                <span class="profile-flow-lane">${testResult
                    ? escapeHtml(profileTestStatusLabel(testResult))
                    : escapeHtml(lanes[flowNode.lane] || flowNode.lane)}</span>
            </div>
            <strong>${escapeHtml(flowNode.title)}</strong>
            <small>${escapeHtml(flowNode.subtitle || '')}</small>
            ${flowNode.model_kind
                ? `<select data-flow-model-slot="${escapeHtml(flowNode.model_slot || flowNode.model_kind)}" aria-label="${escapeHtml(flowNode.title)}模型">${options}</select>`
                : `<span class="profile-flow-fixed">${escapeHtml(stageNames[flowNode.stage] || '固定阶段')}</span>`}
        </article>`;
    }).join('');
    nodes.profileFlowNodes.querySelectorAll('[data-flow-node]').forEach(node => {
        const selectNode = node.querySelector('[data-flow-model-slot]');
        node.addEventListener('click', event => {
            if (event.target === selectNode) return;
            selectedProfileFlowNodeId = node.dataset.flowNode;
            renderProfileFlow();
        });
        node.addEventListener('keydown', event => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            selectedProfileFlowNodeId = node.dataset.flowNode;
            renderProfileFlow();
        });
        selectNode?.addEventListener('change', event => {
            event.stopPropagation();
            profileModelSelections[event.target.dataset.flowModelSlot] = event.target.value;
            selectedProfileFlowNodeId = node.dataset.flowNode;
            renderProfileFlow();
        });
    });
    renderProfileFlowInspector(selectedProfileFlowNodeId);
    scheduleProfileFlowEdges();
}

function renderSettingsProfileList() {
    const profiles = settingsData?.profiles || [];
    nodes.settingsProfileList.innerHTML = profiles.map(profile => {
        const selected = profile.name === selectedSettingsProfileName ? ' selected' : '';
        const active = profile.name === settingsData.active_runtime_profile ? '默认 · ' : '';
        const workflow = settingsData?.schema?.workflows?.[profile.workflow_id || 'video_operation_manual'];
        return `<button class="settings-item${selected}" type="button" data-settings-profile="${escapeHtml(profile.name)}">
            <strong>${escapeHtml(profile.label || profile.name)}</strong>
            <span>${escapeHtml(active + profile.name + (workflow ? ` · ${workflow.label}` : ''))}</span>
            <small>${escapeHtml(profile.description || (profile.built_in ? '内置方案' : '自定义方案'))}</small>
        </button>`;
    }).join('');
    nodes.settingsProfileList.querySelectorAll('[data-settings-profile]').forEach(button => {
        button.addEventListener('click', () => selectSettingsProfile(button.dataset.settingsProfile));
    });
}

function resetProfileEditor(source = null) {
    selectedSettingsProfileName = '';
    profileTestReport = null;
    nodes.profileSettingsForm.reset();
    nodes.profileId.disabled = false;
    nodes.profileId.value = '';
    nodes.profileLabel.value = source ? `${source.label || source.name} 副本` : '';
    nodes.profileWorkflow.innerHTML = Object.values(settingsData?.schema?.workflows || {}).map(workflow => (
        `<option value="${escapeHtml(workflow.id)}">${escapeHtml(workflow.label)}</option>`
    )).join('');
    nodes.profileWorkflow.value = source?.workflow_id || 'video_operation_manual';
    nodes.profileDescription.value = source?.description || '';
    profileModelSelections = {};
    setProfileModelSelections(source);
    selectedProfileFlowNodeId = profileFlowSchema().nodes[0]?.id || '';
    nodes.profilePipelineMode.value = source?.pipeline_mode || 'balanced';
    nodes.profileVlConcurrency.value = source?.vl_concurrency ?? 5;
    nodes.profileOcrConcurrency.value = source?.ocr_concurrency ?? 'auto';
    nodes.profileTextTimeout.value = source?.text_timeout_seconds ?? 600;
    nodes.profileSettingsJson.value = JSON.stringify(source ? profileAdvancedSettings(source) : {}, null, 2);
    nodes.profileEditorTitle.textContent = source ? '复制运行方案' : '新增运行方案';
    nodes.profileEditorMeta.textContent = '组合各阶段模型';
    nodes.duplicateProfileButton.disabled = true;
    nodes.activateProfileButton.disabled = true;
    nodes.deleteProfileButton.disabled = true;
    nodes.profileSettingsMessage.textContent = '';
    renderSettingsProfileList();
    renderProfileFlow();
    renderProfileTestSummary();
}

function selectSettingsProfile(profileName) {
    const profile = (settingsData?.profiles || []).find(item => item.name === profileName);
    if (!profile) return;
    selectedSettingsProfileName = profileName;
    profileTestReport = null;
    nodes.profileId.disabled = true;
    nodes.profileId.value = profile.name;
    nodes.profileLabel.value = profile.label || profile.name;
    nodes.profileWorkflow.innerHTML = Object.values(settingsData?.schema?.workflows || {}).map(workflow => (
        `<option value="${escapeHtml(workflow.id)}">${escapeHtml(workflow.label)}</option>`
    )).join('');
    nodes.profileWorkflow.value = profile.workflow_id || 'video_operation_manual';
    nodes.profileDescription.value = profile.description || '';
    profileModelSelections = {};
    setProfileModelSelections(profile);
    selectedProfileFlowNodeId = profileFlowSchema().nodes[0]?.id || '';
    nodes.profilePipelineMode.value = profile.pipeline_mode || 'balanced';
    nodes.profileVlConcurrency.value = profile.vl_concurrency ?? 5;
    nodes.profileOcrConcurrency.value = profile.ocr_concurrency ?? 'auto';
    nodes.profileTextTimeout.value = profile.text_timeout_seconds ?? 600;
    nodes.profileSettingsJson.value = JSON.stringify(profileAdvancedSettings(profile), null, 2);
    nodes.profileEditorTitle.textContent = profile.label || profile.name;
    nodes.profileEditorMeta.textContent = `${profile.name}${profile.built_in ? ' · 内置方案' : ''}`;
    nodes.duplicateProfileButton.disabled = false;
    nodes.activateProfileButton.disabled = profile.name === settingsData.active_runtime_profile;
    nodes.deleteProfileButton.disabled = profile.name === settingsData.active_runtime_profile;
    nodes.profileSettingsMessage.textContent = '';
    renderSettingsProfileList();
    renderProfileFlow();
    renderProfileTestSummary();
}

async function saveProfileSettings(event) {
    event.preventDefault();
    const profileName = nodes.profileId.value.trim();
    const settings = parseJsonField(nodes.profileSettingsJson, '运行参数');
    settings.pipeline_mode = nodes.profilePipelineMode.value;
    settings.vl_concurrency = Number(nodes.profileVlConcurrency.value || 1);
    settings.ocr_concurrency = nodes.profileOcrConcurrency.value.trim() || 'auto';
    settings.text_timeout_seconds = Number(nodes.profileTextTimeout.value || 600);
    const models = currentProfileModelRefs();
    const payload = {
        name: profileName,
        label: nodes.profileLabel.value.trim(),
        description: nodes.profileDescription.value.trim(),
        workflow_id: selectedWorkflowId(),
        models,
        settings
    };
    nodes.saveProfileButton.disabled = true;
    try {
        await getJson(
            selectedSettingsProfileName
                ? `/api/settings/profiles/${encodeURIComponent(selectedSettingsProfileName)}`
                : '/api/settings/profiles',
            {
                method: selectedSettingsProfileName ? 'PUT' : 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }
        );
        selectedSettingsProfileName = profileName;
        await loadSettings({ profileName });
        await loadOptions();
        nodes.profileSettingsMessage.textContent = '运行方案已保存';
    } finally {
        nodes.saveProfileButton.disabled = false;
    }
}

async function activateSelectedSettingsProfile() {
    if (!selectedSettingsProfileName) return;
    await getJson(`/api/settings/profiles/${encodeURIComponent(selectedSettingsProfileName)}/activate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}'
    });
    await loadSettings({ profileName: selectedSettingsProfileName });
    await loadOptions();
}

async function deleteSelectedSettingsProfile() {
    if (!selectedSettingsProfileName) return;
    const profile = (settingsData?.profiles || []).find(item => item.name === selectedSettingsProfileName);
    if (!window.confirm(`${profile?.built_in ? '从本机禁用内置运行方案' : '删除运行方案'}？\n\n${profile?.label || selectedSettingsProfileName}`)) return;
    await getJson(`/api/settings/profiles/${encodeURIComponent(selectedSettingsProfileName)}`, { method: 'DELETE' });
    selectedSettingsProfileName = '';
    await loadSettings();
    await loadOptions();
}

async function loadSettings(selection = {}) {
    settingsData = await getJson('/api/settings');
    const kinds = Object.keys(settingsData.schema?.kinds || {});
    nodes.settingsModelKindFilter.innerHTML = [
        '<option value="">全部类型</option>',
        ...kinds.map(kind => `<option value="${escapeHtml(kind)}">${escapeHtml(kind)}</option>`)
    ].join('');
    nodes.modelKind.innerHTML = kinds.map(kind => `<option value="${escapeHtml(kind)}">${escapeHtml(kind)}</option>`).join('');
    nodes.settingsSummary.textContent = `${settingsData.models.length} 个模型资源 · ${settingsData.profiles.length} 个运行方案 · 默认 ${settingsData.active_runtime_profile || '-'}`;
    renderSettingsModelList();
    renderSettingsProfileList();
    const modelId = selection.modelId || selectedSettingsModelId;
    const profileName = selection.profileName || selectedSettingsProfileName;
    if (modelId && settingsData.models.some(item => item.id === modelId)) {
        selectSettingsModel(modelId);
    } else if (!selectedSettingsModelId) {
        resetModelEditor();
    }
    if (profileName && settingsData.profiles.some(item => item.name === profileName)) {
        selectSettingsProfile(profileName);
    } else if (!selectedSettingsProfileName) {
        resetProfileEditor();
    }
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

function appendCommonJobFields(formData) {
    formData.append('focus_prompt', nodes.focusPrompt.value.trim());
    formData.append('analysis_mode', document.getElementById('analysisMode').value);
    formData.append('profile', document.getElementById('profile').value);
    formData.append('run_name', document.getElementById('runName').value.trim());
    formData.append('skip_images', document.getElementById('skipImages').checked ? 'true' : 'false');
    formData.append('keep_existing', document.getElementById('keepExisting').checked ? 'true' : 'false');
    formData.append('auto_start', 'true');
}

async function createUploadJob() {
    const file = nodes.mediaFile?.files?.[0];
    if (!file) throw new Error('请选择一个媒体文件');
    if (file.size <= 0) throw new Error('所选媒体文件为空，请重新选择原始文件');
    const formData = new FormData();
    formData.append('media', file);
    appendCommonJobFields(formData);
    return getJson('/api/video-link/jobs/upload', {
        method: 'POST',
        body: formData
    });
}

async function createJob(event) {
    event.preventDefault();
    nodes.formError.textContent = '';
    nodes.batchResult.textContent = '';
    nodes.createButton.disabled = true;
    try {
        if (sourceMode === 'file') {
            const job = await createUploadJob();
            selectJob(job.job_id, true);
            nodes.batchResult.textContent = '已创建 1/1 个文件任务';
            nodes.jobForm.reset();
            pendingUrls = [];
            renderUrlList();
            await loadOptions();
            applyIntent(activeIntent);
            setSourceMode('file');
            await refreshJobs();
            return;
        }
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
        applyIntent(activeIntent);
        await refreshJobs();
    } catch (error) {
        nodes.formError.textContent = error.message;
    } finally {
        nodes.createButton.disabled = false;
    }
}

function selectJob(jobId, updateUrl = true) {
    const changed = selectedJobId !== jobId;
    selectedJobId = jobId;
    selectedLogStage = null;
    if (changed) resetQaMessages();
    if (updateUrl) {
        const url = new URL(window.location.href);
        url.searchParams.set('job', jobId);
        if (currentView !== 'console') url.searchParams.set('view', currentView);
        window.history.replaceState({}, '', url);
    }
    refreshSelectedJob();
}

function setView(view, updateUrl = true) {
    currentView = ['qa', 'vscode', 'settings'].includes(view) ? view : 'console';
    nodes.consoleView.hidden = currentView !== 'console';
    nodes.qaView.hidden = currentView !== 'qa';
    nodes.vscodeView.hidden = currentView !== 'vscode';
    nodes.settingsView.hidden = currentView !== 'settings';
    nodes.consoleView.classList.toggle('active', currentView === 'console');
    nodes.qaView.classList.toggle('active', currentView === 'qa');
    nodes.vscodeView.classList.toggle('active', currentView === 'vscode');
    nodes.settingsView.classList.toggle('active', currentView === 'settings');
    nodes.consoleTab.classList.toggle('active', currentView === 'console');
    nodes.qaTab.classList.toggle('active', currentView === 'qa');
    nodes.vscodeTab.classList.toggle('active', currentView === 'vscode');
    nodes.settingsTab.classList.toggle('active', currentView === 'settings');
    if (updateUrl) {
        const url = new URL(window.location.href);
        if (currentView !== 'console') {
            url.searchParams.set('view', currentView);
        } else {
            url.searchParams.delete('view');
        }
        window.history.replaceState({}, '', url);
    }
    renderQaPanel(currentJob);
    renderVscodePanel(currentJob);
    if (currentView === 'settings' && !settingsData) {
        loadSettings().catch(error => {
            nodes.settingsSummary.textContent = error.message;
        });
    }
    applySkillsFocusMode();
}

function setResourceView(view, updateUrl = true) {
    currentResourceView = view === 'skills' ? 'skills' : 'docs';
    nodes.resourceDocsView.hidden = currentResourceView !== 'docs';
    nodes.resourceSkillsView.hidden = currentResourceView !== 'skills';
    nodes.resourceToolbar.hidden = currentResourceView === 'skills';
    nodes.docsToolbarActions.hidden = currentResourceView !== 'docs';
    nodes.resourceDocsTab.classList.toggle('active', currentResourceView === 'docs');
    nodes.resourceSkillsTab.classList.toggle('active', currentResourceView === 'skills');
    nodes.skillsResourceDocsTab.classList.toggle('active', currentResourceView === 'docs');
    nodes.skillsResourceSkillsTab.classList.toggle('active', currentResourceView === 'skills');
    if (updateUrl) {
        const url = new URL(window.location.href);
        url.searchParams.set('view', 'vscode');
        url.searchParams.set('resource', currentResourceView);
        if (currentResourceView === 'skills') {
            url.searchParams.set('scope', currentSkillsScope);
        } else {
            url.searchParams.delete('scope');
        }
        window.history.replaceState({}, '', url);
    }
    if (currentResourceView === 'skills') {
        renderSkillsWorkspace(currentJob);
    } else {
        renderDocPreviewPanel(currentJob);
        updateLearningDocsLayout();
    }
    applySkillsFocusMode();
}

function setSkillsScope(scope, updateUrl = true) {
    currentSkillsScope = ['projects', 'enabled', 'disabled', 'trash'].includes(scope) ? scope : 'current';
    if (currentSkillsScope !== 'enabled') {
        skillLibraryProjectSkillNames = null;
    }
    nodes.skillsScopeTabs.forEach(button => {
        button.classList.toggle('active', button.dataset.skillsScope === currentSkillsScope);
    });
    nodes.currentSkillsWorkspace.hidden = currentSkillsScope !== 'current';
    nodes.skillProjectsWorkspace.hidden = currentSkillsScope !== 'projects';
    nodes.skillLibraryWorkspace.hidden = currentSkillsScope === 'current' || currentSkillsScope === 'projects';
    nodes.skillLibrarySearchLabel.hidden = currentSkillsScope === 'current' || currentSkillsScope === 'projects';
    if (updateUrl) {
        const url = new URL(window.location.href);
        url.searchParams.set('view', 'vscode');
        url.searchParams.set('resource', 'skills');
        url.searchParams.set('scope', currentSkillsScope);
        window.history.replaceState({}, '', url);
    }
    if (currentSkillsScope === 'current') {
        renderSkillsWorkspace(currentJob);
    } else if (currentSkillsScope === 'projects') {
        loadSkillProjects();
    } else {
        window.requestAnimationFrame(constrainSkillsLayouts);
        selectedLibrarySkillId = '';
        currentLibrarySkill = null;
        resetSkillEditor();
        loadSkillLibrary();
    }
    syncSkillProjectPolling();
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
    return job.display_title || job.title || job.source_name || job.summary?.study?.title || job.video_url || job.job_id || '-';
}

function jobSourceLabel(job) {
    return job.source_name || job.video_url || '-';
}

function renderJobList(jobs) {
    const visibleJobs = jobs.filter(job => (
        showNonRerunFailures
        || job.status !== 'failed'
        || job.failure_disposition?.rerun_recommended
    ));
    const orderedJobs = sortJobsForAttention(visibleJobs);
    nodes.jobList.innerHTML = orderedJobs.length ? orderedJobs.map(job => {
        const selected = job.job_id === selectedJobId ? ' selected' : '';
        const statusClass = job.status ? ` ${job.status}` : '';
        const title = escapeHtml(jobDisplayTitle(job));
        const url = escapeHtml(jobSourceLabel(job));
        const stage = job.current_stage || job.error_summary?.stage || job.next_stage || '-';
        const statusLabel = job.status === 'failed' && job.failure_disposition?.label
            ? job.failure_disposition.label
            : (job.status || '-');
        return `<div class="job-item${selected}${statusClass}" data-job-id="${escapeHtml(job.job_id)}">
            <button class="job-select" type="button" data-job-id="${escapeHtml(job.job_id)}">
                <strong title="${title}">${title}</strong>
                <span class="job-url" title="${url}">${url}</span>
                <span class="job-status-line">${escapeHtml(statusLabel)} · ${escapeHtml(stageNames[stage] || stage)}</span>
            </button>
            <button class="icon-button light delete-job" type="button" data-job-id="${escapeHtml(job.job_id)}" title="删除任务" aria-label="删除任务">×</button>
        </div>`;
    }).join('') : '<div class="empty">当前没有需要续跑的失败任务</div>';
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
    updateProfileTestAvailability();
    renderJobList(jobs);
    renderSelectedJobSnapshot(jobs);
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
    updateProfileTestAvailability();
    renderJobList(jobs);
    renderSelectedJobSnapshot(jobs);
}

function renderGlobal(data) {
    const summary = data.summary || {};
    const counts = summary.counts || {};
    const cells = [
        ['总任务', summary.total ?? data.total ?? 0],
        ['运行中', counts.running || 0],
        ['排队中', counts.queued || 0],
        ['成功', counts.succeeded || 0],
        ['待续跑', summary.rerun_required || 0],
        ['全部失败', counts.failed || 0],
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
    const title = item.source_name || item.video_url || item.job_id || '-';
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
    consoleFlowJobId = '';
    selectedConsoleStage = '';
    selectedConsoleNodeId = '';
    consoleFlowCenteredKey = '';
    consoleFlowRenderKey = '';
    nodes.runButton.disabled = true;
    nodes.runButton.classList.remove('success-action', 'play-action', 'stop-action');
    nodes.runButton.dataset.action = 'run';
    nodes.runButton.title = '';
    nodes.runButton.textContent = '继续运行';
    nodes.selectedTitle.textContent = '未选择任务';
    nodes.selectedSubtitle.textContent = '创建或选择一个任务后查看进度。';
    nodes.stageDurationSummary.textContent = '原视频长度：- · 阶段总耗时：-';
    renderConsoleEmptyState('选择任务后显示执行流程');
    renderQaPanel(null);
}

function renderServiceOffline(error) {
    currentJob = null;
    consoleFlowJobId = '';
    selectedConsoleStage = '';
    selectedConsoleNodeId = '';
    consoleFlowCenteredKey = '';
    consoleFlowRenderKey = '';
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
    renderConsoleEmptyState('服务恢复连接后显示执行流程');
    renderQaPanel(null);
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
    if (consoleFlowJobId !== job.job_id) {
        consoleFlowJobId = job.job_id;
        selectedConsoleStage = '';
        selectedConsoleNodeId = '';
        consoleFlowCenteredKey = '';
        consoleFlowRenderKey = '';
        consoleFlowScale = 1;
    }
    if (sourcePlayerState.jobId && sourcePlayerState.jobId !== job.job_id) {
        sourcePlayerState.jobId = job.job_id;
        sourcePlayerState.seconds = null;
        sourcePlayerState.loaded = false;
    }
    const progress = job.progress || {};
    const stageProgress = job.stage_progress || job.core_progress;
    const queue = job.queue || {};
    const reason = runDisabledReason(job);
    const process = activeProcess(job);
    const runDir = job.summary?.run_dir || job.run_dir;
    const isSucceeded = job.status === 'succeeded';
    const isActive = jobIsActive(job);
    const failureDisposition = job.failure_disposition || {};
    const rerunCore = job.status === 'failed' && failureDisposition.category === 'rerun_core';
    const shouldOpenExisting = job.status === 'failed' && !failureDisposition.rerun_recommended && Boolean(runDir);
    const missingRunDir = isSucceeded && !runDir;
    nodes.selectedTitle.textContent = jobDisplayTitle(job);
    const subtitleReason = missingRunDir ? '资源目录不可用' : reason;
    const subtitleBase = `任务 ID: ${job.job_id} · ${jobSourceLabel(job)}`;
    nodes.selectedSubtitle.textContent = subtitleReason ? `${subtitleBase} · ${subtitleReason}` : subtitleBase;
    nodes.runButton.disabled = Boolean(missingRunDir || (job.status === 'failed' && !failureDisposition.rerun_recommended && !runDir));
    nodes.runButton.dataset.action = isActive
        ? 'stop'
        : ((isSucceeded || shouldOpenExisting) ? 'open-run-dir' : 'run');
    nodes.runButton.classList.toggle('success-action', isSucceeded && !isActive);
    nodes.runButton.classList.toggle('play-action', !isSucceeded && !isActive);
    nodes.runButton.classList.toggle('stop-action', isActive);
    nodes.runButton.textContent = isActive
        ? '停止'
        : (isSucceeded
            ? '成功'
            : (shouldOpenExisting
                ? (failureDisposition.category === 'review_required' ? '打开产物复核' : '查看已有产物')
                : (rerunCore
                    ? '按当前方案重跑核心分析'
                    : '继续运行')));
    nodes.runButton.title = isActive
        ? '停止当前运行任务'
        : ((isSucceeded || shouldOpenExisting) && runDir
            ? `打开资源目录：${runDir}`
            : (failureDisposition.action || '继续运行任务'));
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
    setText(nodes.detailUrl, jobSourceLabel(job));
    setText(nodes.detailRunDir, job.summary?.run_dir || job.run_dir);
    setText(nodes.detailMode, `${job.options?.analysis_mode || '-'} -> ${job.resolved_mode || '-'}`);
    const runtimeSnapshot = job.runtime_profile_snapshot || {};
    const runtimeModels = runtimeSnapshot.models || {};
    setText(
        nodes.detailProfile,
        `${runtimeSnapshot.profile || job.options?.profile || '-'}${runtimeSnapshot.fingerprint ? ` · ${runtimeSnapshot.fingerprint.slice(0, 10)}` : ''}`
    );
    setText(
        nodes.detailModels,
        Object.entries(runtimeModels)
            .map(([kind, value]) => `${kind}: ${value === true ? 'enabled' : value === false ? 'disabled' : value || '-'}`)
            .join(' · ') || '-'
    );
    setText(nodes.detailUpdated, job.updated_at);

    if (job.error_summary) {
        nodes.errorPanel.hidden = false;
        nodes.errorTitle.textContent = failureDisposition.label
            ? `流程状态：${failureDisposition.label}`
            : `流程失败：${job.error_summary.stage_label || job.error_summary.stage || '未知阶段'}`;
        nodes.errorMessage.textContent = [
            failureDisposition.reason,
            failureDisposition.action,
            job.error_summary.message
        ].filter(Boolean).join('；') || '未提供错误信息';
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
    renderStages(job, stageProgress);
    renderStageProgress(stageProgress);
    renderConsoleSummary(job);
    renderCoreDiagnostics(job.core_diagnostics);
    renderArtifacts(job.summary || {});
    renderConsoleSkillPanel(job);
    renderQaPanel(job);
    renderVscodePanel(job);
    loadSelectedLog(job);
}

function stageStatusLabel(status) {
    return {
        pending: '等待',
        queued: '排队',
        running: '运行中',
        succeeded: '成功',
        skipped: '跳过',
        failed: '失败',
        stopped: '已停止',
        created: '待启动'
    }[status] || status || '等待';
}

function preferredConsoleStage(job) {
    const order = job.stage_order || [];
    if (selectedConsoleStage && order.includes(selectedConsoleStage)) return selectedConsoleStage;
    const failed = order.find(stage => job.stages?.[stage]?.status === 'failed');
    const completed = [...order].reverse().find(stage => ['succeeded', 'skipped'].includes(job.stages?.[stage]?.status));
    return job.current_stage || failed || job.next_stage || completed || order[0] || '';
}

function stageProgressFor(stage, progress) {
    return progress?.stage === stage ? progress : null;
}

function normalizedStagePosition(progress) {
    if (!progress) return {};
    if (progress.position?.label) return progress.position;
    const vl = progress.vl || progress.details?.vl;
    const completed = Number(vl?.completed);
    const total = Number(vl?.total_selected);
    if (Number.isFinite(completed) && Number.isFinite(total) && total > 0) {
        const currentFrame = vl.current_frame_number;
        return {
            kind: 'frame',
            label: `VL ${completed}/${total}${currentFrame != null ? ` · 帧 #${currentFrame}` : ''}`,
            current: completed,
            total,
            unit: 'frame',
            percent: clampPercent((completed / total) * 100),
            eta_seconds: vl.eta_seconds,
            detail: currentFrame != null ? `当前帧 #${currentFrame}` : ''
        };
    }
    const current = (progress.steps || []).find(step => step.id === progress.current_step);
    const message = current?.message || '';
    const chapter = message.match(/\[(?:run|skip)\]\s+chapter\s+(\d+)\/(\d+)(?::\s*(.*))?/i);
    if (chapter) {
        return {
            kind: 'chapter',
            label: `章节 ${chapter[1]}/${chapter[2]}`,
            current: Number(chapter[1]),
            total: Number(chapter[2]),
            unit: 'chapter',
            percent: clampPercent((Number(chapter[1]) / Number(chapter[2])) * 100),
            detail: chapter[3] || ''
        };
    }
    const download = message.match(/^\[download\]\s+(\d+(?:\.\d+)?)%.*?(?:\sat\s+(.+?))?\s+ETA\s+(\d+:\d+(?::\d+)?)\s*$/i);
    if (download) {
        return {
            kind: 'download',
            label: `下载 ${download[1]}%`,
            current: Number(download[1]),
            total: 100,
            unit: 'percent',
            percent: Number(download[1]),
            detail: [download[2], `ETA ${download[3]}`].filter(Boolean).join(' · ')
        };
    }
    return {};
}

function stagePositionText(job, stage, info, progress) {
    const activeProgress = stageProgressFor(stage, progress);
    const position = normalizedStagePosition(activeProgress);
    if (position.label) return position.label;
    if (activeProgress?.current_label) return activeProgress.current_label;
    if (info.status === 'queued') {
        const position = info.queue_position || job.queue?.position;
        const size = job.queue?.size;
        const resource = info.queued_for || job.queue?.resource || '资源';
        return `等待 ${resource}${position ? ` #${position}${size ? `/${size}` : ''}` : ''}`;
    }
    if (info.status === 'failed') return info.error || '阶段失败';
    if (info.status === 'running') return '正在处理';
    if (info.status === 'succeeded') return '阶段已完成';
    if (info.status === 'skipped') return '无需执行';
    return '等待前序阶段';
}

function stageElapsedMarkup(job, stage, info) {
    const status = info.status || 'pending';
    if (status === 'running') {
        const startedAt = info.started_at || (job.current_stage === stage ? job.runner?.started_at : null);
        return startedAt ? consoleElapsedMarkup(startedAt, null, '已运行 ') : '运行中';
    }
    if (status === 'queued') {
        const queuedAt = info.queued_at || (job.current_stage === stage ? job.runner?.updated_at : null);
        return queuedAt ? consoleElapsedMarkup(queuedAt, null, '已等待 ') : '等待资源';
    }
    const seconds = Number(info.duration_seconds);
    if (Number.isFinite(seconds) && seconds >= 0) return `耗时 ${formatClock(seconds)}`;
    if (status === 'skipped') return '已跳过';
    if (status === 'failed') return '执行中断';
    return '尚未开始';
}

function renderStages(job, stageProgress) {
    renderConsoleStageFlow(job, stageProgress);
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
            selectedConsoleStage = button.dataset.stage;
            const matchingNode = (job.execution_flow?.nodes || []).find(node => node.stage === selectedConsoleStage);
            if (matchingNode) selectedConsoleNodeId = matchingNode.id;
            renderConsoleStageFlow(job, stageProgress);
            loadSelectedLog(job);
        });
    });
    nodes.stageDurationSummary.textContent = stageDurationSummary(job);
}

function renderConsoleStageFlow(job, stageProgress) {
    const flow = job.execution_flow;
    if (!flow?.nodes?.length) {
        renderConsoleEmptyState('当前任务没有可显示的执行拓扑');
        return;
    }
    selectedConsoleNodeId = preferredConsoleNode(job);
    const selectedNode = flow.nodes.find(node => node.id === selectedConsoleNodeId);
    selectedConsoleStage = selectedNode?.stage || preferredConsoleStage(job);
    nodes.consoleFlowSummary.innerHTML = consoleFlowHeadline(job, stageProgress);
    renderConsoleStageInspector(job, stageProgress);
    void renderConsoleExecutionDiagram(job);
}

function consoleFlowHeadline(job, stageProgress) {
    const flow = job.execution_flow || {};
    const active = (flow.active_node_ids || [])
        .map(nodeId => (flow.nodes || []).find(node => node.id === nodeId)?.title)
        .filter(Boolean);
    const failed = (flow.failed_node_ids || [])
        .map(nodeId => (flow.nodes || []).find(node => node.id === nodeId)?.title)
        .filter(Boolean);
    const location = active.length
        ? `正在执行：${active.join('、')}`
        : failed.length
            ? `失败节点：${failed.join('、')}`
            : job.status === 'succeeded'
                ? '模型执行与文档生成均已完成'
                : job.next_stage
                    ? `下一阶段：${stageNames[job.next_stage] || job.next_stage}`
                    : '等待继续运行';
    const runner = job.runner || {};
    const elapsed = runner.started_at
        ? consoleElapsedMarkup(runner.started_at, runner.finished_at, runner.finished_at ? '总耗时 ' : '已运行 ')
        : '-';
    return `<span>${escapeHtml(location)}</span><span class="console-flow-summary-time">${elapsed}</span>`;
}

function preferredConsoleNode(job) {
    const flow = job.execution_flow || {};
    const flowNodes = flow.nodes || [];
    if (selectedConsoleNodeId && flowNodes.some(node => node.id === selectedConsoleNodeId)) {
        return selectedConsoleNodeId;
    }
    return flow.active_node_ids?.[0]
        || flow.failed_node_ids?.[0]
        || (job.status === 'succeeded'
            ? flowNodes.find(node => node.id === 'input')?.id
            : flowNodes.find(node => node.status === 'pending')?.id)
        || [...flowNodes].reverse().find(node => node.status === 'succeeded')?.id
        || flowNodes[0]?.id
        || '';
}

function consoleFlowNodeElement(nodeId) {
    if (!nodeId || !nodes.consoleStageFlow) return null;
    return Array.from(nodes.consoleStageFlow.querySelectorAll('g.node')).find(node => {
        const id = node.getAttribute('id') || '';
        return id === nodeId
            || id.startsWith(`flowchart-${nodeId}-`)
            || id.includes(`-${nodeId}-`);
    }) || null;
}

async function renderConsoleExecutionDiagram(job) {
    const flow = job.execution_flow || {};
    const source = String(flow.mermaid || '').trim();
    const renderKey = `${job.job_id}:${source}`;
    if (renderKey === consoleFlowRenderKey && nodes.consoleStageFlow.querySelector('svg')) {
        applyConsoleFlowSelection(job);
        applyConsoleFlowScale();
        return;
    }
    if (!source) {
        nodes.consoleStageFlow.innerHTML = '<div class="console-stage-flow-empty">当前任务没有执行图定义</div>';
        return;
    }
    const fallback = renderSimpleMermaidFlowchart(source);
    if (!initializeMermaid()) {
        nodes.consoleStageFlow.innerHTML = fallback || '<div class="console-stage-flow-empty">Mermaid 渲染器未加载</div>';
        return;
    }
    const sequence = ++consoleFlowRenderSequence;
    nodes.consoleStageFlow.classList.add('loading');
    nodes.consoleStageFlow.innerHTML = '<div class="console-stage-flow-empty">执行图渲染中...</div>';
    try {
        const result = await window.mermaid.render(`console-execution-${Date.now()}-${sequence}`, source);
        if (sequence !== consoleFlowRenderSequence || currentJob?.job_id !== job.job_id) return;
        nodes.consoleStageFlow.classList.remove('loading');
        nodes.consoleStageFlow.innerHTML = result?.svg || fallback;
        consoleFlowRenderKey = renderKey;
        bindConsoleFlowDiagram(job);
        applyConsoleFlowScale();
        applyConsoleFlowSelection(job);
        const currentNodeId = flow.active_node_ids?.[0] || selectedConsoleNodeId;
        const centerKey = `${job.job_id}:${currentNodeId}`;
        if (currentNodeId && centerKey !== consoleFlowCenteredKey) {
            consoleFlowCenteredKey = centerKey;
            window.requestAnimationFrame(() => scrollConsoleNodeIntoView(currentNodeId, 'auto'));
        }
    } catch (error) {
        console.warn('Console execution flow render failed', error);
        if (sequence !== consoleFlowRenderSequence) return;
        nodes.consoleStageFlow.classList.remove('loading');
        nodes.consoleStageFlow.innerHTML = `${fallback || ''}<div class="console-stage-flow-empty">执行图渲染失败：${escapeHtml(error?.message || String(error))}</div>`;
    }
}

function bindConsoleFlowDiagram(job) {
    const flow = job.execution_flow || {};
    (flow.nodes || []).forEach(flowNode => {
        const graphNode = consoleFlowNodeElement(flowNode.id);
        if (!graphNode) return;
        graphNode.dataset.consoleNodeId = flowNode.id;
        graphNode.classList.add('console-flow-svg-node', `status-${flowNode.status || 'pending'}`);
        if (flowNode.node_kind) graphNode.classList.add(`kind-${flowNode.node_kind}`);
        graphNode.setAttribute('role', 'button');
        graphNode.setAttribute('tabindex', '0');
        graphNode.setAttribute(
            'aria-label',
            `${flowNode.title || flowNode.id}，${stageStatusLabel(flowNode.status)}`
        );
        const select = () => selectConsoleFlowNode(job, flowNode.id);
        graphNode.addEventListener('click', select);
        graphNode.addEventListener('keydown', event => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            select();
        });
    });
    const edgeElements = Array.from(nodes.consoleStageFlow.querySelectorAll('g.edgePath'));
    edgeElements.forEach((element, index) => {
        const edge = (flow.edges || [])[index];
        if (!edge) return;
        element.dataset.edgeFrom = edge.from;
        element.dataset.edgeTo = edge.to;
    });
    const edgeLabels = Array.from(nodes.consoleStageFlow.querySelectorAll('g.edgeLabel'));
    edgeLabels.forEach((element, index) => {
        const edge = (flow.edges || [])[index];
        if (!edge) return;
        element.dataset.edgeFrom = edge.from;
        element.dataset.edgeTo = edge.to;
    });
}

function selectConsoleFlowNode(job, nodeId) {
    selectedConsoleNodeId = nodeId;
    const flowNode = (job.execution_flow?.nodes || []).find(node => node.id === nodeId);
    selectedConsoleStage = flowNode?.stage || '';
    applyConsoleFlowSelection(job);
    renderConsoleStageInspector(job, job.stage_progress);
    scrollConsoleNodeIntoView(nodeId);
}

function consoleFlowAncestors(flow, nodeId) {
    const related = new Set([nodeId]);
    let changed = true;
    while (changed) {
        changed = false;
        (flow.edges || []).forEach(edge => {
            if (related.has(edge.to) && !related.has(edge.from)) {
                related.add(edge.from);
                changed = true;
            }
        });
    }
    return related;
}

function applyConsoleFlowSelection(job) {
    const flow = job.execution_flow || {};
    const selected = (flow.nodes || []).find(node => node.id === selectedConsoleNodeId);
    const showLineage = selected?.node_kind === 'output';
    const related = showLineage ? consoleFlowAncestors(flow, selectedConsoleNodeId) : null;
    (flow.nodes || []).forEach(flowNode => {
        const graphNode = consoleFlowNodeElement(flowNode.id);
        if (!graphNode) return;
        graphNode.classList.toggle('selected', flowNode.id === selectedConsoleNodeId);
        graphNode.classList.toggle('lineage-related', Boolean(related?.has(flowNode.id)));
        graphNode.classList.toggle('lineage-dimmed', Boolean(related && !related.has(flowNode.id)));
    });
    nodes.consoleStageFlow.querySelectorAll('[data-edge-from][data-edge-to]').forEach(edge => {
        const inLineage = related?.has(edge.dataset.edgeFrom) && related?.has(edge.dataset.edgeTo);
        edge.classList.toggle('lineage-related', Boolean(inLineage));
        edge.classList.toggle('lineage-dimmed', Boolean(related && !inLineage));
    });
}

function scrollConsoleNodeIntoView(nodeId, behavior = 'smooth') {
    if (!nodes.consoleStageFlowViewport || !nodeId) return;
    const target = consoleFlowNodeElement(nodeId);
    if (!target) return;
    const viewportRect = nodes.consoleStageFlowViewport.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const left = nodes.consoleStageFlowViewport.scrollLeft
        + targetRect.left
        - viewportRect.left
        - Math.max(0, (viewportRect.width - targetRect.width) / 2);
    nodes.consoleStageFlowViewport.scrollTo({ left: Math.max(0, left), behavior });
}

function applyConsoleFlowScale() {
    const svg = nodes.consoleStageFlow?.querySelector('svg');
    if (!svg) return;
    const viewBoxWidth = Number(svg.viewBox?.baseVal?.width) || 2200;
    const width = Math.max(960, viewBoxWidth * consoleFlowScale);
    svg.style.width = `${width}px`;
    svg.style.maxWidth = 'none';
    svg.style.height = 'auto';
}

function fitConsoleFlow() {
    const svg = nodes.consoleStageFlow?.querySelector('svg');
    if (!svg || !nodes.consoleStageFlowViewport) return;
    const viewBoxWidth = Number(svg.viewBox?.baseVal?.width) || 2200;
    consoleFlowScale = Math.max(0.55, Math.min(1, (nodes.consoleStageFlowViewport.clientWidth - 24) / viewBoxWidth));
    applyConsoleFlowScale();
    nodes.consoleStageFlowViewport.scrollTo({ left: 0, behavior: 'smooth' });
}

function renderConsoleStageInspector(job, stageProgress) {
    const flow = job.execution_flow || {};
    const flowNode = (flow.nodes || []).find(node => node.id === preferredConsoleNode(job));
    if (!flowNode) {
        nodes.consoleStageInspector.hidden = true;
        return;
    }
    selectedConsoleNodeId = flowNode.id;
    const stage = flowNode.stage || '';
    selectedConsoleStage = stage;
    const order = job.stage_order || [];
    const info = job.stages?.[stage] || {};
    const status = flowNode.status || info.status || 'pending';
    const model = flowNode.model || {};
    const nodeDuration = Number(flowNode.duration_seconds);
    const elapsed = Number.isFinite(nodeDuration)
        ? formatClock(nodeDuration)
        : flowNode.started_at
            ? consoleElapsedMarkup(flowNode.started_at, flowNode.finished_at, '')
            : '-';
    const workerText = [
        model.worker_count != null ? `${model.worker_count} worker` : '',
        model.concurrency != null ? `并发 ${model.concurrency}` : ''
    ].filter(Boolean).join(' · ') || '-';
    const metrics = [
        ['节点状态', stageStatusLabel(status)],
        ['使用模型', model.label || (flowNode.node_kind === 'output' ? '文档产物' : '规则处理')],
        ['提供方', model.provider || '-'],
        ['部署位置', model.deployment || '-'],
        ['Worker / 并发', workerText],
        ['所属阶段', stageNames[stage] || stage || '输出'],
        ['节点耗时', elapsed],
        ['节点进度', flowNode.progress != null ? `${flowNode.progress}%` : '-']
    ];
    const message = [
        flowNode.message,
        model.inherited_from ? `模型继承自 ${model.inherited_from}` : '',
        ...(flowNode.metrics || []).map(metric => `${metric.label} ${metric.value}`),
        info.retry_reason,
        info.warning
    ].filter(Boolean).join('；');
    nodes.consoleStageInspector.hidden = false;
    nodes.consoleStageInspectorKicker.textContent = flowNode.node_kind === 'output'
        ? '最终文档推导'
        : model.role_label || `第 ${Math.max(1, order.indexOf(stage) + 1)} / ${order.length} 阶段`;
    nodes.consoleStageInspectorTitle.textContent = flowNode.title || flowNode.id;
    nodes.consoleStageInspectorStatus.innerHTML = `<span class="status ${escapeHtml(status)}">${escapeHtml(stageStatusLabel(status))}</span>`;
    nodes.consoleStageInspectorMetrics.innerHTML = metrics.map(([label, value]) => `<div>
        <span>${escapeHtml(label)}</span>
        <strong>${label === '节点耗时' ? value : escapeHtml(value)}</strong>
    </div>`).join('');
    nodes.consoleStageInspectorMessage.textContent = message || flowNode.subtitle || '当前节点暂无额外运行信号。';
    nodes.consoleStageInspectorArtifacts.innerHTML = (flowNode.artifacts || []).length
        ? (flowNode.artifacts || []).map(artifact => artifact.url
            ? `<a class="secondary-link" href="${escapeHtml(artifact.url)}" target="_blank" rel="noreferrer">${escapeHtml(artifact.path)}</a>`
            : `<span>${escapeHtml(artifact.path)}${artifact.file_count != null ? ` · ${escapeHtml(artifact.file_count)} 个文件` : ''}</span>`
        ).join('')
        : '<span class="muted">暂无节点产物</span>';
    nodes.consoleStageLogButton.disabled = !info.log_path;
    nodes.consoleStageLogButton.dataset.stage = stage;
}

function renderConsoleSummary(job) {
    const order = job.stage_order || [];
    const completed = order.filter(stage => ['succeeded', 'skipped'].includes(job.stages?.[stage]?.status)).length;
    const failedStage = order.find(stage => job.stages?.[stage]?.status === 'failed');
    const totalSeconds = totalStageDuration(job);
    const queueSeconds = order.reduce((total, stage) => {
        const info = job.stages?.[stage] || {};
        const seconds = info.queued_at
            ? elapsedSeconds(info.queued_at, info.started_at || info.finished_at || (info.status === 'queued' ? null : job.updated_at))
            : null;
        return seconds == null ? total : total + seconds;
    }, 0);
    const bottleneck = order
        .map(stage => {
            const info = job.stages?.[stage] || {};
            const recorded = Number(info.duration_seconds);
            const seconds = Number.isFinite(recorded)
                ? recorded
                : (info.status === 'running' ? elapsedSeconds(info.started_at) : null);
            return { stage, seconds };
        })
        .filter(item => Number.isFinite(item.seconds) && item.seconds >= 0)
        .sort((left, right) => right.seconds - left.seconds)[0];
    const runner = job.runner || {};
    const overallElapsed = runner.started_at
        ? consoleElapsedMarkup(runner.started_at, runner.finished_at, runner.finished_at ? '' : '')
        : '-';
    const coreCounts = job.summary?.core_counts || {};
    const chapterCount = job.summary?.multidoc?.chapter_count ?? job.summary?.study?.chapter_count;
    const markdownCount = (job.summary?.markdown_files || []).length;
    const imageCount = (job.summary?.final_images || []).length;
    const warningCount = (job.warnings || []).length;
    const metrics = [
        ['本轮耗时', overallElapsed, runner.started_at ? (runner.finished_at ? '本轮已结束' : '持续更新中') : '尚未启动'],
        ['阶段完成', `${completed} / ${order.length}`, `${job.progress?.percent || 0}%`],
        ['处理耗时', totalSeconds > 0 ? formatClock(totalSeconds) : '-', '各阶段执行时间合计'],
        ['排队等待', queueSeconds > 0 ? formatClock(queueSeconds) : '0:00', '可识别的资源等待时间'],
        ['最耗时阶段', bottleneck ? stageNames[bottleneck.stage] || bottleneck.stage : '-', bottleneck ? formatClock(bottleneck.seconds) : '暂无'],
        ['视频长度', durationMinutes(Number(job.preview?.duration_seconds)), '源素材时长'],
        ['文档产物', String(markdownCount), `${imageCount} 张最终图片`],
        ['流程信号', failedStage ? '存在失败' : (warningCount ? `${warningCount} 条警告` : '正常'), failedStage ? stageNames[failedStage] || failedStage : '']
    ];
    nodes.consoleSummaryHeadline.textContent = failedStage
        ? `流程停止在 ${stageNames[failedStage] || failedStage}，下方保留失败前的处理数据。`
        : (job.status === 'succeeded'
            ? `全部 ${order.length} 个阶段已处理完成。`
            : `已完成 ${completed} / ${order.length} 个阶段，数据会随任务运行持续更新。`);
    nodes.consoleSummaryGrid.innerHTML = metrics.map(([label, value, detail]) => `<div class="console-summary-metric">
        <span>${escapeHtml(label)}</span>
        <strong>${label === '本轮耗时' ? value : escapeHtml(value)}</strong>
        <small>${escapeHtml(detail || '')}</small>
    </div>`).join('');

    const results = [
        ['扫描帧', coreCounts.scan_frames],
        ['候选帧', coreCounts.frames_extracted],
        ['OCR候选', coreCounts.ocr_candidate_frames],
        ['OCR帧', coreCounts.ocr_keyframes],
        ['OCR事件', coreCounts.ocr_text_events],
        ['VL帧', coreCounts.vl_frames],
        ['章节', chapterCount],
        ['最终图片', imageCount]
    ].filter(([, value]) => value != null && (job.status === 'succeeded' || Number(value) > 0));
    nodes.consoleResultSummary.innerHTML = results.length
        ? results.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join('')
        : '<div class="console-result-empty">处理量与产物统计将在核心分析写出结果后显示。</div>';
}

function renderConsoleEmptyState(message) {
    if (nodes.consoleFlowSummary) nodes.consoleFlowSummary.textContent = message;
    if (nodes.consoleStageFlow) nodes.consoleStageFlow.innerHTML = `<div class="console-stage-flow-empty">${escapeHtml(message)}</div>`;
    if (nodes.consoleStageInspector) nodes.consoleStageInspector.hidden = true;
    if (nodes.consoleStageInspectorArtifacts) nodes.consoleStageInspectorArtifacts.innerHTML = '';
    if (nodes.consoleSummaryHeadline) nodes.consoleSummaryHeadline.textContent = '选择任务后查看耗时、处理量和产物概况';
    if (nodes.consoleSummaryGrid) nodes.consoleSummaryGrid.innerHTML = '';
    if (nodes.consoleResultSummary) nodes.consoleResultSummary.innerHTML = '<div class="console-result-empty">暂无运行数据</div>';
}

function renderStageProgress(progress) {
    const hasVisibleStep = progress && (progress.steps || []).some(step => step.status !== 'pending');
    nodes.corePanel.hidden = !hasVisibleStep;
    if (!hasVisibleStep) return;
    const percentText = progress.percent != null ? ` · 约 ${progress.percent}%` : '';
    nodes.corePanelTitle.textContent = progress.stage_label ? `${progress.stage_label}子项${percentText}` : `阶段子项${percentText}`;
    const summary = progress.summary || progress.current_label || progress.last_signal_label || '';
    const vl = progress.vl || progress.details?.vl;
    const vlTotal = Number(vl?.total_selected);
    const vlCompleted = Number(vl?.completed);
    const vlAverageSeconds = Number(vl?.average_frame_seconds);
    const vlEtaSeconds = Number(vl?.eta_seconds);
    const vlPercent = Number.isFinite(vlTotal) && vlTotal > 0 && Number.isFinite(vlCompleted)
        ? clampPercent((vlCompleted / vlTotal) * 100)
        : 0;
    const vlSpeed = Number.isFinite(vlAverageSeconds) && vlAverageSeconds > 0
        ? `${(60 / vlAverageSeconds).toFixed(2)} 帧/分钟`
        : '-';
    const vlDetail = vl ? `<div class="vl-progress-detail">
        <div><span>VL 帧进度</span><strong>${escapeHtml(vlCompleted || 0)} / ${escapeHtml(vlTotal || 0)} (${escapeHtml(vlPercent.toFixed(1))}%)</strong></div>
        <div><span>断点复用</span><strong>${escapeHtml(vl.reused || 0)} 帧</strong></div>
        <div><span>本轮失败</span><strong>${escapeHtml(vl.failed || 0)} 帧</strong></div>
        <div><span>处理速度</span><strong>${escapeHtml(vlSpeed)}</strong></div>
        <div><span>单帧中位耗时</span><strong>${Number.isFinite(vlAverageSeconds) && vlAverageSeconds > 0 ? escapeHtml(`${vlAverageSeconds.toFixed(1)} 秒`) : '-'}</strong></div>
        <div><span>预计剩余</span><strong>${Number.isFinite(vlEtaSeconds) && vlEtaSeconds >= 0 ? escapeHtml(formatClock(vlEtaSeconds)) : '-'}</strong></div>
    </div>` : '';
    const summaryRow = `<tr class="stage-progress-meta ${progress.live ? 'live' : ''} ${progress.stale ? 'stale' : ''}">
        <td colspan="4">
            <div class="stage-progress-head">
                <span>${escapeHtml(summary)}</span>
                <strong>${escapeHtml(progress.percent ?? 0)}%</strong>
            </div>
            <div class="bar stage-progress-bar"><div style="width:${escapeHtml(progress.percent ?? 0)}%"></div></div>
            ${vlDetail}
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

function renderConsoleSkillPanel(job) {
    if (!nodes.consoleSkillSummary) return;
    const skill = job?.summary?.skill_distillation || job?.summary?.skill_candidate || {};
    const status = skill.status || 'not_started';
    const statusLabels = {
        not_started: '尚未开始',
        running: '蒸馏运行中',
        waiting_overview_review: '等待骨架确认',
        waiting_candidate_review: '等待候选确认',
        succeeded: skill.enabled ? '已完成并启用' : '蒸馏完成',
        completed_no_skills: '未生成可交付 Skill',
        failed: '蒸馏失败',
        interrupted: '蒸馏中断',
        cancelled: '蒸馏已取消',
        ready: '等待继续'
    };
    const progress = Number(skill.progress?.percent || 0);
    const passed = Number(skill.skills?.passed || 0);
    nodes.consoleSkillSummary.textContent = job
        ? `${statusLabels[status] || status} · ${progress}% · 通过 ${passed} 个`
        : '选择任务后查看蒸馏状态';
    nodes.openSkillsWorkspaceButton.disabled = !job;
}

function renderQaPanel(job) {
    if (!nodes.qaView) return;
    const qa = job?.summary?.qa || {};
    const available = Boolean(qa.available || qa.answer_index?.exists);
    const chunkCount = qa.answer_index?.chunk_count || qa.chunk_count || 0;
    const runDir = job?.summary?.run_dir || job?.run_dir || '';
    nodes.qaAskButton.disabled = !job || !available;
    nodes.qaQuestion.disabled = !job || !available;
    if (!job) {
        nodes.qaSummary.textContent = '选择一个已生成问答索引的任务';
        nodes.qaWarnings.textContent = '等待问答索引';
        resetQaMessages();
        return;
    }
    if (!available) {
        nodes.qaSummary.textContent = runDir ? '问答索引尚未生成' : '资源目录尚未生成';
        nodes.qaWarnings.textContent = '等待“问答证据索引”阶段完成';
        resetQaMessages();
        return;
    }
    nodes.qaSummary.textContent = `已索引 ${chunkCount} 个证据片段 · ${runDir}`;
    const warnings = qa.answer_index?.quality_warnings || qa.quality_warnings || qa.warnings || [];
    nodes.qaWarnings.innerHTML = warnings.length
        ? warnings.map(item => `<div class="qa-warning">${escapeHtml(item.message || item.code || item)}</div>`).join('')
        : '<div class="qa-ok">未发现明显证据边界警告</div>';
    loadFrameTimeMap(job.job_id);
    loadQaHistory(job.job_id);
}

function resetQaMessages() {
    if (!nodes.qaMessages) return;
    loadedQaHistoryKey = '';
    nodes.qaMessages.innerHTML = '<div class="qa-empty">暂无对话</div>';
}

function clearQaMessages() {
    if (!nodes.qaMessages) return;
    nodes.qaMessages.innerHTML = '<div class="qa-empty">暂无对话</div>';
}

function skillActivityMessage(skill) {
    const stage = skill.current_stage || '';
    if (stage === 'source') return '正在整理 transcript、OCR、视觉和页面证据';
    if (stage === 'overview') return '正在调用模型生成内容整体理解';
    if (stage === 'extract') return '五路提取器正在并行生成候选';
    if (stage === 'verify') return '正在执行独立语境、迁移性和多模态验证';
    if (stage === 'build') {
        const current = skill.skills?.test_progress?.current_skill;
        return current ? `正在构造 ${current}` : '正在调用模型构造 Skill 定义';
    }
    if (stage === 'link') return '正在整理 Skill 关系和触发边界';
    if (stage === 'test') {
        const testProgress = skill.skills?.test_progress || {};
        const phase = skillTestPhaseNames[testProgress.phase] || '模型调用中';
        return testProgress.current_skill ? `${phase} · ${testProgress.current_skill}` : phase;
    }
    if (stage === 'deliver') return '正在生成索引、摘要和最终交付包';
    return '蒸馏任务正在运行';
}

function skillActivityStartedAt(skill) {
    const current = skill.current_stage || '';
    return skill.progress?.stages?.[current]?.started_at || skill.updated_at || skill.created_at || '';
}

function renderSkillStageRail(skill) {
    if (!nodes.skillStageRail) return;
    const status = skill.status || 'not_started';
    if (status === 'not_started') {
        nodes.skillStageRail.hidden = true;
        nodes.skillStageRail.innerHTML = '';
        return;
    }
    const stages = skill.progress?.stages || {};
    nodes.skillStageRail.hidden = false;
    nodes.skillStageRail.innerHTML = Object.entries(skillStageNames).map(([name, label]) => {
        const stageStatus = stages[name]?.status || (name === skill.current_stage ? 'running' : 'pending');
        const normalized = ['succeeded', 'running', 'failed'].includes(stageStatus) ? stageStatus : 'pending';
        return `<div class="skill-stage-marker ${normalized}${name === skill.current_stage ? ' current' : ''}">
            <span aria-hidden="true"></span>
            <em>${escapeHtml(label)}</em>
        </div>`;
    }).join('');
}

function renderSkillLiveActivity(skill) {
    ensureSkillActivityNodes();
    if (!nodes.skillLiveActivity || !nodes.skillProgress) return;
    const status = skill.status || 'not_started';
    const active = status === 'running';
    const waiting = status === 'waiting_overview_review' || status === 'waiting_candidate_review';
    nodes.skillProgress.classList.toggle('running', active);
    nodes.skillProgress.classList.toggle('waiting', waiting);
    renderSkillStageRail(skill);
    if (!active && !waiting) {
        nodes.skillLiveActivity.hidden = true;
        nodes.skillLiveActivity.innerHTML = '';
        return;
    }
    const stageLabel = skillStageNames[skill.current_stage] || skill.current_stage || '准备中';
    const title = waiting
        ? (status === 'waiting_overview_review' ? '等待确认整体理解' : '等待确认候选')
        : `正在${stageLabel}`;
    const detail = waiting
        ? '模型任务已暂停，确认后继续执行'
        : skillActivityMessage(skill);
    nodes.skillLiveActivity.hidden = false;
    nodes.skillLiveActivity.className = `skill-live-activity ${active ? 'running' : 'waiting'}`;
    nodes.skillLiveActivity.dataset.startedAt = active ? skillActivityStartedAt(skill) : '';
    nodes.skillLiveActivity.dataset.updatedAt = skill.updated_at || '';
    nodes.skillLiveActivity.innerHTML = `<div class="skill-activity-signal" aria-hidden="true">
        <span></span><span></span><span></span>
    </div>
    <div class="skill-activity-copy">
        <strong>${escapeHtml(title)}</strong>
        <span>${escapeHtml(detail)}</span>
    </div>
    <div class="skill-activity-time">
        <strong data-skill-elapsed>${active ? '已运行 --' : '等待操作'}</strong>
        <span data-skill-heartbeat>状态刚刚更新</span>
    </div>`;
    updateSkillLiveClock();
}

function formatSkillElapsed(totalSeconds) {
    const seconds = Math.max(0, Math.floor(totalSeconds));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainder = seconds % 60;
    if (hours > 0) return `${hours}小时 ${minutes}分 ${remainder}秒`;
    if (minutes > 0) return `${minutes}分 ${remainder}秒`;
    return `${remainder}秒`;
}

function updateSkillLiveClock() {
    const activity = nodes.skillLiveActivity;
    if (!activity || activity.hidden) return;
    const now = Date.now();
    const elapsedNode = activity.querySelector('[data-skill-elapsed]');
    const heartbeatNode = activity.querySelector('[data-skill-heartbeat]');
    const startedAt = Date.parse(activity.dataset.startedAt || '');
    const updatedAt = Date.parse(activity.dataset.updatedAt || '');
    if (elapsedNode && Number.isFinite(startedAt)) {
        elapsedNode.textContent = `已运行 ${formatSkillElapsed((now - startedAt) / 1000)}`;
    }
    if (heartbeatNode && Number.isFinite(updatedAt)) {
        const ageSeconds = Math.max(0, Math.floor((now - updatedAt) / 1000));
        heartbeatNode.textContent = ageSeconds < 5
            ? '状态刚刚更新'
            : ageSeconds < 60
                ? `${ageSeconds} 秒前更新`
                : `${Math.floor(ageSeconds / 60)} 分钟前更新`;
    }
}

function renderSkillCandidatePanel(job) {
    if (!nodes.skillSummary) return;
    const skill = job?.summary?.skill_distillation || job?.summary?.skill_candidate || {};
    const runDir = job?.summary?.run_dir || job?.run_dir || '';
    const status = skill.status || 'not_started';
    const active = status === 'running';
    const waitingOverview = status === 'waiting_overview_review';
    const waitingCandidates = status === 'waiting_candidate_review';
    const resumable = ['failed', 'interrupted', 'cancelled', 'ready'].includes(status);
    const completed = ['succeeded', 'completed_no_skills'].includes(status);
    const passedSkills = Number(skill.skills?.passed || 0);
    const enabled = Boolean(skill.enabled);
    const progress = Number(skill.progress?.percent || 0);
    nodes.skillProgressBar.style.width = `${Math.max(0, Math.min(progress, 100))}%`;
    renderSkillLiveActivity(skill);
    nodes.generateSkillButton.disabled = !job || !runDir || active || waitingOverview || waitingCandidates;
    nodes.createTargetedSkillProjectButton.disabled = !job;
    nodes.generateSkillButton.textContent = completed ? '重新蒸馏' : '开始蒸馏';
    nodes.resumeSkillButton.hidden = !resumable;
    nodes.resumeSkillButton.disabled = !job || active;
    nodes.cancelSkillButton.hidden = !active;
    nodes.cancelSkillButton.disabled = !active;
    nodes.enableSkillButton.disabled = !job || status !== 'succeeded' || passedSkills <= 0 || enabled;
    nodes.skillProfile.disabled = status !== 'not_started';
    if (skill.profile && nodes.skillProfile.querySelector(`option[value="${CSS.escape(skill.profile)}"]`)) {
        nodes.skillProfile.value = skill.profile;
    }
    nodes.skillOverviewReview.hidden = !waitingOverview;
    nodes.skillCandidateReview.hidden = !waitingCandidates;
    if (!waitingCandidates && job?.job_id) {
        skillCandidateDrafts.delete(job.job_id);
    }
    if (!job) {
        nodes.skillSummary.textContent = '选择一个任务后开始蒸馏';
        nodes.skillWarnings.innerHTML = '';
        nodes.skillOverviewReview.hidden = true;
        nodes.skillCandidateReview.hidden = true;
        renderSkillLiveActivity({ status: 'not_started' });
        return;
    }
    if (status === 'not_started') {
        nodes.skillSummary.textContent = runDir ? '尚未开始 · 默认 DeepSeek V4 Pro' : '资源目录尚未生成';
        nodes.skillWarnings.innerHTML = '';
        return;
    }
    const stageLabel = skillStageNames[skill.current_stage] || skill.current_stage || '-';
    const model = skill.review_model && skill.review_model !== skill.generation_model
        ? `${skill.generation_model || '-'} / ${skill.review_model}`
        : (skill.generation_model || '-');
    const modelSummary = skill.vision_model
        ? `${model} · 视觉复核 ${skill.vision_model}`
        : model;
    const statusLabels = {
        running: '运行中',
        waiting_overview_review: '等待骨架确认',
        waiting_candidate_review: '等待候选确认',
        succeeded: enabled ? '已启用' : '蒸馏完成',
        completed_no_skills: '未发现可交付 Skill',
        failed: '失败',
        interrupted: '已中断',
        cancelled: '已取消',
        ready: '等待继续'
    };
    nodes.skillSummary.textContent = `${statusLabels[status] || status} · ${stageLabel} · ${modelSummary}`;
    const warnings = skill.warnings || [];
    const messages = [
        ...(skill.error ? [skill.error] : []),
        ...warnings.map(item => item.message || item.code || item)
    ];
    if (completed) {
        messages.push(`Skills ${skill.skills?.count || 0} · 通过 ${passedSkills} · 未通过 ${skill.skills?.failed || 0}`);
    }
    if (active && skill.current_stage === 'test') {
        const testProgress = skill.skills?.test_progress || {};
        const testStartedAt = skill.progress?.stages?.test?.started_at;
        const elapsedSeconds = testStartedAt
            ? Math.max(0, Math.floor((Date.now() - new Date(testStartedAt).getTime()) / 1000))
            : null;
        const parts = [
            `压力测试 ${testProgress.completed || 0}/${testProgress.total || skill.skills?.count || 0}`,
            skillTestPhaseNames[testProgress.phase] || '模型调用中'
        ];
        if (testProgress.current_skill) parts.push(testProgress.current_skill);
        if (Number.isInteger(testProgress.repair_round) && testProgress.repair_round > 0) {
            parts.push(`修订轮次 ${testProgress.repair_round}`);
        }
        if (elapsedSeconds != null) parts.push(`已运行 ${elapsedSeconds} 秒`);
        messages.push(parts.join(' · '));
    }
    if (waitingCandidates || completed) {
        messages.push(
            `候选：已验证 ${skill.candidates?.accepted_count || 0} · `
            + `单案例 ${skill.candidates?.single_case_count || 0} · `
            + `拒绝 ${skill.candidates?.rejected_count || 0} · `
            + `术语 ${skill.candidates?.glossary_count || 0}`
        );
    }
    nodes.skillWarnings.innerHTML = messages.length
        ? messages.map(item => `<div class="skill-warning">${escapeHtml(item)}</div>`).join('')
        : '<div class="skill-ok">证据与 checkpoint 正常。</div>';
    if (waitingOverview) {
        nodes.skillOverviewPreview.innerHTML = renderMarkdown([
            `# ${skill.overview?.title || '整体理解'}`,
            '',
            skill.overview?.summary || '',
            '',
            `证据分块：${skill.overview?.chunk_count || 0}`
        ].join('\n'));
    }
    if (waitingCandidates) {
        renderSkillCandidates(skill.candidates || {}, job.job_id);
    }
}

function renderSkillCandidates(candidates, jobId) {
    const rows = [
        ...(candidates.accepted || []).map(item => ({ ...item, tier: 'verified' })),
        ...(candidates.single_case || []).map(item => ({ ...item, tier: 'single-case' })),
        ...(candidates.rejected || []).map(item => ({ ...item, tier: 'rejected' }))
    ];
    const candidateKey = rows.map(item => item.id).sort().join('|');
    let draft = skillCandidateDrafts.get(jobId);
    if (!draft || draft.candidateKey !== candidateKey) {
        draft = {
            candidateKey,
            selected: new Set(candidates.selected_ids || [])
        };
        skillCandidateDrafts.set(jobId, draft);
    }
    const selected = draft.selected;
    const scrollTop = nodes.skillCandidateList.scrollTop;
    nodes.skillCandidateList.innerHTML = rows.length
        ? rows.map(item => {
            const reason = item.tier === 'verified'
                ? `已验证 · 独立案例 ${item.v1?.independent_context_count || 0}`
                : item.tier === 'single-case'
                    ? `单案例 · ${item.reason || '需要人工确认'}`
                    : (item.reason || `未通过 ${item.failed_checks?.join(', ') || '验证'}`);
            return `<label class="skill-candidate-choice ${item.tier}">
                <input type="checkbox" value="${escapeHtml(item.id)}" ${selected.has(item.id) ? 'checked' : ''}>
                <span>
                    <strong>${escapeHtml(item.title || item.id)}</strong>
                    <small>${escapeHtml(reason)} · ${escapeHtml((item.source_ids || []).join(', ') || '无证据 ID')}</small>
                </span>
            </label>`;
        }).join('')
        : '<div class="qa-empty">没有候选</div>';
    nodes.skillCandidateList.scrollTop = scrollTop;
}

function updateSkillCandidateDraft(event) {
    const input = event.target.closest('input[type="checkbox"]');
    if (!input || !selectedJobId) return;
    const draft = skillCandidateDrafts.get(selectedJobId);
    if (!draft) return;
    if (input.checked) {
        draft.selected.add(input.value);
    } else {
        draft.selected.delete(input.value);
    }
}

function renderSkillsWorkspace(job) {
    renderSkillCandidatePanel(job);
    if (currentResourceView !== 'skills') return;
    nodes.skillsScopeTabs.forEach(button => {
        button.classList.toggle('active', button.dataset.skillsScope === currentSkillsScope);
    });
    nodes.currentSkillsWorkspace.hidden = currentSkillsScope !== 'current';
    nodes.skillProjectsWorkspace.hidden = currentSkillsScope !== 'projects';
    nodes.skillLibraryWorkspace.hidden = currentSkillsScope === 'current' || currentSkillsScope === 'projects';
    nodes.skillLibrarySearchLabel.hidden = currentSkillsScope === 'current' || currentSkillsScope === 'projects';
    window.requestAnimationFrame(constrainSkillsLayouts);
    if (currentSkillsScope === 'projects') {
        if (!currentSkillProject && !skillProjectWorkbenchLoading) {
            void loadSkillProjects();
        }
        syncSkillProjectPolling();
        return;
    }
    syncSkillProjectPolling();
    if (currentSkillsScope !== 'current') return;
    if (!job?.job_id || !(job.summary?.run_dir || job.run_dir)) {
        loadedSkillsWorkspaceKey = '';
        currentSkillWorkspace = null;
        nodes.skillsCurrentCount.textContent = '0 项';
        nodes.skillsCurrentList.innerHTML = '<div class="skills-empty">选择一个已有运行目录的任务</div>';
        resetCurrentSkillDetail();
        return;
    }
    const skill = job.summary?.skill_distillation || job.summary?.skill_candidate || {};
    const workspaceKey = `${job.job_id}|${skill.updated_at || skill.status || 'not_started'}|${skill.progress?.percent || 0}`;
    if (workspaceKey !== loadedSkillsWorkspaceKey) {
        loadCurrentSkillsWorkspace(job.job_id, workspaceKey);
    }
}

async function loadSkillProjects(preferredProjectId = selectedSkillProjectId) {
    if (currentSkillsScope !== 'projects') return;
    if (skillProjectWorkbenchLoading) return;
    skillProjectWorkbenchLoading = true;
    try {
        const payload = await getJson(
            `/api/skill-projects/workbench?project_id=${encodeURIComponent(preferredProjectId || '')}`
        );
        if (currentSkillsScope !== 'projects') return;
        applySkillProjectWorkbench(payload);
    } catch (error) {
        nodes.skillProjectsList.innerHTML = `<div class="skills-empty error-text">加载失败：${escapeHtml(error.message)}</div>`;
    } finally {
        skillProjectWorkbenchLoading = false;
    }
}

async function selectSkillProject(projectId) {
    selectedSkillProjectId = projectId;
    selectedSkillProjectFlowNodeId = '';
    skillProjectPackagePreview = null;
    await loadSkillProjects(projectId);
}

const skillProjectFlowStatusLabels = {
    pending: '未开始',
    needs_action: '需要操作',
    limited: '可继续但有限制',
    waiting: '等待确认',
    running: '运行中',
    succeeded: '已完成',
    failed: '失败',
    interrupted: '可继续',
    cancelled: '已取消',
    cancelling: '正在停止',
    waiting_resource_decision: '需要资源裁定'
};

function skillProjectListKey(items, selectedId) {
    return JSON.stringify((items || []).map(item => [
        item.id,
        item.revision,
        item.status,
        item.assessment?.summary || '',
        item.id === selectedId
    ]));
}

function applySkillProjectWorkbench(payload) {
    const items = payload.items || [];
    const selectedId = payload.selected_project_id || '';
    const listKey = skillProjectListKey(items, selectedId);
    selectedSkillProjectId = selectedId;
    currentSkillProject = payload.project || null;
    currentSkillProjectWorkspace = payload.workspace || null;
    currentSkillProjectFlow = payload.flow || null;
    if (listKey !== skillProjectListSignature) {
        skillProjectListSignature = listKey;
        nodes.skillProjectsCount.textContent = `${items.length} 项`;
        nodes.skillProjectsList.innerHTML = items.length
            ? items.map(item => `
                <button class="skills-item ${item.id === selectedSkillProjectId ? 'active' : ''}" type="button" data-skill-project-id="${escapeHtml(item.id)}">
                    <strong>${escapeHtml(item.title || item.goal || item.id)}</strong>
                    <span>${escapeHtml(item.status || 'draft')} · ${escapeHtml(item.skill_type || '-')}</span>
                    <small>${escapeHtml(item.assessment?.summary || item.goal || '')}</small>
                </button>
            `).join('')
            : '<div class="skills-empty">暂无项目</div>';
        nodes.skillProjectsList.querySelectorAll('[data-skill-project-id]').forEach(button => {
            button.addEventListener('click', () => {
                void selectSkillProject(button.dataset.skillProjectId || '');
            });
        });
    }
    renderSkillProjectWorkbench(payload.snapshot_version || '');
}

function renderSkillProjectWorkbench(snapshotVersion = '') {
    const project = currentSkillProject;
    const workspace = currentSkillProjectWorkspace;
    if (!project) {
        skillProjectDetailSignature = '';
        selectedSkillProjectFlowNodeId = '';
        nodes.skillProjectFlowSummary.textContent = '创建一个目标，然后导入 Video Analyzer 资料包。';
        nodes.skillProjectFlowStatus.textContent = '未选择';
        nodes.skillProjectFlowStatus.className = 'skill-project-flow-status';
        nodes.skillProjectFlow.innerHTML = '<div class="skills-empty">创建或选择一个 Skill 项目</div>';
        nodes.skillProjectInspectorTitle.textContent = '流程详情';
        nodes.skillProjectInspectorMeta.textContent = '选择流程节点查看当前状态';
        nodes.skillProjectDetail.innerHTML = '<div class="skills-empty">创建或选择一个 Skill 项目</div>';
        nodes.skillProjectSources.innerHTML = '暂无资料';
        nodes.previewSkillProjectPackageButton.disabled = true;
        nodes.importSkillProjectPackageButton.disabled = true;
        nodes.skillProjectPackagePreview.textContent = '输入资料包 ID 后检查。';
        renderSkillProjectLogControls();
        return;
    }
    const assessment = project.assessment || {};
    const sources = project.sources || [];
    const flow = currentSkillProjectFlow || { nodes: [], active_node: 'packages' };
    const flowNodes = flow.nodes || [];
    if (!selectedSkillProjectFlowNodeId || !flowNodes.some(node => node.id === selectedSkillProjectFlowNodeId)) {
        const enableNode = flowNodes.find(node => node.id === 'enable' && node.status === 'succeeded');
        selectedSkillProjectFlowNodeId = enableNode?.id || flow.active_node || flowNodes[0]?.id || '';
    }
    const currentNode = flowNodes.find(node => node.id === selectedSkillProjectFlowNodeId) || flowNodes[0] || null;
    const status = currentNode?.status || project.status || 'pending';
    nodes.skillProjectFlowSummary.textContent = `${project.title || project.goal} · ${sources.length} 个资料来源`;
    nodes.skillProjectFlowStatus.textContent = skillProjectFlowStatusLabels[status] || status;
    nodes.skillProjectFlowStatus.className = `skill-project-flow-status ${status}`;
    const flowKey = JSON.stringify(flowNodes.map(node => [
        node.id,
        node.status,
        node.subtitle,
        node.action,
        node.secondary_action,
        node.id === selectedSkillProjectFlowNodeId
    ]));
    if (flowKey !== nodes.skillProjectFlow.dataset.flowKey) {
        nodes.skillProjectFlow.dataset.flowKey = flowKey;
        nodes.skillProjectFlow.innerHTML = flowNodes.map((node, index) => `
            <button class="skill-project-flow-node ${escapeHtml(node.status || 'pending')} ${node.id === selectedSkillProjectFlowNodeId ? 'active' : ''}"
                type="button" data-skill-project-flow-node="${escapeHtml(node.id)}" aria-pressed="${node.id === selectedSkillProjectFlowNodeId ? 'true' : 'false'}">
                <span class="skill-project-flow-node-top">
                    <span class="skill-project-flow-step">${escapeHtml(String(node.step || index + 1))}</span>
                    <span>${escapeHtml(skillProjectFlowStatusLabels[node.status] || node.status || '未开始')}</span>
                </span>
                <strong>${escapeHtml(node.title || node.id)}</strong>
                <small>${escapeHtml(node.subtitle || '')}</small>
            </button>
            ${index < flowNodes.length - 1 ? '<span class="skill-project-flow-arrow" aria-hidden="true">→</span>' : ''}
        `).join('');
        nodes.skillProjectFlow.querySelectorAll('[data-skill-project-flow-node]').forEach(button => {
            button.addEventListener('click', () => {
                selectedSkillProjectFlowNodeId = button.dataset.skillProjectFlowNode || '';
                renderSkillProjectWorkbench(snapshotVersion);
            });
        });
    }
    nodes.skillProjectSources.innerHTML = sources.length
        ? sources.map(source => `<div>
            <strong>${escapeHtml(source.label || source.filename || source.job_id || source.id)}</strong>
            <small>${escapeHtml(source.kind === 'video_analyzer_package'
                ? `${source.run_dir || ''} · 原始证据 ${Array.from(new Set((source.raw_evidence || []).map(item => item.type))).join(' / ') || '-'} · 辅助资料 ${(source.reference_documents || []).map(item => item.label).join(' / ') || '-'}`
                : source.kind || '')}</small>
            <button class="secondary tiny" type="button" data-remove-skill-source="${escapeHtml(source.id)}">移除</button>
        </div>`).join('')
        : '暂无资料';
    nodes.skillProjectSources.querySelectorAll('[data-remove-skill-source]').forEach(button => {
        button.addEventListener('click', () => {
            void removeSkillProjectSource(button.dataset.removeSkillSource || '');
        });
    });
    renderSkillProjectInspector(currentNode, snapshotVersion);
    renderSkillProjectPackageControls(project);
    renderSkillProjectLogControls();
    void loadSkillProjectLog(project.id);
    syncSkillProjectPolling();
}

function renderSkillProjectPackageControls(project) {
    const preview = skillProjectPackagePreview;
    const belongsToProject = preview?.project_id === project.id;
    nodes.previewSkillProjectPackageButton.disabled = Boolean(skillProjectPackageBusy);
    nodes.previewSkillProjectPackageButton.textContent = skillProjectPackageBusy === 'preview' ? '检查中...' : '检查';
    nodes.importSkillProjectPackageButton.disabled = (
        skillProjectPackageBusy !== ''
        || !belongsToProject
        || !preview?.can_import
    );
    nodes.importSkillProjectPackageButton.textContent = skillProjectPackageBusy === 'import'
        ? '正在导入...'
        : preview?.already_imported
            ? '资料包已导入'
            : '导入此资料包';
    if (!belongsToProject) return;
    const packageInfo = preview.package || {};
    nodes.skillProjectPackagePreview.innerHTML = `
        <strong>${escapeHtml(packageInfo.label || packageInfo.package_id || '')}</strong>
        <small>${escapeHtml(packageInfo.run_dir || '')}</small>
        <small>原始证据：${escapeHtml(Array.from(new Set((packageInfo.raw_evidence || []).map(item => item.type))).join(' / ') || '-')}</small>
        <small>辅助资料：${escapeHtml((packageInfo.reference_documents || []).map(item => item.label).join(' / ') || '-')}</small>
    `;
}

function renderSkillProjectInspector(node, snapshotVersion = '') {
    const project = currentSkillProject;
    const workspace = currentSkillProjectWorkspace || {};
    if (!project || !node) return;
    const assessment = project.assessment || {};
    const candidates = workspace.candidates || [];
    const generatedSkills = workspace.generated_skills || [];
    const progress = workspace.summary?.progress || {};
    const detailKey = JSON.stringify([
        snapshotVersion,
        node.id,
        project.revision,
        workspace.summary?.status,
        workspace.summary?.current_stage,
        progress.percent,
        node.action,
        node.secondary_action,
        skillProjectFlowActionBusy,
        candidates.map(item => [item.id, item.group, item.selected]),
        generatedSkills.map(item => [item.name, item.status, item.pass_rate])
    ]);
    if (detailKey === skillProjectDetailSignature) return;
    skillProjectDetailSignature = detailKey;
    nodes.skillProjectInspectorTitle.textContent = node.title || '流程详情';
    nodes.skillProjectInspectorMeta.textContent = skillProjectFlowStatusLabels[node.status] || node.status || '';
    const actionLabels = {
        package: '前往导入资料包',
        assess: '更新证据评估',
        start: assessment.verdict === 'ready_limited' ? '接受限制后蒸馏' : '开始蒸馏',
        resume: '继续未完成流程',
        cancel: '取消运行',
        'review-overview': '确认骨架',
        'review-candidates': '确认候选',
        enable: '启用 Skills'
    };
    const actionButtons = [
        node.action
            ? `<button type="button" data-skill-project-flow-action="${escapeHtml(node.action)}" ${skillProjectFlowActionBusy ? 'disabled' : ''}>${escapeHtml(actionLabels[node.action] || node.action)}</button>`
            : '',
        node.secondary_action
            ? `<button class="secondary" type="button" data-skill-project-flow-action="${escapeHtml(node.secondary_action)}" ${skillProjectFlowActionBusy ? 'disabled' : ''}>${escapeHtml(actionLabels[node.secondary_action] || node.secondary_action)}</button>`
            : ''
    ].filter(Boolean).join('');
    const action = actionButtons ? `<div class="skill-actions">${actionButtons}</div>` : '';
    if (node.id === 'goal') {
        nodes.skillProjectDetail.innerHTML = `
            <section>
                <h4>${escapeHtml(project.title || project.goal)}</h4>
                <p>${escapeHtml(project.brief?.goal || '')}</p>
                <p>新项目不会自动关联当前任务。导入资料包后才会进入证据评估。</p>
            </section>`;
    } else if (node.id === 'packages') {
        nodes.skillProjectDetail.innerHTML = `
            <section>
                <h4>资料包来源</h4>
                <p>先检查资料包，确认实际选中的运行目录和原始证据后再导入。</p>
                <p>当前已导入 ${(project.sources || []).filter(item => item.kind === 'video_analyzer_package').length} 个资料包。</p>
                ${action}
            </section>`;
    } else if (node.id === 'readiness') {
        const requests = assessment.material_requests || [];
        nodes.skillProjectDetail.innerHTML = `
            <section>
                <h4>${escapeHtml(assessment.summary || '尚未完成证据评估')}</h4>
                <p>原始证据 ${escapeHtml(String(assessment.source_coverage?.high_confidence_records || 0))} 条 · 独立案例 ${escapeHtml(String(assessment.source_coverage?.independent_learning_cases || 0))} 个</p>
                ${requests.length ? `<ul>${requests.map(item => `<li>${escapeHtml(item.reason || '')}；${escapeHtml(item.acceptance || '')}</li>`).join('')}</ul>` : '<p>当前没有额外资料缺口。</p>'}
                ${node.action === 'start' ? '<p>资料已满足条件，可以开始蒸馏。</p>' : ''}
                ${action}
            </section>`;
    } else if (node.id === 'overview') {
        nodes.skillProjectDetail.innerHTML = `
            <section>
                <h4>${escapeHtml(workspace.summary?.overview?.title || '方法骨架')}</h4>
                <p>${escapeHtml(workspace.summary?.overview?.summary || '骨架生成后会在这里等待人工确认。')}</p>
                ${action}
            </section>`;
    } else if (node.id === 'candidates') {
        nodes.skillProjectDetail.innerHTML = workspace.summary?.status === 'waiting_candidate_review'
            ? renderSkillProjectCandidateReview(project.id, candidates)
            : `<section><h4>候选 Review</h4><p>候选 ${candidates.length} 个。流程到达此处后可在这里选择可构建候选。</p>${action}</section>`;
    } else if (node.id === 'build') {
        const currentStage = workspace.summary?.current_stage || '-';
        const runner = workspace.runner || {};
        nodes.skillProjectDetail.innerHTML = `
            <section class="skill-project-run ${node.status === 'running' ? 'running' : ''}">
                <div class="skill-project-run-head">
                    ${node.status === 'running' ? '<span class="skill-project-spinner" aria-hidden="true"></span>' : ''}
                    <div>
                        <h4>${escapeHtml(skillStageNames[currentStage] || currentStage)}</h4>
                        <p>${runner.active ? '后台线程存活' : (runner.status || '等待下一次操作')}</p>
                    </div>
                    <strong>${escapeHtml(String(progress.percent || 0))}%</strong>
                </div>
                <div class="skill-project-progress"><span style="width:${escapeHtml(String(progress.percent || 0))}%"></span></div>
                <p>${escapeHtml(workspace.summary?.error || runner.error || '')}</p>
                ${action}
            </section>`;
    } else {
        const enabledSkills = generatedSkills.filter(item => item.status === 'passed');
        const failedSkills = generatedSkills.filter(item => item.status !== 'passed');
        nodes.skillProjectDetail.innerHTML = `
            <section>
                <h4>启用生成的 Skills</h4>
                <p>以下 ${enabledSkills.length} 项已通过压力测试并写入项目 Skills 目录。</p>
                ${enabledSkills.length ? `
                    <div class="skill-project-enabled-list">
                        ${enabledSkills.map(item => `
                            <article class="skill-project-enabled-item">
                                <strong>${escapeHtml(item.title || item.name)}</strong>
                                <span>${escapeHtml(item.name)} · 通过率 ${escapeHtml(formatPercent(item.pass_rate))}</span>
                            </article>
                        `).join('')}
                    </div>
                    <div class="skill-actions">
                        <button type="button" data-open-project-enabled-skills>打开这 ${enabledSkills.length} 项</button>
                    </div>
                ` : '<p>尚无通过压力测试的 Skill。</p>'}
                ${failedSkills.length ? `<p>另有 ${failedSkills.length} 项未通过压力测试，未启用。</p>` : ''}
                ${action}
            </section>`;
    }
    nodes.skillProjectDetail.querySelectorAll('[data-skill-project-flow-action]').forEach(button => {
        button.addEventListener('click', () => {
            void runSkillProjectFlowAction(button.dataset.skillProjectFlowAction || '');
        });
    });
    nodes.skillProjectDetail.querySelectorAll('[data-open-project-enabled-skills]').forEach(button => {
        button.addEventListener('click', () => {
            skillLibraryProjectSkillNames = new Set(
                (currentSkillProjectWorkspace?.generated_skills || [])
                    .filter(item => item.status === 'passed' && item.name)
                    .map(item => item.name)
            );
            nodes.skillLibrarySearch.value = '';
            setSkillsScope('enabled');
        });
    });
    nodes.skillProjectDetail.querySelectorAll('[data-skill-project-candidate]').forEach(input => {
        input.addEventListener('change', updateSkillProjectCandidateDraft);
    });
    nodes.skillProjectDetail.querySelectorAll('[data-skill-project-selection]').forEach(button => {
        button.addEventListener('click', () => {
            setSkillProjectCandidateSelection(button.dataset.skillProjectSelection || 'none');
        });
    });
    nodes.skillProjectDetail.querySelectorAll('[data-project-candidate-confirm]').forEach(button => {
        button.addEventListener('click', () => {
            void runSkillProjectAction('review-candidates');
        });
    });
    nodes.skillProjectDetail.querySelectorAll('[data-capability-check]').forEach(button => {
        button.addEventListener('click', () => {
            void runSkillProjectCapabilityCheck(button.dataset.capabilityCheck || '');
        });
    });
}

function chooseSkillProjectLogStage(stages, currentStage) {
    if (selectedSkillProjectLogStage && stages.includes(selectedSkillProjectLogStage)) {
        return selectedSkillProjectLogStage;
    }
    return currentStage && stages.includes(currentStage)
        ? currentStage
        : (stages[stages.length - 1] || '');
}

function skillProjectLogPath(stage) {
    return `skills/cangjie_pack/logs/${stage}.jsonl`;
}

function formatSkillProjectLog(text) {
    const lines = String(text || '').trim().split('\n').filter(Boolean);
    if (!lines.length) return '暂无日志输出';
    return lines.map(line => {
        try {
            const item = JSON.parse(line);
            const time = String(item.time || '').replace('T', ' ').replace(/\.\d+\+\d\d:\d\d$/, '');
            return `${time ? `[${time}] ` : ''}${item.event || 'log'}${item.message ? ` · ${item.message}` : ''}`;
        } catch {
            return line;
        }
    }).join('\n');
}

async function loadSkillProjectLog(projectId, requestedStage = '') {
    const logText = nodes.skillProjectLogText;
    const progress = currentSkillProjectWorkspace?.summary?.progress || {};
    const stage = requestedStage || chooseSkillProjectLogStage(
        Object.keys(progress.stages || {}),
        currentSkillProjectWorkspace?.summary?.current_stage || ''
    );
    if (!logText || !projectId || !stage) return;
    const requestId = ++skillProjectLogRequestId;
    try {
        const response = await fetch(
            `/api/skill-projects/${projectId}/resource?path=${encodeURIComponent(skillProjectLogPath(stage))}`,
            { cache: 'no-store' }
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const text = await response.text();
        if (requestId !== skillProjectLogRequestId || selectedSkillProjectId !== projectId) return;
        const formatted = formatSkillProjectLog(text);
        if (logText.textContent !== formatted) {
            const stickToBottom = logText.scrollHeight - logText.scrollTop - logText.clientHeight < 24;
            logText.textContent = formatted;
            if (stickToBottom) logText.scrollTop = logText.scrollHeight;
        }
    } catch (error) {
        if (requestId !== skillProjectLogRequestId || selectedSkillProjectId !== projectId) return;
        logText.textContent = `日志暂不可用：${error.message}`;
    }
}

async function copySkillProjectLog(projectId) {
    const progress = currentSkillProjectWorkspace?.summary?.progress || {};
    const stage = chooseSkillProjectLogStage(
        Object.keys(progress.stages || {}),
        currentSkillProjectWorkspace?.summary?.current_stage || ''
    );
    if (!projectId || !stage) return;
    const button = nodes.copySkillProjectLogButton;
    try {
        const response = await fetch(
            `/api/skill-projects/${projectId}/resource?path=${encodeURIComponent(skillProjectLogPath(stage))}`
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        await copyText(formatSkillProjectLog(await response.text()));
        if (button) button.textContent = '已复制';
    } catch (error) {
        if (button) button.textContent = '复制失败';
    }
}

function renderSkillProjectLogControls() {
    const workspace = currentSkillProjectWorkspace;
    if (!workspace || !selectedSkillProjectId) {
        nodes.skillProjectLogSummary.textContent = '选择项目后显示当前阶段记录。';
        nodes.skillProjectLogTabs.innerHTML = '';
        nodes.skillProjectLogText.textContent = '暂无运行记录';
        nodes.copySkillProjectLogButton.disabled = true;
        return;
    }
    const progress = workspace.summary?.progress || {};
    const stages = Object.keys(progress.stages || {});
    const stage = chooseSkillProjectLogStage(stages, workspace.summary?.current_stage || '');
    if (!stages.length) {
        const starting = skillProjectFlowActionBusy === 'start';
        nodes.skillProjectLogSummary.textContent = starting
            ? '正在启动蒸馏，等待后台建立首个阶段日志。'
            : '尚未开始蒸馏。点击“开始蒸馏”后，此处会显示实时阶段日志。';
        nodes.skillProjectLogTabs.dataset.tabsKey = `${selectedSkillProjectId}|`;
        nodes.skillProjectLogTabs.innerHTML = '';
        nodes.skillProjectLogText.textContent = starting ? '正在启动蒸馏...' : '尚未开始蒸馏。';
        nodes.copySkillProjectLogButton.disabled = true;
        return;
    }
    nodes.skillProjectLogSummary.textContent = stage
        ? `当前显示 ${skillStageNames[stage] || stage} 的日志尾部`
        : '流程尚未写入阶段日志。';
    nodes.copySkillProjectLogButton.disabled = !stage;
    const tabsKey = `${selectedSkillProjectId}|${stage}|${stages.join('|')}`;
    if (nodes.skillProjectLogTabs.dataset.tabsKey === tabsKey) return;
    nodes.skillProjectLogTabs.dataset.tabsKey = tabsKey;
    nodes.skillProjectLogTabs.innerHTML = stages.map(name => `
        <button class="secondary tiny ${name === stage ? 'active' : ''}" type="button"
            data-skill-project-log-stage="${escapeHtml(name)}">${escapeHtml(skillStageNames[name] || name)}</button>
    `).join('');
    nodes.skillProjectLogTabs.querySelectorAll('[data-skill-project-log-stage]').forEach(button => {
        button.addEventListener('click', () => {
            selectedSkillProjectLogStage = button.dataset.skillProjectLogStage || '';
            renderSkillProjectLogControls();
            void loadSkillProjectLog(selectedSkillProjectId, selectedSkillProjectLogStage);
        });
    });
}

function skillProjectCandidateDraft(projectId, candidates) {
    const signature = candidates
        .map(item => `${item.group}:${item.id}:${item.eligible ? '1' : '0'}:${item.selected ? '1' : '0'}`)
        .sort()
        .join('|');
    let draft = skillProjectCandidateDrafts.get(projectId);
    if (!draft || draft.signature !== signature) {
        draft = {
            signature,
            selected: new Set(
                candidates
                    .filter(item => item.eligible && item.selected)
                    .map(item => item.id)
            )
        };
        skillProjectCandidateDrafts.set(projectId, draft);
    }
    return draft;
}

function renderSkillProjectCandidateReview(projectId, candidates) {
    const draft = skillProjectCandidateDraft(projectId, candidates);
    const groups = [
        {
            id: 'accepted',
            title: '已验证候选',
            description: '已通过完整验证，可直接选择。',
            selectable: true,
            collapsible: false
        },
        {
            id: 'single_case',
            title: '可构建候选',
            description: '来自单一案例，需要你确认是否值得先构建。',
            selectable: true,
            collapsible: false
        },
        {
            id: 'glossary',
            title: '术语参考',
            description: '用于理解，不会生成 Skill。',
            selectable: false,
            collapsible: true
        },
        {
            id: 'rejected',
            title: '不建议构建',
            description: '保留拒绝原因，不能提交。',
            selectable: false,
            collapsible: true
        }
    ];
    const eligibleCount = candidates.filter(item => item.eligible).length;
    const selectedCount = draft.selected.size;
    const renderRow = (item, selectable) => {
        const reason = item.reason || item.v1?.reason || '';
        const evidence = item.source_count
            ? `${item.source_count} 条证据`
            : '无证据引用';
        const check = selectable
            ? `<input type="checkbox" data-skill-project-candidate="${escapeHtml(item.id)}" ${draft.selected.has(item.id) ? 'checked' : ''}>`
            : '<span class="skill-project-candidate-marker" aria-hidden="true"></span>';
        return `<label class="skill-project-candidate ${selectable ? 'selectable' : 'reference'}">
            ${check}
            <span>
                <strong>${escapeHtml(item.title || item.id)}</strong>
                <small>${escapeHtml(item.summary || '')}</small>
                <small>${escapeHtml(evidence)}${reason ? ` · ${escapeHtml(reason)}` : ''}</small>
            </span>
        </label>`;
    };
    const groupsHtml = groups.map(group => {
        const items = candidates.filter(item => item.group === group.id);
        const body = items.length
            ? items.map(item => renderRow(item, group.selectable && item.eligible)).join('')
            : '<div class="skills-empty">无</div>';
        const content = `<div class="skill-project-candidate-group-body">${body}</div>`;
        const heading = `<div class="skill-project-candidate-group-head">
            <div><h4>${escapeHtml(group.title)} · ${items.length}</h4><p>${escapeHtml(group.description)}</p></div>
        </div>`;
        return group.collapsible
            ? `<details class="skill-project-candidate-group"><summary>${heading}</summary>${content}</details>`
            : `<section class="skill-project-candidate-group">${heading}${content}</section>`;
    }).join('');
    return `<section class="skill-project-candidate-review">
        <div class="skill-project-candidate-review-head">
            <div><h4>候选 Review · ${candidates.length}</h4><p>只可选择已验证或单案例候选。</p></div>
            <div class="skill-actions">
                <button class="secondary tiny" type="button" data-skill-project-selection="all" ${eligibleCount ? '' : 'disabled'}>全选可构建</button>
                <button class="secondary tiny" type="button" data-skill-project-selection="none" ${selectedCount ? '' : 'disabled'}>清空</button>
                <button type="button" data-project-candidate-confirm ${selectedCount ? '' : 'disabled'}>确认 ${selectedCount} 项</button>
            </div>
        </div>
        ${groupsHtml}
    </section>`;
}

function updateSkillProjectCandidateDraft(event) {
    const candidateId = event.target.dataset.skillProjectCandidate;
    const draft = skillProjectCandidateDrafts.get(selectedSkillProjectId);
    if (!candidateId || !draft) return;
    if (event.target.checked) {
        draft.selected.add(candidateId);
    } else {
        draft.selected.delete(candidateId);
    }
    skillProjectDetailSignature = '';
    renderSkillProjectInspector(
        (currentSkillProjectFlow?.nodes || []).find(node => node.id === 'candidates'),
        ''
    );
}

function setSkillProjectCandidateSelection(action) {
    const candidates = currentSkillProjectWorkspace?.candidates || [];
    const draft = skillProjectCandidateDraft(selectedSkillProjectId, candidates);
    if (action === 'all') {
        draft.selected = new Set(candidates.filter(item => item.eligible).map(item => item.id));
    } else {
        draft.selected.clear();
    }
    skillProjectDetailSignature = '';
    renderSkillProjectInspector(
        (currentSkillProjectFlow?.nodes || []).find(node => node.id === 'candidates'),
        ''
    );
}

function syncSkillProjectPolling() {
    const status = currentSkillProjectWorkspace?.summary?.status;
    const shouldPoll = currentSkillsScope === 'projects'
        && Boolean(selectedSkillProjectId)
        && status === 'running';
    if (!shouldPoll) {
        if (skillProjectPollTimer) {
            window.clearInterval(skillProjectPollTimer);
            skillProjectPollTimer = null;
        }
        return;
    }
    if (skillProjectPollTimer) return;
    skillProjectPollTimer = window.setInterval(async () => {
        if (
            skillProjectPollInFlight
            || document.hidden
            || currentSkillsScope !== 'projects'
            || !selectedSkillProjectId
        ) return;
        skillProjectPollInFlight = true;
        try {
            await loadSkillProjects(selectedSkillProjectId);
        } finally {
            skillProjectPollInFlight = false;
        }
    }, 2000);
}

async function createSkillProject(event) {
    event.preventDefault();
    const goal = nodes.skillProjectGoal.value.trim();
    if (!goal) return;
    const project = await getJson('/api/skill-projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            title: nodes.skillProjectTitle.value.trim(),
            goal
        })
    });
    nodes.skillProjectTitle.value = '';
    nodes.skillProjectGoal.value = '';
    selectedSkillProjectId = project.id;
    await loadSkillProjects(project.id);
}

async function assessSkillProject() {
    if (!selectedSkillProjectId) return;
    await getJson(`/api/skill-projects/${selectedSkillProjectId}/assess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}'
    });
    await loadSkillProjects(selectedSkillProjectId);
}

async function previewSkillProjectPackage() {
    if (!selectedSkillProjectId) return;
    const packageId = nodes.skillProjectPackageId.value.trim();
    if (!packageId) return;
    skillProjectPackageBusy = 'preview';
    nodes.skillProjectPackageStatus.textContent = '正在检查资料包与原始证据...';
    renderSkillProjectPackageControls(currentSkillProject);
    try {
        const preview = await getJson(
            `/api/skill-projects/${selectedSkillProjectId}/packages/preview?package_id=${encodeURIComponent(packageId)}`
        );
        if (selectedSkillProjectId !== currentSkillProject?.id) return;
        skillProjectPackagePreview = { ...preview, project_id: selectedSkillProjectId };
        nodes.skillProjectPackageStatus.textContent = preview.already_imported
            ? '该资料包已导入，可以继续使用。'
            : '检查完成，确认后导入。';
    } catch (error) {
        skillProjectPackagePreview = null;
        nodes.skillProjectPackagePreview.textContent = '未找到可导入的完整资料包。';
        nodes.skillProjectPackageStatus.textContent = `检查失败：${error.message}`;
    } finally {
        skillProjectPackageBusy = '';
        renderSkillProjectPackageControls(currentSkillProject);
    }
}

async function importSkillProjectPackage() {
    if (!selectedSkillProjectId || !skillProjectPackagePreview?.can_import) return;
    const packageId = nodes.skillProjectPackageId.value.trim();
    if (!packageId) return;
    skillProjectPackageBusy = 'import';
    nodes.skillProjectPackageStatus.textContent = '正在冻结资料包参考信息并更新证据评估...';
    renderSkillProjectPackageControls(currentSkillProject);
    try {
        const result = await getJson(`/api/skill-projects/${selectedSkillProjectId}/packages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ package_id: packageId })
        });
        skillProjectPackagePreview = {
            package: result.source,
            project_id: selectedSkillProjectId,
            already_imported: !result.created,
            can_import: false
        };
        nodes.skillProjectPackageStatus.textContent = result.created
            ? '资料包已导入，证据评估已更新。'
            : '资料包已经存在，已保留原来的证据来源。';
        nodes.skillProjectPackageId.value = '';
        applySkillProjectWorkbench(result.workbench);
    } catch (error) {
        nodes.skillProjectPackageStatus.textContent = `导入失败：${error.message}`;
    } finally {
        skillProjectPackageBusy = '';
        renderSkillProjectPackageControls(currentSkillProject);
    }
}

async function removeSkillProjectSource(sourceId) {
    if (!selectedSkillProjectId || !sourceId) return;
    await getJson(`/api/skill-projects/${selectedSkillProjectId}/sources/${encodeURIComponent(sourceId)}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: '{}'
    });
    await loadSkillProjects(selectedSkillProjectId);
}

async function startSkillProjectDistillation() {
    if (!selectedSkillProjectId || !currentSkillProject) return;
    const limited = currentSkillProject.assessment?.verdict === 'ready_limited';
    if (limited && !window.confirm('该项目存在已显示的资料或能力限制。确认接受限制并开始蒸馏？')) return;
    skillProjectFlowActionBusy = 'start';
    skillProjectDetailSignature = '';
    renderSkillProjectInspector(
        (currentSkillProjectFlow?.nodes || []).find(node => node.id === selectedSkillProjectFlowNodeId),
        ''
    );
    renderSkillProjectLogControls();
    try {
        await getJson(`/api/skill-projects/${selectedSkillProjectId}/distillation/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile: 'deepseek_v4_pro', accept_limitations: limited })
        });
        await loadSkillProjects(selectedSkillProjectId);
    } finally {
        skillProjectFlowActionBusy = '';
        skillProjectDetailSignature = '';
        renderSkillProjectWorkbench('');
    }
}

async function runSkillProjectFlowAction(action) {
    try {
        if (action === 'package') {
            nodes.skillProjectPackageId.focus();
            return;
        }
        if (action === 'assess') {
            await assessSkillProject();
            return;
        }
        if (action === 'start') {
            await startSkillProjectDistillation();
            return;
        }
        await runSkillProjectAction(action);
    } catch (error) {
        nodes.skillProjectInspectorMeta.textContent = `操作失败：${error.message}`;
        nodes.skillProjectLogSummary.textContent = `操作失败：${error.message}`;
    }
}

async function runSkillProjectAction(action) {
    if (!selectedSkillProjectId) return;
    const base = `/api/skill-projects/${selectedSkillProjectId}/distillation`;
    if (action === 'enable' && !window.confirm('启用已通过测试的 Skills 并写入项目 .codex/skills？')) return;
    const payload = action === 'review-overview'
        ? { action: 'confirm' }
        : action === 'review-candidates'
            ? {
                selected_ids: Array.from(
                    skillProjectCandidateDraft(
                        selectedSkillProjectId,
                        currentSkillProjectWorkspace?.candidates || []
                    ).selected
                )
            }
            : action === 'enable'
                ? { overwrite: false }
                : {};
    const endpoint = action === 'review-overview'
        ? 'review-overview'
        : action === 'review-candidates'
            ? 'review-candidates'
            : action;
    await getJson(`${base}/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    await loadSkillProjects(selectedSkillProjectId);
}

async function runSkillProjectCapabilityCheck(checkId) {
    if (!selectedSkillProjectId || !currentSkillProject) return;
    if (!window.confirm('将执行已列出的本地能力 smoke test。确认继续？')) return;
    await getJson(
        `/api/skill-projects/${selectedSkillProjectId}/capability-checks/${encodeURIComponent(checkId)}/run`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                confirm: true,
                assessment_revision: currentSkillProject.assessment?.project_revision || ''
            })
        }
    );
    await loadSkillProjects(selectedSkillProjectId);
}

async function loadCurrentSkillsWorkspace(jobId, workspaceKey) {
    loadedSkillsWorkspaceKey = workspaceKey;
    nodes.skillsCurrentList.innerHTML = '<div class="skills-empty">正在加载候选与成品...</div>';
    try {
        const workspace = await getJson(`/api/video-link/jobs/${jobId}/skill-distillation/workspace`);
        if (selectedJobId !== jobId || currentSkillsScope !== 'current') return;
        currentSkillWorkspace = workspace;
        renderCurrentSkillsList();
    } catch (error) {
        if (loadedSkillsWorkspaceKey === workspaceKey) loadedSkillsWorkspaceKey = '';
        nodes.skillsCurrentList.innerHTML = `<div class="skills-empty error-text">加载失败：${escapeHtml(error.message)}</div>`;
    }
}

function renderCurrentSkillsList() {
    const generated = currentSkillWorkspace?.generated_skills || [];
    const candidates = currentSkillWorkspace?.candidates || [];
    const items = [...generated, ...candidates];
    nodes.skillsCurrentCount.textContent = `${items.length} 项`;
    if (!items.length) {
        nodes.skillsCurrentList.innerHTML = '<div class="skills-empty">蒸馏尚未产生候选或成品</div>';
        resetCurrentSkillDetail();
        return;
    }
    if (!selectedCurrentSkillItemId || !items.some(item => item.item_id === selectedCurrentSkillItemId)) {
        selectedCurrentSkillItemId = generated[0]?.item_id || candidates[0]?.item_id || '';
        currentSkillItemDetail = null;
    }
    const groupLabels = {
        accepted: '已验证',
        single_case: '单案例',
        rejected: '已拒绝',
        glossary: '术语'
    };
    nodes.skillsCurrentList.innerHTML = [
        generated.length ? `<section class="skills-item-group">
            <h4>生成的 Skills</h4>
            ${generated.map(item => `<button class="skills-item${item.item_id === selectedCurrentSkillItemId ? ' active' : ''}" type="button" data-current-skill-item="${escapeHtml(item.item_id)}">
                <strong>${escapeHtml(item.title || item.name)}</strong>
                <span>${escapeHtml(item.name)} · 通过率 ${formatPercent(item.pass_rate)}</span>
            </button>`).join('')}
        </section>` : '',
        candidates.length ? `<section class="skills-item-group">
            <h4>候选</h4>
            ${candidates.map(item => `<button class="skills-item ${escapeHtml(item.group || '')}${item.item_id === selectedCurrentSkillItemId ? ' active' : ''}" type="button" data-current-skill-item="${escapeHtml(item.item_id)}">
                <strong>${escapeHtml(item.title || item.id)}</strong>
                <span>${escapeHtml(groupLabels[item.group] || item.group)} · ${item.source_count || 0} 条证据</span>
            </button>`).join('')}
        </section>` : ''
    ].join('');
    nodes.skillsCurrentList.querySelectorAll('[data-current-skill-item]').forEach(button => {
        button.addEventListener('click', () => selectCurrentSkillItem(button.dataset.currentSkillItem || ''));
    });
    if (!currentSkillItemDetail || currentSkillItemDetail.item_id !== selectedCurrentSkillItemId) {
        selectCurrentSkillItem(selectedCurrentSkillItemId);
    }
}

function formatPercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '-';
    return `${Math.round(number * 100)}%`;
}

async function selectCurrentSkillItem(itemId) {
    if (!selectedJobId || !itemId) return;
    selectedCurrentSkillItemId = itemId;
    nodes.skillsCurrentList.querySelectorAll('[data-current-skill-item]').forEach(button => {
        button.classList.toggle('active', button.dataset.currentSkillItem === itemId);
    });
    nodes.skillsDetailTitle.textContent = '加载中...';
    nodes.skillsDetailPreview.textContent = '';
    nodes.skillsInspectorBody.textContent = '正在读取详情...';
    try {
        const detail = await getJson(
            `/api/video-link/jobs/${selectedJobId}/skill-distillation/items/${encodeURIComponent(itemId)}`
        );
        if (selectedCurrentSkillItemId !== itemId) return;
        currentSkillItemDetail = detail;
        renderCurrentSkillDetail();
    } catch (error) {
        nodes.skillsDetailTitle.textContent = '加载失败';
        nodes.skillsInspectorBody.textContent = error.message;
    }
}

function resetCurrentSkillDetail() {
    selectedCurrentSkillItemId = '';
    currentSkillItemDetail = null;
    nodes.skillsDetailTitle.textContent = '选择一个候选或 Skill';
    nodes.skillsDetailMeta.textContent = '证据、审计和测试结果会显示在这里';
    nodes.skillsDetailPreview.textContent = '暂无预览';
    nodes.skillsInspectorBody.textContent = '选择左侧项目查看详情';
}

function renderCurrentSkillDetail() {
    const detail = currentSkillItemDetail;
    if (!detail) {
        resetCurrentSkillDetail();
        return;
    }
    if (detail.kind === 'skill') {
        nodes.skillsDetailTitle.textContent = detail.skill?.title || detail.name;
        nodes.skillsDetailMeta.textContent = `${detail.name} · ${detail.skill?.status || '-'} · 通过率 ${formatPercent(detail.skill?.pass_rate)}`;
        nodes.skillsDetailPreview.innerHTML = renderMarkdown(stripSkillFrontmatter(detail.markdown || ''), selectedJobId);
        renderMarkdownMath(nodes.skillsDetailPreview);
    } else {
        const candidate = detail.candidate || {};
        nodes.skillsDetailTitle.textContent = candidate.title || candidate.id || '候选';
        nodes.skillsDetailMeta.textContent = `${detail.group || '-'} · ${(candidate.source_ids || []).length} 条证据`;
        nodes.skillsDetailPreview.innerHTML = renderCandidatePreview(candidate);
    }
    renderCurrentSkillInspector();
}

function renderCandidatePreview(candidate) {
    const sections = [
        candidate.summary ? `<h3>摘要</h3><p>${escapeHtml(candidate.summary)}</p>` : '',
        candidate.reason ? `<h3>判定</h3><p>${escapeHtml(candidate.reason)}</p>` : '',
        candidate.source_quote ? `<h3>原始摘录</h3><blockquote>${escapeHtml(candidate.source_quote)}</blockquote>` : '',
        (candidate.boundaries || []).length
            ? `<h3>边界</h3><ul>${candidate.boundaries.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`
            : '',
        candidate.execution_hint ? `<h3>执行提示</h3><p>${escapeHtml(candidate.execution_hint)}</p>` : ''
    ];
    return `<article class="candidate-preview">${sections.join('')}</article>`;
}

function setCurrentSkillDetailTab(tab) {
    currentSkillDetailTab = ['audit', 'tests', 'raw'].includes(tab) ? tab : 'evidence';
    nodes.skillsDetailTabs.forEach(button => {
        button.classList.toggle('active', button.dataset.skillDetailTab === currentSkillDetailTab);
    });
    renderCurrentSkillInspector();
}

function renderCurrentSkillInspector() {
    const detail = currentSkillItemDetail;
    if (!detail) return;
    if (currentSkillDetailTab === 'raw') {
        nodes.skillsInspectorBody.innerHTML = `<pre>${escapeHtml(JSON.stringify(detail, null, 2))}</pre>`;
        return;
    }
    if (detail.kind === 'skill') {
        const files = detail.files || {};
        if (currentSkillDetailTab === 'tests') {
            const prompts = files['test-prompts.json'] || {};
            const results = files['test-results.json'] || {};
            nodes.skillsInspectorBody.innerHTML = renderSkillTests(prompts, results);
            return;
        }
        if (currentSkillDetailTab === 'audit') {
            nodes.skillsInspectorBody.innerHTML = '<div class="skills-empty">最终 Skill 的多模态审计请从对应候选查看。</div>';
            return;
        }
        nodes.skillsInspectorBody.innerHTML = files['skill.json']
            ? `<pre>${escapeHtml(JSON.stringify(files['skill.json'], null, 2))}</pre>`
            : '<div class="skills-empty">暂无结构化 Skill 元数据</div>';
        return;
    }
    if (currentSkillDetailTab === 'audit') {
        nodes.skillsInspectorBody.innerHTML = renderMultimodalAudit(detail);
        return;
    }
    if (currentSkillDetailTab === 'tests') {
        nodes.skillsInspectorBody.innerHTML = '<div class="skills-empty">候选通过构建后才会生成压力测试。</div>';
        return;
    }
    nodes.skillsInspectorBody.innerHTML = renderCandidateEvidence(detail);
}

function renderCandidateEvidence(detail) {
    const records = detail.evidence || [];
    const frames = detail.frames || [];
    const frameHtml = frames.length
        ? `<div class="skill-frame-grid">${frames.map(frame => `<button type="button" class="skill-frame-button" data-image-viewer-src="${escapeHtml(frame.url)}" data-image-viewer-alt="${escapeHtml(frame.path)}">
            <img src="${escapeHtml(frame.url)}" alt="${escapeHtml(frame.path)}" loading="lazy">
            <span>${escapeHtml(frame.path)}</span>
        </button>`).join('')}</div>`
        : '';
    const evidenceHtml = records.length
        ? records.map(record => `<article class="skill-evidence-record">
            <strong>${escapeHtml(record.id || '-')}</strong>
            <span>${escapeHtml(record.kind || record.source_type || '-')} · ${escapeHtml(record.timestamp_label || record.timestamp || '')}</span>
            <p>${escapeHtml(record.text || record.content || record.quote || '')}</p>
        </article>`).join('')
        : '<div class="skills-empty">未找到对应证据记录</div>';
    return `${frameHtml}${evidenceHtml}`;
}

function renderMultimodalAudit(detail) {
    const audit = detail.multimodal_audit || {};
    if (!Object.keys(audit).length) return '<div class="skills-empty">暂无多模态审计</div>';
    const unsupported = audit.unsupported_details || [];
    return `<div class="skill-audit-summary">
        <div><span>声明支持</span><strong>${audit.claim_supported ? '是' : '否'}</strong></div>
        <div><span>执行支持</span><strong>${audit.execution_supported ? '是' : '否'}</strong></div>
        <div><span>教学价值</span><strong>${escapeHtml(audit.instructional_value || '-')}</strong></div>
        <div><span>置信度</span><strong>${escapeHtml(audit.confidence ?? '-')}</strong></div>
    </div>
    <h4>口播支持</h4><p>${escapeHtml(audit.transcript_support || '无')}</p>
    <h4>视觉支持</h4><p>${escapeHtml(audit.visual_support || '无')}</p>
    <h4>未证实细节</h4>${unsupported.length
        ? `<ul>${unsupported.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`
        : '<p>无</p>'}`;
}

function renderSkillTests(prompts, results) {
    const cases = prompts.test_cases || [];
    const resultMap = new Map((results.results || []).map(item => [item.id, item]));
    if (!cases.length) return '<div class="skills-empty">暂无压力测试</div>';
    return `<div class="skill-test-summary">
        <strong>${results.passed ? '通过' : '未通过'}</strong>
        <span>${results.passed_count || 0}/${results.total || cases.length} · ${formatPercent(results.pass_rate)}</span>
    </div>${cases.map(testCase => {
        const result = resultMap.get(testCase.id) || {};
        return `<article class="skill-test-case ${result.passed ? 'passed' : 'failed'}">
            <strong>${escapeHtml(testCase.id)} · ${escapeHtml(testCase.type)}</strong>
            <p>${escapeHtml(testCase.prompt || '')}</p>
            <span>期望 ${escapeHtml(testCase.expected_skill || '不触发')} · 实际 ${escapeHtml(result.selected_skill || '不触发')}</span>
        </article>`;
    }).join('')}`;
}

async function loadSkillLibrary() {
    if (currentResourceView !== 'skills' || currentSkillsScope === 'current') return;
    const query = nodes.skillLibrarySearch.value.trim();
    const labels = { enabled: '已启用 Skills', disabled: '已禁用 Skills', trash: '回收站' };
    const projectNames = skillLibraryProjectSkillNames ? new Set(skillLibraryProjectSkillNames) : null;
    nodes.skillLibraryTitle.textContent = projectNames
        ? '本项目已启用 Skills'
        : labels[currentSkillsScope] || '项目 Skills';
    nodes.skillLibraryList.innerHTML = '<div class="skills-empty">正在加载...</div>';
    try {
        const data = await getJson(`/api/skills?state=${encodeURIComponent(currentSkillsScope)}&query=${encodeURIComponent(query)}`);
        if (data.state !== currentSkillsScope) return;
        const items = projectNames
            ? (data.items || []).filter(item => projectNames.has(item.name))
            : data.items || [];
        nodes.skillLibraryCount.textContent = `${items.length} 项`;
        renderSkillLibraryList(items);
    } catch (error) {
        nodes.skillLibraryList.innerHTML = `<div class="skills-empty error-text">加载失败：${escapeHtml(error.message)}</div>`;
    }
}

function renderSkillLibraryList(items) {
    if (!items.length) {
        nodes.skillLibraryList.innerHTML = '<div class="skills-empty">这里还没有 Skill</div>';
        resetSkillEditor();
        return;
    }
    nodes.skillLibraryList.innerHTML = items.map(item => `<button class="skills-item${item.id === selectedLibrarySkillId ? ' active' : ''}" type="button" data-library-skill="${escapeHtml(item.id)}">
        <strong>${escapeHtml(item.title || item.name)}</strong>
        <span>${escapeHtml(item.name)} · ${escapeHtml(item.updated_at || '')}</span>
        <small>${escapeHtml(item.description || '')}</small>
    </button>`).join('');
    nodes.skillLibraryList.querySelectorAll('[data-library-skill]').forEach(button => {
        button.addEventListener('click', () => loadLibrarySkill(button.dataset.librarySkill || ''));
    });
    if (!selectedLibrarySkillId || !items.some(item => item.id === selectedLibrarySkillId)) {
        loadLibrarySkill(items[0].id);
    }
}

async function loadLibrarySkill(skillId) {
    if (!skillId || currentSkillsScope === 'current') return;
    selectedLibrarySkillId = skillId;
    nodes.skillEditorTitle.textContent = '加载中...';
    try {
        const detail = await getJson(`/api/skills/${encodeURIComponent(currentSkillsScope)}/${encodeURIComponent(skillId)}`);
        if (selectedLibrarySkillId !== skillId) return;
        currentLibrarySkill = detail;
        renderLibrarySkillDetail();
        loadSkillLibrary();
    } catch (error) {
        nodes.skillEditorTitle.textContent = '加载失败';
        nodes.skillEditorMessage.textContent = error.message;
    }
}

function resetSkillEditor() {
    currentLibrarySkill = null;
    selectedLibrarySkillId = '';
    nodes.skillEditorTitle.textContent = '选择一个 Skill';
    nodes.skillEditorMeta.textContent = '仅 SKILL.md 可编辑';
    nodes.skillEditor.value = '';
    nodes.skillEditor.disabled = true;
    nodes.skillRenderedPreview.innerHTML = '暂无预览';
    nodes.saveSkillButton.disabled = true;
    nodes.disableSkillButton.hidden = true;
    nodes.restoreSkillButton.hidden = true;
    nodes.deleteSkillButton.hidden = true;
    nodes.permanentDeleteSkillButton.hidden = true;
    nodes.skillAuxiliaryFiles.textContent = '暂无文件';
    nodes.skillVersionList.textContent = '暂无版本';
    nodes.skillEditorMessage.textContent = '';
}

function renderLibrarySkillDetail() {
    const skill = currentLibrarySkill;
    if (!skill) {
        resetSkillEditor();
        return;
    }
    nodes.skillEditorTitle.textContent = skill.title || skill.name;
    nodes.skillEditorMeta.textContent = `${skill.name} · ${skill.state} · ${skill.revision.slice(0, 10)}`;
    nodes.skillEditor.value = skill.markdown || '';
    nodes.skillEditor.disabled = !['enabled', 'disabled'].includes(skill.state);
    nodes.saveSkillButton.disabled = nodes.skillEditor.disabled;
    nodes.disableSkillButton.hidden = skill.state !== 'enabled';
    nodes.restoreSkillButton.hidden = !['disabled', 'trash'].includes(skill.state);
    nodes.deleteSkillButton.hidden = !['enabled', 'disabled'].includes(skill.state);
    nodes.permanentDeleteSkillButton.hidden = skill.state !== 'trash';
    nodes.skillRenderedPreview.innerHTML = renderMarkdown(stripSkillFrontmatter(skill.markdown || ''));
    renderMarkdownMath(nodes.skillRenderedPreview);
    nodes.skillAuxiliaryFiles.innerHTML = (skill.auxiliary_files || []).length
        ? skill.auxiliary_files.map(file => `<details class="skill-auxiliary-file">
            <summary>${escapeHtml(file.path)} · ${formatBytes(file.size_bytes)}</summary>
            ${file.content == null ? '<p>二进制或文件过大，界面不展开。</p>' : `<pre>${escapeHtml(file.content)}</pre>`}
        </details>`).join('')
        : '<div class="skills-empty">无辅助文件</div>';
    nodes.skillVersionList.innerHTML = (skill.versions || []).length
        ? skill.versions.map(version => `<div class="skill-version-item">
            <span>${escapeHtml(version.id)}</span>
            <button class="secondary tiny" type="button" data-restore-version="${escapeHtml(version.id)}">恢复</button>
        </div>`).join('')
        : '<div class="skills-empty">保存后会自动生成版本快照</div>';
    nodes.skillVersionList.querySelectorAll('[data-restore-version]').forEach(button => {
        button.addEventListener('click', () => restoreSkillVersion(button.dataset.restoreVersion || ''));
    });
    nodes.skillEditorMessage.textContent = '';
    setSkillEditorTab(currentSkillEditorTab);
}

function setSkillEditorTab(tab) {
    currentSkillEditorTab = tab === 'preview' ? 'preview' : 'edit';
    nodes.skillEditorTabs.forEach(button => {
        button.classList.toggle('active', button.dataset.skillEditorTab === currentSkillEditorTab);
    });
    nodes.skillEditor.hidden = currentSkillEditorTab !== 'edit';
    nodes.skillRenderedPreview.hidden = currentSkillEditorTab !== 'preview';
    if (currentSkillEditorTab === 'preview') {
        nodes.skillRenderedPreview.innerHTML = renderMarkdown(stripSkillFrontmatter(nodes.skillEditor.value || ''));
        renderMarkdownMath(nodes.skillRenderedPreview);
    }
}

async function saveCurrentLibrarySkill() {
    if (!currentLibrarySkill || nodes.saveSkillButton.disabled) return;
    nodes.saveSkillButton.disabled = true;
    nodes.skillEditorMessage.textContent = '正在保存...';
    try {
        currentLibrarySkill = await getJson(
            `/api/skills/${encodeURIComponent(currentLibrarySkill.state)}/${encodeURIComponent(currentLibrarySkill.id)}`,
            {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    markdown: nodes.skillEditor.value,
                    revision: currentLibrarySkill.revision
                })
            }
        );
        renderLibrarySkillDetail();
        nodes.skillEditorMessage.textContent = '已保存并创建版本快照';
        loadSkillLibrary();
    } catch (error) {
        nodes.skillEditorMessage.textContent = `保存失败：${error.message}`;
        nodes.saveSkillButton.disabled = false;
    }
}

async function changeLibrarySkillState(action) {
    const skill = currentLibrarySkill;
    if (!skill) return;
    let url = '';
    let method = 'POST';
    let body = null;
    if (action === 'disable') {
        if (!window.confirm(`禁用 ${skill.name}？`)) return;
        url = `/api/skills/enabled/${encodeURIComponent(skill.name)}/disable`;
    } else if (action === 'restore') {
        url = skill.state === 'trash'
            ? `/api/skills/trash/${encodeURIComponent(skill.id)}/restore`
            : `/api/skills/disabled/${encodeURIComponent(skill.name)}/restore`;
    } else if (action === 'delete') {
        if (!window.confirm(`将 ${skill.name} 移入回收站？`)) return;
        url = `/api/skills/${encodeURIComponent(skill.state)}/${encodeURIComponent(skill.id)}`;
        method = 'DELETE';
        body = {};
    } else if (action === 'permanent-delete') {
        const confirmation = window.prompt(`永久删除无法恢复。请输入 Skill 名称：${skill.name}`);
        if (confirmation !== skill.name) return;
        url = `/api/skills/trash/${encodeURIComponent(skill.id)}`;
        method = 'DELETE';
        body = { confirmation };
    }
    try {
        await getJson(url, {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: body == null ? undefined : JSON.stringify(body)
        });
        resetSkillEditor();
        await loadSkillLibrary();
    } catch (error) {
        nodes.skillEditorMessage.textContent = `操作失败：${error.message}`;
    }
}

async function restoreSkillVersion(versionId) {
    if (!currentLibrarySkill || !versionId) return;
    if (!window.confirm(`恢复版本 ${versionId}？当前内容会先创建快照。`)) return;
    try {
        currentLibrarySkill = await getJson(
            `/api/skills/${encodeURIComponent(currentLibrarySkill.state)}/${encodeURIComponent(currentLibrarySkill.id)}/versions/${encodeURIComponent(versionId)}/restore`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ revision: currentLibrarySkill.revision })
            }
        );
        renderLibrarySkillDetail();
        nodes.skillEditorMessage.textContent = `已恢复版本 ${versionId}`;
    } catch (error) {
        nodes.skillEditorMessage.textContent = `恢复失败：${error.message}`;
    }
}

function appendQaMessage(role, html) {
    const empty = nodes.qaMessages.querySelector('.qa-empty');
    if (empty) empty.remove();
    const message = document.createElement('article');
    message.className = `qa-message ${role}`;
    message.innerHTML = html;
    nodes.qaMessages.appendChild(message);
    nodes.qaMessages.scrollTop = nodes.qaMessages.scrollHeight;
}

async function loadQaHistory(jobId) {
    if (!nodes.qaMessages || loadedQaHistoryKey === jobId) return;
    loadedQaHistoryKey = jobId;
    try {
        const history = await getJson(`/api/video-link/jobs/${jobId}/qa/history?limit=50`);
        if (currentJob?.job_id !== jobId) return;
        renderQaHistory(history.messages || []);
    } catch (error) {
        if (currentJob?.job_id !== jobId) return;
        clearQaMessages();
    }
}

function renderQaHistory(records) {
    clearQaMessages();
    if (!records.length) return;
    for (const record of records) {
        appendQaMessage('user', `<p>${escapeHtml(record.question || '')}</p>`);
        appendQaMessage('assistant', renderQaAnswer(record));
    }
}

function renderQaAnswer(result) {
    const warningHtml = (result.warnings || []).length
        ? `<div class="qa-answer-warnings">${(result.warnings || []).map(item => escapeHtml(item.message || item.code || item)).join('<br>')}</div>`
        : '';
    const savedAt = result.created_at ? `<div class="qa-saved-at">已保存 · ${escapeHtml(formatDateTime(result.created_at))}</div>` : '';
    return `
        ${warningHtml}
        <div class="qa-answer">${renderMarkdown(result.answer || '没有生成答案。')}</div>
        ${renderQaCitations(result.citations || [])}
        ${savedAt}
    `;
}

function renderQaCitations(citations) {
    if (!citations?.length) return '';
    return `<div class="qa-citations">
        ${citations.map(item => {
            const source = item.source || item.name || item.path || item.label || '-';
            const score = item.score == null ? '' : ` · ${Number(item.score).toFixed(2)}`;
            const lowConfidence = item.confidence === 'low' || item.source_type === 'comments';
            const times = citationTimes(item);
            return `<div class="${lowConfidence ? 'low-confidence' : ''}">
                <strong>${escapeHtml(source)}</strong><span>${escapeHtml(score)}</span>
                ${times.length ? `<div class="qa-citation-times">${times.map(seconds => videoTimeButtonHtml(selectedJobId, seconds)).join('')}</div>` : ''}
            </div>`;
        }).join('')}
    </div>`;
}

function citationTimes(item) {
    const values = [];
    for (const value of item.timestamps || []) {
        const seconds = typeof value === 'string' ? parseTimestampToSeconds(value) : Number(value);
        if (Number.isFinite(seconds)) values.push(seconds);
    }
    for (const frame of item.frames || []) {
        const seconds = Number(frame?.timestamp_sec ?? frame?.timestamp);
        if (Number.isFinite(seconds)) {
            values.push(seconds);
            continue;
        }
        const mapped = frameTimestampFromMap(selectedJobId, frame?.path || frame?.frame_path || frame);
        if (mapped != null) values.push(mapped);
    }
    return [...new Set(values.map(value => Math.max(0, Math.floor(value))))].slice(0, 4);
}

function videoTimeButtonHtml(jobId, seconds) {
    return `<button class="video-time-link" type="button" data-job-id="${escapeHtml(jobId || '')}" data-seconds="${escapeHtml(seconds)}">${escapeHtml(formatTimestampFromSeconds(seconds))}</button>`;
}

async function askQa(event) {
    event.preventDefault();
    if (!selectedJobId || nodes.qaAskButton.disabled) return;
    const question = nodes.qaQuestion.value.trim();
    if (!question) return;
    nodes.qaQuestion.value = '';
    nodes.qaAskButton.disabled = true;
    appendQaMessage('user', `<p>${escapeHtml(question)}</p>`);
    try {
        const result = await getJson(`/api/video-link/jobs/${selectedJobId}/qa/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        appendQaMessage('assistant', renderQaAnswer(result));
    } catch (error) {
        appendQaMessage('assistant error', `<p>${escapeHtml(error.message)}</p>`);
    } finally {
        renderQaPanel(currentJob);
    }
}

function formatDateTime(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString('zh-CN', { hour12: false });
}

async function generateSkillCandidate() {
    if (!selectedJobId || nodes.generateSkillButton.disabled) return;
    const skill = currentJob?.summary?.skill_distillation || currentJob?.summary?.skill_candidate || {};
    const force = ['succeeded', 'completed_no_skills'].includes(skill.status);
    if (force && !window.confirm('重新蒸馏会清空当前蒸馏包和 checkpoint。确认继续？')) return;
    nodes.generateSkillButton.disabled = true;
    nodes.skillSummary.textContent = '正在启动蒸馏...';
    try {
        const result = await getJson(`/api/video-link/jobs/${selectedJobId}/skill-distillation/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                profile: nodes.skillProfile.value || 'deepseek_v4_pro',
                force
            })
        });
        if (currentJob?.summary) {
            currentJob.summary.skill_candidate = result;
            currentJob.summary.skill_distillation = result;
        }
        renderSkillCandidatePanel(currentJob);
        await refreshSelectedJob();
    } catch (error) {
        nodes.skillSummary.textContent = `启动失败：${error.message}`;
        renderSkillCandidatePanel(currentJob);
    }
}

async function reviewSkillOverview(action) {
    if (!selectedJobId) return;
    const buttons = [nodes.regenerateSkillOverviewButton, nodes.confirmSkillOverviewButton];
    buttons.forEach(button => { button.disabled = true; });
    try {
        await getJson(`/api/video-link/jobs/${selectedJobId}/skill-distillation/review-overview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action,
                feedback: nodes.skillOverviewFeedback.value.trim()
            })
        });
        nodes.skillOverviewFeedback.value = '';
        await refreshSelectedJob();
    } catch (error) {
        nodes.skillSummary.textContent = `审核失败：${error.message}`;
    } finally {
        buttons.forEach(button => { button.disabled = false; });
    }
}

async function confirmSkillCandidates() {
    if (!selectedJobId) return;
    const selectedIds = Array.from(
        nodes.skillCandidateList.querySelectorAll('input[type="checkbox"]:checked')
    ).map(input => input.value);
    nodes.confirmSkillCandidatesButton.disabled = true;
    try {
        await getJson(`/api/video-link/jobs/${selectedJobId}/skill-distillation/review-candidates`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ selected_ids: selectedIds })
        });
        skillCandidateDrafts.delete(selectedJobId);
        await refreshSelectedJob();
    } catch (error) {
        nodes.skillSummary.textContent = `候选确认失败：${error.message}`;
    } finally {
        nodes.confirmSkillCandidatesButton.disabled = false;
    }
}

async function resumeSkillDistillation() {
    if (!selectedJobId || nodes.resumeSkillButton.disabled) return;
    nodes.resumeSkillButton.disabled = true;
    try {
        await getJson(`/api/video-link/jobs/${selectedJobId}/skill-distillation/resume`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}'
        });
        await refreshSelectedJob();
    } catch (error) {
        nodes.skillSummary.textContent = `继续失败：${error.message}`;
        nodes.resumeSkillButton.disabled = false;
    }
}

async function cancelSkillDistillation() {
    if (!selectedJobId || nodes.cancelSkillButton.disabled) return;
    nodes.cancelSkillButton.disabled = true;
    try {
        await getJson(`/api/video-link/jobs/${selectedJobId}/skill-distillation/cancel`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}'
        });
        await refreshSelectedJob();
    } catch (error) {
        nodes.skillSummary.textContent = `取消失败：${error.message}`;
        nodes.cancelSkillButton.disabled = false;
    }
}

async function enableSkillCandidate() {
    if (!selectedJobId || nodes.enableSkillButton.disabled) return;
    const skill = currentJob?.summary?.skill_distillation || currentJob?.summary?.skill_candidate || {};
    const count = skill.skills?.passed || 0;
    if (!window.confirm(`启用 ${count} 个已通过压力测试的 Skills？\n\n将写入项目 .codex/skills。`)) return;
    nodes.enableSkillButton.disabled = true;
    nodes.skillSummary.textContent = '正在启用...';
    try {
        const result = await getJson(`/api/video-link/jobs/${selectedJobId}/skill-distillation/enable`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ overwrite: false })
        });
        if (currentJob?.summary) {
            currentJob.summary.skill_candidate = result;
            currentJob.summary.skill_distillation = result;
        }
        renderSkillCandidatePanel(currentJob);
        await refreshSelectedJob();
    } catch (error) {
        if (/already exist|already exists|已存在/i.test(error.message)
            && window.confirm(`${error.message}\n\n覆盖这些现有 Skills？`)) {
            try {
                await getJson(`/api/video-link/jobs/${selectedJobId}/skill-distillation/enable`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ overwrite: true })
                });
                await refreshSelectedJob();
                return;
            } catch (overwriteError) {
                nodes.skillSummary.textContent = `覆盖失败：${overwriteError.message}`;
            }
        } else {
            nodes.skillSummary.textContent = `启用失败：${error.message}`;
        }
    }
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
        renderSourcePlayer(job);
        renderStudyPanel(job);
        renderDocPreviewPanel(job);
        renderSkillsWorkspace(job);
        return;
    }
    if (!runDir) {
        nodes.vscodeSummary.textContent = '资源包目录尚未生成';
        resetVscodeShell();
        renderSourcePlayer(job);
        renderStudyPanel(job);
        renderDocPreviewPanel(job);
        renderSkillsWorkspace(job);
        return;
    }
    nodes.vscodeSummary.textContent = ready
        ? `WebIDE 已就绪 · PID ${preview.pid} · ${runDir}`
        : `${vscodeStarting ? '正在启动 WebIDE' : '资源包目录'} · ${runDir}`;
    resetVscodeShell();
    renderSourcePlayer(job);
    renderStudyPanel(job);
    renderDocPreviewPanel(job);
    renderSkillsWorkspace(job);
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
        const [guide] = await Promise.all([
            getJson(`/api/video-link/jobs/${jobId}/study-guide`),
            loadFrameTimeMap(jobId)
        ]);
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
    const webEvidenceHtml = renderWebEvidenceSummary(guide.web_evidence);
    if (!chapters.length) {
        return `<section class="study-learning-shell">
            <div class="study-content-summary">
                <span>内容总结</span>
                <p>${escapeHtml(overview.summary || '暂无学习总览')}</p>
                ${webEvidenceHtml}
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
                ${webEvidenceHtml}
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

function renderWebEvidenceSummary(webEvidence) {
    const summary = webEvidence?.summary || {};
    const processed = Number(summary.processed_gaps || 0);
    if (!processed) return '';
    const external = Number(summary.resolved_by_external || 0);
    const partial = Number(summary.partial_external_support || 0);
    const unresolved = Number(summary.unresolved || 0) + Number(summary.video_only_gap || 0);
    return `<div class="study-evidence-boundary">
        <strong>联网补证据</strong>
        <span>已处理 ${escapeHtml(processed)} 个缺口 · 外部补强 ${escapeHtml(external)} · 部分补强 ${escapeHtml(partial)} · 仍需复核 ${escapeHtml(unresolved)}</span>
    </div>`;
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
    const frameSeconds = Number(frame.timestamp_sec ?? frame.timestamp ?? frameTimestampFromMap(jobId, path));
    const secondsAttr = Number.isFinite(frameSeconds) ? ` data-frame-seconds="${escapeHtml(frameSeconds)}"` : '';
    const playButton = Number.isFinite(frameSeconds)
        ? `<button class="image-frame-play" type="button" data-job-id="${escapeHtml(jobId)}" data-seconds="${escapeHtml(frameSeconds)}">播放此帧</button>`
        : '';
    return `<img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="lazy" data-image-viewer-src="${escapeHtml(src)}" data-image-viewer-alt="${escapeHtml(alt)}" data-job-id="${escapeHtml(jobId)}" data-source-path="${escapeHtml(path)}"${secondsAttr}>${playButton}`;
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
    nodes.studyBody.querySelectorAll('.image-frame-play').forEach(button => {
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
    if (selectedJobId !== jobId) {
        selectedJobId = jobId;
        await refreshSelectedJob();
    }
    if (!['qa', 'vscode'].includes(currentView)) setView('vscode');
    sourcePlayerState.jobId = jobId;
    sourcePlayerState.seconds = Math.max(0, Math.floor(seconds));
    sourcePlayerState.loaded = true;
    renderSourcePlayer(currentJob, sourcePlayerState.seconds);
    const activePanel = currentView === 'qa' ? nodes.qaSourcePlayerPanel : nodes.sourcePlayerPanel;
    activePanel?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderSourcePlayer(job, seconds = sourcePlayerState.seconds) {
    const targets = sourcePlayerTargets();
    if (!targets.length) return;
    const player = job?.source_player || {};
    const value = Number(seconds);
    const seekSeconds = Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0;
    const label = formatTimestampFromSeconds(seekSeconds);
    if (!job || !player.source_url) {
        renderSourcePlayerEmpty(targets, '选择一个任务后播放', '暂无可嵌入播放器');
        return;
    }

    const watchUrl = sourceWatchUrl(player, seekSeconds);
    const summary = player.can_embed
        ? `${sourceProviderLabel(player.provider)} · ${label || '0:00'}`
        : '当前来源不支持内嵌播放';
    renderSourcePlayerHeaders(targets, summary, watchUrl || player.source_url, Boolean(watchUrl));

    if (!sourcePlayerState.loaded) {
        renderSourcePlayerBody(targets, 'empty', '', `<div class="source-player-empty">
            选择文档或问答中的时间戳后加载播放器；不会自动播放。
        </div>`);
        return;
    }

    if (!player.can_embed || !player.embed_url) {
        renderSourcePlayerBody(targets, 'empty', '', `<div class="source-player-empty">
            当前平台不能稳定内嵌播放，请使用“原站打开”跳到 ${escapeHtml(label || '0:00')}。
        </div>`);
        return;
    }

    const embedUrl = sourceEmbedUrl(player, seekSeconds);
    if (!embedUrl) {
        renderSourcePlayerBody(targets, 'empty', '', '<div class="source-player-empty">播放器地址无效，请使用“原站打开”。</div>');
        return;
    }
    const html = `<div class="source-player-frame-wrap">
        <iframe
            title="${escapeHtml(jobDisplayTitle(job))}"
            src="${escapeHtml(embedUrl)}"
            allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            allowfullscreen></iframe>
    </div>`;
    renderSourcePlayerBody(targets, 'iframe', embedUrl, html);
}

function sourcePlayerTargets() {
    return [
        {
            summary: nodes.qaSourcePlayerSummary,
            openLink: nodes.qaSourcePlayerOpenLink,
            stopButton: nodes.qaSourcePlayerStopButton,
            body: nodes.qaSourcePlayerBody
        },
        {
            summary: nodes.sourcePlayerSummary,
            openLink: nodes.sourcePlayerOpenLink,
            stopButton: nodes.sourcePlayerStopButton,
            body: nodes.sourcePlayerBody
        }
    ].filter(target => target.summary && target.openLink && target.body);
}

function renderSourcePlayerEmpty(targets, summary, message) {
    renderSourcePlayerHeaders(targets, summary, '', false);
    renderSourcePlayerBody(targets, 'empty', '', `<div class="source-player-empty">${escapeHtml(message)}</div>`);
}

function renderSourcePlayerHeaders(targets, summary, href, showLink) {
    targets.forEach(target => {
        target.summary.textContent = summary;
        target.openLink.href = href || '#';
        target.openLink.hidden = !showLink;
        if (target.stopButton) target.stopButton.hidden = !sourcePlayerState.loaded;
    });
}

function renderSourcePlayerBody(targets, playerKind, playerUrl, html) {
    targets.forEach(target => {
        if (
            target.body.dataset.playerKind === playerKind
            && target.body.dataset.playerUrl === playerUrl
        ) return;
        target.body.dataset.playerKind = playerKind;
        target.body.dataset.playerUrl = playerUrl;
        target.body.innerHTML = html;
    });
}

function pauseSourcePlayer() {
    sourcePlayerState.loaded = false;
    renderSourcePlayer(currentJob);
}

function sourceProviderLabel(provider) {
    return {
        youtube: 'YouTube',
        bilibili: 'Bilibili',
        external: '原站'
    }[provider] || '原站';
}

function sourceEmbedUrl(player, seconds = 0) {
    let url;
    try {
        url = new URL(player.embed_url, window.location.href);
    } catch (_error) {
        return '';
    }
    const value = Math.max(0, Math.floor(Number(seconds) || 0));
    if (player.provider === 'youtube') {
        url.searchParams.set('start', String(value));
        url.searchParams.set('rel', '0');
    } else if (player.provider === 'bilibili') {
        url.searchParams.set('t', String(value));
    }
    return url.toString();
}

function sourceWatchUrl(player, seconds = 0) {
    const raw = player.watch_url || player.source_url || '';
    if (!raw) return '';
    const value = Math.max(0, Math.floor(Number(seconds) || 0));
    let url;
    try {
        url = new URL(raw, window.location.href);
    } catch (_error) {
        return raw;
    }
    if (player.provider === 'youtube') {
        url.searchParams.set('t', `${value}s`);
    } else if (player.provider === 'bilibili') {
        url.searchParams.set('t', String(value));
    } else if (value > 0) {
        url.hash = `t=${value}`;
    }
    return url.toString();
}

function resourceUrl(jobId, path) {
    return `/api/video-link/jobs/${jobId}/resource?path=${encodeURIComponent(path)}`;
}

function resourcePathUrl(jobId, path) {
    const encodedPath = String(path || '')
        .split('/')
        .filter(Boolean)
        .map(part => encodeURIComponent(part))
        .join('/');
    return `/api/video-link/jobs/${jobId}/resources/${encodedPath}`;
}

async function loadFrameTimeMap(jobId) {
    if (!jobId) return {};
    if (frameTimeMaps[jobId]) return frameTimeMaps[jobId];
    try {
        const payload = await getJson(`/api/video-link/jobs/${jobId}/frame-time-map`);
        if (payload.available) {
            frameTimeMaps[jobId] = payload.frames || {};
            return frameTimeMaps[jobId];
        }
    } catch (_error) {
        return {};
    }
    return {};
}

function frameTimestampFromMap(jobId, sourcePath) {
    const map = frameTimeMaps[jobId] || {};
    const keys = framePathKeys(sourcePath);
    for (const key of keys) {
        const entry = map[key];
        const seconds = Number(entry?.timestamp_sec);
        if (Number.isFinite(seconds)) return seconds;
    }
    return null;
}

function framePathKeys(sourcePath) {
    const value = String(sourcePath || '').replace(/\\/g, '/');
    if (!value) return [];
    const keys = new Set([value]);
    try {
        const url = new URL(value, window.location.href);
        const resourcePath = url.searchParams.get('path');
        if (resourcePath) keys.add(resourcePath.replace(/\\/g, '/'));
        keys.add(decodeURIComponent(url.pathname.split('/').pop() || ''));
    } catch (_error) {
        keys.add(value.split('/').pop() || value);
    }
    const basename = value.split('/').pop();
    if (basename) keys.add(basename);
    return [...keys].filter(Boolean);
}

function parseTimestampToSeconds(value) {
    const match = String(value || '').trim().match(/^(\d{1,2}):([0-5]\d)(?::([0-5]\d))?$/);
    if (!match) return null;
    const first = Number(match[1]);
    const second = Number(match[2]);
    const third = match[3] == null ? null : Number(match[3]);
    return third == null ? first * 60 + second : first * 3600 + second * 60 + third;
}

const DOCUMENT_DERIVATION_PATH = '__document_derivation__';

function formatBytes(value) {
    const bytes = Number(value || 0);
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    const units = ['KB', 'MB', 'GB'];
    let amount = bytes / 1024;
    let unit = units.shift();
    while (amount >= 1024 && units.length) {
        amount /= 1024;
        unit = units.shift();
    }
    return `${amount.toFixed(amount >= 10 ? 0 : 1)} ${unit}`;
}

function docKindFromPath(path) {
    const value = String(path || '').toLowerCase();
    if (value.endsWith('.pdf')) return 'pdf';
    if (value.endsWith('.html') || value.endsWith('.htm')) return 'html';
    if (value.endsWith('.json')) return 'json';
    return 'markdown';
}

function normalizePreviewItems(items, group) {
    return (items || []).map(item => ({
        ...item,
        group,
        kind: item.type === 'directory' ? 'directory' : docKindFromPath(item.path),
        previewable: item.type !== 'directory'
    }));
}

function fallbackPreviewableDocs(job) {
    const summary = job?.summary || {};
    const markdownFiles = (summary.markdown_files || []).map(path => ({
        path,
        title: path,
        description: '历史 Markdown 产物',
        group: 'process',
        kind: 'markdown',
        previewable: true
    }));
    const pdfFiles = (summary.export_files || [])
        .filter(path => path.toLowerCase().endsWith('.pdf'))
        .map(path => ({
            path,
            title: path,
            description: '历史 PDF 导出',
            group: 'process',
            kind: 'pdf',
            previewable: true
        }));
    const preferred = [
        'operation_manual.md',
        'docs_analysis_chapters/knowledge_notes_v2.md',
        'docs_analysis_chapters/deep_report_v2.md',
        'manual_evidence.md',
        'evidence_index.md',
        'study_overview.md',
        'study_cards.md',
        'exports/operation_manual.pdf',
        'exports/knowledge_notes_v2.pdf',
        'exports/deep_report_v2.pdf',
        'exports/manual_evidence.pdf'
    ];
    return [...markdownFiles, ...pdfFiles].sort((left, right) => {
        const leftIndex = preferred.indexOf(left.path);
        const rightIndex = preferred.indexOf(right.path);
        const leftRank = leftIndex === -1 ? 1000 : leftIndex;
        const rightRank = rightIndex === -1 ? 1000 : rightIndex;
        if (leftRank !== rightRank) return leftRank - rightRank;
        if (left.kind !== right.kind) return left.kind === 'markdown' ? -1 : 1;
        return left.path.localeCompare(right.path);
    });
}

function previewableDocs(job) {
    const preview = job?.document_preview || {};
    const primary = normalizePreviewItems(preview.primary, 'primary');
    const evidence = normalizePreviewItems(preview.evidence, 'evidence');
    const process = normalizePreviewItems(preview.process, 'process');
    const assets = normalizePreviewItems(preview.assets, 'assets');
    const derivation = preview.derivation ? [{
        type: 'mindmap',
        path: DOCUMENT_DERIVATION_PATH,
        title: '文档推导脑图',
        description: '理解重点文档、证据文件和过程文件之间的递进关系',
        group: 'mindmap',
        kind: 'mindmap',
        previewable: true
    }] : [];
    const grouped = [...primary, ...derivation, ...evidence, ...process, ...assets];
    return grouped.length ? grouped : fallbackPreviewableDocs(job);
}

function docGroupDefinitions(docs) {
    const groups = [
        ['primary', '重点阅读', '最终用户最应该先看的结论文档'],
        ['mindmap', '文档推导', '这些文件如何一步步生成'],
        ['evidence', '证据审计', '抽帧、OCR/VL、发布判断等可追溯证据'],
        ['process', '过程文件', '中间分析、QA、调试与结构化产物'],
        ['assets', '素材目录', '帧图、截图和报告素材目录']
    ];
    return groups
        .map(([key, title, description]) => ({
            key,
            title,
            description,
            docs: docs.filter(doc => doc.group === key)
        }))
        .filter(group => group.docs.length);
}

function docTypeLabel(doc) {
    if (doc.kind === 'mindmap') return '图';
    if (doc.kind === 'directory') return '目录';
    if (doc.kind === 'pdf') return 'PDF';
    if (doc.kind === 'html') return 'HTML';
    if (doc.kind === 'json') return 'JSON';
    return 'MD';
}

function docMetaText(doc) {
    const parts = [];
    if (doc.file_count != null) parts.push(`${doc.file_count} 个文件`);
    const size = formatBytes(doc.size_bytes);
    if (size) parts.push(size);
    if (doc.updated_at) parts.push(String(doc.updated_at).replace('T', ' ').replace(/\+00:00$/, 'Z'));
    return parts.join(' · ');
}

function findPreviewDoc(job, path) {
    return previewableDocs(job).find(doc => doc.path === path);
}

function docPreviewUrl(job, doc) {
    if (!job?.job_id || !doc?.path) return '';
    return doc.url || resourcePathUrl(job.job_id, doc.path);
}

let mermaidRenderSequence = 0;

function initializeMermaid() {
    if (!window.mermaid) return false;
    if (!window.mermaid.__videoAnalyzerInitialized) {
        window.mermaid.initialize({
            startOnLoad: false,
            securityLevel: 'antiscript',
            theme: 'default',
            flowchart: {
                htmlLabels: true,
                curve: 'basis',
                useMaxWidth: true
            }
        });
        window.mermaid.__videoAnalyzerInitialized = true;
    }
    return true;
}

async function renderMermaidDiagram(container, diagram) {
    const target = container?.querySelector('[data-mermaid-diagram]');
    if (!target) return;
    const source = String(diagram || '').trim();
    if (!source) {
        target.innerHTML = '<div class="mindmap-empty">暂无 Mermaid 图定义。</div>';
        return;
    }
    const fallbackHtml = renderSimpleMermaidFlowchart(source);
    if (!initializeMermaid()) {
        target.innerHTML = fallbackHtml || '<div class="mindmap-empty">Mermaid 渲染器未加载，请刷新页面。</div>';
        return;
    }
    target.classList.add('loading');
    target.textContent = 'Mermaid 渲染中...';
    try {
        const id = `video-doc-mermaid-${Date.now()}-${mermaidRenderSequence}`;
        mermaidRenderSequence += 1;
        const result = await window.mermaid.render(id, source);
        target.classList.remove('loading');
        target.innerHTML = result?.svg || '<div class="mindmap-empty">Mermaid 未返回可显示图形。</div>';
    } catch (error) {
        console.warn('Mermaid render failed', error);
        target.classList.remove('loading');
        const message = error?.message || String(error || '未知错误');
        target.innerHTML = `${fallbackHtml || ''}<div class="mindmap-empty">Mermaid 渲染失败：${escapeHtml(message)}</div>`;
    }
}

function renderDocMindmap(derivation) {
    const nodes = derivation?.nodes || [];
    const edges = derivation?.edges || [];
    const mermaid = derivation?.mermaid || 'flowchart LR';
    const tiers = [...new Set(nodes.map(node => Number(node.tier || 0)))].sort((left, right) => left - right);
    const tierHtml = tiers.map(tier => {
        const tierNodes = nodes.filter(node => Number(node.tier || 0) === tier);
        return `<div class="mindmap-tier">
            ${tierNodes.map(node => `<div class="mindmap-node${node.available ? '' : ' missing'}">
                <strong>${escapeHtml(node.label || node.id || '')}</strong>
                <span>${escapeHtml(node.description || '')}</span>
            </div>`).join('')}
        </div>`;
    }).join('');
    const edgeHtml = edges.map(edge => `<li>${escapeHtml(edge.from_label || edge.from)} -> ${escapeHtml(edge.to_label || edge.to)}</li>`).join('');
    return `<section class="mindmap-preview">
        <h2>文档推导脑图</h2>
        <p>重点文档来自前面的转写、抽帧、OCR/VL 与证据审计；过程文件默认折叠，只在排查或追溯时打开。</p>
        <h3>Mermaid 预览</h3>
        <div class="mindmap-mermaid" data-mermaid-diagram></div>
        <h3>生成层级</h3>
        <div class="mindmap-canvas">${tierHtml}</div>
        <h3>递进关系</h3>
        <ol>${edgeHtml}</ol>
        <h3>Mermaid 源码</h3>
        <pre class="mindmap-source">${escapeHtml(mermaid)}</pre>
    </section>`;
}

function loadLearningPanelVisibility() {
    try {
        const stored = JSON.parse(localStorage.getItem(panelVisibilityStorageKey) || '{}') || {};
        ['docList', 'study', 'sourcePlayer', 'docContent'].forEach(key => {
            if (typeof stored[key] === 'boolean') learningPanelVisibility[key] = stored[key];
        });
    } catch (_error) {
        // Keep the default all-on layout when localStorage is unavailable or stale.
    }
}

function saveLearningPanelVisibility() {
    localStorage.setItem(panelVisibilityStorageKey, JSON.stringify(learningPanelVisibility));
}

function syncLearningPanelToggles() {
    if (nodes.toggleDocListPanel) nodes.toggleDocListPanel.checked = learningPanelVisibility.docList;
    if (nodes.toggleStudyPanel) nodes.toggleStudyPanel.checked = learningPanelVisibility.study;
    if (nodes.toggleSourcePlayerPanel) nodes.toggleSourcePlayerPanel.checked = learningPanelVisibility.sourcePlayer;
    if (nodes.toggleDocContentPanel) nodes.toggleDocContentPanel.checked = learningPanelVisibility.docContent;
}

function hasDocContent() {
    return Boolean(selectedDocPath && learningPanelVisibility.docContent);
}

function updateLearningDocsLayout() {
    if (!nodes.vscodeDocs) return;
    const docListVisible = learningPanelVisibility.docList;
    const studyVisible = Boolean(
        learningPanelVisibility.study
        && currentJob?.summary?.study?.available
    );
    const contentVisible = hasDocContent();
    const playerVisible = Boolean(nodes.sourcePlayerPanel && learningPanelVisibility.sourcePlayer);
    const visiblePanelCount = [docListVisible, studyVisible, playerVisible, contentVisible].filter(Boolean).length;
    const columns = [];

    nodes.docListPanel.hidden = !docListVisible;
    nodes.studyPanel.hidden = !studyVisible;
    if (nodes.sourcePlayerPanel) nodes.sourcePlayerPanel.hidden = !playerVisible;
    nodes.docPreviewPanel.hidden = !contentVisible;
    nodes.vscodeDocs.classList.toggle('preview-open', contentVisible);
    nodes.vscodeDocs.classList.toggle('empty-layout', visiblePanelCount === 0);

    const docListIsRightmost = docListVisible && !studyVisible && !playerVisible && !contentVisible;
    if (docListVisible) {
        columns.push(
            docListIsRightmost
                ? 'minmax(220px, 1fr)'
                : 'minmax(220px, var(--doc-list-pane-width, 300px))'
        );
    }
    const docListNeedsHandle = docListVisible && (studyVisible || playerVisible || contentVisible);
    nodes.docListResizer?.classList.toggle('active', docListNeedsHandle);
    if (nodes.docListResizer) nodes.docListResizer.hidden = !docListNeedsHandle;
    if (docListNeedsHandle) columns.push('14px');

    const studyIsRightmost = studyVisible && !playerVisible && !contentVisible;
    if (studyVisible) {
        columns.push(
            studyIsRightmost
                ? 'minmax(280px, 1fr)'
                : 'minmax(280px, var(--study-pane-width, 1fr))'
        );
    }
    const studyNeedsHandle = studyVisible && (playerVisible || contentVisible);
    nodes.studyResizer?.classList.toggle('active', studyNeedsHandle);
    if (nodes.studyResizer) nodes.studyResizer.hidden = !studyNeedsHandle;
    if (studyNeedsHandle) columns.push('14px');

    const sourcePlayerIsRightmost = playerVisible && !contentVisible;
    if (playerVisible) {
        columns.push(
            sourcePlayerIsRightmost
                ? 'minmax(320px, 1fr)'
                : 'minmax(320px, var(--source-player-pane-width, 560px))'
        );
    }
    const sourcePlayerNeedsHandle = playerVisible && (docListVisible || studyVisible || contentVisible);
    nodes.sourcePlayerResizer?.classList.toggle('active', sourcePlayerNeedsHandle);
    if (nodes.sourcePlayerResizer) nodes.sourcePlayerResizer.hidden = !sourcePlayerNeedsHandle;
    if (sourcePlayerNeedsHandle) columns.push('14px');

    if (contentVisible) columns.push('minmax(320px, 1fr)');
    nodes.vscodeDocs.style.gridTemplateColumns = columns.join(' ') || '1fr';
}

function setLearningPanelVisibility(panel, visible, persist = true) {
    learningPanelVisibility[panel] = Boolean(visible);
    syncLearningPanelToggles();
    updateLearningDocsLayout();
    if (persist) saveLearningPanelVisibility();
}

function bindLearningPanelToggles() {
    loadLearningPanelVisibility();
    syncLearningPanelToggles();
    nodes.toggleDocListPanel?.addEventListener('change', event => {
        setLearningPanelVisibility('docList', event.target.checked);
    });
    nodes.toggleStudyPanel?.addEventListener('change', event => {
        setLearningPanelVisibility('study', event.target.checked);
    });
    nodes.toggleSourcePlayerPanel?.addEventListener('change', event => {
        setLearningPanelVisibility('sourcePlayer', event.target.checked);
    });
    nodes.toggleDocContentPanel?.addEventListener('change', event => {
        setLearningPanelVisibility('docContent', event.target.checked);
    });
    updateLearningDocsLayout();
}

function renderDocPreviewPanel(job) {
    if (!nodes.docList) return;
    const docs = previewableDocs(job);
    const primaryCount = docs.filter(doc => doc.group === 'primary').length;
    const evidenceCount = docs.filter(doc => doc.group === 'evidence').length;
    const processCount = docs.filter(doc => doc.group === 'process').length;
    nodes.docPreviewSummary.textContent = docs.length
        ? `${primaryCount} 个重点 · ${evidenceCount} 个证据 · ${processCount} 个过程`
        : '无文档';
    if (!job || !docs.length) {
        selectedDocPath = '';
        renderedDocListKey = '';
        loadedDocPreviewKey = '';
        nodes.docList.innerHTML = '<div class="doc-empty">暂无 Markdown/PDF</div>';
        hideDocPreview();
        return;
    }
    if (selectedDocPath && !docs.some(doc => doc.path === selectedDocPath)) selectedDocPath = '';
    const listKey = `${job.job_id}|${selectedDocPath}|${docs.map(doc => `${doc.group}:${doc.kind}:${doc.path}`).join('|')}`;
    if (renderedDocListKey !== listKey) {
        renderedDocListKey = listKey;
        nodes.docList.innerHTML = docGroupDefinitions(docs).map(group => {
            const body = group.docs.map(doc => {
                const active = doc.path === selectedDocPath ? ' active' : '';
                const disabled = doc.previewable ? '' : ' disabled';
                const meta = docMetaText(doc);
                return `<button class="doc-item ${escapeHtml(doc.group)}${active}" type="button" data-doc-path="${escapeHtml(doc.path)}"${disabled}>
                    <span class="doc-kind">${escapeHtml(docTypeLabel(doc))}</span>
                    <strong>${escapeHtml(doc.title || doc.path)}</strong>
                    <small>${escapeHtml(doc.description || doc.path)}</small>
                    <em>${escapeHtml(meta || doc.path)}</em>
                </button>`;
            }).join('');
            const content = `<div class="doc-group-body">${body}</div>`;
            if (group.key === 'process' || group.key === 'assets') {
                return `<details class="doc-group" ${group.key === 'process' ? '' : ''}>
                    <summary>
                        <strong>${escapeHtml(group.title)}</strong>
                        <span>${group.docs.length} 项</span>
                    </summary>
                    <p>${escapeHtml(group.description)}</p>
                    ${content}
                </details>`;
            }
            return `<section class="doc-group">
                <header>
                    <strong>${escapeHtml(group.title)}</strong>
                    <span>${group.docs.length} 项</span>
                </header>
                <p>${escapeHtml(group.description)}</p>
                ${content}
            </section>`;
        }).join('');
        nodes.docList.querySelectorAll('.doc-item').forEach(button => {
            button.addEventListener('click', () => {
                if (button.disabled) return;
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
    if (!learningPanelVisibility.docContent) {
        setLearningPanelVisibility('docContent', true);
        return;
    }
    updateLearningDocsLayout();
}

function hideDocPreview() {
    nodes.docPreviewTitle.textContent = '选择文档';
    nodes.docOpenLink.hidden = true;
    nodes.docOpenLink.removeAttribute('href');
    nodes.docPreviewBody.className = 'doc-preview-body';
    nodes.docPreviewBody.innerHTML = '';
    updateLearningDocsLayout();
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
    const doc = findPreviewDoc(job, path) || { path, kind: docKindFromPath(path), title: path, previewable: true };
    if (doc.kind === 'mindmap') {
        nodes.docPreviewTitle.textContent = doc.title || '文档推导脑图';
        nodes.docOpenLink.hidden = true;
        nodes.docOpenLink.removeAttribute('href');
        nodes.docPreviewBody.className = 'doc-preview-body mindmap';
        nodes.docPreviewBody.innerHTML = renderDocMindmap(job.document_preview?.derivation || {});
        await renderMermaidDiagram(nodes.docPreviewBody, job.document_preview?.derivation?.mermaid || 'flowchart LR');
        loadedDocPreviewKey = previewKey;
        return;
    }
    const url = docPreviewUrl(job, doc);
    nodes.docPreviewTitle.textContent = doc.title || path;
    nodes.docOpenLink.href = url;
    nodes.docOpenLink.hidden = false;
    if (path.toLowerCase().endsWith('.pdf')) {
        nodes.docPreviewBody.className = 'doc-preview-body pdf';
        nodes.docPreviewBody.innerHTML = `<iframe title="${escapeHtml(path)}" src="${escapeHtml(url)}"></iframe>`;
        loadedDocPreviewKey = previewKey;
        return;
    }
    if (path.toLowerCase().endsWith('.html') || path.toLowerCase().endsWith('.htm')) {
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
        const text = await response.text();
        if (selectedDocPath !== path) return;
        nodes.docPreviewBody.className = 'doc-preview-body markdown';
        if (path.toLowerCase().endsWith('.json')) {
            nodes.docPreviewBody.innerHTML = `<pre>${escapeHtml(text)}</pre>`;
        } else {
            nodes.docPreviewBody.innerHTML = renderMarkdown(text, job.job_id, path);
        }
        renderMarkdownMath(nodes.docPreviewBody);
        await enhanceTimestampTargets(nodes.docPreviewBody, job.job_id);
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
        ADD_ATTR: ['target', 'rel', 'loading', 'data-image-viewer-src', 'data-image-viewer-alt', 'data-job-id', 'data-source-path', 'data-frame-seconds']
    });
}

function stripSkillFrontmatter(markdown) {
    const value = String(markdown || '');
    if (!value.startsWith('---')) return value;
    const match = value.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/);
    return match ? value.slice(match[0].length) : value;
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
            if (jobId) token.attrSet('data-job-id', jobId);
            if (assetPath) token.attrSet('data-source-path', assetPath);
            const seconds = frameTimestampFromMap(jobId, assetPath);
            if (seconds != null) token.attrSet('data-frame-seconds', String(seconds));
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

async function enhanceTimestampTargets(container, jobId) {
    if (!container || !jobId) return;
    await loadFrameTimeMap(jobId);
    container.querySelectorAll('img[data-image-viewer-src]').forEach(image => {
        if (!image.dataset.jobId) image.dataset.jobId = jobId;
        const seconds = Number(image.dataset.frameSeconds);
        if (Number.isFinite(seconds)) return;
        const mapped = frameTimestampFromMap(jobId, image.dataset.sourcePath || image.dataset.imageViewerSrc || image.currentSrc || image.src);
        if (mapped != null) image.dataset.frameSeconds = String(mapped);
    });
    linkTimestampTextNodes(container, jobId);
}

function linkTimestampTextNodes(container, jobId) {
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
            const parent = node.parentElement;
            if (!parent || parent.closest('button, a, pre, code, script, style, .katex')) {
                return NodeFilter.FILTER_REJECT;
            }
            return /\b\d{1,2}:[0-5]\d(?::[0-5]\d)?\b/.test(node.nodeValue || '')
                ? NodeFilter.FILTER_ACCEPT
                : NodeFilter.FILTER_REJECT;
        }
    });
    const nodesToReplace = [];
    while (walker.nextNode()) nodesToReplace.push(walker.currentNode);
    for (const node of nodesToReplace) {
        const fragment = document.createDocumentFragment();
        const text = node.nodeValue || '';
        let cursor = 0;
        text.replace(/\b\d{1,2}:[0-5]\d(?::[0-5]\d)?\b/g, (match, offset) => {
            if (offset > cursor) fragment.append(document.createTextNode(text.slice(cursor, offset)));
            const seconds = parseTimestampToSeconds(match);
            if (seconds == null) {
                fragment.append(document.createTextNode(match));
            } else {
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'video-time-link';
                button.dataset.jobId = jobId;
                button.dataset.seconds = String(seconds);
                button.textContent = match;
                fragment.append(button);
            }
            cursor = offset + match.length;
            return match;
        });
        if (cursor < text.length) fragment.append(document.createTextNode(text.slice(cursor)));
        node.replaceWith(fragment);
    }
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
        const seconds = frameTimestampFromMap(jobId, imagePath);
        const secondsAttr = seconds != null ? ` data-frame-seconds="${escapeHtml(seconds)}"` : '';
        html += src
            ? `<figure class="markdown-image"><img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="lazy" data-image-viewer-src="${escapeHtml(src)}" data-image-viewer-alt="${escapeHtml(alt)}" data-job-id="${escapeHtml(jobId)}" data-source-path="${escapeHtml(imagePath)}"${secondsAttr}><figcaption>${escapeHtml(alt)}</figcaption></figure>`
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
    const match = cleaned.match(/^([A-Za-z][A-Za-z0-9_-]*)(.*)$/);
    if (!match) return { id: cleaned, label: '', shape: 'box' };
    const id = match[1];
    const rest = (match[2] || '').trim();
    let label = '';
    let shape = 'box';
    if (rest) {
        const wrappers = [
            [/^\[\[(.*)\]\]$/, 'box'],
            [/^\[(.*)\]$/, 'box'],
            [/^\{\{(.*)\}\}$/, 'decision'],
            [/^\{(.*)\}$/, 'decision'],
            [/^\(\((.*)\)\)$/, 'circle'],
            [/^\((.*)\)$/, 'box'],
        ];
        const wrapper = wrappers.find(([pattern]) => pattern.test(rest));
        if (!wrapper) return { id: cleaned, label: '', shape: 'box' };
        label = rest.match(wrapper[0])?.[1] || '';
        shape = wrapper[1];
    }
    label = label.trim();
    if (label.length >= 2 && label[0] === '"' && label[label.length - 1] === '"') {
        label = label.slice(1, -1);
    }
    return {
        id,
        label,
        shape
    };
}

function mermaidLabelHtml(label) {
    return escapeHtml(label).replace(/&lt;br\s*\/?&gt;/gi, '<br>');
}

function renderSimpleMermaidFlowchart(diagram) {
    const lines = normalizeMermaidPreviewLines(diagram)
        .flatMap(line => line.split(';'))
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
        if (/^(subgraph|end\b|direction\b|style\b|classDef\b|class\b|click\b|linkStyle\b)/i.test(line)) continue;
        if (/^[};\s]+$/.test(line)) continue;
        const edgeMatch = parseMermaidEdge(line);
        if (!edgeMatch) {
            const node = parseMermaidNode(line);
            if (!rememberNode(node)) return '';
            if (!standaloneNodes.includes(node.id)) standaloneNodes.push(node.id);
            continue;
        }
        const left = parseMermaidNode(edgeMatch.left);
        const right = parseMermaidNode(edgeMatch.right);
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
            const circle = shapes.get(node) === 'circle' ? ' circle' : '';
            const label = mermaidLabelHtml(labels.get(node) || node);
            parts.push(`<div class="flow-node${shape}${circle}"><span class="flow-index">${index}</span>${label}</div>`);
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

function normalizeMermaidPreviewLines(diagram) {
    return String(diagram || '')
        .split(/\r?\n/)
        .map(line => line.replace(/\s+%%.*$/, ''));
}

function parseMermaidEdge(line) {
    const value = String(line || '').trim().replace(/;$/, '');
    const patterns = [
        /^(.+?)\s*--\s*[^-<>|]+?\s*-->\s*(.+?)$/,
        /^(.+?)\s*-->\s*(?:\|.*?\|\s*)?(.+?)$/,
        /^(.+?)\s*-.->\s*(?:\|.*?\|\s*)?(.+?)$/,
        /^(.+?)\s*==>\s*(?:\|.*?\|\s*)?(.+?)$/,
        /^(.+?)\s*--[ox]\s*(?:\|.*?\|\s*)?(.+?)$/,
    ];
    for (const pattern of patterns) {
        const match = value.match(pattern);
        if (match) return { left: match[1].trim(), right: match[2].trim() };
    }
    return null;
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
    if (log.history_fallback) {
        nodes.logHint.textContent = `显示：${stageNames[stage] || stage} 的上一轮尝试日志；当前尝试尚未写入输出`;
    }
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
                    <button type="button" data-image-viewer-play hidden>播放此帧</button>
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
    imageViewer.playButton = viewer.querySelector('[data-image-viewer-play]');
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
            return;
        }
        if (event.target.closest('[data-image-viewer-play]')) {
            if (imageViewer.jobId && Number.isFinite(imageViewer.seconds)) {
                jumpToVideoTime(imageViewer.jobId, imageViewer.seconds);
            }
        }
    });
    viewer.querySelector('.image-viewer-stage')?.addEventListener('wheel', event => {
        event.preventDefault();
        zoomImageViewer(event.deltaY < 0 ? 0.1 : -0.1);
    }, { passive: false });
    return viewer;
}

function openImageViewer(src, alt = '', jobId = '', seconds = '') {
    if (!src) return;
    const viewer = ensureImageViewer();
    imageViewer.image.src = src;
    imageViewer.image.alt = alt || '图片预览';
    imageViewer.jobId = jobId || '';
    imageViewer.seconds = Number(seconds);
    const canPlay = imageViewer.jobId && Number.isFinite(imageViewer.seconds);
    if (imageViewer.playButton) {
        imageViewer.playButton.hidden = !canPlay;
        imageViewer.playButton.textContent = canPlay ? `播放此帧 ${formatTimestampFromSeconds(imageViewer.seconds)}` : '播放此帧';
    }
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
        openImageViewer(
            image.dataset.imageViewerSrc || image.currentSrc || image.src,
            image.dataset.imageViewerAlt || image.alt || '',
            image.dataset.jobId || selectedJobId || '',
            image.dataset.frameSeconds || ''
        );
    });
    document.addEventListener('keydown', event => {
        if (!imageViewer.node || imageViewer.node.hidden) return;
        if (event.key === 'Escape') closeImageViewer();
        if (event.key === '+' || event.key === '=') zoomImageViewer(0.2);
        if (event.key === '-' || event.key === '_') zoomImageViewer(-0.2);
        if (event.key === '0') setImageViewerScale(1);
    });
}

function bindVideoTimeLinks() {
    document.addEventListener('click', async event => {
        const button = event.target.closest('.video-time-link, .image-frame-play, [data-video-time-seconds]');
        if (!button) return;
        event.preventDefault();
        const jobId = button.dataset.jobId || selectedJobId;
        const seconds = Number(button.dataset.seconds || button.dataset.videoTimeSeconds);
        if (jobId && Number.isFinite(seconds)) await jumpToVideoTime(jobId, seconds);
    });
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function loadPaneLayout() {
    let layout = {};
    try {
        layout = JSON.parse(localStorage.getItem(paneLayoutStorageKey) || '{}') || {};
    } catch (_error) {
        layout = {};
    }
    applyPaneWidth('study', savedPaneSize(layout.study));
    applyPaneWidth('doc-list', savedPaneSize(layout.docList));
    applyPaneWidth('source-player', savedPaneSize(layout.sourcePlayer));
    applyPaneWidth('qa-source', savedPaneSize(layout.qaSource));
    applySourcePlayerHeight(savedPaneSize(layout.qaSourceHeight));
    applySkillsPaneWidth('skills-current-left', savedPaneSize(layout.skillsCurrentLeft));
    applySkillsPaneWidth('skills-current-right', savedPaneSize(layout.skillsCurrentRight));
    applySkillsPaneHeight('skills-current-height', savedPaneSize(layout.skillsCurrentHeight));
    applySkillsPaneWidth('skills-library-left', savedPaneSize(layout.skillsLibraryLeft));
    applySkillsPaneWidth('skills-library-right', savedPaneSize(layout.skillsLibraryRight));
    applySkillsPaneHeight('skills-library-height', savedPaneSize(layout.skillsLibraryHeight));
}

function savePaneLayout() {
    const vscodeStyle = nodes.vscodeDocs?.style;
    const qaStyle = nodes.qaLayout?.style;
    const layout = {
        study: parsePanePixels(vscodeStyle?.getPropertyValue('--study-pane-width')),
        docList: parsePanePixels(vscodeStyle?.getPropertyValue('--doc-list-pane-width')),
        sourcePlayer: parsePanePixels(vscodeStyle?.getPropertyValue('--source-player-pane-width')),
        qaSource: parsePanePixels(qaStyle?.getPropertyValue('--qa-source-pane-width')),
        qaSourceHeight: parsePanePixels(nodes.qaSourcePlayerPanel?.style.getPropertyValue('--qa-source-player-height')),
        skillsCurrentLeft: parsePanePixels(nodes.currentSkillsGrid?.style.getPropertyValue('--skills-left-pane-width')),
        skillsCurrentRight: parsePanePixels(nodes.currentSkillsGrid?.style.getPropertyValue('--skills-right-pane-width')),
        skillsCurrentHeight: parsePanePixels(nodes.currentSkillsGrid?.style.getPropertyValue('--skills-pane-height')),
        skillsLibraryLeft: parsePanePixels(nodes.skillLibraryGrid?.style.getPropertyValue('--skills-left-pane-width')),
        skillsLibraryRight: parsePanePixels(nodes.skillLibraryGrid?.style.getPropertyValue('--skills-right-pane-width')),
        skillsLibraryHeight: parsePanePixels(nodes.skillLibraryGrid?.style.getPropertyValue('--skills-pane-height'))
    };
    localStorage.setItem(paneLayoutStorageKey, JSON.stringify(layout));
}

function parsePanePixels(value) {
    const number = Number.parseFloat(String(value || '').replace('px', ''));
    return Number.isFinite(number) ? Math.round(number) : null;
}

function savedPaneSize(value) {
    if (value === null || value === undefined || value === '') return Number.NaN;
    return Number(value);
}

function applyPaneWidth(pane, width) {
    if (!Number.isFinite(width)) return;
    if (pane === 'qa-source') {
        nodes.qaLayout?.style.setProperty('--qa-source-pane-width', `${Math.round(width)}px`);
        return;
    }
    if (!nodes.vscodeDocs) return;
    const property = paneWidthProperty(pane);
    nodes.vscodeDocs.style.setProperty(property, `${Math.round(width)}px`);
}

function paneWidth(pane) {
    if (pane === 'qa-source') {
        const explicit = parsePanePixels(nodes.qaLayout?.style.getPropertyValue('--qa-source-pane-width'));
        if (explicit) return explicit;
        return nodes.qaSidePanel?.getBoundingClientRect().width || 0;
    }
    if (!nodes.vscodeDocs) return 0;
    const property = paneWidthProperty(pane);
    const explicit = parsePanePixels(nodes.vscodeDocs.style.getPropertyValue(property));
    if (explicit) return explicit;
    const selector = paneSelector(pane);
    return nodes.vscodeDocs.querySelector(selector)?.getBoundingClientRect().width || 0;
}

function paneWidthProperty(pane) {
    if (pane === 'doc-list') return '--doc-list-pane-width';
    if (pane === 'source-player') return '--source-player-pane-width';
    return '--study-pane-width';
}

function paneSelector(pane) {
    if (pane === 'doc-list') return '.doc-list-panel';
    if (pane === 'source-player') return '.source-player-panel';
    return '.study-panel';
}

function sourcePlayerVisible() {
    return Boolean(nodes.sourcePlayerPanel && !nodes.sourcePlayerPanel.hidden);
}

function studyCanResizeWidth(
    docListVisible = learningPanelVisibility.docList,
    playerVisible = sourcePlayerVisible(),
    contentVisible = hasDocContent()
) {
    return Boolean(docListVisible || playerVisible || contentVisible);
}

function studyRightReserve() {
    const sourceReserve = sourcePlayerVisible() ? 14 + (paneWidth('source-player') || 560) : 0;
    const contentReserve = hasDocContent() ? 14 + 320 : 0;
    const rightPanelReserve = sourceReserve + contentReserve;
    if (rightPanelReserve) return rightPanelReserve;
    return nodes.studyResizer && !nodes.studyResizer.hidden ? 14 : 0;
}

function resizeStudyPane(clientX) {
    const docs = nodes.vscodeDocs;
    if (!docs || !learningPanelVisibility.study) return;
    const rect = docs.getBoundingClientRect();
    const docListOffset = learningPanelVisibility.docList ? (paneWidth('doc-list') || 300) + 14 : 0;
    const leftEdge = rect.left + docListOffset;
    const max = Math.max(280, rect.right - leftEdge - studyRightReserve());
    applyPaneWidth('study', clamp(clientX - leftEdge, 280, max));
}

function resizeDocListPane(clientX) {
    const docs = nodes.vscodeDocs;
    if (!docs || !learningPanelVisibility.docList) return;
    const rect = docs.getBoundingClientRect();
    const studyReserve = learningPanelVisibility.study ? 14 + 280 : 0;
    const sourceReserve = sourcePlayerVisible() ? 14 + (paneWidth('source-player') || 560) : 0;
    const contentReserve = hasDocContent() ? 14 + 320 : 0;
    const max = Math.max(220, rect.width - studyReserve - sourceReserve - contentReserve);
    applyPaneWidth('doc-list', clamp(clientX - rect.left, 220, max));
}

function resizeSourcePlayerPane(clientX) {
    const docs = nodes.vscodeDocs;
    if (!docs || !sourcePlayerVisible()) return;
    const rect = docs.getBoundingClientRect();
    const docListOffset = learningPanelVisibility.docList ? (paneWidth('doc-list') || 300) + 14 : 0;
    const studyOffset = learningPanelVisibility.study ? (paneWidth('study') || 420) + 14 : 0;
    const leftEdge = rect.left + docListOffset + studyOffset;
    const contentReserve = hasDocContent() ? 14 + 320 : 0;
    const max = Math.max(320, rect.right - leftEdge - contentReserve);
    applyPaneWidth('source-player', clamp(clientX - leftEdge, 320, max));
}

function resizePaneFromPointer(pane, clientX) {
    if (pane?.startsWith('skills-')) {
        resizeSkillsPane(pane, clientX);
        return;
    }
    if (pane === 'qa-source') {
        resizeQaSourcePane(clientX);
        return;
    }
    if (pane === 'doc-list') {
        resizeDocListPane(clientX);
    } else if (pane === 'source-player') {
        resizeSourcePlayerPane(clientX);
    } else {
        resizeStudyPane(clientX);
    }
}

function adjustPaneWidth(pane, delta) {
    if (pane?.startsWith('skills-')) {
        adjustSkillsPaneWidth(pane, delta);
        return;
    }
    if (pane === 'qa-source') {
        const rect = nodes.qaLayout?.getBoundingClientRect();
        if (!rect) return;
        const max = Math.max(320, rect.width - 360);
        applyPaneWidth('qa-source', clamp(paneWidth('qa-source') + delta, 320, max));
        return;
    }
    if (!nodes.vscodeDocs) return;
    const rect = nodes.vscodeDocs.getBoundingClientRect();
    if (pane === 'doc-list') {
        const studyReserve = learningPanelVisibility.study ? 14 + 280 : 0;
        const sourceReserve = sourcePlayerVisible() ? 14 + (paneWidth('source-player') || 560) : 0;
        const contentReserve = hasDocContent() ? 14 + 320 : 0;
        const max = Math.max(220, rect.width - studyReserve - sourceReserve - contentReserve);
        applyPaneWidth('doc-list', clamp(paneWidth('doc-list') + delta, 220, max));
        return;
    }
    if (pane === 'source-player') {
        const docListReserve = learningPanelVisibility.docList ? (paneWidth('doc-list') || 300) + 14 : 0;
        const studyReserve = learningPanelVisibility.study ? (paneWidth('study') || 420) + 14 : 0;
        const contentReserve = hasDocContent() ? 14 + 320 : 0;
        const max = Math.max(320, rect.width - docListReserve - studyReserve - contentReserve);
        applyPaneWidth('source-player', clamp(paneWidth('source-player') + delta, 320, max));
        return;
    }
    const docListReserve = learningPanelVisibility.docList ? (paneWidth('doc-list') || 300) + 14 : 0;
    const max = Math.max(280, rect.width - docListReserve - studyRightReserve());
    applyPaneWidth('study', clamp(paneWidth('study') + delta, 280, max));
}

function resizeQaSourcePane(clientX) {
    const layout = nodes.qaLayout;
    if (!layout) return;
    const rect = layout.getBoundingClientRect();
    const max = Math.max(320, rect.width - 360);
    applyPaneWidth('qa-source', clamp(rect.right - clientX, 320, max));
}

function sourcePlayerHeight() {
    const explicit = parsePanePixels(nodes.qaSourcePlayerPanel?.style.getPropertyValue('--qa-source-player-height'));
    if (explicit) return explicit;
    return nodes.qaSourcePlayerBody?.getBoundingClientRect().height || 260;
}

function applySourcePlayerHeight(height) {
    if (!nodes.qaSourcePlayerPanel || !Number.isFinite(height)) return;
    nodes.qaSourcePlayerPanel.style.setProperty('--qa-source-player-height', `${Math.round(height)}px`);
}

function resizeSourcePlayerHeight(clientY) {
    const panel = nodes.qaSourcePlayerPanel;
    const header = panel?.querySelector('.source-player-header');
    if (!panel || !header) return;
    const rect = panel.getBoundingClientRect();
    const headerHeight = header.getBoundingClientRect().height;
    const max = Math.max(220, window.innerHeight - rect.top - 180);
    applySourcePlayerHeight(clamp(clientY - rect.top - headerHeight, 180, max));
}

function adjustSourcePlayerHeight(delta) {
    const max = Math.max(220, window.innerHeight - (nodes.qaSourcePlayerPanel?.getBoundingClientRect().top || 0) - 180);
    applySourcePlayerHeight(clamp(sourcePlayerHeight() + delta, 180, max));
}

function skillsGridForPane(pane) {
    return pane?.startsWith('skills-library-') ? nodes.skillLibraryGrid : nodes.currentSkillsGrid;
}

function skillsPaneProperty(pane) {
    return pane?.endsWith('-right') ? '--skills-right-pane-width' : '--skills-left-pane-width';
}

function skillsPaneWidth(pane) {
    const grid = skillsGridForPane(pane);
    if (!grid) return 0;
    const property = skillsPaneProperty(pane);
    const explicit = parsePanePixels(grid.style.getPropertyValue(property));
    if (explicit) return explicit;
    const selector = pane.endsWith('-right')
        ? '.skills-inspector-pane'
        : '.skills-item-pane';
    return grid.querySelector(selector)?.getBoundingClientRect().width || 0;
}

function applySkillsPaneWidth(pane, width) {
    const grid = skillsGridForPane(pane);
    if (!grid || !Number.isFinite(width)) return;
    grid.style.setProperty(skillsPaneProperty(pane), `${Math.round(width)}px`);
}

function skillsPaneLimits(pane) {
    const grid = skillsGridForPane(pane);
    if (!grid) return null;
    const rect = grid.getBoundingClientRect();
    const isLibrary = pane.startsWith('skills-library-');
    const leftMin = 210;
    const rightMin = 250;
    const centerMin = isLibrary ? 420 : 360;
    const handles = 20;
    return { grid, rect, leftMin, rightMin, centerMin, handles };
}

function resizeSkillsPane(pane, clientX) {
    if (pane.endsWith('-height')) return;
    const limits = skillsPaneLimits(pane);
    if (!limits) return;
    const { rect, leftMin, rightMin, centerMin, handles } = limits;
    if (pane.endsWith('-left')) {
        const right = skillsPaneWidth(pane.replace(/-left$/, '-right')) || 340;
        const max = Math.max(leftMin, rect.width - right - centerMin - handles);
        applySkillsPaneWidth(pane, clamp(clientX - rect.left, leftMin, max));
        return;
    }
    const left = skillsPaneWidth(pane.replace(/-right$/, '-left')) || 270;
    const max = Math.max(rightMin, rect.width - left - centerMin - handles);
    applySkillsPaneWidth(pane, clamp(rect.right - clientX, rightMin, max));
}

function adjustSkillsPaneWidth(pane, delta) {
    if (pane.endsWith('-height')) return;
    const limits = skillsPaneLimits(pane);
    if (!limits) return;
    const { rect, leftMin, rightMin, centerMin, handles } = limits;
    if (pane.endsWith('-left')) {
        const right = skillsPaneWidth(pane.replace(/-left$/, '-right')) || 340;
        const max = Math.max(leftMin, rect.width - right - centerMin - handles);
        applySkillsPaneWidth(pane, clamp(skillsPaneWidth(pane) + delta, leftMin, max));
        return;
    }
    const left = skillsPaneWidth(pane.replace(/-right$/, '-left')) || 270;
    const max = Math.max(rightMin, rect.width - left - centerMin - handles);
    applySkillsPaneWidth(pane, clamp(skillsPaneWidth(pane) - delta, rightMin, max));
}

function skillsPaneHeight(pane) {
    const grid = skillsGridForPane(pane);
    if (!grid) return 0;
    const explicit = parsePanePixels(grid.style.getPropertyValue('--skills-pane-height'));
    return explicit || grid.getBoundingClientRect().height || 520;
}

function applySkillsPaneHeight(pane, height) {
    const grid = skillsGridForPane(pane);
    if (!grid || !Number.isFinite(height)) return;
    grid.style.setProperty('--skills-pane-height', `${Math.round(height)}px`);
}

function resizeSkillsPaneHeight(pane, clientY) {
    const grid = skillsGridForPane(pane);
    if (!grid) return;
    const rect = grid.getBoundingClientRect();
    applySkillsPaneHeight(pane, clamp(clientY - rect.top, 360, 1200));
}

function adjustSkillsPaneHeight(pane, delta) {
    applySkillsPaneHeight(pane, clamp(skillsPaneHeight(pane) + delta, 360, 1200));
}

function constrainSkillsGrid(grid, centerMin) {
    if (!grid || grid.hidden || window.innerWidth <= 980) return;
    const rect = grid.getBoundingClientRect();
    if (rect.width <= 0) return;
    const leftMin = 210;
    const rightMin = 250;
    const available = Math.max(leftMin + rightMin, rect.width - centerMin - 20);
    let left = parsePanePixels(grid.style.getPropertyValue('--skills-left-pane-width')) || 280;
    let right = parsePanePixels(grid.style.getPropertyValue('--skills-right-pane-width')) || 360;
    left = Math.max(leftMin, left);
    right = Math.max(rightMin, right);
    const excess = left + right - available;
    if (excess > 0) {
        const leftRoom = left - leftMin;
        const rightRoom = right - rightMin;
        const room = leftRoom + rightRoom;
        if (room > 0) {
            left -= excess * (leftRoom / room);
            right -= excess * (rightRoom / room);
        }
    }
    grid.style.setProperty('--skills-left-pane-width', `${Math.round(Math.max(leftMin, left))}px`);
    grid.style.setProperty('--skills-right-pane-width', `${Math.round(Math.max(rightMin, right))}px`);
}

function constrainSkillsLayouts() {
    constrainSkillsGrid(nodes.currentSkillsGrid, 360);
    constrainSkillsGrid(nodes.skillLibraryGrid, 420);
}

function isHorizontalPane(pane) {
    return pane === 'qa-source-height' || pane?.endsWith('-height');
}

function bindPaneResizers() {
    loadPaneLayout();
    window.addEventListener('resize', constrainSkillsLayouts);
    document.querySelectorAll('.pane-resizer').forEach(handle => {
        handle.addEventListener('pointerdown', event => {
            paneResizeState.active = handle.dataset.resizePane || 'study';
            paneResizeState.pointerId = event.pointerId;
            handle.setPointerCapture?.(event.pointerId);
            resizerContainer(paneResizeState.active)?.classList.add('resizing');
            if (isHorizontalPane(paneResizeState.active)) {
                if (paneResizeState.active === 'qa-source-height') {
                    resizeSourcePlayerHeight(event.clientY);
                } else {
                    resizeSkillsPaneHeight(paneResizeState.active, event.clientY);
                }
            } else {
                resizePaneFromPointer(paneResizeState.active, event.clientX);
            }
            event.preventDefault();
        });
        handle.addEventListener('keydown', event => {
            const pane = handle.dataset.resizePane || 'study';
            if (isHorizontalPane(pane)) {
                if (!['ArrowUp', 'ArrowDown'].includes(event.key)) return;
                const delta = event.key === 'ArrowDown' ? 24 : -24;
                if (pane === 'qa-source-height') {
                    adjustSourcePlayerHeight(delta);
                } else {
                    adjustSkillsPaneHeight(pane, delta);
                }
            } else {
                if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
                const delta = event.key === 'ArrowRight' ? 24 : -24;
                adjustPaneWidth(pane, delta);
            }
            savePaneLayout();
            event.preventDefault();
        });
    });
    document.addEventListener('pointermove', event => {
        if (!paneResizeState.active) return;
        if (isHorizontalPane(paneResizeState.active)) {
            if (paneResizeState.active === 'qa-source-height') {
                resizeSourcePlayerHeight(event.clientY);
            } else {
                resizeSkillsPaneHeight(paneResizeState.active, event.clientY);
            }
        } else {
            resizePaneFromPointer(paneResizeState.active, event.clientX);
        }
    });
    document.addEventListener('pointerup', event => {
        if (!paneResizeState.active) return;
        if (paneResizeState.pointerId !== null && event.pointerId !== paneResizeState.pointerId) return;
        resizerContainer(paneResizeState.active)?.classList.remove('resizing');
        paneResizeState.active = null;
        paneResizeState.pointerId = null;
        savePaneLayout();
    });
}

function resizerContainer(pane) {
    if (pane?.startsWith('qa-source')) return nodes.qaLayout;
    if (pane === 'skills-library-height') return nodes.skillLibraryWorkspace;
    if (pane === 'skills-current-height') return nodes.currentSkillsWorkspace;
    if (pane?.startsWith('skills-library-')) return nodes.skillLibraryGrid;
    if (pane?.startsWith('skills-current-')) return nodes.currentSkillsGrid;
    return nodes.vscodeDocs;
}

async function runSelectedJob() {
    if (!selectedJobId) return;
    nodes.runButton.disabled = true;
    try {
        const action = nodes.runButton.dataset.action || (currentJob?.status === 'succeeded' ? 'open-run-dir' : 'run');
        const rerunCore = (
            action === 'run'
            && currentJob?.status === 'failed'
            && currentJob?.failure_disposition?.category === 'rerun_core'
        );
        const endpoint = rerunCore
            ? `/api/video-link/jobs/${selectedJobId}/stages/analyze-core/rerun`
            : `/api/video-link/jobs/${selectedJobId}/${action}`;
        const payload = rerunCore
            ? {
                profile: currentJob?.options?.profile || currentJob?.runtime_profile_snapshot?.profile || '',
                refresh_runtime_profile: true
            }
            : {};
        await getJson(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (action === 'run' || action === 'stop' || rerunCore) {
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
    ensureSkillActivityNodes();
    nodes.consoleTab.addEventListener('click', () => setView('console'));
    nodes.qaTab.addEventListener('click', () => setView('qa'));
    nodes.vscodeTab.addEventListener('click', () => setView('vscode'));
    nodes.settingsTab.addEventListener('click', () => setView('settings'));
    nodes.settingsModelsTab.addEventListener('click', () => setSettingsSection('models'));
    nodes.settingsProfilesTab.addEventListener('click', () => setSettingsSection('profiles'));
    nodes.settingsModelKindFilter.addEventListener('change', renderSettingsModelList);
    nodes.settingsModelSearch.addEventListener('input', renderSettingsModelList);
    nodes.newModelButton.addEventListener('click', () => resetModelEditor());
    nodes.modelKind.addEventListener('change', () => syncModelProtocolOptions());
    nodes.modelSettingsForm.addEventListener('submit', event => {
        saveModelSettings(event).catch(error => {
            nodes.modelSettingsMessage.textContent = error.message;
            nodes.modelSettingsMessage.classList.add('error-text');
        });
    });
    nodes.testModelButton.addEventListener('click', () => {
        testSelectedSettingsModel().catch(error => {
            nodes.modelSettingsMessage.textContent = error.message;
            nodes.modelSettingsMessage.classList.add('error-text');
        });
    });
    nodes.deleteModelButton.addEventListener('click', () => {
        deleteSelectedSettingsModel().catch(error => {
            nodes.modelSettingsMessage.textContent = error.message;
            nodes.modelSettingsMessage.classList.add('error-text');
        });
    });
    nodes.newProfileButton.addEventListener('click', () => resetProfileEditor());
    nodes.profileWorkflow.addEventListener('change', () => {
        profileTestReport = null;
        profileModelSelections = {};
        setProfileModelSelections();
        selectedProfileFlowNodeId = profileFlowSchema().nodes[0]?.id || '';
        renderProfileFlow();
        renderProfileTestSummary();
    });
    nodes.duplicateProfileButton.addEventListener('click', () => {
        const profile = (settingsData?.profiles || []).find(item => item.name === selectedSettingsProfileName);
        resetProfileEditor(profile || null);
    });
    nodes.profileSettingsForm.addEventListener('submit', event => {
        saveProfileSettings(event).catch(error => {
            nodes.profileSettingsMessage.textContent = error.message;
            nodes.profileSettingsMessage.classList.add('error-text');
        });
    });
    nodes.activateProfileButton.addEventListener('click', () => {
        activateSelectedSettingsProfile().catch(error => {
            nodes.profileSettingsMessage.textContent = error.message;
            nodes.profileSettingsMessage.classList.add('error-text');
        });
    });
    nodes.deleteProfileButton.addEventListener('click', () => {
        deleteSelectedSettingsProfile().catch(error => {
            nodes.profileSettingsMessage.textContent = error.message;
            nodes.profileSettingsMessage.classList.add('error-text');
        });
    });
    nodes.testProfileButton.addEventListener('click', () => {
        testCurrentProfile().catch(error => {
            nodes.profileTestSummary.hidden = false;
            nodes.profileTestSummary.classList.add('failed');
            nodes.profileTestSummary.innerHTML = `<div><strong>通路测试失败</strong><span>${escapeHtml(error.message)}</span></div>`;
        });
    });
    nodes.profileFlowViewport?.addEventListener('scroll', scheduleProfileFlowEdges, { passive: true });
    window.addEventListener('resize', scheduleProfileFlowEdges);
    if (window.ResizeObserver && nodes.profileFlowCanvas) {
        profileFlowResizeObserver = new ResizeObserver(scheduleProfileFlowEdges);
        profileFlowResizeObserver.observe(nodes.profileFlowCanvas);
    }
    nodes.resourceDocsTab.addEventListener('click', () => setResourceView('docs'));
    nodes.resourceSkillsTab.addEventListener('click', () => setResourceView('skills'));
    nodes.skillsResourceDocsTab.addEventListener('click', () => setResourceView('docs'));
    nodes.skillsResourceSkillsTab.addEventListener('click', () => setResourceView('skills'));
    nodes.toggleSkillsFocusButton.addEventListener('click', () => toggleSkillsFocusMode());
    document.addEventListener('keydown', event => {
        if (event.key !== 'Escape' || !nodes.appShell?.classList.contains('skills-focus-mode')) return;
        toggleSkillsFocusMode(false);
    });
    nodes.openSkillsWorkspaceButton.addEventListener('click', () => {
        setView('vscode', false);
        setResourceView('skills', false);
        setSkillsScope('current', true);
    });
    nodes.createTargetedSkillProjectButton.addEventListener('click', () => {
        setSkillsScope('projects');
        nodes.skillProjectGoal.focus();
    });
    nodes.skillsScopeTabs.forEach(button => {
        button.addEventListener('click', () => {
            skillLibraryProjectSkillNames = null;
            setSkillsScope(button.dataset.skillsScope || 'current');
        });
    });
    nodes.skillsDetailTabs.forEach(button => {
        button.addEventListener('click', () => setCurrentSkillDetailTab(button.dataset.skillDetailTab || 'evidence'));
    });
    nodes.skillEditorTabs.forEach(button => {
        button.addEventListener('click', () => setSkillEditorTab(button.dataset.skillEditorTab || 'edit'));
    });
    nodes.skillLibrarySearch.addEventListener('input', () => {
        window.clearTimeout(skillLibrarySearchTimer);
        skillLibrarySearchTimer = window.setTimeout(loadSkillLibrary, 180);
    });
    nodes.skillEditor.addEventListener('input', () => {
        if (!currentLibrarySkill) return;
        nodes.saveSkillButton.disabled = nodes.skillEditor.disabled;
        nodes.skillEditorMessage.textContent = nodes.skillEditor.value === currentLibrarySkill.markdown
            ? ''
            : '有未保存的修改';
        if (currentSkillEditorTab === 'preview') setSkillEditorTab('preview');
    });
    nodes.saveSkillButton.addEventListener('click', saveCurrentLibrarySkill);
    nodes.disableSkillButton.addEventListener('click', () => changeLibrarySkillState('disable'));
    nodes.restoreSkillButton.addEventListener('click', () => changeLibrarySkillState('restore'));
    nodes.deleteSkillButton.addEventListener('click', () => changeLibrarySkillState('delete'));
    nodes.permanentDeleteSkillButton.addEventListener('click', () => changeLibrarySkillState('permanent-delete'));
    nodes.jobForm.addEventListener('submit', createJob);
    nodes.urlSourceTab?.addEventListener('click', () => setSourceMode('url'));
    nodes.fileSourceTab?.addEventListener('click', () => setSourceMode('file'));
    nodes.intentCards.forEach(button => button.addEventListener('click', () => applyIntent(button.dataset.intent || 'smart')));
    nodes.templateSearch?.addEventListener('input', renderTemplateList);
    nodes.templateCategory?.addEventListener('change', renderTemplateList);
    nodes.clearTemplateButton?.addEventListener('click', clearTemplateSelection);
    nodes.addUrlButton.addEventListener('click', addPendingUrls);
    nodes.videoUrlInput.addEventListener('keydown', event => {
        if (event.key === 'Enter') {
            event.preventDefault();
            addPendingUrls();
        }
    });
    renderUrlList();
    nodes.refreshJobsButton.addEventListener('click', refreshJobs);
    nodes.showNonRerunFailures?.addEventListener('change', event => {
        showNonRerunFailures = Boolean(event.target.checked);
        renderJobList(latestJobs);
    });
    nodes.qaForm.addEventListener('submit', askQa);
    nodes.generateSkillButton.addEventListener('click', generateSkillCandidate);
    nodes.resumeSkillButton.addEventListener('click', resumeSkillDistillation);
    nodes.cancelSkillButton.addEventListener('click', cancelSkillDistillation);
    nodes.regenerateSkillOverviewButton.addEventListener('click', () => reviewSkillOverview('regenerate'));
    nodes.confirmSkillOverviewButton.addEventListener('click', () => reviewSkillOverview('confirm'));
    nodes.skillCandidateList.addEventListener('change', updateSkillCandidateDraft);
    nodes.confirmSkillCandidatesButton.addEventListener('click', confirmSkillCandidates);
    nodes.enableSkillButton.addEventListener('click', enableSkillCandidate);
    nodes.skillProjectForm.addEventListener('submit', event => {
        createSkillProject(event).catch(error => {
            nodes.skillProjectDetail.innerHTML = `<div class="skills-empty error-text">创建失败：${escapeHtml(error.message)}</div>`;
        });
    });
    nodes.previewSkillProjectPackageButton.addEventListener('click', () => {
        previewSkillProjectPackage().catch(error => {
            nodes.skillProjectPackageStatus.textContent = `检查失败：${error.message}`;
        });
    });
    nodes.skillProjectPackageId.addEventListener('keydown', event => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        previewSkillProjectPackage().catch(error => {
            nodes.skillProjectPackageStatus.textContent = `检查失败：${error.message}`;
        });
    });
    nodes.importSkillProjectPackageButton.addEventListener('click', () => {
        importSkillProjectPackage().catch(error => {
            nodes.skillProjectPackageStatus.textContent = `导入失败：${error.message}`;
        });
    });
    nodes.copySkillProjectLogButton.addEventListener('click', () => {
        copySkillProjectLog(selectedSkillProjectId).catch(() => {});
    });
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && currentSkillsScope === 'projects' && selectedSkillProjectId) {
            void loadSkillProjects(selectedSkillProjectId);
        }
    });
    nodes.startVscodeButton.addEventListener('click', () => ensureVscodeSession(false));
    nodes.restartVscodeButton.addEventListener('click', () => ensureVscodeSession(true));
    nodes.stopVscodeButton.addEventListener('click', stopVscodeSession);
    nodes.docPreviewClose.addEventListener('click', closeDocPreview);
    nodes.qaSourcePlayerStopButton?.addEventListener('click', pauseSourcePlayer);
    nodes.sourcePlayerStopButton?.addEventListener('click', pauseSourcePlayer);
    nodes.consoleFlowPrevious?.addEventListener('click', () => {
        nodes.consoleStageFlowViewport?.scrollBy({
            left: -Math.max(260, nodes.consoleStageFlowViewport.clientWidth * 0.7),
            behavior: 'smooth'
        });
    });
    nodes.consoleFlowNext?.addEventListener('click', () => {
        nodes.consoleStageFlowViewport?.scrollBy({
            left: Math.max(260, nodes.consoleStageFlowViewport.clientWidth * 0.7),
            behavior: 'smooth'
        });
    });
    nodes.consoleFlowZoomOut?.addEventListener('click', () => {
        consoleFlowScale = Math.max(0.55, consoleFlowScale - 0.1);
        applyConsoleFlowScale();
    });
    nodes.consoleFlowFit?.addEventListener('click', fitConsoleFlow);
    nodes.consoleFlowZoomIn?.addEventListener('click', () => {
        consoleFlowScale = Math.min(1.6, consoleFlowScale + 0.1);
        applyConsoleFlowScale();
    });
    nodes.consoleFlowCurrent?.addEventListener('click', () => {
        const nodeId = currentJob?.execution_flow?.active_node_ids?.[0] || selectedConsoleNodeId;
        scrollConsoleNodeIntoView(nodeId);
    });
    nodes.consoleStageLogButton?.addEventListener('click', () => {
        const stage = nodes.consoleStageLogButton.dataset.stage || selectedConsoleStage;
        if (!currentJob || !stage) return;
        selectedLogStage = stage;
        loadSelectedLog(currentJob);
        nodes.logText?.closest('.panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    nodes.runButton.addEventListener('click', runSelectedJob);
    nodes.copyLogButton.addEventListener('click', copySelectedLog);
    bindImageViewer();
    bindVideoTimeLinks();
    bindLearningPanelToggles();
    bindPaneResizers();
    loadSkillsFocusMode();
    await loadOptions();
    await loadPromptTemplates();
    applyIntent(activeIntent);
    setSourceMode(sourceMode);
    setView(currentView, true);
    setResourceView(currentResourceView, currentView === 'vscode');
    setSkillsScope(currentSkillsScope, false);
    await refreshJobs();
    if (selectedJobId) await refreshSelectedJob();
    window.setInterval(updateSkillLiveClock, 1000);
    window.setInterval(updateConsoleElapsedClock, 1000);
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
