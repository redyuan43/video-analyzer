import { Terminal } from './vendor/xterm/xterm.mjs';
import { FitAddon } from './vendor/xterm/addon-fit.mjs';

const token = document.querySelector('meta[name="web-debug-token"]')?.content || '';
if (!token) throw new Error('Web debug console token is missing');

const markdownRendererReady = createMarkdownRenderer();

const state = {
    panel: null,
    activeTab: 'terminal',
    config: null,
    terminal: null,
    fitAddon: null,
    terminalSessionId: null,
    terminalSequence: 0,
    terminalPoll: false,
    inputBuffer: '',
    inputTimer: null,
    debugSessionId: null,
    debugSequence: 0,
    debugPoll: false,
    debugHistoryKey: null,
};

function loadVendorScript(relativePath, globalName) {
    if (window[globalName]) return Promise.resolve(window[globalName]);
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = new URL(relativePath, import.meta.url).href;
        script.onload = () => {
            if (window[globalName]) {
                resolve(window[globalName]);
                return;
            }
            reject(new Error(`${globalName} did not initialize`));
        };
        script.onerror = () => reject(new Error(`Failed to load ${relativePath}`));
        document.head.appendChild(script);
    });
}

async function createMarkdownRenderer() {
    const [markdownit, DOMPurify] = await Promise.all([
        loadVendorScript('./vendor/markdown-it/markdown-it.min.js', 'markdownit'),
        loadVendorScript('./vendor/dompurify/purify.min.js', 'DOMPurify'),
    ]);
    const renderer = markdownit({
        html: false,
        breaks: true,
        linkify: true,
        typographer: false,
    });
    return markdown => DOMPurify.sanitize(renderer.render(String(markdown || '')));
}

function currentJobId() {
    return new URL(window.location.href).searchParams.get('job') || '';
}

async function api(path, options = {}) {
    const response = await fetch(`/devtools/api${path}`, {
        ...options,
        headers: {
            'X-Debug-Token': token,
            ...(options.body ? { 'Content-Type': 'application/json' } : {}),
            ...(options.headers || {}),
        },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
}

function buildConsole() {
    const root = document.createElement('section');
    root.className = 'web-debug-console';
    root.innerHTML = `
        <div class="web-debug-launcher" aria-label="开发工具">
            <button type="button" data-debug-open="terminal" title="终端" aria-label="打开终端">&gt;_</button>
            <button type="button" data-debug-open="debug" title="Debug" aria-label="打开 Debug">AI</button>
        </div>
        <div class="web-debug-backdrop" hidden></div>
        <section class="web-debug-panel" hidden aria-label="开发工具面板">
            <header class="web-debug-header">
                <div class="web-debug-tabs" role="tablist">
                    <button class="active" type="button" data-debug-tab="terminal">终端</button>
                    <button type="button" data-debug-tab="debug">Debug</button>
                </div>
                <div class="web-debug-header-actions">
                    <span class="web-debug-cwd" title=""></span>
                    <button class="web-debug-icon" type="button" data-debug-close title="关闭" aria-label="关闭">×</button>
                </div>
            </header>
            <section class="web-debug-tab active" data-debug-view="terminal">
                <div class="web-debug-toolbar">
                    <select data-terminal-tool aria-label="终端工具">
                        <option value="shell">Shell</option>
                        <option value="codex">Codex CLI</option>
                        <option value="claude">Claude CLI</option>
                    </select>
                    <button type="button" data-terminal-start>启动</button>
                    <button class="secondary" type="button" data-terminal-stop disabled>停止</button>
                    <span data-terminal-status>未启动</span>
                </div>
                <div class="web-debug-terminal"></div>
            </section>
            <section class="web-debug-tab" data-debug-view="debug" hidden>
                <div class="web-debug-toolbar">
                    <select data-debug-sandbox aria-label="Debug 权限">
                        <option value="workspace-write">可修改项目</option>
                        <option value="read-only">只读定位</option>
                    </select>
                    <button class="secondary" type="button" data-debug-reset>新会话</button>
                    <span data-debug-status>未连接</span>
                </div>
                <div class="web-debug-messages" aria-live="polite"></div>
                <form class="web-debug-form">
                    <textarea rows="3" placeholder="描述当前问题"></textarea>
                    <button type="submit">发送</button>
                </form>
            </section>
        </section>`;
    document.body.appendChild(root);
    state.panel = root;
    return root;
}

function node(selector) {
    return state.panel?.querySelector(selector);
}

function setStatus(selector, text, tone = '') {
    const target = node(selector);
    if (!target) return;
    target.textContent = text;
    target.dataset.tone = tone;
}

async function refreshConfig() {
    state.config = await api(`/config?job=${encodeURIComponent(currentJobId())}`);
    const cwd = node('.web-debug-cwd');
    if (cwd) {
        cwd.textContent = state.config.cwd;
        cwd.title = state.config.cwd;
    }
    const select = node('[data-terminal-tool]');
    if (select) {
        for (const option of select.options) {
            option.disabled = option.value !== 'shell' && !state.config.tools[option.value];
        }
    }
}

function switchTab(tab) {
    state.activeTab = tab;
    state.panel.querySelectorAll('[data-debug-tab]').forEach(button => {
        button.classList.toggle('active', button.dataset.debugTab === tab);
    });
    state.panel.querySelectorAll('[data-debug-view]').forEach(view => {
        const active = view.dataset.debugView === tab;
        view.hidden = !active;
        view.classList.toggle('active', active);
    });
    if (tab === 'terminal' && state.fitAddon) {
        requestAnimationFrame(() => {
            state.fitAddon.fit();
            resizeTerminal();
        });
    }
}

async function openConsole(tab = 'terminal') {
    await refreshConfig();
    if (tab === 'debug') await loadDebugHistory();
    node('.web-debug-panel').hidden = false;
    node('.web-debug-backdrop').hidden = false;
    document.body.classList.add('web-debug-open');
    switchTab(tab);
}

function closeConsole() {
    node('.web-debug-panel').hidden = true;
    node('.web-debug-backdrop').hidden = true;
    document.body.classList.remove('web-debug-open');
}

function ensureTerminal() {
    if (state.terminal) return;
    state.terminal = new Terminal({
        cursorBlink: true,
        convertEol: false,
        fontFamily: '"SFMono-Regular", Consolas, "Liberation Mono", monospace',
        fontSize: 13,
        lineHeight: 1.2,
        scrollback: 5000,
        theme: {
            background: '#090d14',
            foreground: '#d9e1ec',
            cursor: '#38bdf8',
            selectionBackground: '#1d4ed880',
            black: '#111827',
            red: '#f87171',
            green: '#4ade80',
            yellow: '#facc15',
            blue: '#60a5fa',
            magenta: '#c084fc',
            cyan: '#22d3ee',
            white: '#e5e7eb',
        },
    });
    state.fitAddon = new FitAddon();
    state.terminal.loadAddon(state.fitAddon);
    state.terminal.open(node('.web-debug-terminal'));
    state.terminal.onData(queueTerminalInput);
    requestAnimationFrame(() => state.fitAddon.fit());
}

function queueTerminalInput(data) {
    if (!state.terminalSessionId) return;
    state.inputBuffer += data;
    if (state.inputTimer) return;
    state.inputTimer = setTimeout(async () => {
        const pending = state.inputBuffer;
        state.inputBuffer = '';
        state.inputTimer = null;
        try {
            await api(`/terminal/sessions/${state.terminalSessionId}/input`, {
                method: 'POST',
                body: JSON.stringify({ data: pending }),
            });
        } catch (error) {
            setStatus('[data-terminal-status]', error.message, 'error');
        }
    }, 20);
}

async function startTerminal() {
    ensureTerminal();
    if (state.terminalSessionId) await stopTerminal();
    state.terminal.reset();
    state.fitAddon.fit();
    const tool = node('[data-terminal-tool]').value;
    setStatus('[data-terminal-status]', '启动中');
    const result = await api('/terminal/sessions', {
        method: 'POST',
        body: JSON.stringify({
            job_id: currentJobId(),
            cwd: state.config.cwd,
            tool,
            rows: state.terminal.rows,
            cols: state.terminal.cols,
        }),
    });
    state.terminalSessionId = result.session_id;
    state.terminalSequence = 0;
    node('[data-terminal-stop]').disabled = false;
    setStatus('[data-terminal-status]', `PID ${result.pid}`, 'ok');
    state.terminal.focus();
    pollTerminal();
}

async function pollTerminal() {
    if (state.terminalPoll || !state.terminalSessionId) return;
    state.terminalPoll = true;
    try {
        while (state.terminalSessionId) {
            const result = await api(
                `/terminal/sessions/${state.terminalSessionId}/output?after=${state.terminalSequence}&wait=20`
            );
            state.terminalSequence = result.sequence;
            if (result.output) state.terminal.write(result.output);
            if (!result.running) {
                setStatus('[data-terminal-status]', `已退出 ${result.exit_code ?? ''}`.trim());
                state.terminalSessionId = null;
                node('[data-terminal-stop]').disabled = true;
                break;
            }
        }
    } catch (error) {
        setStatus('[data-terminal-status]', error.message, 'error');
    } finally {
        state.terminalPoll = false;
    }
}

async function resizeTerminal() {
    if (!state.terminalSessionId || !state.terminal) return;
    await api(`/terminal/sessions/${state.terminalSessionId}/resize`, {
        method: 'POST',
        body: JSON.stringify({ rows: state.terminal.rows, cols: state.terminal.cols }),
    }).catch(() => {});
}

async function stopTerminal() {
    const sessionId = state.terminalSessionId;
    state.terminalSessionId = null;
    if (sessionId) {
        await api(`/terminal/sessions/${sessionId}`, { method: 'DELETE' }).catch(() => {});
    }
    node('[data-terminal-stop]').disabled = true;
    setStatus('[data-terminal-status]', '已停止');
}

function appendDebugMessage(kind, text, details = '') {
    const messages = node('.web-debug-messages');
    const item = document.createElement('article');
    item.className = `web-debug-message ${kind}`;
    const body = document.createElement('div');
    body.className = 'web-debug-message-body';
    body.textContent = text;
    item.appendChild(body);
    if (kind === 'assistant') {
        markdownRendererReady
            .then(render => {
                if (body.isConnected) body.innerHTML = render(text);
            })
            .catch(() => {});
    }
    if (details) {
        const pre = document.createElement('pre');
        pre.textContent = details;
        item.appendChild(pre);
    }
    messages.appendChild(item);
    messages.scrollTop = messages.scrollHeight;
}

function renderPersistedDebugEvent(event) {
    if (event.type === 'user') {
        appendDebugMessage('user', event.text || '');
        return;
    }
    renderDebugEvent(event);
}

async function loadDebugHistory(force = false) {
    const historyKey = currentJobId() || '__project__';
    if (!force && state.debugHistoryKey === historyKey) return;
    const result = await api(`/debug/history?job=${encodeURIComponent(currentJobId())}`);
    const messages = node('.web-debug-messages');
    messages.replaceChildren();
    (result.messages || []).forEach(renderPersistedDebugEvent);
    state.debugHistoryKey = historyKey;
    setStatus(
        '[data-debug-status]',
        result.thread_id ? '历史已加载' : '未连接',
        result.thread_id ? 'ok' : ''
    );
}

async function ensureDebugSession() {
    if (state.debugSessionId) return;
    setStatus('[data-debug-status]', '连接中');
    const result = await api('/debug/sessions', {
        method: 'POST',
        body: JSON.stringify({
            job_id: currentJobId(),
            cwd: state.config.cwd,
            sandbox: node('[data-debug-sandbox]').value,
        }),
    });
    state.debugSessionId = result.session_id;
    state.debugSequence = 0;
    setStatus('[data-debug-status]', result.resumed ? '已恢复' : '已连接', 'ok');
    pollDebugEvents();
}

async function resetDebugSession() {
    const sessionId = state.debugSessionId;
    state.debugSessionId = null;
    if (sessionId) {
        await api(`/debug/sessions/${sessionId}`, { method: 'DELETE' }).catch(() => {});
    }
    await api(`/debug/history?job=${encodeURIComponent(currentJobId())}`, {
        method: 'DELETE',
    });
    node('.web-debug-messages').replaceChildren();
    state.debugHistoryKey = currentJobId() || '__project__';
    setStatus('[data-debug-status]', '未连接');
}

async function sendDebugMessage(event) {
    event.preventDefault();
    const input = node('.web-debug-form textarea');
    const message = input.value.trim();
    if (!message) return;
    await ensureDebugSession();
    appendDebugMessage('user', message);
    input.value = '';
    setStatus('[data-debug-status]', '分析中');
    await api(`/debug/sessions/${state.debugSessionId}/messages`, {
        method: 'POST',
        body: JSON.stringify({ message }),
    });
}

function renderDebugEvent(event) {
    if (event.type === 'assistant') {
        appendDebugMessage('assistant', event.text || '');
        return;
    }
    if (event.type === 'command') {
        appendDebugMessage(
            'activity',
            `${event.status || 'completed'}: ${event.command || 'command'}`,
            event.output || ''
        );
        return;
    }
    if (event.type === 'file_change') {
        const paths = (event.changes || []).map(change => change.path).filter(Boolean).join('\n');
        appendDebugMessage('activity', `文件变更: ${event.status || '-'}`, paths);
        return;
    }
    if (event.type === 'activity') {
        setStatus('[data-debug-status]', event.label || '执行中');
        return;
    }
    if (event.type === 'done') {
        setStatus('[data-debug-status]', event.status === 'completed' ? '完成' : event.status || '结束', event.status === 'completed' ? 'ok' : 'error');
        return;
    }
    if (event.type === 'error' || event.type === 'closed') {
        appendDebugMessage('error', event.message || 'Debug 会话异常');
        setStatus('[data-debug-status]', '异常', 'error');
    }
}

async function pollDebugEvents() {
    if (state.debugPoll || !state.debugSessionId) return;
    state.debugPoll = true;
    try {
        while (state.debugSessionId) {
            const result = await api(
                `/debug/sessions/${state.debugSessionId}/events?after=${state.debugSequence}&wait=20`
            );
            state.debugSequence = result.sequence;
            result.events.forEach(renderDebugEvent);
            if (!result.running) {
                state.debugSessionId = null;
                break;
            }
        }
    } catch (error) {
        appendDebugMessage('error', error.message);
    } finally {
        state.debugPoll = false;
    }
}

function bindConsole() {
    const root = buildConsole();
    root.querySelectorAll('[data-debug-open]').forEach(button => {
        button.addEventListener('click', () => openConsole(button.dataset.debugOpen));
    });
    root.querySelectorAll('[data-debug-tab]').forEach(button => {
        button.addEventListener('click', () => {
            const tab = button.dataset.debugTab;
            const loading = tab === 'debug' ? loadDebugHistory() : Promise.resolve();
            loading
                .then(() => switchTab(tab))
                .catch(error => setStatus('[data-debug-status]', error.message, 'error'));
        });
    });
    node('[data-debug-close]').addEventListener('click', closeConsole);
    node('.web-debug-backdrop').addEventListener('click', closeConsole);
    node('[data-terminal-start]').addEventListener('click', () => startTerminal().catch(error => setStatus('[data-terminal-status]', error.message, 'error')));
    node('[data-terminal-stop]').addEventListener('click', stopTerminal);
    node('[data-debug-reset]').addEventListener('click', () => {
        resetDebugSession().catch(error => {
            appendDebugMessage('error', error.message);
            setStatus('[data-debug-status]', '异常', 'error');
        });
    });
    node('.web-debug-form').addEventListener('submit', event => sendDebugMessage(event).catch(error => {
        appendDebugMessage('error', error.message);
        setStatus('[data-debug-status]', '异常', 'error');
    }));
    window.addEventListener('resize', () => {
        if (!node('.web-debug-panel').hidden && state.activeTab === 'terminal' && state.fitAddon) {
            state.fitAddon.fit();
            resizeTerminal();
        }
    });
    window.addEventListener('pagehide', () => {
        if (state.terminalSessionId) {
            fetch(`/devtools/api/terminal/sessions/${state.terminalSessionId}`, {
                method: 'DELETE',
                headers: { 'X-Debug-Token': token },
                keepalive: true,
            }).catch(() => {});
        }
        if (state.debugSessionId) {
            fetch(`/devtools/api/debug/sessions/${state.debugSessionId}`, {
                method: 'DELETE',
                headers: { 'X-Debug-Token': token },
                keepalive: true,
            }).catch(() => {});
        }
    });
    window.webDebugConsole = { open: openConsole, close: closeConsole };
}

bindConsole();
