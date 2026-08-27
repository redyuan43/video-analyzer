/*
 * Mobile / iOS behaviour layer.
 *
 * Self-contained: main.js is untouched. This only adds the two things that
 * cannot be done in CSS alone — an off-canvas drawer for the console sidebar,
 * and picking up a URL handed over by the iOS Share Extension.
 */
(function () {
    'use strict';

    var MOBILE_QUERY = '(max-width: 860px)';
    var OPEN_CLASS = 'mobile-drawer-open';

    var shell = document.querySelector('.app-shell');
    var topbar = document.getElementById('appTopbar');
    var sidebar = document.querySelector('.sidebar');
    if (!shell || !topbar || !sidebar) return;

    var media = window.matchMedia(MOBILE_QUERY);

    // ------------------------------------------------------------- drawer

    var backdrop = document.createElement('div');
    backdrop.className = 'mobile-drawer-backdrop';
    backdrop.setAttribute('aria-hidden', 'true');
    shell.appendChild(backdrop);

    var toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'mobile-drawer-toggle';
    toggle.setAttribute('aria-label', '打开任务面板');
    toggle.setAttribute('aria-expanded', 'false');
    toggle.textContent = '☰';
    topbar.insertBefore(toggle, topbar.firstChild);

    function setDrawer(open) {
        shell.classList.toggle(OPEN_CLASS, open);
        // The page itself scrolls at this width, so freeze it while the drawer
        // is up — otherwise dragging the backdrop scrolls the console behind it.
        document.body.classList.toggle('mobile-drawer-locked', open);
        toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
        toggle.setAttribute('aria-label', open ? '关闭任务面板' : '打开任务面板');
    }

    toggle.addEventListener('click', function () {
        setDrawer(!shell.classList.contains(OPEN_CLASS));
    });

    backdrop.addEventListener('click', function () {
        setDrawer(false);
    });

    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') setDrawer(false);
    });

    // Picking a job or submitting the form is the point of the drawer, so get
    // out of the way once that happens and let the result panel show through.
    sidebar.addEventListener('click', function (event) {
        if (event.target.closest('.job-list [data-job-id], .job-list li, .job-list button')) {
            setDrawer(false);
        }
    });

    var jobForm = document.getElementById('jobForm');
    if (jobForm) {
        jobForm.addEventListener('submit', function () {
            setDrawer(false);
        });
    }

    // The drawer is a phone-only concept; never leave it latched on a wide layout.
    function syncToViewport() {
        if (!media.matches) setDrawer(false);
    }

    if (typeof media.addEventListener === 'function') {
        media.addEventListener('change', syncToViewport);
    } else if (typeof media.addListener === 'function') {
        media.addListener(syncToViewport);
    }
    syncToViewport();

    // -------------------------------------------------------- shared URL

    /*
     * The iOS Share Extension hands the URL over as `?share=<encoded>`.
     * We prefill the field and open the drawer, but deliberately do NOT submit:
     * a job kicks off a long GPU pipeline, and a share gesture is not consent
     * for that. The user still taps the existing button.
     */
    function applySharedUrl() {
        var shared = new URL(window.location.href).searchParams.get('share');
        if (!shared) return;

        var input = document.getElementById('videoUrlInput');
        if (input) {
            input.value = shared;
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
        }

        if (media.matches) setDrawer(true);
        if (input) {
            input.scrollIntoView({ block: 'center' });
            input.focus();
        }

        // Keep it out of the history so a reload does not re-prefill.
        var url = new URL(window.location.href);
        url.searchParams.delete('share');
        window.history.replaceState({}, '', url);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applySharedUrl);
    } else {
        applySharedUrl();
    }
})();
