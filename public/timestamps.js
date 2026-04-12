/**
 * Injects timestamps on chat messages.
 * Uses a MutationObserver to watch for new messages and adds
 * a human-readable timestamp below each one.
 */
(function () {
    'use strict';

    const TIMESTAMP_CLASS = 'msg-timestamp';

    function formatTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function addTimestamp(msgEl) {
        if (msgEl.querySelector('.' + TIMESTAMP_CLASS)) return;

        const ts = document.createElement('span');
        ts.className = TIMESTAMP_CLASS;
        ts.textContent = formatTime(new Date());
        ts.style.cssText =
            'display:block;font-size:0.65rem;opacity:0.45;margin-top:2px;' +
            'font-family:var(--font-sans, inherit);user-select:none;';

        // Append timestamp at the bottom of the message container
        msgEl.appendChild(ts);
    }

    function processMessages() {
        // Target both user and assistant message steps
        const messages = document.querySelectorAll(
            '[class*="message"]:not([data-ts-processed])'
        );

        messages.forEach((msg) => {
            // Only add timestamps to actual chat message bubbles
            const isUserMsg = msg.closest('[class*="user"]') || msg.getAttribute('data-testid')?.includes('user');
            const isAssistantMsg = msg.closest('[class*="assistant"]') || msg.getAttribute('data-testid')?.includes('assistant');

            if (isUserMsg || isAssistantMsg) {
                addTimestamp(msg);
                msg.setAttribute('data-ts-processed', 'true');
            }
        });
    }

    // Observe DOM changes for dynamically added messages
    const observer = new MutationObserver(function (mutations) {
        let shouldProcess = false;
        for (const m of mutations) {
            if (m.addedNodes.length > 0) {
                shouldProcess = true;
                break;
            }
        }
        if (shouldProcess) {
            requestAnimationFrame(processMessages);
        }
    });

    // Start observing once the DOM is ready
    function init() {
        const target = document.getElementById('root') || document.body;
        observer.observe(target, { childList: true, subtree: true });
        processMessages();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
