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
        // Target Chainlit message steps using robust selectors
        const messages = document.querySelectorAll(
            '.step:not([data-ts-processed]), [class*="message"]:not([data-ts-processed])'
        );

        messages.forEach((msg) => {
            // Check if it's a message bubble
            // In Chainlit, user messages and assistant messages usually reside in a container
            const isUserMsg = msg.closest('[class*="user"]') || msg.getAttribute('data-testid')?.includes('user');
            const isAssistantMsg = msg.closest('[class*="assistant"]') || msg.getAttribute('data-testid')?.includes('assistant') || msg.closest('.step') !== null;

            if (isUserMsg || isAssistantMsg) {
                // To avoid placing timestamp in tiny nested divs, place it at the main step container if possible
                const targetEl = msg.querySelector('.message-content') || msg;
                addTimestamp(targetEl);
                msg.setAttribute('data-ts-processed', 'true');
            }
        });
    }

    // Observe DOM changes for dynamically added messages
    const observer = new MutationObserver(function (mutations) {
        let shouldProcess = false;
        for (const m of mutations) {
            if (m.addedNodes.length > 0) {
                for (const node of m.addedNodes) {
                    if (node.nodeType === 1 && (node.className?.includes?.('message') || node.className?.includes?.('step'))) {
                        shouldProcess = true;
                        break;
                    } else if (node.nodeType === 1 && node.querySelector && (node.querySelector('.step') || node.querySelector('[class*="message"]'))) {
                        shouldProcess = true;
                        break;
                    }
                }
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
        // process immediately in case messages are already rendered
        setTimeout(processMessages, 500);
        setInterval(processMessages, 3000); // Fail-safe check
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
