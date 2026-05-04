// Tiny DOM helpers shared across host.js and viewer.js so neither page
// has to ship its own copy of the same string sanitization.

(() => {
    /**
     * HTML-escape a string for safe insertion via innerHTML. Coerces null /
     * undefined to '' so callers don't need their own guard.
     *
     * @param {unknown} text
     * @returns {string}
     */
    function escapeHtml(text) {
        return String(text == null ? '' : text)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }

    /** @type {any} */ (window).LinguaGapDom = { escapeHtml };
})();
