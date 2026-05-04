// Text translation page (WebTranslateV3) — split panel, debounced
// auto-translate. POSTs to /api/translate; pulls the language list from
// /api/languages so it stays in sync with the server's authoritative
// registry.

(() => {
    const FLAGS = {
        de: '🇩🇪', en: '🇬🇧', fr: '🇫🇷', es: '🇪🇸', it: '🇮🇹',
        pl: '🇵🇱', ro: '🇷🇴', hr: '🇭🇷', bg: '🇧🇬', tr: '🇹🇷',
        ru: '🇷🇺', uk: '🇺🇦', hu: '🇭🇺', sr: '🇷🇸', sq: '🇦🇱',
        ar: '🇸🇦', fa: '🇮🇷', nl: '🇳🇱', pt: '🇵🇹', cs: '🇨🇿',
    };

    /**
     * @param {HTMLSelectElement} selectEl
     * @param {Array<{ code: string; label: string }>} options
     * @param {string} defaultCode
     */
    function fillLangSelect(selectEl, options, defaultCode) {
        for (const { code, label } of options) {
            const opt = document.createElement('option');
            opt.value = code;
            opt.textContent = label;
            if (code === defaultCode) opt.selected = true;
            selectEl.appendChild(opt);
        }
    }

    const srcLang = /** @type {HTMLSelectElement} */ (document.getElementById('srcLang'));
    const tgtLang = /** @type {HTMLSelectElement} */ (document.getElementById('tgtLang'));
    const sourceText = /** @type {HTMLTextAreaElement} */ (document.getElementById('sourceText'));
    const printBtn = /** @type {HTMLButtonElement} */ (document.getElementById('printBtn'));
    const clearSrcBtn = /** @type {HTMLButtonElement} */ (document.getElementById('clearSrcBtn'));
    const srcFlag = document.getElementById('srcFlag');
    const tgtFlag = document.getElementById('tgtFlag');
    const result = /** @type {HTMLElement} */ (document.getElementById('result'));
    const errorBanner = /** @type {HTMLElement} */ (document.getElementById('errorBanner'));

    /** @param {string} msg */
    function showError(msg) {
        errorBanner.textContent = msg;
        errorBanner.classList.add('visible');
    }
    function clearError() {
        errorBanner.classList.remove('visible');
    }

    function updateFlags() {
        if (srcFlag) srcFlag.textContent = FLAGS[srcLang.value] || '🌐';
        if (tgtFlag) tgtFlag.textContent = FLAGS[tgtLang.value] || '🌐';
    }

    function setResultEmpty() {
        result.classList.add('empty');
        result.classList.remove('pending');
        result.textContent = 'Hier erscheint die Übersetzung.';
        printBtn.disabled = true;
    }
    function setResultPending() {
        result.classList.remove('empty');
        result.classList.add('pending');
        result.textContent = 'übersetze…';
        printBtn.disabled = true;
    }
    function setResultText(output) {
        result.classList.remove('empty', 'pending');
        result.textContent = output;
        printBtn.disabled = false;
    }

    let inflightController = null;
    let translateSeq = 0;

    async function runTranslate() {
        const text = sourceText.value.trim();
        clearError();
        if (!text) {
            setResultEmpty();
            return;
        }
        if (srcLang.value === tgtLang.value) {
            setResultEmpty();
            showError('Quell- und Zielsprache sind identisch.');
            return;
        }
        const seq = ++translateSeq;
        if (inflightController) inflightController.abort();
        inflightController = new AbortController();
        setResultPending();
        try {
            const resp = await fetch('/api/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    src_lang: srcLang.value,
                    tgt_lang: tgtLang.value,
                }),
                signal: inflightController.signal,
            });
            if (seq !== translateSeq) return;
            if (resp.status === 401) {
                globalThis.location.href = '/login';
                return;
            }
            if (!resp.ok) {
                let detail = '';
                try {
                    detail = (await resp.json()).detail || '';
                } catch {
                    /* ignore */
                }
                const detailSuffix = detail ? `: ${detail}` : '';
                showError(`Übersetzung fehlgeschlagen${detailSuffix}.`);
                setResultEmpty();
                return;
            }
            const data = await resp.json();
            const output = (data.output || '').trim();
            if (output) {
                setResultText(output);
            } else {
                setResultEmpty();
                result.textContent = '(leere Übersetzung)';
            }
        } catch (e) {
            if (e instanceof DOMException && e.name === 'AbortError') return;
            if (seq !== translateSeq) return;
            showError('Netzwerkfehler. Bitte erneut versuchen.');
            setResultEmpty();
        }
    }

    let debounceTimer = null;
    function scheduleTranslate() {
        if (debounceTimer !== null) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            debounceTimer = null;
            runTranslate();
        }, 500);
    }

    sourceText.addEventListener('input', scheduleTranslate);
    srcLang.addEventListener('change', () => {
        updateFlags();
        scheduleTranslate();
    });
    tgtLang.addEventListener('change', () => {
        updateFlags();
        scheduleTranslate();
    });
    clearSrcBtn?.addEventListener('click', () => {
        sourceText.value = '';
        setResultEmpty();
        clearError();
        sourceText.focus();
    });

    // Populate dropdowns from the server's language registry.
    (async () => {
        const resp = await fetch('/api/languages?scope=translate');
        if (!resp.ok) {
            showError('Sprachliste konnte nicht geladen werden.');
            return;
        }
        const options = await resp.json();
        fillLangSelect(srcLang, options, 'de');
        fillLangSelect(tgtLang, options, 'en');
        updateFlags();
    })();

    printBtn.addEventListener('click', () => globalThis.print());

    LinguaGapAuth.requireUser();
    LinguaGapAuth.wireLogoutButton('logoutBtn');
})();
