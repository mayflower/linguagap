// Host page main script — recording control, WebSocket transport,
// segment rendering, translation display, QR sharing, summary download.

(() => {
    // Auth: load branding (admin link) + wire logout via the shared guard.
    LinguaGapAuth.requireUser().then((user) => {
        if (user?.is_admin) {
            const adminLink = document.getElementById('adminLink');
            if (adminLink) adminLink.style.display = '';
        }
    });
    LinguaGapAuth.wireLogoutButton('logoutBtn');

    // Translations for UI internationalization
    const TRANSLATIONS = {
        de: {
            title: 'Echtzeit-Sprachübersetzung',
            subtitle: 'Echtzeit-Transkription & Übersetzung',
            defaultMic: 'Standard-Mikrofon',
            selectLanguage: '-- Sprache wählen --',
            selectLanguageFirst: 'Bitte wählen Sie zuerst eine Sprache aus',
            startRecording: 'Gespräch starten',
            stopRecording: 'Gespräch beenden',
            mute: 'Stumm',
            unmute: 'Ton an',
            clear: 'Löschen',
            statusReady: 'Klicken Sie auf "Gespräch starten"',
            foreignLang: 'Fremdsprache',
            german: 'Deutsch',
            shareTitle: 'Live mitlesen',
            shareSubtitle: 'Gäste scannen den Code, um auf dem Handy mitzulesen.',
            statusMicRequest: 'Mikrofonzugriff wird angefordert...',
            statusConnecting: 'Verbindung zum Server...',
            statusRecording: 'Verbunden - Aufnahme läuft...',
            statusSpeaking: 'Aufnahme... (spricht: {lang})',
            statusStopped: 'Aufnahme beendet',
            statusConnError: 'Verbindungsfehler',
            statusConnClosed: 'Verbindung geschlossen',
            errorHttps: 'Fehler: Mikrofon erfordert HTTPS oder localhost',
            errorBrowser: 'Fehler: Browser unterstützt keinen Mikrofonzugriff',
            errorMicDenied: 'Mikrofonzugriff verweigert',
            speaker: 'Sprecher {n}',
            translationFailed: 'Übersetzung fehlgeschlagen',
            pttLabel: 'Sprechtaste',
            pttHint: 'Leertaste halten zum Sprechen',
            guestSpeaking: 'Gast spricht...',
            translating: 'Übersetzung läuft…',
            statusFinalizing: 'Gespräch wird abgeschlossen, bitte warten…',
            downloadTranscript: 'Gesprächsprotokoll herunterladen',
            downloadFailed: 'Download fehlgeschlagen',
            hostTranscriptLabel: 'Protokoll',
            hostTranscriptTitle:
                'Protokoll erstellen (Download am Sitzungsende, Zustimmung des Gasts vorausgesetzt)',
        },
        en: {
            title: 'Real-time Speech Translation',
            subtitle: 'Real-time Speech Transcription & Translation',
            defaultMic: 'Default Microphone',
            selectLanguage: '-- Select Language --',
            selectLanguageFirst: 'Please select a language first',
            startRecording: 'Start Recording',
            stopRecording: 'Stop Recording',
            mute: 'Mute',
            unmute: 'Unmute',
            clear: 'Clear',
            statusReady: 'Click "Start Recording" to begin',
            foreignLang: 'Foreign Language',
            german: 'German',
            shareTitle: 'Read along live',
            shareSubtitle: 'Guests scan the code to read along on their phone.',
            statusMicRequest: 'Requesting microphone access...',
            statusConnecting: 'Connecting to server...',
            statusRecording: 'Connected - Recording...',
            statusSpeaking: 'Recording... (speaking: {lang})',
            statusStopped: 'Recording stopped',
            statusConnError: 'Connection error',
            statusConnClosed: 'Connection closed',
            errorHttps: 'Error: Microphone requires HTTPS or localhost access',
            errorBrowser: 'Error: Browser does not support microphone access',
            errorMicDenied: 'Microphone access denied',
            speaker: 'Speaker {n}',
            translationFailed: 'Translation failed',
            pttLabel: 'Push-to-Talk',
            pttHint: 'Hold spacebar to speak',
            guestSpeaking: 'Guest is speaking...',
            translating: 'Translation in progress…',
            statusFinalizing: 'Finishing up, please wait…',
            downloadTranscript: 'Download conversation protocol',
            downloadFailed: 'Download failed',
            hostTranscriptLabel: 'Protocol',
            hostTranscriptTitle:
                'Create a protocol (downloadable after the session, requires guest consent)',
        },
    };

    // Current UI language (default: German)
    let currentUiLang = localStorage.getItem('uiLang') || 'de';

    /**
     * Translation lookup — delegates to the shared resolver in
     * static/js/lib/i18n.js so the fallback chain (current → en → de → key)
     * stays consistent with viewer.js.
     *
     * @param {string} key
     * @param {Record<string, string | number>} [replacements]
     */
    function t(key, replacements = {}) {
        return LinguaGapI18n.t([TRANSLATIONS], currentUiLang, key, replacements);
    }

    const startBtn = /** @type {HTMLButtonElement} */ (document.getElementById('startBtn'));
    const muteBtn = /** @type {HTMLButtonElement} */ (document.getElementById('muteBtn'));
    const clearBtn = /** @type {HTMLButtonElement} */ (document.getElementById('clearBtn'));
    const languageSelect = /** @type {HTMLSelectElement} */ (
        document.getElementById('languageSelect')
    );
    const audioInputSelect = /** @type {HTMLSelectElement} */ (
        document.getElementById('audioInputSelect')
    );
    const statusEl = document.getElementById('status');
    const statusTextEl = document.getElementById('statusText');
    if (!statusEl || !statusTextEl) {
        console.error('Status elements missing from DOM — UI feedback will be broken');
    }
    const qrCodeEl = document.getElementById('qrCode');
    const viewerUrlText = document.getElementById('viewerUrlText');
    const sgaThread = document.getElementById('sgaThread');
    const sgaStage = document.getElementById('sgaStage');
    const railToggleBtn = document.getElementById('railToggleBtn');
    const heroFromLang = document.getElementById('heroFromLang');
    const heroToLang = document.getElementById('heroToLang');
    const leftLangLabel = document.getElementById('leftLangLabel');
    const rightLangLabel = document.getElementById('rightLangLabel');
    const uiLangSelect = /** @type {HTMLSelectElement} */ (document.getElementById('uiLangSelect'));
    const subtitleEl = document.getElementById('subtitle');
    const pttToggle = /** @type {HTMLInputElement} */ (document.getElementById('pttToggle'));
    const pttLabelText = document.getElementById('pttLabelText');
    const pttToggleLabel = document.getElementById('pttToggleLabel');
    const hostTranscriptToggle = /** @type {HTMLInputElement} */ (
        document.getElementById('hostTranscriptToggle')
    );
    const hostTranscriptLabelText = document.getElementById('hostTranscriptLabelText');
    const hostTranscriptLabel = document.getElementById('hostTranscriptLabel');
    let hostTranscriptConsent = false;
    const viewerSpeakingIndicator = document.getElementById('viewerSpeakingIndicator');
    const viewerSpeakingText = document.getElementById('viewerSpeakingText');
    const translatingIndicator = document.getElementById('translatingIndicator');
    const translatingText = document.getElementById('translatingText');
    // Map<segmentId, enqueuedAtMs>. Pending translations get evicted after
    // PENDING_TRANSLATION_TIMEOUT_MS so the "Übersetzung läuft" pill can't
    // stick forever when a server-side MT call hangs without emitting a
    // translation_error.
    const pendingTranslations = new Map();
    const PENDING_TRANSLATION_TIMEOUT_MS = 60 * 1000;

    function prunePendingTranslations() {
        const cutoff = Date.now() - PENDING_TRANSLATION_TIMEOUT_MS;
        for (const [id, ts] of pendingTranslations) {
            if (ts < cutoff) pendingTranslations.delete(id);
        }
    }

    function updateTranslatingIndicator() {
        prunePendingTranslations();
        translatingIndicator.classList.toggle('visible', pendingTranslations.size > 0);
    }

    setInterval(() => {
        if (pendingTranslations.size > 0) updateTranslatingIndicator();
    }, 5000);

    function refreshPendingFromSegments(segments) {
        // Host target is German. A segment is pending iff it is finalized,
        // its source language is non-German, and no 'de' translation has
        // arrived yet.
        const now = Date.now();
        for (const seg of segments || []) {
            const needsGerman = seg.final && seg.src_lang && seg.src_lang !== 'de';
            const hasGerman = seg.translations?.de;
            if (needsGerman && !hasGerman) {
                if (!pendingTranslations.has(seg.id)) pendingTranslations.set(seg.id, now);
            } else {
                pendingTranslations.delete(seg.id);
            }
        }
        updateTranslatingIndicator();
    }

    // Display name lookup. Always includes German (for transcript
    // labels, foreign-lang fallback). Filled from /api/languages on
    // page load — see populateLanguages() below. Other code reads from
    // it lazily via `LANG_NAMES[code] || code`, so it's safe before fetch
    // resolves.
    const LANG_NAMES = { de: 'German' };

    async function populateLanguages() {
        const resp = await fetch('/api/languages');
        if (!resp.ok) return;
        const langs = await resp.json();
        const select = document.getElementById('languageSelect');
        for (const { code, label } of langs) {
            LANG_NAMES[code] = label;
            const opt = document.createElement('option');
            opt.value = code;
            opt.textContent = label;
            select.appendChild(opt);
        }
    }
    populateLanguages();

    // Authoritative copy of the last segments list from the server,
    // kept in sync by the `segments` and `translation` handlers. Used
    // to build the transcript export at session end.
    let allSegments = [];
    let viewerConsentedTranscript = false;
    let isStoppingRecording = false;
    let stopWatchdogTimer = null;

    // Hard upper bound on how long the host waits for the server to flush
    // the final segments + summary. 60s matches _handle_request_summary's
    // translation_queue.join timeout on the server; +15s budget for the
    // summariser. If we hit it, the user gets their download (from local
    // state) and is unblocked.
    const STOP_WATCHDOG_MS = 75 * 1000;

    function updateAllSegmentsFromMessage(segs) {
        allSegments = Array.isArray(segs)
            ? segs.map((s) => ({ ...s, translations: { ...s.translations } }))
            : [];
    }

    function applyTranslationToAllSegments(segmentId, tgtLang, text) {
        const seg = allSegments.find((s) => s.id === segmentId);
        if (!seg) return;
        if (!seg.translations) seg.translations = {};
        seg.translations[tgtLang] = text;
    }

    function maybeShowTranscriptDownload() {
        const bar = document.getElementById('transcriptDownloadBar');
        const btn = document.getElementById('transcriptDownloadBtn');
        if (!bar || !btn) return;
        const hasFinal = allSegments.some((s) => s.final);
        // Both parties must consent: the host opts in via the controls-bar
        // toggle, the viewer via the checkbox in their privacy popup.
        if (!hostTranscriptConsent || !viewerConsentedTranscript || !foreignLang || !hasFinal)
            return;
        btn.textContent = t('downloadTranscript');
        bar.classList.add('visible');
    }

    function downloadTranscriptFile() {
        try {
            TranscriptExport.download({
                segments: allSegments,
                foreignLang: foreignLang,
                source: 'host',
                langNames: LANG_NAMES,
            });
        } catch (e) {
            console.error('Transcript download failed:', e);
            setStatus(t('downloadFailed'), 'error');
        }
    }

    let isRecording = false;
    let isMuted = false;
    let foreignLang = null; // The detected/selected foreign (non-German) language
    let ws = null;
    let audioContext = null;
    let mediaStream = null;
    let processor = null;
    let sessionCleared = false; // Guard flag to prevent renderSegments after clear
    let pttMode = false;
    let spaceHeld = false;

    const SAMPLE_RATE = 16000;

    // Generate session token on page load
    function generateToken() {
        // Use crypto.randomUUID if available, otherwise fallback
        if (crypto.randomUUID) {
            return crypto.randomUUID().replaceAll('-', '');
        }
        // Fallback for older browsers
        return Array.from(crypto.getRandomValues(new Uint8Array(16)))
            .map((b) => b.toString(16).padStart(2, '0'))
            .join('');
    }

    const sessionToken = generateToken();

    function generateQRCode(url) {
        qrCodeEl.innerHTML = '';
        const qr = qrcode(0, 'M');
        qr.addData(url);
        qr.make();

        // Create canvas for better quality
        const moduleCount = qr.getModuleCount();
        const cellSize = 5;
        const margin = 3 * cellSize;
        const size = moduleCount * cellSize + margin * 2;

        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');

        // White background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, size, size);

        // Draw QR modules
        ctx.fillStyle = '#000000';
        for (let row = 0; row < moduleCount; row++) {
            for (let col = 0; col < moduleCount; col++) {
                if (qr.isDark(row, col)) {
                    ctx.fillRect(
                        col * cellSize + margin,
                        row * cellSize + margin,
                        cellSize,
                        cellSize
                    );
                }
            }
        }

        qrCodeEl.appendChild(canvas);
    }

    // Generate and display QR code immediately on page load
    function initQRCode() {
        const viewerUrl = `${globalThis.location.origin}/viewer/${sessionToken}`;
        generateQRCode(viewerUrl);
        viewerUrlText.textContent = viewerUrl;
    }

    // Initialize QR code on page load
    initQRCode();

    function applyControlButtonTranslations() {
        if (startBtn) startBtn.textContent = isRecording ? t('stopRecording') : t('startRecording');
        if (muteBtn) muteBtn.textContent = isMuted ? t('unmute') : t('mute');
        if (clearBtn) clearBtn.textContent = t('clear');
    }

    function applyDropdownPlaceholderTranslations() {
        const defaultMicOption = audioInputSelect?.querySelector('option[value=""]');
        if (defaultMicOption) defaultMicOption.textContent = t('defaultMic');

        const selectLanguageOption = languageSelect?.querySelector('option[value=""]');
        if (selectLanguageOption) selectLanguageOption.textContent = t('selectLanguage');
    }

    function applyStatusTranslation() {
        // Only swap the ready text; leave other status messages alone.
        if (
            statusTextEl &&
            (statusTextEl.textContent.includes('Start') ||
                statusTextEl.textContent.includes('starten'))
        ) {
            statusTextEl.textContent = t('statusReady');
        }
    }

    function applyPaneLabelTranslations() {
        // The lang-pair chip uses 2-letter codes, not full names.
        if (!foreignLang && leftLangLabel) leftLangLabel.textContent = '—';
        if (rightLangLabel) rightLangLabel.textContent = 'DE';
    }

    function applyQrSidebarTranslations() {
        const qrTitle = document.getElementById('qrTitle');
        const qrSubtitle = document.getElementById('qrSubtitle');
        if (qrTitle) qrTitle.textContent = t('shareTitle');
        if (qrSubtitle) qrSubtitle.textContent = t('shareSubtitle');
    }

    function applyToggleTranslations() {
        if (pttLabelText) pttLabelText.textContent = t('pttLabel');
        if (hostTranscriptLabelText) hostTranscriptLabelText.textContent = t('hostTranscriptLabel');
        if (hostTranscriptLabel) hostTranscriptLabel.title = t('hostTranscriptTitle');
        if (viewerSpeakingText) {
            viewerSpeakingText.textContent = `● ${t('guestSpeaking').replace(/[…\.]+$/, '')}`;
        }
        if (translatingText) translatingText.textContent = t('translating');
    }

    // Apply translations to all UI elements
    function applyTranslations() {
        // The host page now hides the subtitle/crumb (top bar shows just
        // "Synia SGA"), but keep the element + text so screen readers can
        // announce the page purpose.
        if (subtitleEl) subtitleEl.textContent = t('subtitle');
        applyDropdownPlaceholderTranslations();
        applyControlButtonTranslations();
        applyStatusTranslation();
        applyPaneLabelTranslations();
        applyQrSidebarTranslations();
        applyToggleTranslations();
    }

    // Initialize UI language
    uiLangSelect.value = currentUiLang;
    applyTranslations();

    document
        .getElementById('transcriptDownloadBtn')
        .addEventListener('click', downloadTranscriptFile);

    // Handle UI language change
    uiLangSelect.addEventListener('change', (e) => {
        currentUiLang = /** @type {HTMLSelectElement} */ (e.target).value;
        localStorage.setItem('uiLang', currentUiLang);
        applyTranslations();
    });

    // Host opts in to the bilingual transcript. The toggle is the gate
    // that lets the server ask the viewer for consent; without it, the
    // viewer sees no banner and the download button never surfaces.
    hostTranscriptToggle.addEventListener('change', () => {
        hostTranscriptConsent = hostTranscriptToggle.checked;
        if (ws?.readyState === WebSocket.OPEN) {
            ws.send(
                JSON.stringify({
                    type: 'host_transcript_requested',
                    enabled: hostTranscriptConsent,
                })
            );
        }
        if (!hostTranscriptConsent) {
            // Re-toggle later must re-prompt the viewer explicitly.
            viewerConsentedTranscript = false;
        }
    });

    // PTT toggle handler
    pttToggle.addEventListener('change', () => {
        pttMode = pttToggle.checked;
        // Notify server
        if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ptt_mode', enabled: pttMode }));
        }
        if (pttMode) {
            // PTT on: default to muted, hide mute button
            isMuted = true;
            muteBtn.style.display = 'none';
            if (isRecording) {
                setStatus(t('pttHint'), 'connected');
            }
        } else {
            // PTT off: if mid-speech, release
            if (spaceHeld) {
                spaceHeld = false;
                if (ws?.readyState === WebSocket.OPEN) {
                    ws.send(
                        JSON.stringify({
                            type: 'speaking_state',
                            party: 'host',
                            speaking: false,
                        })
                    );
                }
            }
            isMuted = false;
            if (isRecording) {
                muteBtn.style.display = '';
                muteBtn.textContent = t('mute');
                muteBtn.classList.remove('muted');
                setStatus(t('statusRecording'), 'connected');
            }
        }
    });

    // Spacebar PTT handler
    document.addEventListener('keydown', (e) => {
        if (!pttMode || !isRecording) return;
        if (e.code !== 'Space') return;
        if (e.repeat) return;
        e.preventDefault();
        spaceHeld = true;
        isMuted = false;
        startBtn.classList.add('recording');
        setStatus(t('statusRecording'), 'connected');
        if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'speaking_state', party: 'host', speaking: true }));
        }
    });

    document.addEventListener('keyup', (e) => {
        if (!pttMode || !isRecording) return;
        if (e.code !== 'Space') return;
        e.preventDefault();
        spaceHeld = false;
        isMuted = true;
        startBtn.classList.remove('recording');
        setStatus(t('pttHint'), 'connected');
        if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'speaking_state', party: 'host', speaking: false }));
        }
    });

    // Release PTT if tab loses focus or becomes hidden
    function releasePTT() {
        if (!pttMode || !spaceHeld || !isRecording) return;
        spaceHeld = false;
        isMuted = true;
        startBtn.classList.remove('recording');
        setStatus(t('pttHint'), 'connected');
        if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'speaking_state', party: 'host', speaking: false }));
        }
    }
    window.addEventListener('blur', releasePTT);
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) releasePTT();
    });

    async function enumerateDevices() {
        try {
            // Request permission first to get device labels
            const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            tempStream.getTracks().forEach((track) => track.stop());

            const devices = await navigator.mediaDevices.enumerateDevices();

            // Clear existing options (except default)
            audioInputSelect.innerHTML = `<option value="">${t('defaultMic')}</option>`;

            devices.forEach((device) => {
                if (device.kind === 'audioinput') {
                    const option = document.createElement('option');
                    option.value = device.deviceId;
                    option.textContent =
                        device.label || `Microphone (${device.deviceId.slice(0, 8)}...)`;
                    audioInputSelect.appendChild(option);
                }
            });
        } catch (error) {
            console.error('Error enumerating devices:', error);
        }
    }

    // Enumerate devices on page load
    enumerateDevices();
    navigator.mediaDevices.addEventListener('devicechange', enumerateDevices);

    function setStatus(msg, type = '') {
        // #status holds sibling elements (text + dualChannelBadge); writing
        // textContent on the parent would wipe the siblings, so update only
        // the text span. Warn loudly if the span is missing rather than
        // silently dropping every status message in the app.
        if (statusTextEl) {
            statusTextEl.textContent = msg;
        } else {
            console.warn('setStatus called but statusText element is missing:', msg);
        }
        if (statusEl) {
            statusEl.className = `sga-status-row ${type}`;
        }
    }

    const escapeHtml = LinguaGapDom.escapeHtml;

    function computeSegmentTexts(seg, isGermanSpeaker) {
        const segLang = seg.src_lang;
        const translations = seg.translations || {};
        if (isGermanSpeaker) {
            return {
                leftText: translations[foreignLang] || '...',
                rightText: segLang === 'de' ? seg.src : translations.de || '...',
            };
        }
        const foreignSourceReliable = !!foreignLang && segLang === foreignLang;
        return {
            leftText: translations[foreignLang] || (foreignSourceReliable ? seg.src : '...'),
            rightText: translations.de || (segLang === 'de' ? seg.src : '...'),
        };
    }

    function speakerLabelFor(speakerId) {
        if (!speakerId || typeof speakerId !== 'string') return '';
        const speakerNum = Number.parseInt(speakerId.replace('SPEAKER_', ''), 10);
        if (Number.isNaN(speakerNum)) return '';
        return t('speaker', { n: speakerNum + 1 });
    }

    // Render a single ConversationTurn — primary bubble in the host's
    // language, secondary line under it in the foreign language. Low-conf
    // tokens wrapped in [brackets] get a wavy underline.
    function renderContent(target, text, onTeal) {
        target.textContent = '';
        if (!text) {
            const placeholder = document.createElement('span');
            placeholder.style.opacity = '0.5';
            placeholder.textContent = '…';
            target.appendChild(placeholder);
            return;
        }
        const cls = onTeal ? 'sga-low-conf-on-teal' : 'sga-low-conf';
        const parts = text.split(/(\[[^\]]+\])/);
        for (const part of parts) {
            if (!part) continue;
            if (part.startsWith('[') && part.endsWith(']')) {
                const span = document.createElement('span');
                span.className = cls;
                span.textContent = part.slice(1, -1);
                target.appendChild(span);
            } else {
                target.appendChild(document.createTextNode(part));
            }
        }
    }

    function buildSecondaryInner(turn, code, text, pending) {
        const sec = document.createElement('div');
        sec.className = pending ? 'turn-secondary pending' : 'turn-secondary';

        if (pending) {
            const dots = document.createElement('span');
            dots.className = 'three-dots';
            for (let i = 0; i < 3; i++) dots.appendChild(document.createElement('span'));
            sec.appendChild(dots);
            sec.appendChild(document.createTextNode(` übersetze ${code}…`));
            return sec;
        }

        const inner = document.createElement('div');
        inner.className = 'turn-secondary-inner';
        const tag = document.createElement('span');
        tag.className = 'lang-tag';
        tag.textContent = code;
        inner.appendChild(tag);
        const content = document.createElement('span');
        content.className = 'bubble-content';
        renderContent(content, text, false);
        inner.appendChild(content);
        sec.appendChild(inner);
        return sec;
    }

    function buildTurn(seg) {
        const speakerRole = seg.speaker_role || (seg.src_lang === 'de' ? 'german' : 'foreign');
        const isGermanSpeaker = speakerRole === 'german';
        // Host's view: primary bubble is always German. Foreign-spoken
        // turns are displayed translated; German-spoken turns display the
        // original.
        const turnSide = isGermanSpeaker ? 'right' : 'left';
        const speakerLabel = speakerLabelFor(seg.speaker_id);
        const { leftText, rightText } = computeSegmentTexts(seg, isGermanSpeaker);
        const primaryText = rightText; // German always primary on host
        const secondaryText = leftText; // Foreign always secondary
        const inProgress = !seg.final;
        const translated = !isGermanSpeaker;
        const foreignCode = (foreignLang || '').toUpperCase() || '—';

        const turn = document.createElement('div');
        turn.className = `turn turn-${turnSide}`;
        turn.dataset.id = seg.id;
        turn.dataset.srcLang = seg.src_lang;
        turn.dataset.speakerRole = speakerRole;
        if (seg.speaker_id) turn.dataset.speakerId = seg.speaker_id;

        const inner = document.createElement('div');
        inner.className = 'turn-inner';

        const meta = document.createElement('div');
        meta.className = 'turn-meta';
        if (speakerLabel) meta.appendChild(document.createTextNode(speakerLabel));
        if (translated) {
            const badge = document.createElement('span');
            badge.className = 'badge-mt';
            badge.textContent = `aus ${foreignCode} übersetzt`;
            meta.appendChild(badge);
        }
        inner.appendChild(meta);

        const primary = document.createElement('div');
        primary.className = 'bubble-primary';
        const content = document.createElement('span');
        content.className = 'bubble-content';
        renderContent(
            content,
            primaryText && primaryText !== '...' ? primaryText : '',
            isGermanSpeaker
        );
        primary.appendChild(content);
        if (inProgress) {
            const caret = document.createElement('span');
            caret.className = 'bubble-caret';
            primary.appendChild(caret);
        }
        inner.appendChild(primary);

        const pending = inProgress && (!secondaryText || secondaryText === '...');
        inner.appendChild(
            buildSecondaryInner(turn, foreignCode, pending ? '' : secondaryText, pending)
        );

        turn.appendChild(inner);
        return turn;
    }

    function flagFor(lang) {
        const FLAGS = {
            de: '🇩🇪',
            en: '🇬🇧',
            fr: '🇫🇷',
            es: '🇪🇸',
            it: '🇮🇹',
            pl: '🇵🇱',
            ro: '🇷🇴',
            hr: '🇭🇷',
            bg: '🇧🇬',
            tr: '🇹🇷',
            ru: '🇷🇺',
            uk: '🇺🇦',
            hu: '🇭🇺',
            sr: '🇷🇸',
            sq: '🇦🇱',
            ar: '🇸🇦',
            fa: '🇮🇷',
            nl: '🇳🇱',
            pt: '🇵🇹',
            cs: '🇨🇿',
        };
        return FLAGS[lang] || '🌐';
    }

    function updateLangPair() {
        const code = (foreignLang || '').toUpperCase();
        if (leftLangLabel) leftLangLabel.textContent = code || '—';
        if (heroFromLang) {
            heroFromLang.textContent = '';
            if (code) {
                heroFromLang.appendChild(document.createTextNode(`${flagFor(foreignLang)} `));
                const span = document.createElement('span');
                span.textContent = LANG_NAMES[foreignLang] || code;
                heroFromLang.appendChild(span);
            }
        }
        if (heroToLang) {
            heroToLang.textContent = '';
            heroToLang.appendChild(document.createTextNode('🇩🇪 '));
            const span = document.createElement('span');
            span.textContent = t('german');
            heroToLang.appendChild(span);
        }
    }

    function renderSegments(segments, serverForeignLang) {
        if (serverForeignLang && !foreignLang) {
            foreignLang = serverForeignLang;
            updateLangPair();
        }

        sgaThread.textContent = '';
        segments.forEach((seg) => sgaThread.appendChild(buildTurn(seg)));

        requestAnimationFrame(() => {
            sgaThread.scrollTop = sgaThread.scrollHeight;
        });
    }

    function updateBubbleContent(turn, newText, where) {
        const isPrimary = where === 'primary';
        const onTeal = isPrimary && turn.classList.contains('turn-right');
        let target;
        if (isPrimary) {
            target = turn.querySelector('.bubble-primary .bubble-content');
        } else {
            ensureSecondaryShell(turn);
            target = turn.querySelector('.turn-secondary-inner .bubble-content');
        }
        if (target) renderContent(target, newText, onTeal);
    }

    function applyFailed(turn, where) {
        const isPrimary = where === 'primary';
        const target = turn.querySelector(isPrimary ? '.bubble-primary' : '.turn-secondary-inner');
        if (!target) return;
        target.classList.add('failed');
        const content = target.querySelector('.bubble-content');
        if (content && (!content.textContent || content.textContent === '…')) {
            content.textContent = `✗ ${t('translationFailed') || 'translation failed'}`;
        }
    }

    function ensureSecondaryShell(turn) {
        const sec = turn.querySelector('.turn-secondary');
        if (!sec) return;
        if (sec.querySelector('.turn-secondary-inner')) return;
        const code = (foreignLang || '').toUpperCase() || '—';
        const replacement = buildSecondaryInner(turn, code, '', false);
        sec.replaceWith(replacement);
    }

    function updateTranslation(segmentId, tgtLang, text) {
        const turn = /** @type {HTMLElement | null} */ (
            sgaThread.querySelector(`.turn[data-id="${segmentId}"]`)
        );
        if (!turn) return;

        const speakerRole =
            turn.dataset.speakerRole || (turn.dataset.srcLang === 'de' ? 'german' : 'foreign');
        const isGermanSpeaker = speakerRole === 'german';

        if (isGermanSpeaker) {
            if (foreignLang && tgtLang === foreignLang) {
                updateBubbleContent(turn, text, 'secondary');
            }
        } else if (tgtLang === 'de') {
            updateBubbleContent(turn, text, 'primary');
        } else if (foreignLang && tgtLang === foreignLang) {
            updateBubbleContent(turn, text, 'secondary');
        }
    }

    function markTranslationFailed(segmentId, tgtLang) {
        const turn = /** @type {HTMLElement | null} */ (
            sgaThread.querySelector(`.turn[data-id="${segmentId}"]`)
        );
        if (!turn) return;

        const speakerRole =
            turn.dataset.speakerRole || (turn.dataset.srcLang === 'de' ? 'german' : 'foreign');
        const isGermanSpeaker = speakerRole === 'german';

        if (isGermanSpeaker) {
            if (!tgtLang || tgtLang === foreignLang) {
                ensureSecondaryShell(turn);
                applyFailed(turn, 'secondary');
            }
        } else {
            if (!tgtLang || tgtLang === 'de') applyFailed(turn, 'primary');
            if (tgtLang === foreignLang && foreignLang) {
                ensureSecondaryShell(turn);
                applyFailed(turn, 'secondary');
            }
        }
    }

    const { downsampleBuffer, floatTo16BitPCM } = LinguaGapAudio;

    async function startRecording() {
        // Reset the cleared flag when starting a new session
        sessionCleared = false;
        // Reset transcript export state for the new session
        allSegments = [];
        viewerConsentedTranscript = false;
        isStoppingRecording = false;
        const dlBar = document.getElementById('transcriptDownloadBar');
        if (dlBar) dlBar.classList.remove('visible');

        // Check for secure context (HTTPS or localhost)
        if (!globalThis.isSecureContext) {
            setStatus(t('errorHttps'), 'error');
            alert(
                `Microphone access requires a secure context.\n\nOptions:\n1. Access via http://localhost:8000\n2. Use SSH tunnel: ssh -L 8000:localhost:8000 user@server\n3. Launch Chrome with: --unsafely-treat-insecure-origin-as-secure=http://${globalThis.location.host}`
            );
            return;
        }

        // Check if getUserMedia is available
        if (!navigator.mediaDevices?.getUserMedia) {
            setStatus(t('errorBrowser'), 'error');
            return;
        }

        try {
            setStatus(t('statusMicRequest'), '');

            const audioConstraints = {
                channelCount: 1,
                sampleRate: { ideal: 48000 },
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                // Legacy Chrome constraints that often help with aggressive room echo
                googEchoCancellation: true,
                googNoiseSuppression: true,
                googHighpassFilter: true,
            };

            // Use selected input device if specified
            const selectedInputId = audioInputSelect.value;
            if (selectedInputId) {
                audioConstraints.deviceId = { exact: selectedInputId };
            }

            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: audioConstraints,
            });

            audioContext = new AudioContext({ sampleRate: 48000 });
            const source = audioContext.createMediaStreamSource(mediaStream);

            await audioContext.audioWorklet.addModule('/static/js/audio_capture_worklet.js');
            processor = new AudioWorkletNode(audioContext, 'audio-capture-processor');

            const wsProtocol = globalThis.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${globalThis.location.host}/ws`;

            setStatus(t('statusConnecting'), '');
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                // Keep "Connecting..." status until config_ack is received

                // Set foreign language from selection if not "auto" or "de"
                const selectedLang = languageSelect.value;
                if (selectedLang !== 'auto' && selectedLang !== 'de') {
                    foreignLang = selectedLang;
                    updateLangPair();
                }

                const config = {
                    type: 'config',
                    sample_rate: SAMPLE_RATE,
                    src_lang: 'de',
                    foreign_lang:
                        selectedLang !== 'auto' && selectedLang !== 'de' ? selectedLang : null,
                    token: sessionToken,
                    // Carry the pre-start PTT toggle into the config so the
                    // server can activate with the right mode and pending
                    // viewers learn about it in the session_active
                    // broadcast (no race with a later ptt_mode message).
                    ptt_mode: pttMode,
                };
                ws.send(JSON.stringify(config));

                source.connect(processor);
                processor.connect(audioContext.destination);
            };

            const handleConfigAck = (data) => {
                console.log('Config acknowledged:', data.status);
                if (pttMode) {
                    ws.send(JSON.stringify({ type: 'ptt_mode', enabled: true }));
                    setStatus(t('pttHint'), 'connected');
                } else {
                    setStatus(t('statusRecording'), 'connected');
                }
                if (hostTranscriptConsent) {
                    ws.send(
                        JSON.stringify({
                            type: 'host_transcript_requested',
                            enabled: true,
                        })
                    );
                }
            };

            const handleSegments = (data) => {
                console.log('Calling renderSegments with', data.segments.length, 'segments');
                updateAllSegmentsFromMessage(data.segments);
                renderSegments(data.segments, data.foreign_lang);
                refreshPendingFromSegments(data.segments);
                console.log('renderSegments completed');
                const badge = document.getElementById('dualChannelBadge');
                if (badge) badge.classList.toggle('active', !!data.dual_channel);
                if (data.src_lang && data.src_lang !== 'unknown') {
                    const langName = LANG_NAMES[data.src_lang] || data.src_lang;
                    setStatus(t('statusSpeaking', { lang: langName }), 'connected');
                }
            };

            const handleTranslation = (data) => {
                applyTranslationToAllSegments(data.segment_id, data.tgt_lang, data.text);
                updateTranslation(data.segment_id, data.tgt_lang, data.text);
                if (data.tgt_lang === 'de') {
                    pendingTranslations.delete(data.segment_id);
                    updateTranslatingIndicator();
                }
            };

            const handleTranslationError = (data) => {
                console.error('Translation failed for segment', data.segment_id, data.error);
                markTranslationFailed(data.segment_id, data.tgt_lang);
                if (data.tgt_lang === 'de') {
                    pendingTranslations.delete(data.segment_id);
                    updateTranslatingIndicator();
                }
            };

            const handleSpeakingState = (data) => {
                if (data.party !== 'viewer') return;
                viewerSpeakingIndicator.classList.toggle('live', !!data.speaking);
                // In PTT mode the status text latches onto the last
                // "spricht: XX" from a segments message. Reset it when the
                // guest stops so the host doesn't see a stale attribution.
                if (!data.speaking && isRecording) {
                    setStatus(t(pttMode ? 'pttHint' : 'statusRecording'), 'connected');
                }
            };

            const dispatchMessage = (data) => {
                switch (data.type) {
                    case 'config_ack':
                        handleConfigAck(data);
                        return;
                    case 'error':
                        console.error('Server error:', data.message);
                        setStatus(data.message || t('statusConnError'), 'error');
                        return;
                    case 'segments':
                        if (data.segments) handleSegments(data);
                        return;
                    case 'translation':
                        handleTranslation(data);
                        return;
                    case 'translation_error':
                        handleTranslationError(data);
                        return;
                    case 'transcript_consent':
                        viewerConsentedTranscript = !!data.enabled;
                        return;
                    case 'speaking_state':
                        handleSpeakingState(data);
                        return;
                    default:
                }
            };

            ws.onmessage = (event) => {
                if (sessionCleared) {
                    console.log('WS message ignored (session cleared)');
                    return;
                }
                try {
                    const data = JSON.parse(event.data);
                    console.log('WS message:', data.type, data.segments?.length || 0);
                    dispatchMessage(data);
                } catch (e) {
                    console.error('WebSocket message error:', e, e.stack);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setStatus(t('statusConnError'), 'error');
            };

            ws.onclose = () => {
                if (isRecording) {
                    setStatus(t('statusConnClosed'), 'error');
                }
                // Server closes the WS after drain + summary. Surface the
                // transcript download at that point if the viewer consented.
                if (isStoppingRecording) {
                    finalizeAfterStop();
                } else {
                    pendingTranslations.clear();
                    updateTranslatingIndicator();
                }
                ws = null;
            };

            let frameCount = 0;
            processor.port.onmessage = (e) => {
                if (isMuted) return;
                if (ws?.readyState !== WebSocket.OPEN) return;
                const inputData = /** @type {Float32Array} */ (e.data);

                // Debug: log audio levels every ~50 ms (one chunk = 128 samples).
                frameCount++;
                if (frameCount % 50 === 1) {
                    const rms = Math.sqrt(
                        inputData.reduce((sum, x) => sum + x * x, 0) / inputData.length
                    );
                    const max = Math.max(...inputData.map(Math.abs));
                    console.log(
                        `Audio frame ${frameCount}: rms=${rms.toFixed(4)}, max=${max.toFixed(4)}, len=${inputData.length}`
                    );
                }

                const downsampled = downsampleBuffer(
                    inputData,
                    audioContext.sampleRate,
                    SAMPLE_RATE
                );
                const pcm16 = floatTo16BitPCM(downsampled);
                ws.send(pcm16);
            };

            isRecording = true;
            // PTT starts muted (press-to-talk gates speech). Non-PTT
            // starts live so the host can speak immediately; the mute
            // button is available to silence the mic on demand.
            isMuted = pttMode;
            startBtn.textContent = t('stopRecording');
            if (!pttMode) startBtn.classList.add('recording');
            muteBtn.style.display = pttMode ? 'none' : '';
            muteBtn.disabled = false;
            muteBtn.textContent = isMuted ? t('unmute') : t('mute');
            muteBtn.classList.toggle('muted', isMuted);
            languageSelect.disabled = true;
            audioInputSelect.disabled = true;
            if (pttMode) {
                setStatus(t('pttHint'), 'connected');
            }
        } catch (error) {
            console.error('Error starting recording:', error);
            setStatus(t('errorMicDenied'), 'error');
        }
    }

    function stopRecording() {
        isRecording = false;
        isMuted = false;
        spaceHeld = false;
        isStoppingRecording = true;
        muteBtn.style.display = 'none';
        muteBtn.classList.remove('muted');
        viewerSpeakingIndicator.classList.remove('live');

        // Stop audio capture
        if (processor) {
            processor.disconnect();
            processor = null;
        }

        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }

        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => track.stop());
            mediaStream = null;
        }

        // Ask server to drain translations and send final segments + summary.
        // We intentionally keep the socket open so those messages can arrive;
        // the server closes it once summarization is done.
        if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'request_summary' }));
            setStatus(t('statusFinalizing'), 'connected');
            // Watchdog: if the server never closes the socket, force-exit
            // the finalising state so the user isn't stranded.
            stopWatchdogTimer = setTimeout(() => {
                stopWatchdogTimer = null;
                console.warn('stopRecording watchdog fired — forcing finalize');
                try {
                    try {
                        if (ws) ws.close();
                    } catch (e) {
                        console.warn('ws.close from watchdog failed:', e);
                    }
                    finalizeAfterStop();
                } finally {
                    // Guarantee the user escapes "finalizing" even if
                    // finalizeAfterStop throws for any reason.
                    if (isStoppingRecording) {
                        isStoppingRecording = false;
                        setStatus(t('statusStopped'), '');
                    }
                }
            }, STOP_WATCHDOG_MS);
        } else {
            // No live connection — finalize immediately so the user isn't stuck.
            finalizeAfterStop();
        }

        startBtn.textContent = t('startRecording');
        startBtn.classList.remove('recording');
        startBtn.disabled = false;
        languageSelect.disabled = false;
        audioInputSelect.disabled = false;
    }

    function finalizeAfterStop() {
        if (!isStoppingRecording) return;
        isStoppingRecording = false;
        if (stopWatchdogTimer !== null) {
            clearTimeout(stopWatchdogTimer);
            stopWatchdogTimer = null;
        }
        pendingTranslations.clear();
        updateTranslatingIndicator();
        setStatus(t('statusStopped'), '');
        maybeShowTranscriptDownload();
    }

    // Disable start button until language is selected
    function updateStartButtonState() {
        const hasLanguage = languageSelect.value !== '';
        startBtn.disabled = !hasLanguage && !isRecording;
        if (!hasLanguage && !isRecording) {
            startBtn.title = t('selectLanguageFirst');
        } else {
            startBtn.title = '';
        }
    }

    languageSelect.addEventListener('change', () => {
        updateStartButtonState();
        // Reflect the picked language on the pill immediately, so the user
        // sees their selection before recording starts. The server will
        // confirm/override on session_active, which triggers the same path.
        if (leftLangLabel && languageSelect.value) {
            leftLangLabel.textContent = languageSelect.value.toUpperCase();
        }
    });
    updateStartButtonState(); // Initial state

    startBtn.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            if (!languageSelect.value) {
                alert(t('selectLanguageFirst'));
                return;
            }
            startRecording();
        }
    });

    muteBtn.addEventListener('click', () => {
        isMuted = !isMuted;
        muteBtn.textContent = isMuted ? t('unmute') : t('mute');
        muteBtn.classList.toggle('muted', isMuted);
    });

    clearBtn.addEventListener('click', () => {
        // Set guard flag FIRST to block any pending WebSocket messages
        sessionCleared = true;
        console.log('Cleared: sessionCleared =', sessionCleared);

        // Stop recording to prevent new messages
        if (isRecording) {
            // Force stop without server notification
            isRecording = false;
            if (processor) {
                processor.disconnect();
                processor = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            if (mediaStream) {
                mediaStream.getTracks().forEach((track) => track.stop());
                mediaStream = null;
            }
            if (ws) {
                ws.close();
                ws = null;
            }
            startBtn.textContent = t('startRecording');
            startBtn.classList.remove('recording');
            isMuted = false;
            spaceHeld = false;
            muteBtn.style.display = 'none';
            muteBtn.classList.remove('muted');
            languageSelect.disabled = false;
            audioInputSelect.disabled = false;
        }
        // Cancel any pending stop watchdog and close a still-draining socket.
        if (stopWatchdogTimer !== null) {
            clearTimeout(stopWatchdogTimer);
            stopWatchdogTimer = null;
        }
        if (ws) {
            try {
                ws.close();
            } catch (e) {
                console.warn('ws.close during clear failed:', e);
            }
            ws = null;
        }
        // Reset PTT state
        pttMode = false;
        pttToggle.checked = false;
        viewerSpeakingIndicator.classList.remove('live');
        pendingTranslations.clear();
        updateTranslatingIndicator();
        // Reset transcript download state
        allSegments = [];
        viewerConsentedTranscript = false;
        isStoppingRecording = false;
        const dlBar = document.getElementById('transcriptDownloadBar');
        if (dlBar) dlBar.classList.remove('visible');
        sgaThread.textContent = '';
        foreignLang = null;
        updateLangPair();
        setStatus(t('statusReady'), '');
    });

    // PTT segmented control — buttons drive the (hidden) checkbox so the
    // existing change-handler wiring stays intact.
    const pttSegOn = pttToggleLabel?.querySelector('.ptt-on');
    const pttSegOff = pttToggleLabel?.querySelector('.ptt-off');
    function reflectPttSegmented() {
        if (!pttSegOn || !pttSegOff) return;
        pttSegOn.classList.toggle('on', pttToggle.checked);
        pttSegOff.classList.toggle('on', !pttToggle.checked);
    }
    reflectPttSegmented();
    pttSegOn?.addEventListener('click', () => {
        if (pttToggle.checked) return;
        pttToggle.checked = true;
        pttToggle.dispatchEvent(new Event('change'));
        reflectPttSegmented();
    });
    pttSegOff?.addEventListener('click', () => {
        if (!pttToggle.checked) return;
        pttToggle.checked = false;
        pttToggle.dispatchEvent(new Event('change'));
        reflectPttSegmented();
    });

    // Protokoll toggle — clicking the .sga-toggle label flips the input
    // and applies the .on visual state.
    function reflectProtokoll() {
        hostTranscriptLabel?.classList.toggle('on', hostTranscriptToggle.checked);
    }
    reflectProtokoll();
    hostTranscriptLabel?.addEventListener('click', (e) => {
        // Native label-for handling fires the input's change event for us;
        // we just need to reflect the new state visually.
        if (e.target instanceof HTMLInputElement) return;
        hostTranscriptToggle.checked = !hostTranscriptToggle.checked;
        hostTranscriptToggle.dispatchEvent(new Event('change'));
        reflectProtokoll();
        e.preventDefault();
    });

    // Side rail toggle — persisted across reloads.
    const RAIL_KEY = 'sgaRailOpen';
    function setRailOpen(open) {
        sgaStage?.classList.toggle('with-rail', open);
        if (railToggleBtn) {
            railToggleBtn.textContent = open ? 'Seitenleiste ausblenden ›' : '‹ Teilen & Export';
        }
        try {
            localStorage.setItem(RAIL_KEY, open ? '1' : '0');
        } catch (e) {
            console.warn('localStorage write failed:', e);
        }
    }
    setRailOpen(localStorage.getItem(RAIL_KEY) !== '0');
    railToggleBtn?.addEventListener('click', () => {
        setRailOpen(!sgaStage?.classList.contains('with-rail'));
    });

    // Initial lang-pair render so the chip + hero text aren't blank.
    updateLangPair();
})();
