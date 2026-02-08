(() => {
    if (window.__vashaOverlayInitialized) return;
    window.__vashaOverlayInitialized = true;

    const OVERLAY_ID = "vasha-subtitle-overlay";

    function createOverlay() {
        let overlay = document.getElementById(OVERLAY_ID);
        if (overlay) return overlay;

        overlay = document.createElement("div");
        overlay.id = OVERLAY_ID;
        overlay.style.position = "fixed";
        overlay.style.left = "50%";
        overlay.style.bottom = "8%";
        overlay.style.transform = "translateX(-50%)";
        overlay.style.zIndex = "2147483647";
        overlay.style.maxWidth = "80%";
        overlay.style.padding = "10px 14px";
        overlay.style.borderRadius = "8px";
        overlay.style.background = "rgba(0, 0, 0, 0.72)";
        overlay.style.color = "#fff";
        overlay.style.fontSize = "16px";
        overlay.style.lineHeight = "1.4";
        overlay.style.fontFamily = "Segoe UI, sans-serif";
        overlay.style.boxShadow = "0 4px 18px rgba(0,0,0,0.3)";
        overlay.style.textAlign = "center";
        overlay.style.pointerEvents = "auto";
        overlay.style.transition = "opacity 0.2s ease";
        overlay.style.opacity = "0";

        const controller = document.createElement("div");
        controller.id = "vasha-subtitle-controller";
        controller.style.display = "flex";
        controller.style.justifyContent = "center";
        controller.style.alignItems = "center";
        controller.style.gap = "6px";
        controller.style.marginBottom = "6px";
        controller.style.pointerEvents = "auto";

        const langLabel = document.createElement("span");
        langLabel.style.fontSize = "11px";
        langLabel.style.opacity = "0.8";
        langLabel.textContent = "Lang:";

        const langSelect = document.createElement("select");
        langSelect.id = "vasha-lang-select";
        langSelect.style.fontSize = "11px";
        langSelect.style.background = "rgba(255,255,255,0.1)";
        langSelect.style.color = "#fff";
        langSelect.style.border = "1px solid rgba(255,255,255,0.2)";
        langSelect.style.borderRadius = "4px";
        langSelect.style.padding = "2px 6px";

        controller.appendChild(langLabel);
        controller.appendChild(langSelect);

        const status = document.createElement("div");
        status.id = "vasha-subtitle-status";
        status.style.fontSize = "12px";
        status.style.opacity = "0.8";
        status.style.marginBottom = "4px";
        status.style.pointerEvents = "none";
        status.textContent = "Listening";

        const model = document.createElement("div");
        model.id = "vasha-subtitle-model";
        model.style.fontSize = "11px";
        model.style.opacity = "0.7";
        model.style.marginBottom = "6px";
        model.style.pointerEvents = "none";
        model.textContent = "ASR: --";

        const text = document.createElement("div");
        text.id = "vasha-subtitle-text";
        text.style.pointerEvents = "none";
        text.style.display = "flex";
        text.style.flexDirection = "column";
        text.style.gap = "6px";
        text.textContent = "";

        overlay.appendChild(controller);
        overlay.appendChild(status);
        overlay.appendChild(model);
        overlay.appendChild(text);

        document.documentElement.appendChild(overlay);
        return overlay;
    }

    const overlay = createOverlay();
    const statusEl = overlay.querySelector("#vasha-subtitle-status");
    const modelEl = overlay.querySelector("#vasha-subtitle-model");
    const textEl = overlay.querySelector("#vasha-subtitle-text");
    const langSelect = overlay.querySelector("#vasha-lang-select");

    let lastSegmentId = null;
    let partialEnabled = true;
    let availableTargets = ["en"];
    let wordHighlightEnabled = false;
    let activeWordTimer = null;

    const MAX_LINES = 3;
    const MAX_CHARS_PER_LINE = 60;
    const MIN_LINE_MS = 1600;

    function showOverlay() {
        overlay.style.opacity = "1";
    }

    function hideOverlay() {
        overlay.style.opacity = "0";
    }

    function setStatus(state) {
        if (!state) return;
        const label = state === "transcribing" ? "Transcribing"
            : state === "finalized" ? "Finalized"
            : "Listening";
        statusEl.textContent = label;
        showOverlay();
    }

    function setModel(model, reason) {
        if (!model) return;
        let text = `ASR: ${model}`;
        if (reason) text += ` (${reason})`;
        modelEl.textContent = text;
    }

    function updateText(message) {
        if (!partialEnabled && !message.is_final) return;

        if (message.segment_id !== null && message.segment_id !== undefined) {
            if (lastSegmentId !== null && message.segment_id <= lastSegmentId && !message.is_final) return;
        }

        const text = message.text || "";
        const words = Array.isArray(message.words) ? message.words : null;

        if (message.is_final) {
            lastSegmentId = message.segment_id;
            pushFinalLine(message.segment_id, text, words, message);
            setStatus("finalized");
        } else {
            updatePartialLine(message.segment_id, text, words, message);
            setStatus("transcribing");
        }
        showOverlay();
    }

    function pushFinalLine(segmentId, text, words, message) {
        removePartialLine();
        const lines = splitForPacing(text);
        lines.forEach((line, idx) => {
            const el = createLineElement(segmentId, line, true, words, idx === 0);
            textEl.appendChild(el);
        });
        trimLines();
        scheduleLineFade(text, lines.length);
        startWordHighlight(words);
    }

    function updatePartialLine(segmentId, text, words, message) {
        let lineEl = textEl.querySelector("[data-partial='true']");
        if (!lineEl) {
            lineEl = createLineElement(segmentId, text, false, words, true);
            lineEl.dataset.partial = "true";
            textEl.appendChild(lineEl);
            trimLines();
        } else {
            lineEl.dataset.segmentId = segmentId ?? lineEl.dataset.segmentId;
            lineEl.innerHTML = "";
            renderLineContent(lineEl, text, words, true);
        }
        startWordHighlight(words);
    }

    function removePartialLine() {
        const lineEl = textEl.querySelector("[data-partial='true']");
        if (lineEl) lineEl.remove();
    }

    function createLineElement(segmentId, text, isFinal, words, isPrimary) {
        const line = document.createElement("div");
        line.style.opacity = isFinal ? "1" : "0.85";
        line.style.fontWeight = isPrimary ? "600" : "500";
        line.style.fontSize = isPrimary ? "16px" : "14px";
        line.dataset.segmentId = segmentId ?? "";
        line.dataset.partial = isFinal ? "false" : "true";
        renderLineContent(line, text, words, isPrimary);
        return line;
    }

    function renderLineContent(lineEl, text, words, isPrimary) {
        if (wordHighlightEnabled && words && words.length) {
            const frag = document.createDocumentFragment();
            words.forEach((w, idx) => {
                const span = document.createElement("span");
                span.textContent = w.word ? `${w.word} ` : "";
                span.dataset.wordIndex = String(idx);
                span.style.transition = "color 120ms ease";
                frag.appendChild(span);
            });
            lineEl.appendChild(frag);
        } else {
            lineEl.textContent = text;
        }
    }

    function splitForPacing(text) {
        if (!text) return [""];
        if (text.length <= MAX_CHARS_PER_LINE) return [text];
        const parts = [];
        let remaining = text;
        while (remaining.length > MAX_CHARS_PER_LINE) {
            let cut = remaining.lastIndexOf(" ", MAX_CHARS_PER_LINE);
            if (cut <= 0) cut = MAX_CHARS_PER_LINE;
            parts.push(remaining.slice(0, cut).trim());
            remaining = remaining.slice(cut).trim();
        }
        if (remaining) parts.push(remaining);
        return parts;
    }

    function trimLines() {
        const lines = Array.from(textEl.children).filter(el => el.dataset.partial !== "true");
        while (lines.length > MAX_LINES) {
            const first = lines.shift();
            if (first) first.remove();
        }
    }

    function scheduleLineFade(text, lineCount) {
        const duration = Math.max(MIN_LINE_MS, Math.min(6000, text.length * 45));
        const lines = Array.from(textEl.children).slice(-lineCount);
        lines.forEach(line => {
            line.style.transition = "opacity 600ms ease";
            setTimeout(() => {
                if (line.dataset.partial === "false") line.style.opacity = "0.4";
            }, duration);
        });
    }

    function startWordHighlight(words) {
        if (!wordHighlightEnabled || !words || !words.length) return;
        if (activeWordTimer) cancelAnimationFrame(activeWordTimer);
        const start = performance.now();
        const lineEl = textEl.querySelector("[data-partial='true']") || textEl.lastElementChild;
        if (!lineEl) return;
        const spans = Array.from(lineEl.querySelectorAll("[data-word-index]"));
        const loop = () => {
            const elapsed = (performance.now() - start) / 1000;
            spans.forEach((span, idx) => {
                const w = words[idx];
                if (!w) return;
                const active = elapsed >= w.start_time && elapsed <= w.end_time;
                span.style.color = active ? "#a8dadc" : "#fff";
            });
            activeWordTimer = requestAnimationFrame(loop);
        };
        activeWordTimer = requestAnimationFrame(loop);
    }

    function updateLangOptions(targetLang) {
        availableTargets = targetLang ? [targetLang] : ["en"];
        langSelect.innerHTML = "";
        availableTargets.forEach(lang => {
            const opt = document.createElement("option");
            opt.value = lang;
            opt.textContent = lang.toUpperCase();
            langSelect.appendChild(opt);
        });
        langSelect.value = availableTargets[0];
    }

    langSelect.addEventListener("change", () => {
        chrome.runtime.sendMessage({
            type: "UPDATE_TARGET_LANG",
            targetLang: langSelect.value
        });
    });

    chrome.runtime.onMessage.addListener((message) => {
        if (message.type === "SUBTITLE_UPDATE") {
            updateText(message);
            if (message.asr_model) setModel(message.asr_model);
        } else if (message.type === "ASR_STATE") {
            setStatus(message.state);
        } else if (message.type === "ASR_MODEL_UPDATE") {
            setModel(message.model, message.reason);
        } else if (message.type === "TARGET_LANG_UPDATE") {
            updateLangOptions(message.targetLang || "en");
        } else if (message.type === "SUBTITLE_CLEAR") {
            textEl.textContent = "";
            setStatus("listening");
            hideOverlay();
        } else if (message.type === "TOGGLE_PARTIAL") {
            partialEnabled = !!message.enabled;
        } else if (message.type === "TOGGLE_WORD_TS") {
            wordHighlightEnabled = !!message.enabled;
        }
    });
})();
