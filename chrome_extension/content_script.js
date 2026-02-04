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
        overlay.style.pointerEvents = "none";
        overlay.style.transition = "opacity 0.2s ease";
        overlay.style.opacity = "0";

        const status = document.createElement("div");
        status.id = "vasha-subtitle-status";
        status.style.fontSize = "12px";
        status.style.opacity = "0.8";
        status.style.marginBottom = "4px";
        status.textContent = "Listening";

        const model = document.createElement("div");
        model.id = "vasha-subtitle-model";
        model.style.fontSize = "11px";
        model.style.opacity = "0.7";
        model.style.marginBottom = "6px";
        model.textContent = "ASR: --";

        const text = document.createElement("div");
        text.id = "vasha-subtitle-text";
        text.textContent = "";

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

    let lastSegmentId = null;
    let partialEnabled = true;

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

        if (message.is_final) {
            lastSegmentId = message.segment_id;
            textEl.textContent = message.text || "";
            setStatus("finalized");
        } else {
            textEl.textContent = message.text || "";
            setStatus("transcribing");
        }
        showOverlay();
    }

    chrome.runtime.onMessage.addListener((message) => {
        if (message.type === "SUBTITLE_UPDATE") {
            updateText(message);
            if (message.asr_model) setModel(message.asr_model);
        } else if (message.type === "ASR_STATE") {
            setStatus(message.state);
        } else if (message.type === "ASR_MODEL_UPDATE") {
            setModel(message.model, message.reason);
        } else if (message.type === "SUBTITLE_CLEAR") {
            textEl.textContent = "";
            setStatus("listening");
            hideOverlay();
        } else if (message.type === "TOGGLE_PARTIAL") {
            partialEnabled = !!message.enabled;
        }
    });
})();
