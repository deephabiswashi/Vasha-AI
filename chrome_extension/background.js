// background.js

// background.js

let targetLang = 'en';
let inputMode = 'tab';
let asrModel = 'faster_whisper';
let partialEnabled = true;
let wordTimestamps = false;
let lastStableText = "";
let activeTabId = null;

const SESSION_KEYS = [
    'asrModel',
    'partialEnabled',
    'wordTimestamps'
];

async function getSessionStore() {
    if (chrome.storage && chrome.storage.session) return chrome.storage.session;
    return chrome.storage.local;
}

async function loadSessionState() {
    const store = await getSessionStore();
    const state = await store.get(SESSION_KEYS);
    if (state.asrModel) asrModel = state.asrModel;
    if (typeof state.partialEnabled === 'boolean') partialEnabled = state.partialEnabled;
    if (typeof state.wordTimestamps === 'boolean') wordTimestamps = state.wordTimestamps;
}

async function saveSessionState() {
    const store = await getSessionStore();
    await store.set({
        asrModel,
        partialEnabled,
        wordTimestamps
    });
}

// Listen for messages from Popup/Offscreen
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    // IMPORTANT: Return true for async sendResponse to work
    (async () => {
        if (request.type === "START_RECORDING") {
            try {
                targetLang = request.targetLang;
                inputMode = request.inputMode;
                if (request.asrModel) asrModel = request.asrModel;
                if (typeof request.partialEnabled === 'boolean') partialEnabled = request.partialEnabled;
                if (typeof request.wordTimestamps === 'boolean') wordTimestamps = request.wordTimestamps;

                await saveSessionState();

                // Set optimistic state
                await chrome.storage.local.set({
                    transcriptHistory: [],
                    isRecording: true
                });

                await startCapture(request.tabId);
                sendResponse({ success: true });

            } catch (e) {
                console.error("Start Failed:", e);
                await chrome.storage.local.set({ isRecording: false });
                sendResponse({ success: false, error: e.message });
                // Also broadcast error to popup if open
                chrome.runtime.sendMessage({ type: "ERROR", message: "Start failed: " + e.message });
            }
        } else if (request.type === "STOP_RECORDING") {
            try {
                stopCapture();
                await chrome.storage.local.set({ isRecording: false });
                sendResponse({ success: true });
            } catch (e) {
                await chrome.storage.local.set({ isRecording: false }); // safety
                sendResponse({ success: true });
            }

        } else if (request.type === "GET_STATUS") {
            const result = await chrome.storage.local.get(['isRecording']);
            await loadSessionState();
            sendResponse({
                isRecording: result.isRecording || false,
                asrModel,
                partialEnabled,
                wordTimestamps
            });

        } else if (request.type === "TRANSCRIPTION_UPDATE") {
            const result = await chrome.storage.local.get(['transcriptHistory']);
            let history = result.transcriptHistory || [];
            history.push(request);
            if (history.length > 50) history.shift();
            await chrome.storage.local.set({ transcriptHistory: history });
            if (request.is_final && request.text) lastStableText = request.text;

        } else if (request.type === "GET_HISTORY") {
            const result = await chrome.storage.local.get(['transcriptHistory']);
            sendResponse({ history: result.transcriptHistory || [] });
        } else if (request.type === "UPDATE_ASR_PREFS") {
            if (request.asrModel) asrModel = request.asrModel;
            if (typeof request.partialEnabled === 'boolean') partialEnabled = request.partialEnabled;
            if (typeof request.wordTimestamps === 'boolean') wordTimestamps = request.wordTimestamps;
            await saveSessionState();
            broadcastToUIs({
                type: "ASR_MODEL_UPDATE",
                model: asrModel
            });
            broadcastToUIs({
                type: "TOGGLE_PARTIAL",
                enabled: partialEnabled
            });
            chrome.runtime.sendMessage({
                type: "UPDATE_PREFS_OFFSCREEN",
                asrModel,
                partialEnabled,
                wordTimestamps
            });
            sendResponse({ success: true });
        } else if (request.type === "ASR_OVERRIDE") {
            // Backend override due to language switching
            asrModel = request.model || asrModel;
            await saveSessionState();
            broadcastToUIs({
                type: "ASR_MODEL_UPDATE",
                model: asrModel,
                reason: request.reason || "backend_override",
                detected_language: request.detected_language || null
            });
            chrome.runtime.sendMessage({
                type: "UPDATE_PREFS_OFFSCREEN",
                asrModel,
                partialEnabled,
                wordTimestamps
            });
        } else if (request.type === "ASR_PARTIAL" || request.type === "ASR_FINAL") {
            const payload = {
                type: "SUBTITLE_UPDATE",
                text: request.text || "",
                segment_id: request.segment_id,
                is_final: request.type === "ASR_FINAL",
                words: request.words || null,
                state: request.state || (request.type === "ASR_FINAL" ? "finalized" : "transcribing"),
                asr_model: request.asr_model || asrModel,
                detected_language: request.detected_language || null,
                segment_start_time: request.segment_start_time || null,
                segment_end_time: request.segment_end_time || null
            };

            // Persist for popup history
            const result = await chrome.storage.local.get(['transcriptHistory']);
            let history = result.transcriptHistory || [];
            history.push(payload);
            if (history.length > 50) history.shift();
            await chrome.storage.local.set({ transcriptHistory: history });

            if (payload.is_final && payload.text) lastStableText = payload.text;
            broadcastToUIs(payload);
        } else if (request.type === "ASR_STATE") {
            broadcastToUIs({
                type: "ASR_STATE",
                state: request.state
            });
        } else if (request.type === "GET_LAST_STABLE") {
            sendResponse({ text: lastStableText });
        } else if (request.type === "ERROR") {
            broadcastToUIs({
                type: "ERROR",
                message: request.message || "Unknown error"
            });
            if (lastStableText) {
                broadcastToUIs({
                    type: "SUBTITLE_UPDATE",
                    text: lastStableText,
                    is_final: true,
                    segment_id: -1,
                    state: "finalized"
                });
            }
        }
    })();
    return true; // Keep channel open
});

async function startCapture(tabId) {
    // Check if truly recording via storage (double check)
    const state = await chrome.storage.local.get(['isRecording']);
    if (state.isRecording) {
        const contexts = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] });
        if (contexts.length > 0) return;
    }

    await createOffscreenDocument();

    if (inputMode === 'mic') {
        chrome.runtime.sendMessage({
            type: "INIT_RECORDING",
            streamId: null,
            targetLang: targetLang,
            inputMode: inputMode,
            asrModel,
            partialEnabled,
            wordTimestamps
        });
        if (tabId) {
            activeTabId = tabId;
            await injectContentUI(tabId);
        }
        return;
    }

    // Tab or Mixed
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) throw new Error("No active tab found");
    activeTabId = tab.id;

    if (tab.url.startsWith("chrome://") || tab.url.startsWith("edge://") || tab.url.startsWith("about:")) {
        throw new Error("Cannot capture restricted system page");
    }

    // Promisify getMediaStreamId
    const streamId = await new Promise((resolve, reject) => {
        chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id }, (id) => {
            if (chrome.runtime.lastError) {
                reject(new Error(chrome.runtime.lastError.message));
            } else if (!id) {
                reject(new Error("Failed to get stream ID"));
            } else {
                resolve(id);
            }
        });
    });

    // Send stream ID to offscreen
    chrome.runtime.sendMessage({
        type: "INIT_RECORDING",
        streamId: streamId,
        targetLang: targetLang,
        inputMode: inputMode,
        asrModel,
        partialEnabled,
        wordTimestamps
    });

    await injectContentUI(tab.id);
}

function stopCapture() {
    chrome.storage.local.set({ isRecording: false });
    chrome.runtime.sendMessage({ type: "STOP_RECORDING_OFFSCREEN" });
    // Close offscreen doc roughly
    chrome.offscreen.closeDocument().catch(() => { });
    if (activeTabId) {
        chrome.tabs.sendMessage(activeTabId, { type: "SUBTITLE_CLEAR" }).catch(() => { });
    }
}

async function createOffscreenDocument() {
    const existingContexts = await chrome.runtime.getContexts({
        contextTypes: ['OFFSCREEN_DOCUMENT'],
    });

    if (existingContexts.length > 0) {
        return;
    }

    return chrome.offscreen.createDocument({
        url: 'offscreen.html',
        reasons: ['USER_MEDIA'],
        justification: 'Recording tab audio for translation',
    });
}

async function injectContentUI(tabId) {
    try {
        await chrome.scripting.executeScript({
            target: { tabId },
            files: ["content_script.js"]
        });
    } catch (e) {
        // Ignore injection failures for restricted pages
    }
}

function broadcastToUIs(message) {
    chrome.runtime.sendMessage(message);
    if (activeTabId) {
        chrome.tabs.sendMessage(activeTabId, message).catch(() => { });
    }
}

loadSessionState();
