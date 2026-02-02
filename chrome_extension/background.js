// background.js

// background.js

let targetLang = 'en';
let inputMode = 'tab';

// Listen for messages from Popup/Offscreen
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    // IMPORTANT: Return true for async sendResponse to work
    (async () => {
        if (request.type === "START_RECORDING") {
            try {
                targetLang = request.targetLang;
                inputMode = request.inputMode;

                // Set optimistic state
                await chrome.storage.local.set({
                    transcriptHistory: [],
                    isRecording: true
                });

                await startCapture();
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
            sendResponse({ isRecording: result.isRecording || false });

        } else if (request.type === "TRANSCRIPTION_UPDATE") {
            const result = await chrome.storage.local.get(['transcriptHistory']);
            let history = result.transcriptHistory || [];
            history.push(request);
            if (history.length > 50) history.shift();
            await chrome.storage.local.set({ transcriptHistory: history });

        } else if (request.type === "GET_HISTORY") {
            const result = await chrome.storage.local.get(['transcriptHistory']);
            sendResponse({ history: result.transcriptHistory || [] });
        }
    })();
    return true; // Keep channel open
});

async function startCapture() {
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
            inputMode: inputMode
        });
        return;
    }

    // Tab or Mixed
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) throw new Error("No active tab found");

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
        inputMode: inputMode
    });
}

function stopCapture() {
    chrome.storage.local.set({ isRecording: false });
    chrome.runtime.sendMessage({ type: "STOP_RECORDING_OFFSCREEN" });
    // Close offscreen doc roughly
    chrome.offscreen.closeDocument().catch(() => { });
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
