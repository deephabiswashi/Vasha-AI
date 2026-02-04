// popup.js
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusText = document.getElementById('statusText');
const statusDot = document.getElementById('statusDot');
const liveText = document.getElementById('liveText');
const langSelect = document.getElementById('languageSelect');
const lidLang = document.getElementById('lidLang');
const lidConf = document.getElementById('lidConf');
const lidTimeline = document.getElementById('lidTimeline');
const asrModelSelect = document.getElementById('asrModelSelect');
const asrModelStatus = document.getElementById('asrModelStatus');
const partialToggle = document.getElementById('partialToggle');
const wordTsToggle = document.getElementById('wordTsToggle');

// Initial check of state
chrome.runtime.sendMessage({ type: "GET_STATUS" }, (response) => {
    if (response && response.isRecording) {
        showRecordingState();

        // Restore History
        chrome.runtime.sendMessage({ type: "GET_HISTORY" }, (historyRes) => {
            if (historyRes && historyRes.history) {
                historyRes.history.forEach(msg => {
                    if (msg.type === "SUBTITLE_UPDATE" || msg.text) {
                        liveText.innerText = msg.text || liveText.innerText;
                    }
                    if (msg.metadata) updateLID(msg.metadata);
                });
            }
        });
    }

    if (response) {
        if (response.asrModel) {
            asrModelSelect.value = response.asrModel;
            asrModelStatus.innerText = response.asrModel;
        }
        if (typeof response.partialEnabled === 'boolean') partialToggle.checked = response.partialEnabled;
        if (typeof response.wordTimestamps === 'boolean') wordTsToggle.checked = response.wordTimestamps;
    }
});

startBtn.addEventListener('click', () => {
    const lang = langSelect.value;
    const mode = document.getElementById('inputMode').value;
    const asrModel = asrModelSelect.value;
    const partialEnabled = partialToggle.checked;
    const wordTimestamps = wordTsToggle.checked;
    chrome.runtime.sendMessage({
        type: "START_RECORDING",
        targetLang: lang,
        inputMode: mode,
        asrModel,
        partialEnabled,
        wordTimestamps
    }, (response) => {
        if (response.success) {
            showRecordingState();
        }
    });
});

stopBtn.addEventListener('click', () => {
    chrome.runtime.sendMessage({ type: "STOP_RECORDING" }, (response) => {
        showReadyState();
    });
});

asrModelSelect.addEventListener('change', () => {
    chrome.runtime.sendMessage({
        type: "UPDATE_ASR_PREFS",
        asrModel: asrModelSelect.value,
        partialEnabled: partialToggle.checked,
        wordTimestamps: wordTsToggle.checked
    });
    asrModelStatus.innerText = asrModelSelect.value;
});

partialToggle.addEventListener('change', () => {
    chrome.runtime.sendMessage({
        type: "UPDATE_ASR_PREFS",
        asrModel: asrModelSelect.value,
        partialEnabled: partialToggle.checked,
        wordTimestamps: wordTsToggle.checked
    });
});

wordTsToggle.addEventListener('change', () => {
    chrome.runtime.sendMessage({
        type: "UPDATE_ASR_PREFS",
        asrModel: asrModelSelect.value,
        partialEnabled: partialToggle.checked,
        wordTimestamps: wordTsToggle.checked
    });
});

function showRecordingState() {
    startBtn.style.display = 'none';
    stopBtn.style.display = 'block';
    statusText.innerText = "Listening";
    statusDot.classList.add('active');
    liveText.innerText = "Waiting for speech...";
    asrModelStatus.innerText = asrModelSelect.value;
}

function showReadyState() {
    startBtn.style.display = 'block';
    stopBtn.style.display = 'none';
    statusText.innerText = "Ready";
    statusDot.classList.remove('active');
    liveText.innerText = "Stopped.";
    lidTimeline.innerHTML = ""; // Clear timeline
}

// Listen for updates from background
chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "SUBTITLE_UPDATE") {
        if (message.text) liveText.innerText = message.text;
        if (message.state) updateState(message.state);
        if (message.asr_model) {
            asrModelStatus.innerText = message.asr_model;
        }
    } else if (message.type === "ASR_MODEL_UPDATE") {
        if (message.model) {
            asrModelSelect.value = message.model;
            asrModelStatus.innerText = message.model;
        }
    } else if (message.type === "ASR_STATE") {
        updateState(message.state);
    } else if (message.type === "TRANSCRIPTION_UPDATE") {
        liveText.innerText = message.text;
        if (message.metadata) updateLID(message.metadata);
    } else if (message.type === "ERROR") {
        statusText.innerText = "Error";
        statusDot.style.background = "red";
        statusDot.style.boxShadow = "0 0 5px red";
        liveText.innerText = message.message;
        // Reset buttons
        startBtn.style.display = 'block';
        stopBtn.style.display = 'none';
    }
});

function updateState(state) {
    if (!state) return;
    if (state === "transcribing") {
        statusText.innerText = "Transcribing";
    } else if (state === "finalized") {
        statusText.innerText = "Finalized";
    } else {
        statusText.innerText = "Listening";
    }
}

function updateLID(metadata) {
    if (!metadata) return;
    const lang = metadata.detected_language || "?";
    const conf = Math.round((metadata.confidence || 0) * 100);

    lidLang.innerText = lang.toUpperCase();
    lidConf.innerText = conf + "%";

    // Timeline logic
    const seg = document.createElement('div');
    seg.className = 'timeline-seg';
    seg.title = `${lang} (${conf}%)`;

    // Color coding
    const colors = {
        'en': '#a8dadc', 'hi': '#e63946', 'bn': '#457b9d',
        'te': '#2a9d8f', 'ta': '#f4a261', 'unknown': '#555'
    };
    seg.style.backgroundColor = colors[lang] || '#999';

    lidTimeline.appendChild(seg);

    // Keep last 20 chunks
    if (lidTimeline.children.length > 20) {
        lidTimeline.removeChild(lidTimeline.firstChild);
    }
}
