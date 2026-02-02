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

// Initial check of state
chrome.runtime.sendMessage({ type: "GET_STATUS" }, (response) => {
    if (response && response.isRecording) {
        showRecordingState();

        // Restore History
        chrome.runtime.sendMessage({ type: "GET_HISTORY" }, (historyRes) => {
            if (historyRes && historyRes.history) {
                historyRes.history.forEach(msg => {
                    liveText.innerText = msg.text; // Show latest
                    if (msg.metadata) {
                        updateLID(msg.metadata); // Rebuild timeline
                    }
                });
            }
        });
    }
});

startBtn.addEventListener('click', () => {
    const lang = langSelect.value;
    const mode = document.getElementById('inputMode').value;
    chrome.runtime.sendMessage({
        type: "START_RECORDING",
        targetLang: lang,
        inputMode: mode
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

function showRecordingState() {
    startBtn.style.display = 'none';
    stopBtn.style.display = 'block';
    statusText.innerText = "Listening & Translating...";
    statusDot.classList.add('active');
    liveText.innerText = "Processing chunks...";
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
    if (message.type === "TRANSCRIPTION_UPDATE") {
        liveText.innerText = message.text; // Show translated text
        if (message.metadata) {
            updateLID(message.metadata);
        }
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
