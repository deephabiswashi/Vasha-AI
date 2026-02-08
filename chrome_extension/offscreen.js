// offscreen.js - Advanced Audio Capture (Tab, Mic, Mixed), Resampling, VAD + WS ASR

let audioContext;
let workletNode;
let isRecording = false;
let globalStream = null; // Reference to active media stream

// Config
const TARGET_SAMPLE_RATE = 16000;
const VAD_THRESHOLD = 0.01; // RMS Threshold
const PARTIAL_FLUSH_MS = 700;
const PARTIAL_WINDOW_SEC = 2.0;
const FINAL_MIN_SEC = 1.0;
const FINAL_MAX_SEC = 6.0;
const SILENCE_FINAL_MS = 500;
const MAX_BUFFER_SEC = 8.0;

let buffer = []; // Float32 Accumulator
let silenceStart = null;
let speaking = false;
let lastPartialSentAt = 0;
let recordingStartTs = 0;
let currentSegmentStartTs = 0;
let segmentId = 0;

// Backend Config
const WS_URL = "ws://127.0.0.1:5000/stream_audio";
const BACKEND_URL = "http://127.0.0.1:5000/transcribe_translate";
const ENABLE_LEGACY_TRANSLATION = true;

// Session prefs
let targetLang = "en";
let inputMode = "tab";
let asrModel = "faster_whisper";
let partialEnabled = true;
let wordTimestamps = false;
let voiceoverEnabled = true;
let ttsVolume = 1.0;

// WebSocket
let ws = null;
let wsReady = false;
let wsQueue = [];
let reconnectTimer = null;
let legacyFailCount = 0;

chrome.runtime.onMessage.addListener(async (message) => {
    if (message.type === "INIT_RECORDING") {
        startRecording(
            message.streamId,
            message.targetLang,
            message.inputMode,
            message.asrModel,
            message.partialEnabled,
            message.wordTimestamps,
            message.voiceoverEnabled,
            message.ttsVolume
        );
    } else if (message.type === "STOP_RECORDING_OFFSCREEN") {
        stopRecording();
    } else if (message.type === "UPDATE_PREFS_OFFSCREEN") {
        if (message.asrModel) asrModel = message.asrModel;
        if (typeof message.partialEnabled === "boolean") partialEnabled = message.partialEnabled;
        if (typeof message.wordTimestamps === "boolean") wordTimestamps = message.wordTimestamps;
        if (typeof message.voiceoverEnabled === "boolean") voiceoverEnabled = message.voiceoverEnabled;
        if (typeof message.ttsVolume === "number") ttsVolume = message.ttsVolume;
        if (message.targetLang) targetLang = message.targetLang;
        if (wsReady) sendControl("update");
    }
});

async function startRecording(streamId, tLang, mode, model, partial, wordTs, vEnabled, volume) {
    if (isRecording) return;

    targetLang = tLang || targetLang;
    inputMode = mode || inputMode;
    asrModel = model || asrModel;
    partialEnabled = typeof partial === "boolean" ? partial : partialEnabled;
    wordTimestamps = typeof wordTs === "boolean" ? wordTs : wordTimestamps;
    if (typeof vEnabled === "boolean") voiceoverEnabled = vEnabled;
    if (typeof volume === "number") ttsVolume = volume;

    try {
        audioContext = new AudioContext({ sampleRate: 16000 });
    } catch (e) {
        audioContext = new AudioContext();
    }

    // Ensure we load the module
    try {
        await audioContext.audioWorklet.addModule('audio-processor.js');
    } catch (err) {
        console.error("Failed to load audio-processor.js", err);
        chrome.runtime.sendMessage({ type: "ERROR", message: "Failed to load audio processor worklet." });
        return;
    }

    isRecording = true;
    buffer = [];
    silenceStart = null;
    speaking = false;
    lastPartialSentAt = 0;
    recordingStartTs = performance.now();
    currentSegmentStartTs = recordingStartTs;
    segmentId = 0;

    connectWebSocket();

    try {
        const destination = audioContext.createMediaStreamDestination();

        // 1. Get Sources
        if (inputMode === 'tab' || inputMode === 'mixed') {
            try {
                const tabStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        mandatory: {
                            chromeMediaSource: "tab",
                            chromeMediaSourceId: streamId,
                        },
                    },
                    video: false,
                });

                globalStream = tabStream; // Save reference for cleanup

                const tabSource = audioContext.createMediaStreamSource(tabStream);
                tabSource.connect(audioContext.destination); // Tab output to speakers
                tabSource.connect(destination); // Tab output to mix
            } catch (tabErr) {
                console.error("Tab Capture Failed:", tabErr);
                throw new Error("Failed to capture tab audio.");
            }
        }

        if (inputMode === 'mic' || inputMode === 'mixed') {
            try {
                const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                if (!globalStream) globalStream = micStream; // Save reference if not set

                const micSourceNode = audioContext.createMediaStreamSource(micStream);
                micSourceNode.connect(destination);
            } catch (micErr) {
                console.error("Mic Capture Failed:", micErr);
            }
        }

        // 2. Setup AudioWorklet
        const mixedSource = audioContext.createMediaStreamSource(destination.stream);
        workletNode = new AudioWorkletNode(audioContext, 'audio-processor');

        // Handle messages from Worklet
        workletNode.port.onmessage = (event) => {
            if (!isRecording) return;
            handleAudioProcess(event.data, audioContext.sampleRate);
        };

        mixedSource.connect(workletNode);
        workletNode.connect(audioContext.destination);

    } catch (err) {
        console.error("Recording Setup Error:", err);
        chrome.runtime.sendMessage({ type: "ERROR", message: `${err.name}: ${err.message}` });
        isRecording = false;
    }
}

function handleAudioProcess(inputData, inputRate) {
    try {
        // A. Resample to 16kHz if needed
        let processedData = inputData;
        if (inputRate !== TARGET_SAMPLE_RATE) {
            processedData = resample(inputData, inputRate, TARGET_SAMPLE_RATE);
        }

        // B. VAD (Energy)
        const rms = calculateRMS(processedData);

        if (rms > VAD_THRESHOLD) {
            speaking = true;
            silenceStart = null;
        } else {
            if (speaking) {
                if (!silenceStart) silenceStart = Date.now();
            }
        }

        // Append to buffer
        if (processedData && processedData.length > 0) {
            for (let i = 0; i < processedData.length; i++) {
                buffer.push(processedData[i]);
            }
        }

        // Cap buffer to last MAX_BUFFER_SEC
        const maxLen = Math.round(MAX_BUFFER_SEC * TARGET_SAMPLE_RATE);
        if (buffer.length > maxLen) {
            buffer = buffer.slice(buffer.length - maxLen);
        }

        // Check flush
        const durationSec = buffer.length / TARGET_SAMPLE_RATE;
        const silenceDuration = silenceStart ? (Date.now() - silenceStart) : 0;

        // Partial updates
        const now = Date.now();
        if (partialEnabled && speaking && (now - lastPartialSentAt) > PARTIAL_FLUSH_MS && durationSec >= 0.4) {
            lastPartialSentAt = now;
            const windowLen = Math.min(buffer.length, Math.round(PARTIAL_WINDOW_SEC * TARGET_SAMPLE_RATE));
            const startIdx = Math.max(0, buffer.length - windowLen);
            const partialSlice = buffer.slice(startIdx);
            sendChunk(partialSlice, false);
        }

        // Final flush on silence or max duration
        if (durationSec >= FINAL_MAX_SEC || (durationSec >= FINAL_MIN_SEC && silenceDuration > SILENCE_FINAL_MS)) {
            sendChunk(buffer, true);
            buffer = [];
            speaking = false;
            silenceStart = null;
            segmentId += 1;
            currentSegmentStartTs = performance.now();
            chrome.runtime.sendMessage({ type: "ASR_STATE", state: "finalized" });
        } else if (!speaking) {
            chrome.runtime.sendMessage({ type: "ASR_STATE", state: "listening" });
        } else {
            chrome.runtime.sendMessage({ type: "ASR_STATE", state: "transcribing" });
        }
    } catch (err) {
        console.error("Audio Process Error:", err);
    }
}

function stopRecording() {
    isRecording = false;

    // 1. Stop the MediaStream Tracks (Critical for releasing tab capture)
    if (globalStream) {
        globalStream.getTracks().forEach(track => track.stop());
        globalStream = null;
    }

    if (workletNode) workletNode.disconnect();
    if (audioContext) audioContext.close();
    closeWebSocket();
    console.log("Recording Stopped");
}

// --- WebSocket ---

function connectWebSocket() {
    if (ws && (wsReady || ws.readyState === WebSocket.CONNECTING)) return;

    ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        wsReady = true;
        sendControl("init");
        flushQueue();
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === "asr_override") {
                chrome.runtime.sendMessage({
                    type: "ASR_OVERRIDE",
                    model: data.model,
                    reason: data.reason,
                    detected_language: data.detected_language
                });
            } else if (data.type === "asr_partial") {
                chrome.runtime.sendMessage({
                    type: "ASR_PARTIAL",
                    text: data.text,
                    words: data.words || null,
                    segment_id: data.segment_id,
                    asr_model: data.asr_model,
                    detected_language: data.detected_language,
                    segment_start_time: data.segment_start_time,
                    segment_end_time: data.segment_end_time,
                    state: "transcribing"
                });
            } else if (data.type === "asr_final") {
                chrome.runtime.sendMessage({
                    type: "ASR_FINAL",
                    text: data.text,
                    words: data.words || null,
                    segment_id: data.segment_id,
                    asr_model: data.asr_model,
                    detected_language: data.detected_language,
                    segment_start_time: data.segment_start_time,
                    segment_end_time: data.segment_end_time,
                    state: "finalized"
                });
            } else if (data.type === "error") {
                chrome.runtime.sendMessage({ type: "ERROR", message: data.message || "ASR error" });
            }
        } catch (e) {
            // Ignore non-JSON
        }
    };

    ws.onclose = () => {
        wsReady = false;
        scheduleReconnect();
    };

    ws.onerror = () => {
        wsReady = false;
        scheduleReconnect();
    };
}

function closeWebSocket() {
    wsReady = false;
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }
    if (ws) {
        ws.close();
        ws = null;
    }
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        if (isRecording) connectWebSocket();
    }, 1000);
}

function sendControl(action) {
    const payload = {
        type: "control",
        action,
        target_lang: targetLang,
        input_mode: inputMode,
        asr_model: asrModel,
        partial_enabled: partialEnabled,
        word_timestamps: wordTimestamps
    };
    sendWS(payload);
}

function sendChunk(floatData, isFinal) {
    if (!floatData || floatData.length === 0) return;
    const wavBuffer = createWavBuffer(floatData);
    const audioB64 = arrayBufferToBase64(wavBuffer);
    const nowTs = performance.now();
    const payload = {
        type: "audio_chunk",
        segment_id: segmentId,
        is_final: isFinal,
        target_lang: targetLang,
        asr_model: asrModel,
        partial_enabled: partialEnabled,
        word_timestamps: wordTimestamps,
        segment_start_time: (currentSegmentStartTs - recordingStartTs) / 1000.0,
        segment_end_time: (nowTs - recordingStartTs) / 1000.0,
        audio_b64: audioB64
    };
    sendWS(payload);

    if (isFinal && ENABLE_LEGACY_TRANSLATION) {
        const blob = new Blob([wavBuffer], { type: "audio/wav" });
        sendToBackendLegacy(blob, targetLang);
    }
}

function sendWS(payload) {
    if (wsReady && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
    } else {
        wsQueue.push(payload);
        if (wsQueue.length > 10) wsQueue.shift();
    }
}

function flushQueue() {
    if (!wsReady || !ws || ws.readyState !== WebSocket.OPEN) return;
    while (wsQueue.length > 0) {
        ws.send(JSON.stringify(wsQueue.shift()));
    }
}

// --- Utils ---

function resample(data, inputRate, outputRate) {
    if (inputRate === outputRate) return data;
    const ratio = inputRate / outputRate;
    const newLength = Math.round(data.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
        const inputIndex = i * ratio;
        const low = Math.floor(inputIndex);
        const high = Math.min(Math.ceil(inputIndex), data.length - 1);
        const weight = inputIndex - low;
        result[i] = data[low] * (1 - weight) + data[high] * weight;
    }
    return result;
}

function calculateRMS(data) {
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
        sum += data[i] * data[i];
    }
    return Math.sqrt(sum / data.length);
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function createWavBuffer(floatData) {
    const dataLen = floatData.length * 2;
    const bufferArray = new ArrayBuffer(44 + dataLen);
    const view = new DataView(bufferArray);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataLen, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, TARGET_SAMPLE_RATE, true);
    view.setUint32(28, TARGET_SAMPLE_RATE * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataLen, true);

    floatTo16BitPCM(view, 44, floatData);
    return bufferArray;
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

async function sendToBackendLegacy(audioBlob, lang) {
    try {
        const formData = new FormData();
        formData.append("audio", audioBlob, "chunk.wav");
        formData.append("target_lang", lang);
        formData.append("mode", "fast");

        const response = await fetch(BACKEND_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();

        if (data.status === "success" && data.audio_base64) {
            chrome.runtime.sendMessage({
                type: "TRANSCRIPTION_UPDATE",
                text: data.translated_text,
                metadata: data.metadata
            });
            if (voiceoverEnabled) {
                playBase64Audio(data.audio_base64);
            }
        }
        legacyFailCount = 0;
    } catch (e) {
        console.error("Backend Error:", e);
        legacyFailCount += 1;
        if (legacyFailCount >= 3) {
            legacyFailCount = 0;
            chrome.runtime.sendMessage({
                type: "ERROR",
                message: "Backend unreachable. Stopping capture."
            });
            stopRecording();
        }
    }
}

function playBase64Audio(base64String) {
    try {
        const audio = new Audio("data:audio/wav;base64," + base64String);
        audio.volume = Math.max(0, Math.min(1, ttsVolume));
        audio.play().catch(e => console.error("Playback failed:", e));
    } catch (e) {
        console.error("Audio creation failed:", e);
    }
}
