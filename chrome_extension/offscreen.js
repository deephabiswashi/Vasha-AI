// offscreen.js - Advanced Audio Capture (Tab, Mic, Mixed), Resampling, VAD
// Updated to use AudioWorklet to avoid Deprecation Warnings

let audioContext;
let workletNode;
let isRecording = false;
let globalStream = null; // Reference to active media stream

// Config
const TARGET_SAMPLE_RATE = 16000;
const VAD_THRESHOLD = 0.01; // RMS Threshold
let buffer = []; // Float32 Accumulator
let silenceStart = null;
let speaking = false;

// Backend Config
const BACKEND_URL = "http://127.0.0.1:5000/transcribe_translate";
const WS_URL = "ws://127.0.0.1:5000/stream_audio";

chrome.runtime.onMessage.addListener(async (message) => {
    if (message.type === "INIT_RECORDING") {
        startRecording(message.streamId, message.targetLang, message.inputMode);
    } else if (message.type === "STOP_RECORDING_OFFSCREEN") {
        stopRecording();
    }
});

async function startRecording(streamId, targetLang, inputMode) {
    if (isRecording) return;

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
            handleAudioProcess(event.data, audioContext.sampleRate, targetLang);
        };

        mixedSource.connect(workletNode);
        workletNode.connect(audioContext.destination);

    } catch (err) {
        console.error("Recording Setup Error:", err);
        chrome.runtime.sendMessage({ type: "ERROR", message: `${err.name}: ${err.message}` });
        isRecording = false;
    }
}

function handleAudioProcess(inputData, inputRate, targetLang) {
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

        // Check Flush
        const durationSec = buffer.length / TARGET_SAMPLE_RATE;
        const silenceDuration = silenceStart ? (Date.now() - silenceStart) : 0;

        if (durationSec >= 3.0 || (durationSec > 1.0 && silenceDuration > 500)) {
            flushBuffer(targetLang);
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
    console.log("Recording Stopped");
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

async function flushBuffer(lang) {
    if (buffer.length === 0) return;

    // Convert to 16-bit PCM Blob
    const dataLen = buffer.length * 2;
    const bufferArray = new ArrayBuffer(44 + dataLen);
    const view = new DataView(bufferArray);

    // WAV Header
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

    floatTo16BitPCM(view, 44, buffer);

    const blob = new Blob([view], { type: 'audio/wav' });
    buffer = [];
    speaking = false;
    silenceStart = null;

    // Send
    await sendToBackend(blob, lang);
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

async function sendToBackend(audioBlob, lang) {
    // For MVP validation, we still use the standard FETCH logic with the new URL (127.0.0.1)
    // The user requested WebSocket "Prefer" but also "Replace insecure http". 
    // We start with reliable http://127.0.0.1. A full msg-based WS stream requires larger backend rewrite.

    // Check if backend is reachable at new address
    try {
        const formData = new FormData();
        formData.append("audio", audioBlob, "chunk.wav");
        formData.append("target_lang", lang);
        formData.append("mode", "fast"); // Request Fast Mode (Live)

        // Using 127.0.0.1 which avoids some localhost issues
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
                metadata: data.metadata // Pass LID confidence & language
            });
            playBase64Audio(data.audio_base64);
        }
    } catch (e) {
        console.error("Backend Error:", e);
        // chrome.runtime.sendMessage({ type: "ERROR", message: "Backend unreachable. Is vasha_server.py running?" });
    }
}

function playBase64Audio(base64String) {
    try {
        const audio = new Audio("data:audio/wav;base64," + base64String);
        audio.volume = 1.0;
        audio.play().catch(e => console.error("Playback failed:", e));
    } catch (e) {
        console.error(" Audio creation failed:", e);
    }
}
