# Project Requirement Document (PRD): Vasha-AI

## 1. Project Overview
**Project Name**: Vasha-AI (Real-Time AI Speech Translation System)
**Version**: 1.0.0
**Status**: Development / Research Prototype
**Authors/Contributors**: Deep Habiswashi, Soumyadeep Dutta, Sudeshna Mohanty

**Vasha-AI** is a comprehensive, real-time multilingual speech-to-speech translation pipeline designed to break language barriers dynamically. By leveraging state-of-the-art (SOTA) AI models, it enables seamless communication across 200+ global and Indic languages. The system integrates advanced capabilities such as Language Identification (LID), Automatic Speech Recognition (ASR), Machine Translation (MT), and Text-to-Speech (TTS) synthesis into a unified, low-latency framework accessible via a Chrome Extension and a web interface.

This project specifically targets challenges in translating low-resource and code-mixed languages (e.g., Hinglish), preserving named entities (NER), and detecting spoofed audio, making it a robust solution for education, content consumption (YouTube/Twitch), and cross-lingual communication.

---

## 2. Objectives and Goals
### Primary Objectives
1.  **Real-Time Translation**: Enable instant speech-to-speech and speech-to-text translation from any browser tab or microphone input.
2.  **Multilingual Support**: Provide high-accuracy support for 200+ languages, with specialized optimization for Indic languages (Hindi, Bengali, Tamil, etc.).
3.  **Code-Mixing Handling**: Accurately process and translate code-mixed speech (e.g., sentences mixing English and Hindi) common in casual conversation.
4.  **Entity Preservation**: Ensure proper nouns, names, and technical terms (Named Entities) are preserved during translation.
5.  **Accessibility**: Make the technology easily accessible via a lightweight browser extension.

### Research Goals (for Conference Paper)
1.  Evaluate the latency and accuracy trade-offs of combining diverse ASR models (Whisper vs. IndicConformer).
2.  Analyze the effectiveness of NER-preservation techniques in neural machine translation (NMT).
3.  Assess the performance of real-time spoof detection in a translation pipeline.
4.  Demonstrate the viability of client-side audio capture coupled with server-side heavy inference.

---

## 3. Scope and Functional Requirements

### 3.1. Audio Capture & Preprocessing
*   **Input Sources**:
    *   **Tab Audio**: Capture system audio from specific browser tabs (YouTube, Meet, etc.) using `chrome.tabCapture`.
    *   **Microphone**: Capture user voice for dictation or two-way conversation.
    *   **Mixed Mode**: Simultaneous capture of both tab and microphone audio.
*   **Audio Processing**:
    *   **Sampling**: Default capture at system rate, resampled to 16kHz mono for model compatibility.
    *   **VAD (Voice Activity Detection)**: Silence trimming to reduce unnecessary processing.
    *   **Chunking**: Intelligent splitting of audio streams into processable chunks (2-5 seconds) to balance latency and context.

### 3.2. Core Pipeline (The "Vasha Engine")
The backend server orchestrates a sequential pipeline for each audio chunk:

#### A. Language Identification (LID) & Spoof Detection
*   **Function**: Detect the spoken language and verify audio authenticity before processing.
*   **Models**:
    *   Custom finetuned `LanguageIdentifier` (based on Whisper/CLID).
    *   Spoof Detection module to flag AI-generated or synthetic voices.
*   **Output**: ISO Language Code (e.g., `hin_Deva`, `eng_Latn`) and Real/Fake boolean flag.

#### B. Automatic Speech Recognition (ASR)
*   **Function**: Convert audio to text (Speech-to-Text).
*   **Dynamic Model Selection**:
    *   **Indic Languages**: Uses **AI4Bharat IndicConformer** for superior accuracy on Indian languages.
    *   **Global Languages**: Uses **Faster-Whisper (Large-v3)** or **OpenAI Whisper (Small)** based on availability/load.
*   **Features**: Generates word-level timestamps and confidence scores.

#### C. Machine Translation (MT)
*   **Function**: Translate transcribed text to the target language.
*   **Models**:
    *   **Meta NLLB-200 (No Language Left Behind)**: Primary heavy-duty translation model.
    *   **AI4Bharat IndicTrans2**: Specialized for Indic-to-Indic and Indic-to-English translations.
    *   **Google Translate API**: Fast fallback for specific language pairs.
*   **Advanced Logic**:
    *   **NER-Preservation**: Identifies and protects named entities from being translated literally.
    *   **Transliteration**: Option to convert script (e.g., Hindi text in Latin script) without translation.
    *   **Code-Mixed Processing**: Handles sentences varying between languages mid-flow.

#### D. Text-to-Speech (TTS)
*   **Function**: Synthesize translated text back into speech.
*   **Models**:
    *   **Indic-Parler**: Natural-sounding voices for Indian languages.
    *   **Coqui XTTS**: High-quality, clone-capable voices for major languages.
    *   **gTTS (Google TTS)**: Reliable, low-latency fallback.
*   **Output**: Base64 encoded WAV/MP3 audio returned to the client.

### 3.3. User Interface (Chrome Extension)
*   **Architecture**: Manifest V3 compliant.
*   **Popup Interface**:
    *   Language selection (Source & Target).
    *   Start/Stop controls.
    *   Live transcription display (Original & Translated).
*   **Visual Feedback**:
    *   **LID Timeline**: Dynamic color-coded bar showing language changes over time.
    *   **Progress Indicators**: Real-time feedback on processing status.
*   **Overlay (Future)**: Subtitles injected directly into the video player DOM.

---

## 4. System Architecture Summary

The system follows a Client-Server architecture:

1.  **Client (Browser Extension)**:
    *   Captures audio via `chrome.tabCapture` or `navigator.mediaDevices`.
    *   Uses an **Offscreen Document** (Web Audio API / AudioWorklet) to process raw PCM data.
    *   Streams audio chunks via Wi-Fi/Localhost to the Python backend (WebSocket/REST).
    *   Receives and plays back the translated audio and displays text.

2.  **Server (Python Flask)**:
    *   **API Gateway**: Flask application handling REST (`/transcribe_translate`) and WebSocket (`/stream_audio`) requests.
    *   **Model Manager**: Lazy-loading system for heavy AI models to optimize VRAM usage.
    *   **GPU Lock**: Thread-safe management of CUDA resources (`threading.Lock`) to prevent concurrent execution crashes.

3.  **Data Flow**:
    `Audio Source` -> `Extension (Capture)` -> `WebSocket Stream` -> `Server (LID -> ASR -> MT -> TTS)` -> `JSON Response` -> `Extension (Playback/Display)`

---

## 5. Non-Functional Requirements
1.  **Latency**:
    *   End-to-End Latency Target: < 3-5 seconds for live translation.
    *   ASR Latency: < 1 second per chunk.
2.  **Scalability**:
    *   The server is currently designed for single-node execution (local GPU).
    *   Modular design allows for potential containerization (Docker) and horizontal scaling.
3.  **Reliability**:
    *   Fallback mechanisms (e.g., if IndicConformer fails, fallback to Whisper).
    *   Graceful error handling for network interruptions or empty audio chunks.
4.  **Hardware Requirements (Server)**:
    *   **GPU**: NVIDIA GPU with CUDA support (Recommended: 8GB+ VRAM for optimal performance with NLLB/Faster-Whisper).
    *   **RAM**: 16GB+ System RAM.
    *   **Storage**: Models require ~10-20GB depending on cache.

---

## 6. Technology Stack
*   **Frontend**:
    *   HTML5, CSS3, JavaScript (ES6+)
    *   Chrome Extensions API (Manifest V3)
    *   Web Audio API (AudioContext, AudioWorklet)
*   **Backend**:
    *   Python 3.10+
    *   Flask (Web Framework), Flask-Sock (WebSockets), Flask-CORS
*   **AI/ML Libraries**:
    *   **PyTorch**: Core Deep Learning framework.
    *   **Hugging Face Transformers**: Model management.
    *   **Faster-Whisper**: Optimized ASR inference.
    *   **SentencePiece / Sacremoses**: Text tokenization.
    *   **SoundFile / Librosa**: Audio processing.

---

## 7. Future Scope
*   **Speaker Diarization**: Distinguish between multiple speakers in a single audio stream.
*   **Voice Cloning**: Real-time voice cloning to maintaining the speaker's original voice characteristics in the translated audio.
*   **Lip Sync**: (Advanced) AI-generated video modification to sync lip movements with translated audio.
*   **Mobile App**: Porting the client functionality to a React Native mobile application.
