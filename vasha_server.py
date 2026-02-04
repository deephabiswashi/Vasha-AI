import os
import sys
import torch
import warnings
import logging
import tempfile
import base64
import json
from flask import Flask, request, jsonify, send_file, redirect
from flask_cors import CORS
from flasgger import Swagger
from transformers import logging as hf_logging
import threading
import gc

# Global GPU Lock to prevent CUDA collisions in threaded Flask
GPU_LOCK = threading.Lock()

# Silence logs
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.getcwd())

# Import Vasha Modules
from LID_Model.lid import LanguageIdentifier
from ASR_Model.indic_conformer.conformer_asr import IndicConformerASR
from MT_Model.mt_helper import perform_translation, ISO_TO_FLORES
from TTS_Model.tts_common.tts_handler import run_universal_tts
import whisper

app = Flask(__name__)

# Enable CORS with PNA support
# Access-Control-Allow-Private-Network: true is required for Chrome to allow 
# extension/web requests to localhost from non-secure (or public) contexts.
CORS(app, resources={r"/*": {
    "origins": "*",
    "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Private-Network"],
    "methods": ["GET", "POST", "OPTIONS"]
}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

# --- Swagger Config ---
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}

template = {
    "swagger": "2.0",
    "info": {
        "title": "Vasha-AI API",
        "description": "Real-time AI Audio Translation Pipeline",
        "version": "1.0.0"
    }
}

swagger = Swagger(app, config=swagger_config, template=template)

# --- Global Models (Lazy Loading) ---
WHISPER_MODEL = None
FASTER_WHISPER_MODEL = None
INDIC_CONFORMER_MODEL = None
LID_MODEL = None

# ASR Constants
CONFORMER_LANGS = {
    'as','bn','brx','doi','gu','hi','kn','kok','ks','mai','ml','mni','mr','ne',
    'or','pa','sa','sat','sd','ta','te','ur'
}
WHISPER_LANGS = set(['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar', 'tr', 'id'])

print("Vasha-AI Server Starting... Models will load on first request.")

def get_whisper():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("Loading Whisper Model (small)...")
        WHISPER_MODEL = whisper.load_model("small")
    return WHISPER_MODEL

def get_faster_whisper():
    global FASTER_WHISPER_MODEL
    if FASTER_WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            model_size = "large-v3"
            print(f"Loading Faster-Whisper Model ({model_size})...")
            print("First run may download model and take several minutes.")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                torch.cuda.empty_cache()

            FASTER_WHISPER_MODEL = WhisperModel(model_size, device=device, compute_type="float16")
            print("Faster-Whisper Loaded.")
        except Exception as e:
            print(f"Faster-Whisper load failed: {e}")
            return None
    return FASTER_WHISPER_MODEL

def get_indic_conformer():
    global INDIC_CONFORMER_MODEL
    if INDIC_CONFORMER_MODEL is None:
        try:
            print("Loading IndicConformer...")
            INDIC_CONFORMER_MODEL = IndicConformerASR()
        except Exception as e:
            print(f"IndicConformer load failed: {e}")
            return None
    return INDIC_CONFORMER_MODEL

def get_lid():
    global LID_MODEL
    if LID_MODEL is None:
        print("Loading LID Model (Whisper-Small) on CPU to save VRAM...")
        LID_MODEL = LanguageIdentifier(device="cpu")
    return LID_MODEL

# Try to import flask_sock for WebSockets
try:
    from flask_sock import Sock
    sock = Sock(app)
    HAS_WEBSOCKET = True
    print("WebSocket Support Enabled (flask-sock)")
except ImportError:
    HAS_WEBSOCKET = False
    print("flask-sock not installed. WebSocket endpoint disabled. Run `pip install flask-sock`")

@app.route('/')
def home():
    return redirect('/docs')

@app.route('/health', methods=['GET'])
def health():
    """
    Health Check
    ---
    tags:
      - System
    responses:
      200:
        description: System is ready
    """
    return jsonify({
        "status": "ok",
        "message": "Vasha-AI Backend Ready",
        "websocket_enabled": HAS_WEBSOCKET
    })

def run_asr_chunk(input_path, detected_lang, model_choice, word_timestamps):
    text = ""
    asr_used = "unknown"
    words = None

    with GPU_LOCK:
        if model_choice == "indic_conformer":
            conformer = get_indic_conformer()
            if conformer:
                text = conformer.transcribe(input_path, detected_lang, decoder_type="ctc")
                asr_used = "indic_conformer"

        if not text and model_choice == "faster_whisper":
            fw = get_faster_whisper()
            if fw:
                segments, _ = fw.transcribe(
                    input_path,
                    language=detected_lang,
                    beam_size=5,
                    word_timestamps=word_timestamps
                )
                text = " ".join([s.text for s in segments]).strip()
                asr_used = "faster_whisper"
                if word_timestamps:
                    words = []
                    for s in segments:
                        if not s.words:
                            continue
                        for w in s.words:
                            words.append({
                                "word": w.word.strip(),
                                "start_time": w.start,
                                "end_time": w.end
                            })

        if not text:
            model = get_whisper()
            result = model.transcribe(input_path, language=detected_lang)
            text = result['text'].strip()
            asr_used = "whisper_standard"

        gc.collect()
        torch.cuda.empty_cache()

    return text, asr_used, words

if HAS_WEBSOCKET:
    @sock.route('/stream_audio')
    def stream_audio(ws):
        print("WebSocket Connected")

        state = {
            "target_lang": "en",
            "asr_model": "faster_whisper",
            "partial_enabled": True,
            "word_timestamps": False
        }

        try:
            while True:
                data = ws.receive()
                if data is None:
                    break

                if isinstance(data, bytes):
                    continue

                try:
                    msg = json.loads(data)
                except Exception:
                    continue

                if msg.get("type") == "control":
                    state["target_lang"] = msg.get("target_lang", state["target_lang"])
                    state["asr_model"] = msg.get("asr_model", state["asr_model"])
                    state["partial_enabled"] = msg.get("partial_enabled", state["partial_enabled"])
                    state["word_timestamps"] = msg.get("word_timestamps", state["word_timestamps"])
                    continue

                if msg.get("type") != "audio_chunk":
                    continue

                audio_b64 = msg.get("audio_b64")
                if not audio_b64:
                    continue

                is_final = bool(msg.get("is_final", False))
                partial_enabled = bool(msg.get("partial_enabled", state["partial_enabled"]))
                word_ts = bool(msg.get("word_timestamps", state["word_timestamps"]))
                segment_id = msg.get("segment_id", 0)
                segment_start_time = msg.get("segment_start_time", None)
                segment_end_time = msg.get("segment_end_time", None)

                if (not is_final) and (not partial_enabled):
                    continue

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception:
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
                    temp_in.write(audio_bytes)
                    input_path = temp_in.name

                if os.path.getsize(input_path) < 1024:
                    continue

                # LID
                lid = get_lid()
                detected_lang, confidence_dict = lid.detect(input_path, duration_limit=2.0)
                confidence = confidence_dict.get(detected_lang, 0.0)
                if not detected_lang:
                    detected_lang = "en"

                # Determine ASR model (backend override)
                preferred_model = msg.get("asr_model", state["asr_model"])
                effective_model = preferred_model
                indic_set = {'hi', 'bn', 'as', 'or', 'ta', 'te'}
                if detected_lang in indic_set:
                    effective_model = "indic_conformer"
                else:
                    effective_model = "faster_whisper"

                if effective_model != preferred_model:
                    ws.send(json.dumps({
                        "type": "asr_override",
                        "model": effective_model,
                        "reason": "language_switch",
                        "detected_language": detected_lang
                    }))

                # ASR
                text, asr_used, words = run_asr_chunk(
                    input_path,
                    detected_lang,
                    effective_model,
                    word_ts
                )

                if not text:
                    continue

                response = {
                    "type": "asr_final" if is_final else "asr_partial",
                    "segment_id": segment_id,
                    "text": text,
                    "words": words,
                    "asr_model": asr_used,
                    "detected_language": detected_lang,
                    "confidence": confidence,
                    "segment_start_time": segment_start_time,
                    "segment_end_time": segment_end_time
                }
                ws.send(json.dumps(response))
        except Exception as e:
            print(f"WebSocket Error: {e}")
            try:
                ws.send(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass
        finally:
            print("WebSocket Disconnected")

@app.route('/transcribe_translate', methods=['POST'])
def process_audio():
    """
    Process Audio Chunk
    ---
    tags:
      - Pipeline
    consumes:
      - multipart/form-data
    parameters:
      - name: audio
        in: formData
        type: file
        required: true
        description: Audio chunk (WAV/MP3) to process
      - name: target_lang
        in: formData
        type: string
        required: false
        default: en
        description: Target language ISO code (e.g. hi, es, fr)
    responses:
      200:
        description: Processed result
    """
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        target_lang_iso = request.form.get('target_lang', 'en')
        mode = request.form.get('mode', 'fast')

        target_flores = ISO_TO_FLORES.get(target_lang_iso, "eng_Latn")
        if target_lang_iso == "hi": target_flores = "hin_Deva"
        if target_lang_iso == "ja": target_flores = "jpn_Jpan"
        if target_lang_iso == "es": target_flores = "spa_Latn"
        if target_lang_iso == "fr": target_flores = "fra_Latn"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
            audio_file.save(temp_in.name)
            input_path = temp_in.name

        if os.path.getsize(input_path) < 1024:
            return jsonify({"status": "empty", "message": "Audio too short/silent"}), 200

        # 1. Language Identification
        lid = get_lid()
        detected_lang, confidence_dict = lid.detect(input_path, duration_limit=2.0)
        confidence = confidence_dict.get(detected_lang, 0.0)

        if not detected_lang:
            detected_lang = "en"
            print("LID failed, defaulting to 'en'")
        else:
            print(f"Detected {detected_lang} (Confidence: {confidence:.2f})")

        # 2. ASR Selection with GPU LOCK
        text = ""
        asr_used = "unknown"

        INDIC_SET = {'hi', 'bn', 'as', 'or', 'ta', 'te'}

        with GPU_LOCK:
            if detected_lang in INDIC_SET:
                conformer = get_indic_conformer()
                if conformer:
                    text = conformer.transcribe(input_path, detected_lang, decoder_type="ctc")
                    asr_used = "indic_conformer"
                else:
                    print("IndicConformer unavailable, falling back to Whisper")

            elif detected_lang == 'en':
                fw = get_faster_whisper()
                if fw:
                    segments, _ = fw.transcribe(input_path, language=detected_lang, beam_size=5)
                    text = " ".join([s.text for s in segments]).strip()
                    asr_used = "faster_whisper_large"
                else:
                    model = get_whisper()
                    result = model.transcribe(input_path, language=detected_lang)
                    text = result['text'].strip()
                    asr_used = "whisper_standard"
            else:
                fw = get_faster_whisper()
                if fw:
                    segments, _ = fw.transcribe(input_path, language=detected_lang, beam_size=5)
                    text = " ".join([s.text for s in segments]).strip()
                    asr_used = "faster_whisper_large"
                else:
                    model = get_whisper()
                    result = model.transcribe(input_path, language=detected_lang)
                    text = result['text'].strip()
                    asr_used = "whisper_standard"

            gc.collect()
            torch.cuda.empty_cache()

        if not text:
            if asr_used == "unknown" or asr_used == "indic_conformer":
                with GPU_LOCK:
                    model = get_whisper()
                    result = model.transcribe(input_path, language=detected_lang)
                    text = result['text'].strip()
                    asr_used = "whisper_fallback"
                    gc.collect()
                    torch.cuda.empty_cache()

        if not text:
            return jsonify({"status": "empty", "message": "No speech detected"})

        print(f"Transcript ({detected_lang}) [{asr_used}]: {text}")

        # 3. Translation (MT) - Fast vs Quality
        print(f"Translation Mode: {mode.upper()}")

        translated_text = ""
        with GPU_LOCK:
            translated_text = perform_translation(
                text,
                detected_lang,
                target_flores,
                backend_choice="nllb"
            )
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Translated ({target_flores}): {translated_text}")

        # 4. TTS
        output_tts_filename = f"out_{os.path.basename(input_path)}"
        out_dir = os.path.join(os.getcwd(), "sessions", "server_temp")
        os.makedirs(out_dir, exist_ok=True)

        tts_path = ""

        with GPU_LOCK:
            try:
                tts_path = run_universal_tts(
                    text=translated_text,
                    target_lang=target_flores,
                    prefer="xtts",
                    out_dir=out_dir,
                    out_name=output_tts_filename,
                    reference_audio=None
                )
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"TTS Failed: {e}")

        audio_b64 = ""
        if tts_path and os.path.exists(tts_path):
            with open(tts_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            "status": "success",
            "metadata": {
                "detected_language": detected_lang,
                "confidence": confidence,
                "asr_model": asr_used,
                "mt_mode": mode
            },
            "transcribed_text": text,
            "translated_text": translated_text,
            "audio_base64": audio_b64,
            "target_lang": target_flores
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("INITIALIZING VASHA-AI SERVER")
    print("="*50)
    print("Pre-loading critical models to prevent runtime lags...")

    # 1. Load LID (Fast)
    get_lid()

    # 2. Load Faster-Whisper (Heavy)
    fw = get_faster_whisper()
    if fw:
        print("Whisper Large-v3 Ready")
    else:
        print("Whisper Large-v3 Failed to Load (Will retry on request)")

    print("\nSERVER READY - LISTENING FOR REQUESTS")
    print("="*50 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)
