
import os
import sys
import torch
import warnings
import logging
import tempfile
import base64
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
    "specs_route": "/docs"  # This makes the UI available at /docs
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
WHISPER_LANGS = set(['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar', 'tr', 'id']) # Simplified

print("🚀 Vasha-AI Server Starting... Models will load on first request.")

def get_whisper():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("📦 Loading Whisper Model (small)...")
        WHISPER_MODEL = whisper.load_model("small")
    return WHISPER_MODEL

def get_faster_whisper():
    global FASTER_WHISPER_MODEL
    if FASTER_WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            model_size = "large-v3"
            print(f"⚡ Loading Faster-Whisper Model ({model_size})...")
            print("   (This helps download the model if it's the first run, which may take several minutes. Please wait.)")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Attempt to free VRAM before loading a large model
            if device == "cuda":
                torch.cuda.empty_cache()
                
            FASTER_WHISPER_MODEL = WhisperModel(model_size, device=device, compute_type="float16")
            print("✅ Faster-Whisper Loaded!")
        except Exception as e:
            print(f"⚠️ Faster-Whisper load failed: {e}")
            return None
    return FASTER_WHISPER_MODEL

def get_indic_conformer():
    global INDIC_CONFORMER_MODEL
    if INDIC_CONFORMER_MODEL is None:
        try:
            print("🕉️ Loading IndicConformer...")
            INDIC_CONFORMER_MODEL = IndicConformerASR()
        except Exception as e:
            print(f"⚠️ IndicConformer load failed: {e}")
            return None
    return INDIC_CONFORMER_MODEL

def get_lid():
    global LID_MODEL
    if LID_MODEL is None:
        print("🧠 Loading LID Model (Whisper-Small) on CPU to save VRAM...")
        LID_MODEL = LanguageIdentifier(device="cpu")
    return LID_MODEL

# Try to import flask_sock for WebSockets
try:
    from flask_sock import Sock
    sock = Sock(app)
    HAS_WEBSOCKET = True
    print("✅ WebSocket Support Enabled (flask-sock)")
except ImportError:
    HAS_WEBSOCKET = False
    print("⚠️ flask-sock not installed. WebSocket endpoint will be disabled. Run `pip install flask-sock`")

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

if HAS_WEBSOCKET:
    @sock.route('/stream_audio')
    def stream_audio(ws):
        print("🔌 WebSocket Connected")
        try:
            while True:
                data = ws.receive()
                if not data:
                    break
                
                # Expecting JSON with metadata or Raw bytes? 
                # For simplicity, let's assume we receive a JSON with command or raw bytes.
                # If bytes, it's audio. If text, it might be control.
                # Implementation of full streaming pipeline logic here.
                # For this MVP step, we just acknowledge receipt or process minimal chunks.
                pass
        except Exception as e:
            print(f"WebSocket Error: {e}")
        finally:
            print("🔌 WebSocket Disconnected")


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
        mode = request.form.get('mode', 'fast') # fast or quality
        
        target_flores = ISO_TO_FLORES.get(target_lang_iso, "eng_Latn")
        # Quick map overrides
        if target_lang_iso == "hi": target_flores = "hin_Deva"
        if target_lang_iso == "ja": target_flores = "jpn_Jpan"
        if target_lang_iso == "es": target_flores = "spa_Latn"
        if target_lang_iso == "fr": target_flores = "fra_Latn"
        
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
            audio_file.save(temp_in.name)
            input_path = temp_in.name

        # Validation: Check if audio is valid (size > 1KB)
        if os.path.getsize(input_path) < 1024:
            print("⚠️ Audio chunk too small (<1KB), skipping.")
            return jsonify({"status": "empty", "message": "Audio too short/silent"}), 200

        # ---------------------------------------------------------
        # 1. 🔍 Language Identification (Lightweight, First 2s)
        # ---------------------------------------------------------
        lid = get_lid()
        # Run on first 2 seconds only
        detected_lang, confidence_dict = lid.detect(input_path, duration_limit=2.0)
        confidence = confidence_dict.get(detected_lang, 0.0)
        
        if not detected_lang:
            detected_lang = "en"
            print("⚠️ LID failed, defaulting to 'en'")
        else:
            print(f"✅ Realtime Gateway: Detected {detected_lang} (Confidence: {confidence:.2f})")

        # ---------------------------------------------------------
        # 2. 🧠 ASR Selection (Decision Table) with GPU LOCK
        # ---------------------------------------------------------
        text = ""
        asr_used = "unknown"
        
        # Define Indic set for IndicConformer
        INDIC_SET = {'hi', 'bn', 'as', 'or', 'ta', 'te'}
        
        with GPU_LOCK:
            if detected_lang in INDIC_SET:
                conformer = get_indic_conformer()
                if conformer:
                    print(f"📝 ASR Strategy: IndicConformer for {detected_lang}")
                    text = conformer.transcribe(input_path, detected_lang, decoder_type="ctc")
                    asr_used = "indic_conformer"
                else:
                    print("⚠️ IndicConformer unavailable, falling back to Whisper")
                    
            elif detected_lang == 'en':
                fw = get_faster_whisper()
                if fw:
                    print(f"📝 ASR Strategy: Faster-Whisper (Large) for {detected_lang}")
                    segments, _ = fw.transcribe(input_path, language=detected_lang, beam_size=5)
                    text = " ".join([s.text for s in segments]).strip()
                    asr_used = "faster_whisper_large"
                else:
                    model = get_whisper()
                    result = model.transcribe(input_path, language=detected_lang)
                    text = result['text'].strip()
                    asr_used = "whisper_standard"
            else:
                print(f"📝 ASR Strategy: Fallback/General (Whisper Large) for {detected_lang}")
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
            
            # Flush VRAM after heavy ASR
            gc.collect()
            torch.cuda.empty_cache()
        
        # Check if text was generated
        if not text:
             # Final fallback attempt
             if asr_used == "unknown" or asr_used == "indic_conformer":
                 print("⚠️ Primary ASR produced no text, trying Whisper backup...")
                 with GPU_LOCK:
                    model = get_whisper()
                    result = model.transcribe(input_path, language=detected_lang)
                    text = result['text'].strip()
                    asr_used = "whisper_fallback"
                    gc.collect()
                    torch.cuda.empty_cache()

        if not text:
            return jsonify({"status": "empty", "message": "No speech detected"})

        print(f"🎤 Transcript ({detected_lang}) [{asr_used}]: {text}")

        # ---------------------------------------------------------
        # 3. 🌍 Translation (MT) - Fast vs Quality
        # ---------------------------------------------------------
        print(f"🔄 Translation Mode: {mode.upper()}")
        
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
            
        print(f"💬 Translated ({target_flores}): {translated_text}")

        # ---------------------------------------------------------
        # 4. 🔊 TTS (Voiceover) - Unchanged but Locked
        # ---------------------------------------------------------
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
                print(f"❌ TTS Failed: {e}")

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
        print(f"❌ Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 INITIALIZING VASHA-AI SERVER")
    print("="*50)
    print("⏳ Pre-loading critical models to prevent runtime lags...")
    
    # 1. Load LID (Fast)
    get_lid()
    
    # 2. Load Faster-Whisper (Heavy)
    # We load it now so the First Request doesn't time out the browser
    fw = get_faster_whisper()
    if fw:
        print("✅ Whisper Large-v3 Ready")
    else:
        print("⚠️ Whisper Large-v3 Failed to Load (Will retry on request, but expect delays)")

    print("\n✅ SERVER READY - LISTENING FOR REQUESTS")
    print("="*50 + "\n")
    
    # use_reloader=False is CRITICAL for:
    # 1. Allowing Ctrl+C to work properly on Windows
    # 2. Preventing double-loading of heavy models (VRAM issues)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)
