import torch
import sys
import os
import re
import argparse
from datetime import datetime
import shutil
import whisper
import math
import subprocess
import tempfile
from tqdm import tqdm
import importlib
import warnings
from transformers import logging as hf_logging

# Silence verbose HF/model logs during TTS selection/output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

import logging, warnings, os
from transformers import logging as hf_logging

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Optional: silence other loggers
for name in list(logging.root.manager.loggerDict):
    logging.getLogger(name).setLevel(logging.ERROR)

sys.path.append(os.getcwd())

from LID_Model.lid import (
    LanguageIdentifier,
    detect_dialect,
    record_live_audio,
    extract_audio_ffmpeg,
    download_youtube_audio,
    is_spoofed_audio,
    TARGET_LANGS,
)
from ASR_Model.indic_conformer.conformer_asr import IndicConformerASR

# ✅ Import full MT stack
from MT_Model.mt_model import ISO_TO_FLORES, batch_translate_text
from MT_Model.mt_helper import (
    translate_text,
    GLOBAL_LANGS,
    INDIC_LANGS_MENU,
    perform_translation,
    batch_translate_via_perform,   # ✅ centralized batch function
)
from MT_Model import mt_debug  # debug module


"""
Unified TTS integration using `TTS_Model.tts_common.tts_handler.run_universal_tts`.
This selects between Indic-Parler, Coqui XTTS, and gTTS with caching and chunking.
"""
TTS_AVAILABLE = True
try:
    from TTS_Model.tts_common.tts_handler import run_universal_tts
except Exception as e:
    print(f"[WARN] TTS handler unavailable: {e}")
    TTS_AVAILABLE = False

# ===========================
# Supported languages for IndicConformer (ISO codes only)
# ===========================
CONFORMER_LANGS = {
    'as','bn','brx','doi','gu','hi','kn','kok','ks','mai','ml','mni','mr','ne',
    'or','pa','sa','sat','sd','ta','te','ur'
}

# ✅ Supported languages for Whisper (ISO codes in TARGET_LANGS)
WHISPER_LANGS = set(TARGET_LANGS.keys())


# -------------------
# Helper functions
# -------------------
def get_language_for_model(lang_code):
    if lang_code in TARGET_LANGS:
        return lang_code
    name_lower = str(lang_code).lower()
    for code, name in TARGET_LANGS.items():
        if name.lower() == name_lower:
            return code
    return lang_code


def auto_select_asr(lang_code):
    if lang_code in CONFORMER_LANGS:
        return "conformer"
    elif lang_code in WHISPER_LANGS:
        return "whisper"
    else:
        return "faster"


def user_select_asr(lang_code):
    print("\n🤖 Select ASR Mode:")
    print("1. Whisper (OpenAI)")
    print("2. IndicConformer (AI4Bharat)")
    print("3. Faster-Whisper (batched, fast)")
    print("4. Auto-detect (based on language)")
    choice = input("👉 Enter 1, 2, 3 or 4 (default=4): ").strip()

    if choice == "1":
        return "whisper"
    elif choice == "2":
        return "conformer"
    elif choice == "3":
        return "faster"
    else:
        selected = auto_select_asr(lang_code)
        if not selected:
            print(f"⚠️ No ASR supports {lang_code}. Defaulting to Whisper.")
            return "whisper"
        print(f"✅ Auto-detected ASR: {selected}")
        return selected


def user_select_tts():
    """CLI prompt for TTS model: 1=Coqui/XTTS, 2=gTTS, 3=IndicParler."""
    print("\n🔊 Select TTS Model:")
    print("1. Coqui TTS / XTTS (multilingual, voice cloning)")
    print("2. gTTS (Google, fast, many languages)")
    print("3. IndicParler TTS (Indian languages, natural)")
    choice = input("👉 Enter 1, 2 or 3 (default=1): ").strip() or "1"
    if choice == "2":
        return "gtts"
    elif choice == "3":
        return "indic"
    else:
        return "xtts"


def download_youtube_audio_cached(url):
    cache_dir = os.path.join(os.getcwd(), "youtube_cache")
    os.makedirs(cache_dir, exist_ok=True)

    match = re.search(r"(?:v=|be/)([A-Za-z0-9_-]{11})", url)
    if match:
        video_id = match.group(1)
    else:
        video_id = re.sub(r'\W+', '', url)

    cached_path = os.path.join(cache_dir, f"{video_id}.wav")

    if os.path.exists(cached_path):
        if os.path.getsize(cached_path) > 1024:
            print(f"⚡ Using cached YouTube audio: {cached_path}")
            return cached_path
        else:
            print(f"⚠️ Cached file seems corrupted, re-downloading: {cached_path}")
            os.remove(cached_path)

    print("⬇️ Downloading YouTube audio...")
    try:
        path = download_youtube_audio(url)
        shutil.move(path, cached_path)
        return cached_path
    except Exception as e:
        print(f"❌ YouTube download failed: {e}")
        try:
            print("🔁 Retrying download...")
            path = download_youtube_audio(url)
            shutil.move(path, cached_path)
            return cached_path
        except Exception as e2:
            print(f"❌ Fallback YouTube download also failed: {e2}")
            raise RuntimeError("YouTube download failed after retry.")


# ---------- ffmpeg utilities ----------
def get_duration_ffprobe(path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    try:
        return float(out)
    except:
        return 0.0


def extract_chunk_ffmpeg(src, start, duration, dst):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(max(0.0, start)),
        "-t", str(duration),
        "-i", src,
        "-ar", "16000", "-ac", "1", "-vn", dst
    ]
    subprocess.run(cmd, check=True)


def make_overlapped_chunks(audio_path, chunk_length=120, overlap=5):
    total = get_duration_ffprobe(audio_path)
    if total == 0.0:
        return [(audio_path, 0.0)]
    tmpdir = tempfile.mkdtemp(prefix="chunks_")
    starts = []
    step = chunk_length - overlap
    s = 0.0
    while s < total:
        starts.append(s)
        s += step
    chunks = []
    for idx, s in enumerate(starts):
        remaining = max(0.0, total - s)
        dur = min(chunk_length + (overlap if idx > 0 else 0), remaining)
        start_for_extract = max(0.0, s - (overlap if idx > 0 else 0))
        out_path = os.path.join(tmpdir, f"chunk_{idx:03d}.wav")
        extract_chunk_ffmpeg(audio_path, start_for_extract, dur, out_path)
        chunks.append((out_path, s))
    return chunks


# ---------- Whisper + faster-whisper workers ----------
def _transcribe_one_chunk_whisper(model, chunk_path, lang_input, use_word_ts=False):
    """
    use_word_ts=False avoids Whisper's Triton-based timing (median_filter, DTW).
    On Windows Triton often fails -> slow CPU fallbacks + warnings. Segment-level
    timing is still returned; _stitch_segments does not need word-level.
    """
    res = model.transcribe(
        chunk_path,
        task="transcribe",
        language=lang_input,
        word_timestamps=use_word_ts,
        verbose=False
    )
    return {'segments': res.get('segments', []), 'text': res.get('text', '')}


def _transcribe_one_chunk_faster(fw_model, chunk_path, lang_input, beam_size=5, use_word_ts=False):
    segments_gen, info = fw_model.transcribe(
        chunk_path,
        beam_size=beam_size,
        language=lang_input,
        word_timestamps=use_word_ts,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
        temperature=0
    )
    segs = [{'start': seg.start, 'end': seg.end, 'text': seg.text} for seg in segments_gen]
    full_text = " ".join(s['text'] for s in segs)
    return {'segments': segs, 'text': full_text}


def _stitch_segments(all_segment_lists, overlap_guard=0.3):
    merged = []
    last_end = 0.0
    for segments, offset in all_segment_lists:
        for seg in segments:
            s = float(seg.get('start', 0.0)) + offset
            e = float(seg.get('end', s)) + offset
            text = seg.get('text', '').strip()
            if not text:
                continue
            if s < (last_end - overlap_guard):
                if merged and text.startswith(merged[-1]['text'][-30:]):
                    text = text[len(merged[-1]['text'][-30:]):].lstrip()
                if not text:
                    continue
                s = max(s, last_end)
            merged.append({'start': s, 'end': e, 'text': text})
            last_end = max(last_end, e)
    pieces = [m['text'] for m in merged]
    out = " ".join(pieces)
    out = re.sub(r'\s+', ' ', out).strip()
    return out


# -------------------
# Main transcription
# -------------------
def transcribe(audio_path, language_code, session_dir, asr_model, chunk_length=120, overlap=5, workers=1):
    print(f"📝 Transcribing with {asr_model}...")
    lang_input = get_language_for_model(language_code)

    if asr_model == "whisper":
        model = whisper.load_model("large")
        audio_length = math.ceil(os.path.getsize(audio_path) / (16000*2))
        if audio_length > chunk_length:
            print("⏳ Long audio detected. Splitting...")
            chunks = make_overlapped_chunks(audio_path, chunk_length=chunk_length, overlap=overlap)
            results_by_index = {}
            for i, (ch_path, start) in enumerate(tqdm(chunks, desc="Transcribing chunks", unit="chunk"), 0):
                out = _transcribe_one_chunk_whisper(model, ch_path, lang_input)
                results_by_index[i] = (out['segments'], start)
            ordered = [results_by_index[i] for i in sorted(results_by_index.keys())]
            transcribed_text = _stitch_segments(ordered)
        else:
            result = model.transcribe(audio_path, task="transcribe", language=lang_input)
            transcribed_text = result["text"]

    elif asr_model == "faster":
        from faster_whisper import WhisperModel
        model_size = "large-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Loading faster-whisper model ({model_size}) on {device} ...")
        fw_model = WhisperModel(model_size, device=device, compute_type="float16")
        audio_length = math.ceil(os.path.getsize(audio_path) / (16000*2))
        if audio_length > chunk_length:
            chunks = make_overlapped_chunks(audio_path, chunk_length=chunk_length, overlap=overlap)
            results_by_index = {}
            for i, (ch_path, start) in enumerate(tqdm(chunks, desc="Transcribing chunks (faster-whisper)", unit="chunk"), 0):
                try:
                    out = _transcribe_one_chunk_faster(fw_model, ch_path, lang_input)
                except Exception as e:
                    print(f"❌ faster-whisper chunk {i} failed: {e}")
                    out = {'segments': [], 'text': ''}
                results_by_index[i] = (out['segments'], start)
            ordered = [results_by_index[i] for i in sorted(results_by_index.keys())]
            transcribed_text = _stitch_segments(ordered)
        else:
            segments, info = fw_model.transcribe(audio_path, beam_size=5, language=lang_input)
            segs = [{'start': s.start, 'end': s.end, 'text': s.text} for s in segments]
            transcribed_text = " ".join(s['text'] for s in segs)

    elif asr_model == "conformer":
        conformer = IndicConformerASR()
        transcribed_text = conformer.transcribe(audio_path, get_language_for_model(language_code), decoder_type="ctc")
    else:
        raise ValueError("Invalid ASR model selected.")

    print("\n🗣 Transcribed Text:")
    print(transcribed_text)

    output_filename = f"output_{language_code}_{asr_model}.txt"
    output_path = os.path.join(session_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcribed_text)
    print(f"\n💾 Transcription saved to: {output_path}")
    return transcribed_text


# -------------------
# Script entry
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🌍 Real-time LID + ASR + MT pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file')
    group.add_argument('--youtube')
    group.add_argument('--mic', action='store_true')
    parser.add_argument('--duration', type=int, default=5)
    parser.add_argument('--asr', choices=['whisper','conformer','faster'], nargs='?', const=None)
    parser.add_argument('--chunk-len', type=int, default=120)
    parser.add_argument('--overlap', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--backtranslate', action="store_true", help="Enable back-translation debug mode")

    # --- TTS CLI options ---
    parser.add_argument('--tts', action='store_true', help="Enable TTS of the translated text (uses TTS Model if available)")
    parser.add_argument('--tts-model', choices=['xtts', 'gtts', 'indic'], default=None, help="TTS model: xtts (Coqui/XTTS), gtts, or indic (IndicParler). If omitted, prompts interactively.")
    parser.add_argument('--tts-play', action='store_true', help="Attempt to play the TTS output locally (requires ffplay or OS player)")
    parser.add_argument('--tts-desc', type=str, default="", help="Optional natural-language voice description for TTS (overrides defaults)")
    parser.add_argument('--tts-save', type=str, default="", help="Optional filename to save the TTS output (relative to session dir). Example: translated.mp3")
    parser.add_argument('--tts-speaker-wav', type=str, default="", help="Path to a reference WAV file of YOUR voice (used by XTTS v2 for voice cloning across all languages).")

    args = parser.parse_args()

    # Session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = os.path.join(os.getcwd(), "sessions")
    os.makedirs(session_root, exist_ok=True)
    session_dir = os.path.join(session_root, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    # Input
    if args.file:
        audio_path = extract_audio_ffmpeg(args.file) if args.file.lower().endswith(('.mp4','.mkv','.mov','.avi')) else args.file
    elif args.youtube:
        audio_path = download_youtube_audio_cached(args.youtube)
    elif args.mic:
        audio_path = record_live_audio(duration=args.duration)
    else:
        sys.exit(1)

    # LID
    lid = LanguageIdentifier()
    lang_code, _ = lid.detect(audio_path)
    if not lang_code:
        print("❌ Language not detected.")
        sys.exit(1)

    # ASR
    selected_asr = args.asr if args.asr else user_select_asr(lang_code)
    transcribed = transcribe(audio_path, lang_code, session_dir, selected_asr,
                             chunk_length=args.chunk_len, overlap=args.overlap, workers=args.workers)

    # ===================
    # MT Mode Selection
    # ===================
    print("\n🤖 Select MT Mode:")
    print("1. Meta-NLLB (Global)")
    print("2. IndicTrans2 (Indian languages, fallback to NLLB)")
    print("3. Google Translate (free API wrapper)")
    print("4. Transliteration (script-only, no meaning change)")
    print("5. Code-mixed handling (Hindi+English, etc.)")
    mt_choice = input("👉 Enter 1-5 (default=1): ").strip() or "1"

    mt_model_choice = "nllb"
    mode = None
    translit_scheme = None
    code_mixed_english_action = "pass"

    if mt_choice == "2":
        mt_model_choice = "indic"
    elif mt_choice == "3":
        mt_model_choice = "google"
        # Ensure googletrans is available; otherwise fall back to NLLB to avoid [translation_error]
        try:
            from googletrans import Translator
            Translator()
        except Exception:
            print("⚠️ googletrans not available. Install with: pip install googletrans==4.0.0-rc1")
            print("   Falling back to Meta-NLLB for this run. (Or re-run and choose MT mode 1 or 2.)")
            mt_model_choice = "nllb"
    elif mt_choice == "4":
        mode = "transliterate"
    elif mt_choice == "5":
        mode = "code_mixed"

    # Target language menus
    if mode == "transliterate":
        print("\n🔤 Transliteration selected — choose Latin scheme:")
        print("1. ITRANS (default)")
        print("2. IAST")
        print("3. HK (Harvard-Kyoto)")
        ts_choice = input("👉 Enter 1-3 (default=1): ").strip()
        if ts_choice == "2":
            translit_scheme = "IAST"
        elif ts_choice == "3":
            translit_scheme = "HK"
        else:
            translit_scheme = "ITRANS"
        # 🔄 Fix: tgt_lang irrelevant, set equal to src
        tgt_lang = lang_code
    else:
        print("\n🌍 Choose target translation language:")
        print("=== Global Languages ===")
        for key,(name,iso) in GLOBAL_LANGS.items():
            flores = ISO_TO_FLORES.get(iso, iso)
            print(f"{key}. {name} ({flores})")
        print("\n=== Indian Languages ===")
        for key,(name,iso) in INDIC_LANGS_MENU.items():
            flores = ISO_TO_FLORES.get(iso, iso)
            print(f"{key}. {name} ({flores})")
        print("0. Custom FLORES code")

        choice = input("👉 Enter choice (default=1 for English): ").strip()
        if choice == "" or choice == "1":
            tgt_lang = ISO_TO_FLORES["en"]
        elif choice == "0":
            tgt_lang = input("🔤 Enter custom FLORES code: ").strip()
        elif choice in GLOBAL_LANGS:
            iso = GLOBAL_LANGS[choice][1]
            tgt_lang = ISO_TO_FLORES.get(iso, "eng_Latn")
        elif choice in INDIC_LANGS_MENU:
            iso = INDIC_LANGS_MENU[choice][1]
            tgt_lang = ISO_TO_FLORES.get(iso, iso)
        else:
            tgt_lang = ISO_TO_FLORES["en"]

        if mode == "code_mixed":
            print("\n🧩 Code-mixed selected. Handle Latin runs how?")
            print("1. Pass-through (default)")
            print("2. Translate English runs")
            cm_choice = input("👉 Enter 1 or 2 (default=1): ").strip()
            if cm_choice == "2":
                code_mixed_english_action = "translate"

    # ===================
    # Debug Options Menu
    # ===================
    if mode == "transliterate":
        dbg_choice = "2"  # force batch only
    else:
        print("\n🛠️ Debug Options:")
        print("1. Normal translation")
        print("2. Batch translation (default)")
        print("3. Back-translation debug")
        print("4. NER-preservation mode (keep names)")
        dbg_choice = input("👉 Enter 1-4 (default=2): ").strip() or "2"
        
# -------------------
# Final Translation Logic (compose MT + Debug)
# -------------------
translated = ""
if dbg_choice == "3":
    bt = mt_debug.back_translate(
        transcribed,
        lang_code,
        tgt_lang,
        backend_choice=mt_model_choice,   # ✅ ensure same MT backend
        mode=mode,
        translit_scheme=translit_scheme,
        code_mixed_english_action=code_mixed_english_action,
    )
    print("➡️ Forward:", bt["forward"])
    print("⬅️ Backward:", bt["backward"])
    translated = bt["forward"]

elif dbg_choice == "4":
    translated = batch_translate_via_perform(
        transcribed,
        lang_code,
        tgt_lang,
        backend_choice=mt_model_choice,
        mode="ner" if mode is None else mode,
        translit_scheme=translit_scheme,
        code_mixed_english_action=code_mixed_english_action,
    )

elif dbg_choice == "1":
    translated = perform_translation(
        transcribed,
        lang_code,
        tgt_lang,
        backend_choice=mt_model_choice,
        mode=mode,
        translit_scheme=translit_scheme,
        code_mixed_english_action=code_mixed_english_action,
    )

else:
    translated = batch_translate_via_perform(
        transcribed,
        lang_code,
        tgt_lang,
        backend_choice=mt_model_choice,
        mode=mode,
        translit_scheme=translit_scheme,
        code_mixed_english_action=code_mixed_english_action,
    )

# ✅ Save translation (for all MT/debug modes)
translation_file = os.path.join(session_dir, f"translation_{lang_code}_to_{tgt_lang}.txt")
with open(translation_file, "w", encoding="utf-8") as f:
    f.write(translated)
print("\n💬 Translation:")
print(translated)
print(f"💾 Translation saved to: {translation_file}")

# ✅ CLI-based optional back-translation debug
if args.backtranslate:
    bt = mt_debug.back_translate(
        transcribed,
        lang_code,
        tgt_lang,
        backend_choice=mt_model_choice,   # ✅ keep consistent backend
        mode=mode,
        translit_scheme=translit_scheme,
        code_mixed_english_action=code_mixed_english_action,
    )
    print("\n🔁 Back-Translation Debug:")
    print("➡️ Forward:", bt["forward"])
    print("⬅️ Backward:", bt["backward"])


# -------------------
# TTS Hook (new) — only runs if user asked with --tts and TTS module available
# -------------------
if args.tts:
    if not TTS_AVAILABLE:
        print("[WARN] --tts requested but TTS module is not available. Skipping TTS generation.")
    else:
        try:
            # CLI choice: which TTS model to use (1=Coqui/XTTS, 2=gTTS, 3=IndicParler)
            # Use --tts-model if provided, else prompt
            tts_model_choice = args.tts_model if args.tts_model else user_select_tts()

            # Choose a voice description if user didn't provide one
            user_desc = args.tts_desc.strip() if args.tts_desc else ""
            # Simple templates for common target languages (adjust as needed)
            voice_templates = {
                "hin": "Sunita speaks in a calm, neutral Hindi voice with clear audio and no background noise.",
                "hin_Latn": "Sunita speaks in a calm, neutral Hindi voice with clear audio and no background noise.",
                "eng_Latn": "A neutral English speaker speaks with a clear, moderately-paced British/Indian-accented voice.",
                "tam": "Jaya speaks in a calm Tamil voice with natural prosody and clear audio.",
                "tel": "Prakash speaks in a calm Telugu voice with natural prosody and clear audio.",
                "kan": "Suresh speaks in a calm Kannada voice with clear audio and no background noise.",
                # generic fallback
                "default": "The speaker speaks naturally in clear audio with no background noise."
            }

            # pick template by target Flores code (tgt_lang may be an ISO/FLORES code)
            # try simple match by prefix
            desc = user_desc or voice_templates.get(tgt_lang, None)
            if desc is None:
                # try mapping languages by common substrings
                if tgt_lang.startswith("hin") or tgt_lang.startswith("hi"):
                    desc = voice_templates["hin"]
                elif tgt_lang.startswith("eng") or tgt_lang.startswith("en"):
                    desc = voice_templates["eng_Latn"]
                elif tgt_lang.startswith("tam") or tgt_lang.startswith("ta"):
                    desc = voice_templates["tam"]
                elif tgt_lang.startswith("tel") or tgt_lang.startswith("te"):
                    desc = voice_templates["tel"]
                else:
                    desc = voice_templates["default"]

            # output filename
            if args.tts_save:
                tts_filename = args.tts_save
            else:
                # default: save translated text audio into session folder
                safe_name = "translated_audio.wav"
                tts_filename = safe_name

            tts_out_path = os.path.join(session_dir, tts_filename)

            print(f"\n🔊 Generating TTS with {tts_model_choice} (saving to {tts_out_path}) ...")
            final_tts_path = run_universal_tts(
                text=translated,
                target_lang=tgt_lang,
                reference_audio=(args.tts_speaker_wav.strip() or None),
                prefer=tts_model_choice,
                out_dir=session_dir,
                out_name=tts_filename,
                hf_token=os.getenv("HF_TOKEN"),
                device=("cuda" if torch.cuda.is_available() else "cpu"),
                max_chunk_chars=700,
                use_cache=True,
            )
            print(f"[OK] TTS generated: {final_tts_path}")

            # Optional playback
            if args.tts_play:
                # Use ffplay (part of ffmpeg) if present, else attempt platform-specific playback
                ffplay = shutil.which("ffplay")
                if ffplay:
                    print("[INFO] Playing audio with ffplay...")
                    try:
                        subprocess.run([ffplay, "-nodisp", "-autoexit", "-loglevel", "error", final_tts_path], check=False)
                    except Exception as e:
                        print(f"[WARN] ffplay failed to play audio: {e}")
                else:
                    # Try OS default player
                    try:
                        if sys.platform.startswith("win"):
                            os.startfile(final_tts_path)
                        elif sys.platform.startswith("darwin"):
                            subprocess.run(["open", final_tts_path], check=False)
                        else:
                            subprocess.run(["xdg-open", final_tts_path], check=False)
                    except Exception as e:
                        print(f"[WARN] Failed to launch OS player: {e}")

        except Exception as e:
            print(f"[ERROR] TTS generation failed: {e}")
            # do not raise — allow pipeline to finish


# Summary
dialect = detect_dialect(audio_path)
spoofed = is_spoofed_audio(audio_path)
print("\n📄 Summary:")
print(f"🗣 Detected Language: {lang_code}")
print(f"🧬 Dialect: {dialect}")
print(f"🔒 Spoofing Detected: {'Yes' if spoofed else 'No'}")
print(f"📁 Session folder: {session_dir}")