import argparse
import os
import re
import torch
import soundfile as sf
import numpy as np
from torch.serialization import add_safe_globals

# --- ASR + MT imports ---
from asr_pipeline import run_asr
from mt_pipeline import translate_batch as indic_translate
from mt_google import translate_google, normalize_code_for_google

# --- TTS (Coqui + gTTS unified handler) ---
from tts_handler import run_tts   # ✅ new unified TTS handler

# --- Import XTTS config classes for safe deserialization ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Pre-register XTTS classes for deserialization
add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])


# --- Indic LID → IndicTrans mapping ---
LID2INDICTRANS = {
    "hi": "hin_Deva", "mr": "mar_Deva", "gu": "guj_Gujr", "bn": "ben_Beng",
    "pa": "pan_Guru", "as": "asm_Beng", "or": "ory_Orya", "ne": "npi_Deva",
    "sd": "snd_Arab", "sa": "san_Deva", "ur": "urd_Arab",
    "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym",
    "hi_Latn": "hin_Deva", "bn_Latn": "ben_Beng", "pa_Latn": "pan_Guru",
    "ta_Latn": "tam_Taml", "te_Latn": "tel_Telu", "kn_Latn": "kan_Knda",
    "ml_Latn": "mal_Mlym", "en": "eng_Latn"
}


def split_into_sentences(text):
    """Simple Indic-friendly sentence splitter."""
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def batch_translate(text, src_lang, tgt_lang, save_path, backend="indic"):
    """Batch translation with sentence splitting + joining."""
    sentences = split_into_sentences(text)
    print(f"📝 Splitting into {len(sentences)} sentences for batch translation...")

    if backend == "google":
        translated_sentences = translate_google(sentences, src_lang=src_lang, tgt_lang=tgt_lang, save_path=save_path)
    else:
        translated_sentences = indic_translate(sentences, src_lang=src_lang, tgt_lang=tgt_lang, save_path=save_path)

    return " ".join(translated_sentences)


def asr_mt_tts_pipeline(audio=None, mic=False, youtube=None, duration=10,
                        tgt_lang=None, asr_model="whisper", whisper_size="large",
                        decoding="ctc", mt_backend="indic", lid_model="whisper",
                        ref_audio="samples/female_clip.wav", fast_tts=False,
                        tts_prefer="auto"):
    """ASR + MT + TTS pipeline (with XTTS + gTTS fallback)."""

    # Step 1: ASR
    transcription, lang_code, transcription_file = run_asr(
        audio_path=audio, mic=mic, youtube=youtube, duration=duration,
        asr_model=asr_model, whisper_size=whisper_size,
        decoding=decoding, lid_model=lid_model,
    )

    # ✅ Fix wrong Whisper LID detection (common mislabels)
    if re.search(r"[\u4e00-\u9fff]", transcription):
        lang_code = "zho_Hans"
    elif re.search(r"[\u3040-\u30ff]", transcription):
        lang_code = "jpn_Jpan"
    elif re.search(r"[\uac00-\ud7af]", transcription):
        lang_code = "kor_Hang"

    print("\n📝 Raw Transcription:", transcription)
    print("🌐 Detected / Normalized Source Language:", lang_code)

    os.makedirs("asrfiles", exist_ok=True)
    asr_save_path = f"asrfiles/{lang_code}_transcription.txt"
    with open(asr_save_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"💾 ASR saved to: {asr_save_path}")

    # Step 2: MT
    final_translation, tts_out_path = None, None
    if tgt_lang:
        print(f"\n🔄 Translating {lang_code} → {tgt_lang} ...")
        src_lang = LID2INDICTRANS.get(lang_code, lang_code)

        if mt_backend == "google":
            src_lang_norm = normalize_code_for_google(src_lang)
            tgt_lang_norm = normalize_code_for_google(tgt_lang)
        else:
            src_lang_norm, tgt_lang_norm = src_lang, tgt_lang

        os.makedirs("mt_transcript", exist_ok=True)
        base_name = os.path.basename(transcription_file).replace("_transcription.txt", "_translation.txt")
        mt_save_path = os.path.join("mt_transcript", base_name)

        final_translation = batch_translate(
            transcription,
            src_lang=src_lang_norm,
            tgt_lang=tgt_lang_norm,
            save_path=mt_save_path,
            backend=mt_backend,
        )

        print(f"\n💾 Translation saved to: {mt_save_path}")
        print(f"✅ Final Translation: {final_translation}")

        # Step 3: TTS (Unified handler with XTTS + gTTS fallback)
        os.makedirs("tts_output", exist_ok=True)
        tts_out_path = run_tts(
            text=final_translation,
            lang_code=tgt_lang,
            reference_audio=ref_audio,
            out_dir="tts_output",
            out_name=base_name.replace(".txt", "_tts.wav"),
            prefer=tts_prefer,
        )

    return transcription, final_translation, tts_out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR + MT + TTS Pipeline")
    parser.add_argument("--audio", help="Path to input audio file (wav/mp3)")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    parser.add_argument("--youtube", help="YouTube video URL")
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--tgt_lang", help="Target language code (e.g., hin_Deva, eng_Latn)")
    parser.add_argument("--asr_model", choices=["whisper", "ai4bharat", "faster_whisper"], default="whisper")
    parser.add_argument("--whisper_size", default="large")
    parser.add_argument("--decoding", choices=["ctc", "rnnt"], default="ctc")
    parser.add_argument("--mt_backend", choices=["indic", "google"], default="indic")
    parser.add_argument("--lid_model", choices=["whisper", "ai4bharat"], default="whisper")
    parser.add_argument("--ref_audio", type=str, default="samples/soumyavoice.wav", help="Reference speaker audio for TTS")
    parser.add_argument("--fast_tts", action="store_true", help="Enable fast inference mode for long text TTS (XTTS only)")
    parser.add_argument("--tts_prefer", choices=["auto", "xtts", "gtts", "indic"], default="auto", help="Force TTS backend (default auto)")

    args = parser.parse_args()

    asr_mt_tts_pipeline(
        audio=args.audio,
        mic=args.mic,
        youtube=args.youtube,
        duration=args.duration,
        tgt_lang=args.tgt_lang,
        asr_model=args.asr_model,
        whisper_size=args.whisper_size,
        decoding=args.decoding,
        mt_backend=args.mt_backend,
        lid_model=args.lid_model,
        ref_audio=args.ref_audio,
        fast_tts=args.fast_tts,
        tts_prefer=args.tts_prefer,
    )
