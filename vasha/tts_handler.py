import os
import torch
import soundfile as sf
import numpy as np
from TTS.api import TTS
from tts_gtts import run_gtts
from tts_indic_parler import run_indic_tts, INDIC_LANGS

# --- Tokenizer for XTTS ---
from transformers import AutoTokenizer
_xtts_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")

XTTS_LANGS = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
              "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"}

FLORES_TO_ISO = {
    "eng_Latn": "en", "spa_Latn": "es", "fra_Latn": "fr", "ita_Latn": "it",
    "por_Latn": "pt", "deu_Latn": "de", "rus_Cyrl": "ru", "tur_Latn": "tr",
    "fas_Arab": "fa", "ind_Latn": "id", "jpn_Jpan": "ja", "kor_Hang": "ko",
    "zho_Hans": "zh-cn", "ara_Arab": "ar", "hin_Deva": "hi"
}

def run_tts(text, lang_code, reference_audio=None,
            out_dir="tts_output", out_name="final_tts.wav", prefer="auto"):

    os.makedirs(out_dir, exist_ok=True)
    lang = FLORES_TO_ISO.get(lang_code, "en")

    # ✅ 1️⃣ Indic Parler-TTS (priority if user selects or when auto and Indian language)
    if prefer == "indic" or (prefer == "auto" and lang in INDIC_LANGS):
        print(f"🇮🇳 Using Indic Parler-TTS for language: {lang}")
        return run_indic_tts(text, out_dir=out_dir, out_name=out_name)

    # ✅ 2️⃣ XTTS (Coqui Voice Cloning)
    if prefer == "xtts" or (prefer == "auto" and lang in XTTS_LANGS):
        print(f"🎙️ Using XTTS for language: {lang}")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to("cuda" if torch.cuda.is_available() else "cpu")

        # Split text for XTTS safety
        chunks = split_text(text, max_len=180)
        print(f"🔹 Splitting text into {len(chunks)} chunks for XTTS...")

        audios = []
        for i, chunk in enumerate(chunks):
            wav = tts.tts(chunk, speaker_wav=reference_audio, language=lang)
            audios.append(wav)

        # Concatenate and save
        final_wav = np.concatenate(audios)
        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, final_wav, 24000)  # XTTS default: 24kHz
        print(f"✅ XTTS TTS saved to {out_path}")
        return out_path

    # ✅ 3️⃣ Google TTS fallback (gTTS)
    print(f"🌐 Using gTTS as fallback for language: {lang}")
    return run_gtts(
        text,
        lang=lang,
        out_dir=out_dir,
        out_name=out_name.replace(".wav", ".mp3")
    )
