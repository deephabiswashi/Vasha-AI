import os
import torch
import soundfile as sf
import numpy as np
import subprocess
from TTS.api import TTS
from tts_gtts import run_gtts
from tts_indic_parler import run_indic_tts, INDIC_LANGS
from transformers import AutoTokenizer

# =========================================================
# 🧩 Auto-fix missing dependencies for Japanese tokenizer
# =========================================================
try:
    import cutlet
    import fugashi
    import unidic_lite
    os.environ["UNIDIC_DIR"] = unidic_lite.DICDIR
    os.environ["MECABRC"] = ""  # prevent MeCab system lookup
    print("✅ Japanese tokenizer (cutlet + fugashi + unidic-lite) ready.")
except Exception:
    print("⚙️ Installing missing Japanese text dependencies...")
    subprocess.run(["pip", "install", "cutlet", "fugashi[unidic-lite]", "unidic-lite"], check=False)
    import cutlet
    import fugashi
    import unidic_lite
    os.environ["UNIDIC_DIR"] = unidic_lite.DICDIR
    os.environ["MECABRC"] = ""


# =========================================================
# 🪶 Tokenizer for XTTS
# =========================================================
_xtts_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")

# =========================================================
# 🌐 Supported XTTS languages
# =========================================================
XTTS_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"
}

# =========================================================
# 🌏 FLORES → ISO mapping (Indic + Global)
# =========================================================
FLORES_TO_ISO = {
    "eng_Latn": "en", "spa_Latn": "es", "fra_Latn": "fr", "ita_Latn": "it",
    "por_Latn": "pt", "deu_Latn": "de", "rus_Cyrl": "ru", "tur_Latn": "tr",
    "fas_Arab": "fa", "ind_Latn": "id", "jpn_Jpan": "ja", "kor_Hang": "ko",
    "zho_Hans": "zh-cn", "ara_Arab": "ar", "hin_Deva": "hi",

    # Indic
    "hi": "hin_Deva", "mr": "mar_Deva", "gu": "guj_Gujr", "bn": "ben_Beng",
    "pa": "pan_Guru", "as": "asm_Beng", "or": "ory_Orya", "ne": "npi_Deva",
    "sd": "snd_Arab", "sa": "san_Deva", "ur": "urd_Arab",
    "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym",

    # Latin transliterations
    "hi_Latn": "hin_Deva", "bn_Latn": "ben_Beng", "pa_Latn": "pan_Guru",
    "ta_Latn": "tam_Taml", "te_Latn": "tel_Telu", "kn_Latn": "kan_Knda",
    "ml_Latn": "mal_Mlym", "en": "eng_Latn"
}

# =========================================================
# ✂️ Token-safe text splitter
# =========================================================
def split_text_by_tokens(text, max_tokens=350):
    """Split text into chunks <= max_tokens based on XTTS tokenizer."""
    if not text.strip():
        return []
    tokens = _xtts_tokenizer.encode(text)
    total = len(tokens)
    if total <= max_tokens:
        return [text]

    chunks, start = [], 0
    while start < total:
        end = min(start + max_tokens, total)
        sub_text = _xtts_tokenizer.decode(tokens[start:end])
        chunks.append(sub_text.strip())
        start = end
    return chunks


# =========================================================
# 🧠 Smarter sentence splitter for Japanese / Indic
# =========================================================
def smart_split_text(text, lang, max_len=120):
    """Language-aware sentence splitting."""
    text = text.strip().replace("\n", " ")
    if lang == "ja":
        # Japanese sentences often end with 。！？
        import re
        sentences = re.split(r'(?<=[。！？])', text)
    else:
        sentences = text.split(". ")

    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= max_len:
            current += s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks


# =========================================================
# 🔉 Main TTS Runner
# =========================================================
def run_tts(text, lang_code, reference_audio=None,
            out_dir="tts_output", out_name="final_tts.wav", prefer="auto"):

    os.makedirs(out_dir, exist_ok=True)
    lang = FLORES_TO_ISO.get(lang_code, lang_code)
    lang = lang.split("_")[0]  # Reduce to ISO (e.g., "ja", "hi")

    # 🪷 Indic TTS
    if prefer == "indic" or (prefer == "auto" and lang in INDIC_LANGS):
        print(f"🇮🇳 Using Indic Parler-TTS for: {lang}")
        return run_indic_tts(text, out_dir=out_dir, out_name=out_name)

    # 🪷 XTTS
    if prefer == "xtts" or (prefer == "auto" and lang in XTTS_LANGS):
        print(f"🎙️ Using XTTS for language: {lang}")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to("cuda" if torch.cuda.is_available() else "cpu")

        # Reference voice
        ref_wav = None
        if reference_audio and os.path.exists(reference_audio):
            try:
                data, sr = sf.read(reference_audio)
                if len(data) > 0:
                    ref_wav = reference_audio
                    print(f"🎧 Using reference voice: {reference_audio}")
            except Exception as e:
                print(f"⚠️ Reference audio failed: {e}")

        # Split text safely
        chunks = split_text_by_tokens(text, 300)
        if len(chunks) == 1 and len(text) > 200:
            chunks = smart_split_text(text, lang, 120)
        print(f"🔹 {len(chunks)} text chunks prepared for XTTS.")

        audios = []
        for i, chunk in enumerate(chunks):
            try:
                print(f"🗣️ Generating chunk {i+1}/{len(chunks)}...")
                wav = tts.tts(text=chunk, speaker_wav=ref_wav, language=lang)
                audios.append(wav)
            except Exception as e:
                print(f"⚠️ XTTS failed ({e}) — falling back to gTTS.")
                from gtts import gTTS
                tmp_mp3 = os.path.join(out_dir, f"tmp_{i}.mp3")
                gtts_fallback = gTTS(chunk, lang="ja" if lang == "ja" else "en")
                gtts_fallback.save(tmp_mp3)
                data, _ = sf.read(tmp_mp3)
                audios.append(data)

        final_wav = np.concatenate(audios)
        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, final_wav, 24000)
        print(f"✅ XTTS output saved to: {out_path}")
        return out_path

    # 🪷 gTTS Fallback
    print(f"🌐 Using Google TTS fallback for: {lang}")
    return run_gtts(text, lang=lang, out_dir=out_dir, out_name=out_name.replace(".wav", ".mp3"))
