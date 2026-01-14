# =========================================================
# tts_common/tts_utils.py
# =========================================================

import re
from transformers import AutoTokenizer

# ---------------------------------------------------------
# 🌍 FLORES → ISO Mapping
# ---------------------------------------------------------
FLORES_TO_ISO = {
    "eng_Latn": "en", "spa_Latn": "es", "fra_Latn": "fr", "deu_Latn": "de",
    "ita_Latn": "it", "por_Latn": "pt", "rus_Cyrl": "ru", "tur_Latn": "tr",
    "ara_Arab": "ar", "zho_Hans": "zh-cn", "jpn_Jpan": "ja", "kor_Hang": "ko",
    "hin_Deva": "hi", "mar_Deva": "mr", "guj_Gujr": "gu", "ben_Beng": "bn",
    "pan_Guru": "pa", "asm_Beng": "as", "ory_Orya": "or", "npi_Deva": "ne",
    "tam_Taml": "ta", "tel_Telu": "te", "kan_Knda": "kn", "mal_Mlym": "ml",
    "urd_Arab": "ur"
}

# ---------------------------------------------------------
# 🌐 Supported TTS Languages
# ---------------------------------------------------------
XTTS_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"
}

INDIC_LANGS = {
    "hi", "bn", "gu", "ta", "te", "kn", "ml", "mr", "pa", "as", "or", "ne", "ur"
}

# ---------------------------------------------------------
# 🪶 Tokenizer & Text Splitters
# ---------------------------------------------------------
_xtts_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")


def split_text_by_tokens(text, max_tokens=350):
    """Split text safely based on XTTS tokenizer length."""
    text = text.strip()
    if not text:
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


def smart_split_text(text, lang="en", max_len=120):
    """Language-aware sentence splitting."""
    text = text.replace("\n", " ")
    if lang == "ja":
        sentences = re.split(r'(?<=[。！？])', text)
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)

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


# ---------------------------------------------------------
# 🧠 Model Routing Helper
# ---------------------------------------------------------
def resolve_tts_engine(lang_code):
    """
    Decide which TTS engine to use (XTTS / Indic / gTTS).
    """
    lang = FLORES_TO_ISO.get(lang_code, lang_code).split("_")[0]

    if lang in INDIC_LANGS:
        return "indic"
    elif lang in XTTS_LANGS:
        return "xtts"
    else:
        return "gtts"
