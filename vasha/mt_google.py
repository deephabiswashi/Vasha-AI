import os
from googletrans import Translator

translator = Translator()

# Mapping IndicTrans → Google Translate ISO codes
INDICTRANS_TO_GOOGLE = {
    # --- Indo-Aryan ---
    "hin_Deva": "hi",        # Hindi
    "mar_Deva": "mr",        # Marathi
    "guj_Gujr": "gu",        # Gujarati
    "ben_Beng": "bn",        # Bengali
    "pan_Guru": "pa",        # Punjabi
    "asm_Beng": "as",        # Assamese
    "ory_Orya": "or",        # Odia
    "npi_Deva": "ne",        # Nepali
    "kas_Arab": "ks",        # Kashmiri (Arabic) → Google "ks"
    "kas_Deva": "ks",        # Kashmiri (Devanagari) → same "ks"
    "gom_Deva": "gom",       # Konkani → Google "gom"
    "mai_Deva": "mai",       # Maithili
    "snd_Arab": "sd",        # Sindhi (Arabic)
    "snd_Deva": "sd",        # Sindhi (Devanagari)
    "san_Deva": "sa",        # Sanskrit
    "urd_Arab": "ur",        # Urdu
    "brx_Deva": "brx",       # Bodo
    "doi_Deva": "doi",       # Dogri
    "sat_Olck": "sat",       # Santali

    # --- Dravidian ---
    "tam_Taml": "ta",        # Tamil
    "tel_Telu": "te",        # Telugu
    "kan_Knda": "kn",        # Kannada
    "mal_Mlym": "ml",        # Malayalam
    "mni_Beng": "mni",       # Manipuri (Bengali)
    "mni_Mtei": "mni",       # Manipuri (Meitei)

    # --- Global languages ---
    "eng_Latn": "en",        # English
    "spa_Latn": "es",        # Spanish
    "fra_Latn": "fr",        # French
    "ita_Latn": "it",        # Italian
    "por_Latn": "pt",        # Portuguese
    "deu_Latn": "de",        # German
    "rus_Cyrl": "ru",        # Russian
    "tur_Latn": "tr",        # Turkish
    "fas_Arab": "fa",        # Persian (Farsi)
    "ind_Latn": "id",        # Indonesian
    "jpn_Jpan": "ja",        # Japanese
    "kor_Hang": "ko",        # Korean
    "zho_Hans": "zh-cn",     # Simplified Chinese
    "ara_Arab": "ar",        # Arabic
}

def normalize_code_for_google(code: str) -> str:
    """
    Convert IndicTrans style codes (hin_Deva, tam_Taml, eng_Latn, etc.)
    into ISO codes Google Translate understands (hi, ta, en...).
    """
    if not code:
        return "en"
    if code in INDICTRANS_TO_GOOGLE:
        return INDICTRANS_TO_GOOGLE[code]
    return code.split("_")[0]  # fallback: strip suffix


def translate_google(texts, src_lang, tgt_lang, save_path=None):
    """
    Translate using Google Translate API.
    :param texts: list of sentences
    :param src_lang: IndicTrans-style or ISO code (e.g. eng_Latn, hin_Deva, en, hi)
    :param tgt_lang: IndicTrans-style or ISO code (e.g. tam_Taml, eng_Latn, ta, en)
    """
    src = normalize_code_for_google(src_lang)
    tgt = normalize_code_for_google(tgt_lang)

    translations = []
    for t in texts:
        try:
            res = translator.translate(t, src=src, dest=tgt)
            translations.append(res.text)
        except Exception as e:
            print(f"⚠️ Error translating '{t}': {e}")
            translations.append(t)

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for line in translations:
                f.write(line + "\n")
        print(f"\n💾 Google Translation saved to: {save_path}")

    return translations
