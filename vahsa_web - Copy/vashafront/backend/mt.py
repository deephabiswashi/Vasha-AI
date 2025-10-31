import os
import re
from typing import List

from googletrans import Translator
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

translator = Translator()

# Optional IndicProcessor
try:
    from IndicTransToolkit.processor import IndicProcessor
    HAS_INDIC_PROCESSOR = True
except ImportError:
    HAS_INDIC_PROCESSOR = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INDICTRANS_TO_GOOGLE = {
    "hin_Deva": "hi", "mar_Deva": "mr", "guj_Gujr": "gu", "ben_Beng": "bn",
    "pan_Guru": "pa", "asm_Beng": "as", "ory_Orya": "or", "npi_Deva": "ne",
    "kas_Arab": "ks", "kas_Deva": "ks", "gom_Deva": "gom", "mai_Deva": "mai",
    "snd_Arab": "sd", "snd_Deva": "sd", "san_Deva": "sa", "urd_Arab": "ur",
    "brx_Deva": "brx", "doi_Deva": "doi", "sat_Olck": "sat",
    "tam_Taml": "ta", "tel_Telu": "te", "kan_Knda": "kn", "mal_Mlym": "ml",
    "mni_Beng": "mni", "mni_Mtei": "mni",
    "eng_Latn": "en", "spa_Latn": "es", "fra_Latn": "fr", "ita_Latn": "it",
    "por_Latn": "pt", "deu_Latn": "de", "rus_Cyrl": "ru", "tur_Latn": "tr",
    "fas_Arab": "fa", "ind_Latn": "id", "jpn_Jpan": "ja", "kor_Hang": "ko",
    "zho_Hans": "zh-cn", "ara_Arab": "ar",
}

# Build inverse mapping for ISO->IndicTrans where known
GOOGLE_TO_INDICTRANS = {v: k for k, v in INDICTRANS_TO_GOOGLE.items()}

def normalize_code_for_google(code: str) -> str:
    if not code:
        return "en"
    if code in INDICTRANS_TO_GOOGLE:
        return INDICTRANS_TO_GOOGLE[code]
    return code.split("_")[0]


def normalize_code_for_indictrans(code: str) -> str:
    """Convert ISO codes like 'en','hi' to IndicTrans tags like 'eng_Latn','hin_Deva'."""
    if not code:
        return "eng_Latn"
    if "_" in code:
        return code  # already IndicTrans-style
    # Prefer explicit mapping
    if code in GOOGLE_TO_INDICTRANS:
        return GOOGLE_TO_INDICTRANS[code]
    # Common fallbacks
    fallback = {
        "en": "eng_Latn",
        "hi": "hin_Deva",
        "bn": "ben_Beng",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "mr": "mar_Deva",
        "gu": "guj_Gujr",
        "kn": "kan_Knda",
        "ml": "mal_Mlym",
        "pa": "pan_Guru",
        "or": "ory_Orya",
        "ne": "npi_Deva",
        "ur": "urd_Arab",
    }
    return fallback.get(code, "eng_Latn")


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def translate_google(texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
    src = normalize_code_for_google(src_lang)
    tgt = normalize_code_for_google(tgt_lang)
    outputs: List[str] = []
    for t in texts:
        try:
            res = translator.translate(t, src=src, dest=tgt)
            outputs.append(res.text)
        except Exception:
            outputs.append(t)
    return outputs


MODEL_MAP = {
    ("eng_Latn", "indic"): "ai4bharat/indictrans2-en-indic-1B",
    ("indic", "eng_Latn"): "ai4bharat/indictrans2-indic-en-1B",
    ("indic", "indic"): "ai4bharat/indictrans2-indic-indic-1B",
}

def detect_model(src_lang: str, tgt_lang: str) -> str:
    if src_lang == "eng_Latn" and tgt_lang != "eng_Latn":
        return MODEL_MAP[("eng_Latn", "indic")]
    elif src_lang != "eng_Latn" and tgt_lang == "eng_Latn":
        return MODEL_MAP[("indic", "eng_Latn")]
    else:
        return MODEL_MAP[("indic", "indic")]


def translate_indictrans(texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
    # Normalize to IndicTrans tags
    src_lang = normalize_code_for_indictrans(src_lang)
    tgt_lang = normalize_code_for_indictrans(tgt_lang)
    model_name = detect_model(src_lang, tgt_lang)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

    if HAS_INDIC_PROCESSOR:
        ip = IndicProcessor(inference=True)
        batch = ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, use_cache=True, min_length=0, max_length=1024, num_beams=5, num_return_sequences=1)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations = ip.postprocess_batch(decoded, lang=tgt_lang)
    else:
        tagged = [f"<{src_lang}><{tgt_lang}> {s}" for s in texts]
        inputs = tokenizer(tagged, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=1024, num_beams=5, use_cache=True)
        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return translations


def translate_with_fallback(text: str, src_lang: str, tgt_lang: str, primary: str = "indictrans"):
    sentences = split_into_sentences(text)
    try:
        print(f"\n🔁 MT request: primary={primary} src={src_lang} tgt={tgt_lang} sentences={len(sentences)}")
        if primary == "google":
            primary_out = translate_google(sentences, src_lang, tgt_lang)
        else:
            primary_out = translate_indictrans(sentences, src_lang, tgt_lang)
        return " ".join(primary_out), primary
    except Exception:
        fallback = "indictrans" if primary == "google" else "google"
        print(f"⚠️ Primary MT failed, falling back to {fallback}")
        if fallback == "google":
            fb_out = translate_google(sentences, src_lang, tgt_lang)
        else:
            fb_out = translate_indictrans(sentences, src_lang, tgt_lang)
        return " ".join(fb_out), fallback


