import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Try to import IndicProcessor (two possible package names)
try:
    from IndicTransToolkit import IndicProcessor  # pip install IndicTransToolkit
    _HAS_INDIC_TOOLKIT = True
except Exception:
    try:
        from indictrans import IndicProcessor  # alternate packaging
        _HAS_INDIC_TOOLKIT = True
    except Exception:
        IndicProcessor = None
        _HAS_INDIC_TOOLKIT = False

_indic_ip_cache = {}
def _get_indic_processor(model_name: str):
    if not _HAS_INDIC_TOOLKIT:
        return None
    if model_name not in _indic_ip_cache:
        try:
            _indic_ip_cache[model_name] = IndicProcessor(inference=True)
        except TypeError:
            _indic_ip_cache[model_name] = IndicProcessor()
    return _indic_ip_cache[model_name]

# -----------------------------
# Config
# -----------------------------
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load NLLB
# -----------------------------
print(f"[INFO] Loading NLLB model: {NLLB_MODEL_NAME} on {device} ...")
nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME).to(device)

# -----------------------------
# IndicTrans2 model names
# -----------------------------
INDIC_EN_MODEL = "ai4bharat/indictrans2-indic-en-1B"
EN_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-1B"
INDIC_INDIC_MODEL = "ai4bharat/indictrans2-indic-indic-1B"

_indic_tokenizers = {}
_indic_models = {}

def _load_indic_model(model_name):
    if model_name in _indic_tokenizers:
        return _indic_tokenizers[model_name], _indic_models[model_name]
    print(f"⚡ Loading {model_name} ... (this may take a while)")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    _indic_tokenizers[model_name] = tokenizer
    _indic_models[model_name] = model
    return tokenizer, model

# -----------------------------
# Indic Languages (FLORES codes)
# -----------------------------
INDIC_LANGS = {
    "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "guj_Gujr", "hin_Deva",
    "kan_Knda", "kas_Arab", "kas_Deva", "mai_Deva", "mal_Mlym", "mar_Deva",
    "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "sat_Olck", "snd_Arab",
    "snd_Deva", "tam_Taml", "tel_Telu", "urd_Arab", "kok_Deva", "mni_Beng",
    "mni_Mtei", "gom_Deva"
}

# -----------------------------
# ISO → FLORES Mapping
# -----------------------------
ISO_TO_FLORES = {
    "en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "as": "asm_Beng",
    "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva",
    "ne": "npi_Deva", "or": "ory_Orya", "pa": "pan_Guru", "sa": "san_Deva",
    "ta": "tam_Taml", "te": "tel_Telu", "ur": "urd_Arab", "ks": "kas_Arab",
    "sd": "snd_Arab", "brx": "brx_Deva", "doi": "doi_Deva", "mai": "mai_Deva",
    "kok": "kok_Deva", "mni": "mni_Beng", "sat": "sat_Olck",
    "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "ar": "arb_Arab", "fa": "pes_Arab", "tr": "tur_Latn",
    "id": "ind_Latn",
}

# -----------------------------
# Helpers
# -----------------------------
def _split_into_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    return re.split(r'(?<=[\.\?\!])\s+', text)

def _group_sentences(sentences, char_limit=2000):
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        if cur_len + len(s) + 1 <= char_limit:
            cur.append(s)
            cur_len += len(s) + 1
        else:
            chunks.append(" ".join(cur))
            cur, cur_len = [s], len(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# -----------------------------
# Batch Translation API (NEW)
# -----------------------------
def batch_translate_text(text: str, src_flores: str, tgt_flores: str, mt_model_choice="auto", max_chunk_size: int = 1800) -> str:
    """
    Splits input into manageable chunks and translates each sequentially.
    """
    sentences = _split_into_sentences(text)
    chunks = _group_sentences(sentences, char_limit=max_chunk_size)

    outputs = []
    for chunk in chunks:
        try:
            out = translate_text(chunk, src_flores, tgt_flores, mt_model_choice=mt_model_choice)
            outputs.append(out)
        except Exception as e:
            print(f"⚠️ Batch translation error on chunk: {e}")
            outputs.append("[translation_error]")
    return " ".join(outputs).strip()

# -----------------------------
# IndicTrans2 Translation
# -----------------------------
def _encode_for_indictrans(tokenizer, text, src_flores, tgt_flores, model_name, device):
    batch = [text]
    ip = _get_indic_processor(model_name)
    if ip is not None:
        prepped = ip.preprocess_batch(batch, src_lang=src_flores, tgt_lang=tgt_flores)
        enc = tokenizer(prepped, return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(device) for k, v in enc.items()}
    try:
        if hasattr(tokenizer, "set_src_lang"):
            tokenizer.set_src_lang(src_flores)
        else:
            tokenizer.src_lang = src_flores
        if hasattr(tokenizer, "set_tgt_lang"):
            tokenizer.set_tgt_lang(tgt_flores)
        else:
            tokenizer.tgt_lang = tgt_flores
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(device) for k, v in enc.items()}
    except TypeError:
        tagged = [f"{src_flores} {text}"]
        try:
            tokenizer.tgt_lang = tgt_flores
        except Exception:
            pass
        enc = tokenizer(tagged, return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(device) for k, v in enc.items()}

def translate_with_indictrans2(text, src_flores, tgt_flores, use_processor: bool = False):
    if src_flores == "eng_Latn" and tgt_flores in INDIC_LANGS:
        model_name = EN_INDIC_MODEL
    elif src_flores in INDIC_LANGS and tgt_flores == "eng_Latn":
        model_name = INDIC_EN_MODEL
    elif src_flores in INDIC_LANGS and tgt_flores in INDIC_LANGS:
        model_name = INDIC_INDIC_MODEL
    else:
        raise ValueError("Invalid IndicTrans2 translation direction.")

    tokenizer, model = _load_indic_model(model_name)
    sentences = _split_into_sentences(text)
    chunks = _group_sentences(sentences, char_limit=1800)

    outputs = []
    ip = _get_indic_processor(model_name) if use_processor else _get_indic_processor(model_name)
    for chunk in chunks:
        try:
            enc = _encode_for_indictrans(tokenizer, chunk, src_flores, tgt_flores, model_name, device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=512,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    repetition_penalty=2.0,
                    early_stopping=True,
                    use_cache=False,
                )
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            if ip is not None:
                decoded = ip.postprocess_batch(decoded, lang=tgt_flores)
            outputs.append(decoded[0] if decoded else "")
        except Exception as e:
            print(f"⚠️ IndicTrans2 translation chunk failed: {e}")
            outputs.append("[translation_error]")
    return " ".join(outputs).strip()

# -----------------------------
# NLLB Translation
# -----------------------------
def _get_forced_bos_token_id(tokenizer, tgt_flores_code):
    try:
        lang_map = getattr(tokenizer, "lang_code_to_id", None)
        if lang_map and tgt_flores_code in lang_map:
            return lang_map[tgt_flores_code]
    except Exception:
        pass
    try:
        tok_id = tokenizer.convert_tokens_to_ids(tgt_flores_code)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            return tok_id
    except Exception:
        pass
    try:
        vocab = tokenizer.get_vocab()
        if tgt_flores_code in vocab:
            return vocab[tgt_flores_code]
    except Exception:
        pass
    raise ValueError(f"Could not map target FLORES code '{tgt_flores_code}' to a token id.")

def translate_with_nllb(text, src_flores, tgt_flores, max_new_tokens=1024):
    nllb_tokenizer.src_lang = src_flores
    inputs = nllb_tokenizer(text, return_tensors="pt", truncation=True, padding="longest")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    forced_bos_token_id = _get_forced_bos_token_id(nllb_tokenizer, tgt_flores)
    with torch.no_grad():
        gen = nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            early_stopping=True,
            num_beams=4,
            use_cache=False,
        )
    return nllb_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

def translate_long_text(text, src_flores, tgt_flores):
    sentences = _split_into_sentences(text)
    chunks = _group_sentences(sentences)
    outputs = []
    for chunk in chunks:
        try:
            out = translate_with_nllb(chunk, src_flores, tgt_flores)
            outputs.append(out)
        except Exception as e:
            print(f"⚠️ Translation chunk failed: {e}")
            outputs.append("[translation_error]")
    return " ".join(outputs).strip()

def translate_long_text_nllb(text, src_flores, tgt_flores):
    return translate_long_text(text, src_flores, tgt_flores)

# -----------------------------
# Google Translate (Optional)
# -----------------------------
try:
    from googletrans import Translator
    _HAS_GOOGLETRANS = True
    _google_translator = Translator()
except Exception:
    _HAS_GOOGLETRANS = False
    _google_translator = None

def translate_with_google(text: str, src_lang_iso_or_flores: str, tgt_lang_iso_or_flores: str) -> str:
    """
    Uses googletrans (free API) for translation.
    Works with ISO codes. FLORES codes are mapped back to ISO when possible.
    """
    if not _HAS_GOOGLETRANS:
        raise ImportError("googletrans not installed. Run: pip install googletrans==4.0.0-rc1")

    # Map FLORES → ISO if needed
    src_iso = src_lang_iso_or_flores
    tgt_iso = tgt_lang_iso_or_flores
    if "_" in src_iso:
        inv_map = {v: k for k, v in ISO_TO_FLORES.items()}
        src_iso = inv_map.get(src_iso, "auto")
    if "_" in tgt_iso:
        inv_map = {v: k for k, v in ISO_TO_FLORES.items()}
        tgt_iso = inv_map.get(tgt_iso, tgt_iso.split("_")[0])

    try:
        result = _google_translator.translate(text, src=src_iso, dest=tgt_iso)
        return result.text
    except Exception as e:
        print(f"⚠️ Google Translate failed: {e}")
        return "[translation_error]"

# -----------------------------
# Public API
# -----------------------------
def translate_text(text, src_lang_iso_or_flores, tgt_flores="eng_Latn", mt_model_choice: str = None, use_processor: bool = False):
    if "_" in str(src_lang_iso_or_flores):
        src_flores = src_lang_iso_or_flores
    else:
        iso = str(src_lang_iso_or_flores).lower()
        src_flores = ISO_TO_FLORES.get(iso)
        if src_flores is None:
            raise ValueError(f"Unrecognized source language '{src_lang_iso_or_flores}'.")

    if mt_model_choice is None:
        choice = "auto"
    else:
        choice = str(mt_model_choice).strip().lower()

    if choice in ("nllb", "meta-nllb"):
        backend = "nllb"
    elif choice in ("indic", "indictrans2", "ai4bharat"):
        backend = "indic"
    elif choice in ("google", "gt", "googletranslate"):
        backend = "google"
    else:
        if src_flores in INDIC_LANGS or tgt_flores in INDIC_LANGS:
            backend = "indic"
        else:
            backend = "nllb"

    if backend == "indic":
        return translate_with_indictrans2(text, src_flores, tgt_flores, use_processor=use_processor)
    elif backend == "google":
        return translate_with_google(text, src_lang_iso_or_flores, tgt_flores)
    else:
        return translate_long_text(text, src_flores, tgt_flores)
