"""
MT preprocessing utilities:
- NER preservation (keep named entities untranslated)
- Transliteration mode (script <-> script using indic_transliteration.sanscript)
- Simple code-mixed handling (route Indic runs to Indic backend, keep/translate Latin runs)

This module is intentionally independent: it does NOT import mt_helper or mt_model
to avoid circular imports. Instead it accepts a translation callable in functions.

Usage:
    from MT_Model.mt_preprocessor import preprocess_and_translate, transliterate_text
    result = preprocess_and_translate(
        text,
        src_iso='hi',
        tgt_flores='eng_Latn',
        translate_func=perform_translation,   # pass a callable provided by mt_helper
        mode='code_mixed',                     # or None / 'transliterate'
        ner_preserve=True
    )
"""

from typing import Callable, Dict, List, Tuple, Optional
import re
import uuid

# Try common NER backends (transformers pipeline preferred)
_NER_PIPELINE = None
try:
    from transformers import pipeline
    import torch
    _NER_DEVICE = 0 if torch.cuda.is_available() else -1

    def _load_ner_pipeline():
        global _NER_PIPELINE
        if _NER_PIPELINE is None:
            # use aggregated entities (HF newer versions use aggregation_strategy)
            try:
                _NER_PIPELINE = pipeline(
                    "token-classification",
                    model="ai4bharat/IndicNER",
                    aggregation_strategy="simple",
                    device=_NER_DEVICE,
                )
            except TypeError:
                # older transformers use grouped_entities=True
                _NER_PIPELINE = pipeline(
                    "token-classification",
                    model="ai4bharat/IndicNER",
                    grouped_entities=True,
                    device=_NER_DEVICE,
                )
        return _NER_PIPELINE
except Exception:
    _NER_PIPELINE = None

    def _load_ner_pipeline():
        return None


# transliteration
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except Exception:
    sanscript = None
    transliterate = None

# Script detection helpers (broad unicode ranges for Indic scripts)
_INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari (hi, mr, ne, etc)
    (0x0980, 0x09FF),  # Bengali/Assamese
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0D80, 0x0DFF),  # Sinhala (if needed)
    (0x0600, 0x06FF),  # Arabic (covers Urdu/Sindhi script)
]


def _contains_indic_char(s: str) -> bool:
    for ch in s:
        cp = ord(ch)
        for lo, hi in _INDIC_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def _contains_latin(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s))


# -----------------------
# Sanscript scheme helpers
# -----------------------
def _safe_scheme_attr(name: str, fallback_attr: Optional[str] = None):
    """
    Return sanscript.<name> if present, else fallback, else raise.
    """
    if sanscript is None:
        raise RuntimeError("indic_transliteration.sanscript not available.")
    if hasattr(sanscript, name):
        return getattr(sanscript, name)
    if fallback_attr and hasattr(sanscript, fallback_attr):
        return getattr(sanscript, fallback_attr)
    # final fallback to DEVANAGARI or ITRANS
    if hasattr(sanscript, "DEVANAGARI"):
        return getattr(sanscript, "DEVANAGARI")
    if hasattr(sanscript, "ITRANS"):
        return getattr(sanscript, "ITRANS")
    # last resort: return a string literal with common scheme name
    return "devanagari"


# mapping ISO -> sanscript scheme (best-effort)
DEFAULT_ISO_TO_SANSCRIPT = {
    "hi": "DEVANAGARI",
    "mr": "DEVANAGARI",
    "ne": "DEVANAGARI",
    "bn": "BENGALI",
    "as": "BENGALI",
    "pa": "GURMUKHI",
    "gu": "GUJARATI",
    "or": "ORIYA",
    "ta": "TAMIL",
    "te": "TELUGU",
    "kn": "KANNADA",
    "ml": "MALAYALAM",
    "si": "SINHALA",
    # Urdu / Sindhi (Arabic script) — safe fallback since sanscript.URDU does not exist
    "ur": "ARABIC",
    "sd": "ARABIC",
    # fallback for Latin (English): ITRANS or HK/IAST
    "en": "ITRANS",
}


def iso_to_sanscript_scheme(iso: str):
    """
    Return a sanscript scheme constant (string) for given ISO code, robustly.
    """
    name = DEFAULT_ISO_TO_SANSCRIPT.get(iso, None)
    if name is None:
        # default: DEVANAGARI for Indic-ish, else ITRANS
        if iso and iso.lower() in ("en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"):
            name = "ITRANS"
        else:
            name = "DEVANAGARI"
    return _safe_scheme_attr(name, fallback_attr="DEVANAGARI")


# -----------------------
# NER preservation
# -----------------------
def extract_entities(text: str, min_score: float = 0.6) -> List[Dict]:
    """
    Return list of entity dicts with keys: start, end, entity_group (label), word (text)
    Uses HF IndicNER if available; otherwise returns [].
    """
    ner = _load_ner_pipeline()
    if ner is None:
        return []

    try:
        ents = ner(text)
    except Exception:
        return []

    groups_keep = {"PER", "PERSON", "LOC", "LOCATION", "ORG", "ORGANIZATION"}
    out = []
    for e in ents:
        score = e.get("score", e.get("confidence", 0.0))
        label = e.get("entity_group") or e.get("entity") or e.get("label", "")
        normalized = str(label).upper().replace("B-", "").replace("I-", "")
        if score >= min_score and normalized in groups_keep:
            out.append({
                "start": int(e.get("start", 0)),
                "end": int(e.get("end", 0)),
                "label": normalized,
                "text": e.get("word", text[int(e.get("start", 0)):int(e.get("end", 0))]),
                "score": score
            })
    out_sorted = sorted(out, key=lambda x: (x["start"], -x["end"]))
    filtered = []
    last_end = -1
    for ent in out_sorted:
        if ent["start"] >= last_end:
            filtered.append(ent)
            last_end = ent["end"]
    return filtered


def mask_named_entities(text: str, ents: List[Dict]) -> Tuple[str, Dict[str, str]]:
    placeholders = {}
    if not ents:
        return text, placeholders

    out_parts = []
    last = 0
    for i, ent in enumerate(ents):
        s, e = ent["start"], ent["end"]
        if s < last:
            continue
        placeholder = f"__NER_{i}_{uuid.uuid4().hex[:6]}__"
        placeholders[placeholder] = ent["text"]
        out_parts.append(text[last:s])
        out_parts.append(placeholder)
        last = e
    out_parts.append(text[last:])
    return "".join(out_parts), placeholders


def reinstate_placeholders(text: str, placeholders: Dict[str, str]) -> str:
    if not placeholders:
        return text
    for ph, orig in placeholders.items():
        text = text.replace(ph, orig)
    return text


# -----------------------
# Transliteration utility
# -----------------------
def transliterate_text(text: str, src_iso: str, to_scheme: Optional[str] = "ITRANS") -> str:
    if sanscript is None or transliterate is None:
        raise RuntimeError("indic_transliteration package missing or not importable.")

    from_scheme = iso_to_sanscript_scheme(src_iso)
    if isinstance(to_scheme, str):
        if hasattr(sanscript, to_scheme):
            to_scheme_const = getattr(sanscript, to_scheme)
        else:
            to_scheme_const = to_scheme.lower()
    else:
        to_scheme_const = to_scheme

    try:
        return transliterate(text, from_scheme, to_scheme_const)
    except Exception:
        return text


# -----------------------
# Code-mixed handling helper
# -----------------------
def split_by_script_runs(text: str) -> List[Tuple[str, str]]:
    parts = re.split(r'(\s+)', text)  # keep whitespace tokens
    runs = []
    for p in parts:
        if p.strip() == "":
            runs.append(("whitespace", p))
            continue
        if _contains_indic_char(p):
            runs.append(("indic", p))
        elif _contains_latin(p):
            runs.append(("latin", p))
        else:
            runs.append(("other", p))
    return runs


# -----------------------
# High-level preprocess + translate
# -----------------------
def preprocess_and_translate(
    text: str,
    src_iso: str,
    tgt_flores: str,
    translate_func: Callable[[str, str, str], str],
    mode: Optional[str] = None,
    ner_preserve: bool = False,
    translit_target_scheme: str = "ITRANS",       # ✅ matches mt_helper
    english_run_action: str = "pass",             # ✅ matches mt_helper
    backend_choice: Optional[str] = None
) -> str:
    """
    Unified helper that:
      - optionally masks named entities,
      - optionally transliterates,
      - optionally handles code-mixed text,
      - otherwise calls translate_func directly.
    """
    original_text = text

    # NER preserve
    placeholders = {}
    if ner_preserve:
        try:
            ents = extract_entities(text)
            if ents:
                text, placeholders = mask_named_entities(text, ents)
        except Exception:
            placeholders = {}
            text = original_text

    # Mode: transliteration
    if mode == "transliterate":
        try:
            out = transliterate_text(text, src_iso, to_scheme=translit_target_scheme)
        except Exception:
            out = text
        return reinstate_placeholders(out, placeholders)

    # Mode: code-mixed
    elif mode == "code_mixed":
        runs = split_by_script_runs(text)
        out_parts = []
        for typ, run_text in runs:
            if typ == "whitespace":
                out_parts.append(run_text)
                continue
            if typ == "latin":
                if english_run_action == "pass":
                    out_parts.append(run_text)
                else:
                    try:
                        translated_run = translate_func(run_text, src_iso, tgt_flores, mt_model_choice=backend_choice)
                        out_parts.append(translated_run)
                    except Exception:
                        out_parts.append(run_text)
            elif typ == "indic":
                try:
                    translated_run = translate_func(run_text, src_iso, tgt_flores, mt_model_choice=backend_choice)
                    out_parts.append(translated_run)
                except Exception:
                    out_parts.append(run_text)
            else:
                out_parts.append(run_text)
        final = "".join(out_parts)
        return reinstate_placeholders(final, placeholders)

    # Normal mode
    else:
        try:
            translated = translate_func(text, src_iso, tgt_flores, mt_model_choice=backend_choice)
        except TypeError:
            translated = translate_func(text, src_iso, tgt_flores)
        return reinstate_placeholders(translated, placeholders)


# convenience wrapper
def preprocess_translate_wrapper(
    text: str,
    src_iso: str,
    tgt_flores: str,
    translate_callable: Callable,
    **kwargs
):
    return preprocess_and_translate(text, src_iso, tgt_flores, translate_callable, **kwargs)


__all__ = [
    "preprocess_and_translate",
    "preprocess_translate_wrapper",
    "transliterate_text",
    "extract_entities",
    "mask_named_entities",
    "reinstate_placeholders",
    "split_by_script_runs",
]
