from typing import Dict, Tuple, Optional
from MT_Model import mt_model as mm
from MT_Model import mt_google   # ✅ Google Translate backend
from MT_Model import mt_preprocessor  # ✅ Preprocessor (NER, transliteration, code-mix)
import importlib
from tqdm import tqdm   # ✅ added for progress bar

# reload to pick up latest changes if module was mutated
importlib.reload(mm)

ISO_TO_FLORES: Dict[str, str] = getattr(mm, "ISO_TO_FLORES", {})

# -------------------------
# Global Languages Menu
# -------------------------
GLOBAL_LANGS: Dict[str, Tuple[str, str]] = {
    "1": ("English", "en"),
    "2": ("Spanish", "es"),
    "3": ("French", "fr"),
    "4": ("German", "de"),
    "5": ("Italian", "it"),
    "6": ("Portuguese", "pt"),
    "7": ("Russian", "ru"),
    "8": ("Chinese (Simplified)", "zh"),
    "9": ("Japanese", "ja"),
    "10": ("Korean", "ko"),
    "11": ("Arabic", "ar"),
    "12": ("Persian", "fa"),
    "13": ("Turkish", "tr"),
    "14": ("Indonesian", "id"),
}

# -------------------------
# Indic Languages Menu
# -------------------------
INDIC_LANGS_MENU: Dict[str, Tuple[str, str]] = {
    "101": ("Assamese", "as"),
    "102": ("Bengali", "bn"),
    "103": ("Bodo", "brx"),
    "104": ("Dogri", "doi"),
    "105": ("Gujarati", "gu"),
    "106": ("Hindi", "hi"),
    "107": ("Kannada", "kn"),
    "108": ("Kashmiri (Arabic)", "ks"),
    "109": ("Konkani", "kok"),
    "110": ("Maithili", "mai"),
    "111": ("Malayalam", "ml"),
    "112": ("Marathi", "mr"),
    "113": ("Manipuri (Meitei)", "mni"),
    "114": ("Nepali", "ne"),
    "115": ("Odia", "or"),
    "116": ("Punjabi", "pa"),
    "117": ("Sanskrit", "sa"),
    "118": ("Santali", "sat"),
    "119": ("Sindhi (Arabic)", "sd"),
    "120": ("Tamil", "ta"),
    "121": ("Telugu", "te"),
    "122": ("Urdu", "ur"),
}

# -------------------------
# User Menu
# -------------------------
def choose_language_menu() -> Tuple[str, str, str]:
    print("\n🌐 Select Translation Domain:")
    print("1. Global Languages")
    print("2. Indic Languages")
    domain_choice = input("👉 Enter 1 or 2 (default=1): ").strip() or "1"

    if domain_choice == "1":
        langs = GLOBAL_LANGS
        domain = "global"
    else:
        langs = INDIC_LANGS_MENU
        domain = "indic"

    print(f"\n=== {'Global Languages' if domain=='global' else 'Indian Languages'} ===")
    for key, (name, iso) in sorted(langs.items(), key=lambda x: int(x[0])):
        flores = ISO_TO_FLORES.get(iso, "(no mapping)")
        print(f"{key}. {name} ({flores})")
    print("0. Custom FLORES code")

    choice = input("👉 Enter choice (default=1): ").strip()
    if choice == "0":
        custom = input("🔤 Enter custom FLORES code: ").strip()
        return domain, "custom", custom

    if choice == "" or choice not in langs:
        choice = list(langs.keys())[0]

    iso = langs[choice][1]
    flores = ISO_TO_FLORES.get(iso, iso)
    return domain, iso, flores

# -------------------------
# Auto logic
# -------------------------
def auto_select_backend(src_flores: str, tgt_flores: Optional[str] = None) -> str:
    """
    Decide which backend is safest.
    - If both src/tgt are Indic → use IndicTrans2
    - Else → fall back to NLLB (to avoid invalid IndicTrans2 pairs)
    """
    if src_flores in mm.INDIC_LANGS and tgt_flores in mm.INDIC_LANGS:
        return "indic"
    return "nllb"

# -------------------------
# Hybrid + Google + Preprocessing Translation
# -------------------------
def perform_translation(
    text: str,
    src_lang: str,
    tgt_flores: str,
    backend_choice: Optional[str] = None,
    use_processor: bool = False,
    mode: Optional[str] = None,
    translit_scheme: str = "ITRANS",
    code_mixed_english_action: str = "pass",
    ner_preserve: bool = False,
) -> str:
    """
    Unified translation entrypoint with preprocessing (NER, translit, code-mixed).
    """

    src_flores = src_lang if "_" in src_lang else ISO_TO_FLORES.get(src_lang, src_lang)

    # Base translation function for preprocessor to call
    def _base_translate(txt, s, t, **kwargs):
        choice = (backend_choice or auto_select_backend(s, t)).lower()
        if choice == "indic":
            return mm.translate_text(txt, s, t, mt_model_choice="indic", use_processor=use_processor)
        elif choice == "nllb":
            return mm.translate_text(txt, s, t, mt_model_choice="nllb")
        elif choice == "google":
            return mt_google.translate_joined(txt, s, t)
        else:
            # fallback to nllb always instead of auto→indic crash
            return mm.translate_text(txt, s, t, mt_model_choice="nllb")

    # Delegate to preprocessor
    return mt_preprocessor.preprocess_and_translate(
        text,
        src_iso=src_lang,
        tgt_flores=tgt_flores,
        translate_func=_base_translate,
        mode=mode,
        translit_target_scheme=translit_scheme,
        english_run_action=code_mixed_english_action,
        ner_preserve=ner_preserve,
        backend_choice=backend_choice,
    )

# -------------------------
# Batch Translation Wrapper with Progress Bar
# -------------------------
def batch_translate_via_perform(
    text: str,
    src_iso: str,
    tgt_flores: str,
    backend_choice: Optional[str] = None,
    mode: Optional[str] = None,
    use_processor: bool = False,
    translit_scheme: str = "ITRANS",
    code_mixed_english_action: str = "pass",
    ner_preserve: bool = False,
    max_chunk_size: int = 1800,
) -> str:
    sentences = mm._split_into_sentences(text)
    chunks = mm._group_sentences(sentences, char_limit=max_chunk_size)

    outputs = []
    for chunk in tqdm(chunks, desc="🌍 Translating", unit="chunk"):   # ✅ real-time loading bar
        try:
            out = perform_translation(
                chunk,
                src_iso,
                tgt_flores,
                backend_choice=backend_choice,
                use_processor=use_processor,
                mode=mode,
                translit_scheme=translit_scheme,
                code_mixed_english_action=code_mixed_english_action,
                ner_preserve=ner_preserve,
            )
            outputs.append(out)
        except Exception as e:
            print(f"⚠️ Chunk translation failed: {e}")
            outputs.append("[translation_error]")
    return " ".join(outputs).strip()

# -------------------------
# Shim for pipeline
# -------------------------
def translate_text(
    text: str,
    src_lang_iso_or_flores: str,
    tgt_flores: str = "eng_Latn",
    mt_model_choice: Optional[str] = None,
    use_processor: bool = False,
    **kwargs
) -> str:
    return mm.translate_text(
        text=text,
        src_lang_iso_or_flores=src_lang_iso_or_flores,
        tgt_flores=tgt_flores,
        mt_model_choice=mt_model_choice,
        use_processor=use_processor,
    )

__all__ = [
    "ISO_TO_FLORES",
    "GLOBAL_LANGS",
    "INDIC_LANGS_MENU",
    "choose_language_menu",
    "auto_select_backend",
    "perform_translation",
    "batch_translate_via_perform",
    "translate_text",
    "mt_google",
    "mt_preprocessor",
]
