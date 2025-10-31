# MT_Model/mt_google.py
"""
Lightweight Google Translate wrapper (scraped client).
- Accepts source/target as FLORES codes (eng_Latn, hin_Deva, ...) or ISO (en, hi).
- Provides batched translation helpers and a "joiner" that splits long text safely.
- Retries on transient errors and falls back gracefully (returns original chunk on persistent failure).
Notes:
- This uses the community `googletrans` (4.0.0-rc1) client. It's unofficial and can be flaky.
- For production use the official Google Cloud Translate API.
"""

import os
import time
import typing as t

try:
    from googletrans import Translator
except Exception as e:
    Translator = None  # will raise later if used

# Minimal FLORES -> Google ISO mapping.
# Keep this in sync with your ISO_TO_FLORES or add entries if you need more languages.
INDICTRANS_TO_GOOGLE = {
    # Indic / Indo-Aryan
    "hin_Deva": "hi",
    "mar_Deva": "mr",
    "guj_Gujr": "gu",
    "ben_Beng": "bn",
    "pan_Guru": "pa",
    "asm_Beng": "as",
    "ory_Orya": "or",
    "npi_Deva": "ne",
    "kas_Arab": "ks",
    "kas_Deva": "ks",
    "gom_Deva": "gom",
    "mai_Deva": "mai",
    "snd_Arab": "sd",
    "snd_Deva": "sd",
    "san_Deva": "sa",
    "urd_Arab": "ur",
    "brx_Deva": "brx",
    "doi_Deva": "doi",
    "sat_Olck": "sat",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "kan_Knda": "kn",
    "mal_Mlym": "ml",
    "mni_Beng": "mni",
    "mni_Mtei": "mni",

    # Global / FLORES style -> ISO
    "eng_Latn": "en",
    "spa_Latn": "es",
    "fra_Latn": "fr",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "deu_Latn": "de",
    "rus_Cyrl": "ru",
    "zho_Hans": "zh-cn",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "arb_Arab": "ar",
    "pes_Arab": "fa",
    "tur_Latn": "tr",
    "ind_Latn": "id",
}

# Some common aliases Google accepts too (not exhaustive)
FALLBACK_SCRIPT_MAP = {
    "zh": "zh-cn",
}

# global translator instance (lazy)
_google_translator: t.Optional[Translator] = None


def _ensure_translator():
    global _google_translator
    if _google_translator is None:
        if Translator is None:
            raise ImportError(
                "googletrans not available. Install with: pip install googletrans==4.0.0-rc1"
            )
        # You can pass service_urls if translation fails in your region.
        _google_translator = Translator()
    return _google_translator


def _normalize_for_google(code: str) -> str:
    """
    Accepts:
      - FLORES code (eng_Latn, hin_Deva, ...)
      - ISO code (en, hi)
    Returns best-effort Google ISO code (e.g. 'hi', 'en', 'zh-cn').
    """
    if not code:
        return "en"
    code = str(code).strip()
    if "_" in code:
        # FLORES style
        if code in INDICTRANS_TO_GOOGLE:
            return INDICTRANS_TO_GOOGLE[code]
        base = code.split("_", 1)[0]
        return FALLBACK_SCRIPT_MAP.get(base, base)
    # already short ISO — normalize some known variants
    lc = code.lower()
    return FALLBACK_SCRIPT_MAP.get(lc, lc)


def translate_google_list(
    texts: t.List[str],
    src: str,
    tgt: str,
    save_path: t.Optional[str] = None,
    retry: int = 2,
    sleep_between_retries: float = 1.0,
) -> t.List[str]:
    """
    Translate a list of short texts (sentences) using googletrans.
    Returns list of translations (same length).
    On repeated failure for a chunk, returns the original chunk (graceful fallback).
    """
    translator = _ensure_translator()
    src_code = _normalize_for_google(src)
    tgt_code = _normalize_for_google(tgt)

    outputs: t.List[str] = []
    for t_text in texts:
        attempt = 0
        last_exc = None
        while attempt <= retry:
            try:
                # googletrans accepts a single string; sometimes it accepts a list
                res = translator.translate(t_text, src=src_code, dest=tgt_code)
                # res may be a single object
                txt = getattr(res, "text", None)
                if txt is None and isinstance(res, list) and len(res) > 0:
                    txt = getattr(res[0], "text", t_text)
                outputs.append(txt if txt is not None else t_text)
                break
            except Exception as e:
                last_exc = e
                attempt += 1
                time.sleep(sleep_between_retries)
        else:
            # exhausted retries → fallback to original chunk
            print(f"⚠️ Google Translate failed for chunk (src={src_code}, tgt={tgt_code}): {last_exc}")
            outputs.append(t_text)

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as fh:
                for line in outputs:
                    fh.write(line + "\n")
            print(f"💾 Google translations saved to: {save_path}")
        except Exception as e:
            print(f"⚠️ Warning: could not save google translations to {save_path}: {e}")

    return outputs


# convenience helper: chunk by sentences and join results
def translate_joined(
    text: str,
    src_code: str,
    tgt_code: str,
    char_limit: int = 1000,
    **kwargs
) -> str:
    """
    High level convenience:
    - Splits text into sentences (lightweight)
    - Groups sentences into chunks ≤ char_limit
    - Calls translate_google_list for chunked items
    - Re-joins output and returns single string
    """
    import re

    # lightweight sentence split
    sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', text) if s.strip()]
    if not sents:
        # fallback: treat all text as one chunk
        chunks = [text.strip()]
    else:
        # group into chunks
        chunks, cur, cur_len = [], [], 0
        for s in sents:
            if cur_len + len(s) + 1 <= char_limit:
                cur.append(s)
                cur_len += len(s) + 1
            else:
                chunks.append(" ".join(cur))
                cur = [s]
                cur_len = len(s)
        if cur:
            chunks.append(" ".join(cur))

    translations = translate_google_list(chunks, src_code, tgt_code, **kwargs)
    return " ".join(translations).strip()


# quick CLI test

if __name__ == "__main__":
    # basic demo (quick sanity check)
    demo_src = "eng_Latn"
    demo_tgt = "hin_Deva"
    demo_text = "Hello. This is a short demo of Google Translate integration. How are you?"
    print("Demo translating:", demo_text)
    out = translate_joined(demo_text, demo_src, demo_tgt)
    print("Result:", out)
