# tts_common/tts_chunker.py
"""
Utilities for splitting long text into safe chunks for TTS models.
Uses nltk if available, otherwise simple heuristics.
"""

import re

# Try to use nltk sentence tokenizer if present
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except Exception:
    try:
        import nltk
        nltk.download("punkt", quiet=True)
    except Exception:
        nltk = None  # we'll fallback

def sentence_split(text: str):
    text = text.strip()
    if not text:
        return []
    if nltk:
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            pass
    # simple fallback: split on .!? plus newlines
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def join_chunks(chunks):
    return " ".join([c.strip() for c in chunks if c.strip()])

def split_text_by_max_chars(text: str, max_chars: int = 700):
    """
    Split text into chunks each <= max_chars. Prefer sentence boundaries.
    """
    if not text or len(text) <= max_chars:
        return [text.strip()]

    sentences = sentence_split(text)
    chunks = []
    current = ""
    for s in sentences:
        if not current:
            current = s
        elif len(current) + 1 + len(s) <= max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())

    # final safety: ensure no chunk is longer than max_chars — if so, hard-split
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            # hard chop
            for i in range(0, len(c), max_chars):
                final.append(c[i:i+max_chars].strip())
    return final
