"""
Debug utilities for MT quality checking (e.g., back-translation).
"""

from typing import Callable, Optional, Dict
from MT_Model import mt_helper as mh


def back_translate(
    text: str,
    src_lang: str,
    tgt_flores: str,
    translate_callable: Optional[Callable[..., str]] = None,
    backend_choice: Optional[str] = "auto",
    **kwargs
) -> Dict[str, str]:
    """
    Translate src → tgt → src again.

    Args:
        text: Input text to test
        src_lang: Source language (ISO or FLORES code)
        tgt_flores: Target FLORES code
        translate_callable: Function used for translation
            (default = mh.perform_translation)
        backend_choice: which backend to use ("auto","nllb","indic","google")
        **kwargs: Extra args forwarded into translate_callable
                  (e.g., mode, ner_preserve, translit_scheme, etc.)

    Returns:
        dict with forward, backward, and a simple quality note
    """
    if translate_callable is None:
        translate_callable = mh.perform_translation

    # forward translation
    forward = translate_callable(
        text, src_lang, tgt_flores, backend_choice=backend_choice, **kwargs
    )

    # back translation
    backward = translate_callable(
        forward, tgt_flores, src_lang, backend_choice=backend_choice, **kwargs
    )

    quality_note = "✅ Looks consistent"
    if len(backward.split()) < 0.7 * len(text.split()):
        quality_note = "⚠️ Possible information loss"

    return {
        "source": text,
        "forward": forward,
        "backward": backward,
        "note": quality_note,
    }


if __name__ == "__main__":
    # Example quick test
    sample = "Hello, how are you?"
    result = back_translate(sample, "en", "hin_Deva", backend_choice="nllb")
    print(result)
