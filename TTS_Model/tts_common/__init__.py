# tts_common/__init__.py
"""
Unified TTS common utilities.

Public entrypoint:
    from tts_common import run_universal_tts
"""
from .tts_handler import run_universal_tts

__all__ = ["run_universal_tts"]
