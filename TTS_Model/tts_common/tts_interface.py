# tts_common/tts_interface.py
"""
Low-level interface that wraps model-specific calls.
Provides:
- synthesize_indic_parler(...)
- synthesize_coqui_xtts(...)
These functions return a path to the generated audio file (wav) and sampling rate where applicable.
"""

import os
import numpy as np
import soundfile as sf
from typing import Optional

# Model singletons to avoid reloading
_INDIC_TTS_INSTANCE = None
_COQUI_TTS_INSTANCE = None

# Provide lazy import so package doesn't fail when model dependencies missing
def _get_indic_instance(hf_token: Optional[str] = None, device: Optional[str] = None):
    global _INDIC_TTS_INSTANCE
    if _INDIC_TTS_INSTANCE is None:
        # Always use the underscore module path (hyphens are invalid in Python module names)
        from TTS_Model.indic_parler_tts.model_loader import IndicParlerTTS  # type: ignore
        _INDIC_TTS_INSTANCE = IndicParlerTTS(device=device, hf_token=hf_token)
    return _INDIC_TTS_INSTANCE

def synthesize_indic_parler(text: str, description: str = "", out_path: str = "out_indic.wav",
                            hf_token: Optional[str] = None, device: Optional[str] = None):
    """
    Generate TTS using Indic-Parler-TTS model loader. Returns (out_path, sampling_rate).
    """
    inst = _get_indic_instance(hf_token=hf_token, device=device)
    audio_np, sr = inst.synthesize(text, description=description, save_path=out_path)
    # If model already saved file inside synthesize, ensure out_path exists; otherwise write here
    if not os.path.exists(out_path):
        sf.write(out_path, audio_np, sr)
    return out_path, sr

def _get_coqui_instance(model_name="tts_models/multilingual/multi-dataset/xtts_v2", device=None):
    """
    Return a Coqui TTS instance (TTS api). Keep as singleton.
    """
    global _COQUI_TTS_INSTANCE
    if _COQUI_TTS_INSTANCE is None:
        try:
            from TTS.api import TTS
        except Exception as e:
            raise RuntimeError("Coqui TTS package not installed or broken. Install `TTS` from Coqui.") from e
        _COQUI_TTS_INSTANCE = TTS(model_name)
        # Move model to device if possible
        try:
            _COQUI_TTS_INSTANCE.to(device if device else ("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        except Exception:
            pass
    return _COQUI_TTS_INSTANCE

def synthesize_coqui_xtts(text: str, language: str = "en", speaker_wav: Optional[str] = None,
                          out_path: str = "out_xtts.wav", model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                          device: Optional[str] = None):
    """
    Synthesize with Coqui XTTS. Must provide either `speaker_wav` (voice cloning) or `speaker` identifier.
    Returns (out_path, sampling_rate).
    """
    tts = _get_coqui_instance(model_name=model_name, device=device)

    # Coqui API: use `tts_to_file` or `tts` depending on environment.
    # If tts has 'tts_to_file', call that; else attempt tts.tts(...) returning numpy array.
    if speaker_wav and not os.path.exists(speaker_wav):
        speaker_wav = None

    # prefer tts_to_file which handles multi-sentence
    if hasattr(tts, "tts_to_file"):
        # generate file
        try:
            tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=out_path)
            # Coqui defaults often sample rate 24000 or 22050; try to read
            try:
                import soundfile as sf
                data, sr = sf.read(out_path)
                return out_path, sr
            except Exception:
                return out_path, 24000
        except Exception as e:
            # try fallback tts method (returns array)
            pass

    # fallback: call tts.tts which often returns waveform array
    try:
        wav = tts.tts(text=text, speaker_wav=speaker_wav, language=language)
        # write wav
        import soundfile as sf
        sr = 24000
        if isinstance(wav, tuple) and len(wav) >= 2:
            # some tts returns (audio, sr)
            arr, sr = wav[0], int(wav[1])
        else:
            arr = wav
        arr = np.asarray(arr)
        sf.write(out_path, arr, sr)
        return out_path, sr
    except Exception as e:
        raise RuntimeError(f"Coqui XTTS synthesis failed: {e}") from e
