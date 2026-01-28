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
    Fixes PyTorch 2.6+ compatibility issue with weights_only parameter.
    """
    global _COQUI_TTS_INSTANCE
    if _COQUI_TTS_INSTANCE is None:
        # Fix for PyTorch 2.6+: Patch TTS library's load_fsspec to use weights_only=False
        # This is needed because Coqui TTS checkpoints contain custom classes that require
        # weights_only=False (PyTorch 2.6 changed default from False to True)
        try:
            import TTS.utils.io as tts_io
            
            # Patch load_fsspec if not already patched (singleton pattern)
            if not hasattr(tts_io, '_tts_patched_load_fsspec'):
                original_load_fsspec = tts_io.load_fsspec
                
                def patched_load_fsspec(path, map_location=None, **kwargs):
                    """Patched load_fsspec that explicitly sets weights_only=False for PyTorch 2.6+"""
                    kwargs['weights_only'] = False
                    return original_load_fsspec(path, map_location=map_location, **kwargs)
                
                tts_io.load_fsspec = patched_load_fsspec
                tts_io._tts_patched_load_fsspec = True
                print("[INFO] Patched TTS.utils.io.load_fsspec for PyTorch 2.6+ compatibility")
        except Exception as patch_error:
            print(f"[WARN] Could not patch TTS load_fsspec: {patch_error}")
            # Will try torch.load patch as fallback
        
        try:
            from TTS.api import TTS
        except Exception as e:
            raise RuntimeError("Coqui TTS package not installed or broken. Install `TTS` from Coqui.") from e
        
        try:
            print("[INFO] Loading Coqui TTS model...")
            _COQUI_TTS_INSTANCE = TTS(model_name)
            print("[INFO] Coqui TTS model loaded successfully")
        except Exception as load_error:
            error_str = str(load_error)
            # Check if it's a weights_only error
            if "weights_only" in error_str or "WeightsUnpickler" in error_str:
                print(f"[WARN] TTS loading failed with weights_only error, trying torch.load patch...")
                # Fallback: patch torch.load temporarily during loading
                try:
                    import torch
                    original_torch_load = torch.load
                    
                    def patched_torch_load(*args, **kwargs):
                        """Patched torch.load that defaults weights_only=False for TTS"""
                        if 'weights_only' not in kwargs:
                            kwargs['weights_only'] = False
                        return original_torch_load(*args, **kwargs)
                    
                    torch.load = patched_torch_load
                    try:
                        print("[INFO] Retrying with patched torch.load...")
                        _COQUI_TTS_INSTANCE = TTS(model_name)
                        print("[INFO] Coqui TTS model loaded successfully with torch.load patch")
                    finally:
                        # Restore original torch.load
                        torch.load = original_torch_load
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load Coqui TTS model. PyTorch 2.6+ compatibility issue. "
                        f"Original error: {load_error}, Workaround error: {e2}. "
                        f"Consider downgrading PyTorch to <2.6 or updating Coqui TTS."
                    ) from e2
            else:
                raise
        
        # Move model to device if possible
        try:
            _COQUI_TTS_INSTANCE.to(device if device else ("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        except Exception:
            pass
    return _COQUI_TTS_INSTANCE

def synthesize_coqui_xtts(text: str, language: str = "en", speaker_wav: Optional[str] = None,
                          out_path: str = "out_xtts.wav", model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                          device: Optional[str] = None, speaker: Optional[str] = None):
    """
    Synthesize with Coqui XTTS (XTTS v2).

    IMPORTANT: XTTS v2 voice cloning requires a `speaker_wav` reference audio.
    (If you want a “default speaker” without providing a voice sample, use a different single-speaker TTS model.)
    Returns (out_path, sampling_rate).
    """
    tts = _get_coqui_instance(model_name=model_name, device=device)

    # Coqui API: use `tts_to_file` or `tts` depending on environment.
    # If tts has 'tts_to_file', call that; else attempt tts.tts(...) returning numpy array.
    if not speaker_wav:
        raise ValueError(
            "XTTS v2 requires a reference voice WAV (`speaker_wav`). "
            "Record your voice to a .wav file and pass it to the pipeline (see `--tts-speaker-wav`)."
        )
    if not os.path.exists(speaker_wav):
        raise ValueError(f"XTTS speaker_wav not found: {speaker_wav}")

    # Prefer tts_to_file
    if hasattr(tts, "tts_to_file"):
        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=out_path,
            )
            try:
                data, sr = sf.read(out_path)
                return out_path, sr
            except Exception:
                return out_path, 24000
        except Exception as e:
            raise RuntimeError(f"Coqui XTTS synthesis failed: {e}") from e

    # Fallback: array-returning API
    try:
        wav = tts.tts(text=text, speaker_wav=speaker_wav, language=language)
        sr = 24000
        if isinstance(wav, tuple) and len(wav) >= 2:
            arr, sr = wav[0], int(wav[1])
        else:
            arr = wav
        arr = np.asarray(arr)
        sf.write(out_path, arr, sr)
        return out_path, sr
    except Exception as e:
        raise RuntimeError(f"Coqui XTTS synthesis failed: {e}") from e
