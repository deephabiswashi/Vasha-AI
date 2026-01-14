# tts_common/tts_handler.py
"""
High-level TTS handler for pipeline integration.

Function: run_universal_tts(text, target_lang, reference_audio=None, prefer='auto', out_dir, out_name)
This will:
 - normalize language (FLORES or ISO if mapping provided)
 - choose engine (XTTS, Indic-Parler, or fallback to gTTS)
 - check cache
 - chunk text if required
 - produce final single WAV file and return its path
"""

import os
import shutil
from typing import Optional
from .tts_cache import exists_in_cache, save_to_cache, cache_filepath, make_cache_dir
from .tts_chunker import split_text_by_max_chars
from .tts_fallbacks import run_gtts, convert_mp3_to_wav
from .tts_interface import synthesize_indic_parler, synthesize_coqui_xtts
# optional mapping module (user may provide a more complete tts_utils)
try:
    from TTS_Model.tts_common.tts_utils import FLORES_TO_ISO, INDIC_LANGS, XTTS_LANGS  # type: ignore
except Exception:
    # minimal defaults
    FLORES_TO_ISO = {
        "eng_Latn": "en", "hin_Deva": "hi", "jpn_Jpan": "ja", "zho_Hans": "zh-cn"
    }
    INDIC_LANGS = {"hi", "bn", "gu", "mr", "ta", "te", "kn", "ml", "pa", "or", "as", "ne"}
    XTTS_LANGS = {"en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh-cn", "hi"}

def _normalize_lang(code: str) -> str:
    if not code:
        return ""
    # handle Flores-style codes or iso
    if code in FLORES_TO_ISO:
        # Flores -> ISO (or mapping value could be iso-like)
        mapped = FLORES_TO_ISO[code]
        # some entries map to xx_XXX -> reduce to part before underscore where appropriate
        if isinstance(mapped, str) and "_" in mapped:
            return mapped.split("_")[0]
        return mapped
    # reduce e.g. eng_Latn if passed
    if "_" in code:
        return code.split("_")[0]
    return code

def _assemble_wav_parts(parts, out_path, sample_rate=24000):
    """
    Concatenate audio parts (file paths or numpy arrays) into single WAV
    parts: list of file paths. For simplicity we expect file paths.
    """
    import soundfile as sf
    import numpy as np

    all_audio = []
    sr = None
    for p in parts:
        if isinstance(p, (list, tuple)):
            # maybe (wav_path, sr)
            p, src = p if len(p) > 1 else (p[0], None)
        if not os.path.exists(p):
            continue
        data, s = sf.read(p)
        if sr is None:
            sr = s
        # resample? we will assume same sr or rely on model defaults
        all_audio.append(data)
    if not all_audio:
        raise RuntimeError("No audio parts to assemble.")
    concat = np.concatenate(all_audio)
    dst_sr = int(sr or sample_rate)
    sf.write(out_path, concat, dst_sr)
    return out_path, dst_sr

def run_universal_tts(
    text: str,
    target_lang: str,
    reference_audio: Optional[str] = None,
    prefer: str = "auto",
    out_dir: Optional[str] = None,
    out_name: Optional[str] = None,
    hf_token: Optional[str] = None,
    device: Optional[str] = None,
    max_chunk_chars: int = 700,
    use_cache: bool = True
) -> str:
    """
    Top-level function to synthesize `text` in `target_lang` (FLORES or ISO).
    Returns path to saved audio (wav or mp3 depending on engine).
    """

    out_dir = out_dir or os.path.join(os.getcwd(), "tts_output")
    os.makedirs(out_dir, exist_ok=True)
    out_name = out_name or "tts_out.wav"
    out_path = os.path.join(out_dir, out_name)

    # 1) normalize language
    lang_norm = _normalize_lang(target_lang)

    # 2) Decide engine
    engine = None
    if prefer and prefer != "auto":
        engine = prefer.lower()
    else:
        # auto heuristic: indic languages -> indic parler; XTTS supported -> xtts else gtts
        if lang_norm in INDIC_LANGS:
            engine = "indic"
        elif lang_norm in XTTS_LANGS:
            engine = "xtts"
        else:
            # default try xtts first for global, then indic, then gtts
            engine = "xtts"

    # 3) check cache
    if use_cache:
        cached = exists_in_cache(text, lang_norm, desc="", engine=engine, base_dir=None, ext=".wav")
        if cached:
            # copy cached to out_path (so pipeline always finds it)
            shutil.copyfile(cached, out_path)
            return out_path

    # 4) chunk and synthesize
    chunks = split_text_by_max_chars(text, max_chars=max_chunk_chars)
    if not chunks:
        raise ValueError("Empty text passed to TTS.")

    part_paths = []
    for idx, chunk in enumerate(chunks):
        tmp_name = f"part_{idx:03d}.wav"
        tmp_path = os.path.join(out_dir, tmp_name)
        try:
            if engine == "indic":
                # Indic-Parler uses description voice prompts; best-effort default
                desc = "The speaker speaks naturally with clear audio and neutral tone."
                p_out, sr = synthesize_indic_parler(chunk, description=desc, out_path=tmp_path, hf_token=hf_token, device=device)
                part_paths.append(p_out)
            elif engine in ("xtts", "coqui", "coqui_xtts"):
                # Coqui XTTS - pass reference if given for cloning
                p_out, sr = synthesize_coqui_xtts(chunk, language=lang_norm, speaker_wav=reference_audio, out_path=tmp_path, device=device)
                part_paths.append(p_out)
            else:
                # fallback to gTTS
                mp3_name = f"gtts_part_{idx:03d}.mp3"
                mp3_path = run_gtts(chunk, lang=lang_norm or "en", out_dir=out_dir, out_name=mp3_name)
                wav_tmp = os.path.join(out_dir, tmp_name)
                try:
                    convert_mp3_to_wav(mp3_path, wav_tmp)
                    part_paths.append(wav_tmp)
                except Exception:
                    # if conversion fails, return mp3
                    part_paths.append(mp3_path)
        except Exception as e:
            # If primary engine fails for chunk, try fallback engine order
            # fallback order: xtts -> indic -> gtts
            fallback_used = None
            last_exc = e
            try:
                if engine != "xtts":
                    p_out, sr = synthesize_coqui_xtts(chunk, language=lang_norm, speaker_wav=reference_audio, out_path=tmp_path, device=device)
                    part_paths.append(p_out)
                    fallback_used = "xtts"
                else:
                    raise RuntimeError("Primary xtts failed and fallback not attempted.")
            except Exception as e2:
                last_exc = e2
                try:
                    if engine != "indic":
                        p_out, sr = synthesize_indic_parler(chunk, description="The speaker speaks naturally.", out_path=tmp_path, hf_token=hf_token, device=device)
                        part_paths.append(p_out)
                        fallback_used = "indic"
                    else:
                        raise RuntimeError("Indic fallback failed too.")
                except Exception as e3:
                    last_exc = e3
                    # final fallback to gTTS
                    mp3_name = f"gtts_fallback_part_{idx:03d}.mp3"
                    mp3_path = run_gtts(chunk, lang=lang_norm or "en", out_dir=out_dir, out_name=mp3_name)
                    wav_tmp = os.path.join(out_dir, tmp_name)
                    try:
                        convert_mp3_to_wav(mp3_path, wav_tmp)
                        part_paths.append(wav_tmp)
                        fallback_used = "gtts"
                    except Exception:
                        part_paths.append(mp3_path)
                        fallback_used = "gtts_mp3"
            # continue after fallback

    # 5) assemble
    final_path, final_sr = _assemble_wav_parts(part_paths, out_path)
    # 6) save to cache
    try:
        save_to_cache(final_path, text, lang_norm, desc="", engine=engine)
    except Exception:
        pass
    return final_path
