# tts_common/tts_fallbacks.py
"""
Fallback TTS utilities: gTTS wrapper and placeholders for cloud TTS.
"""

import os
import tempfile

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def run_gtts(text: str, lang: str = "en", out_dir: str = "tts_output", out_name: str = "gtts_out.mp3", slow: bool = False):
    """
    Use gTTS to create an MP3 fallback. Returns path to generated file.
    """
    try:
        from gtts import gTTS
    except Exception as e:
        raise RuntimeError("gTTS not installed. Install with `pip install gTTS`.") from e

    out_dir = _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, out_name)
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(out_path)
    return out_path

def convert_mp3_to_wav(mp3_path: str, wav_path: str, sample_rate: int = 22050):
    """
    Convert mp3 to wav using pydub if available; fallback to ffmpeg command if not.
    """
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(mp3_path)
        seg = seg.set_frame_rate(sample_rate).set_channels(1)
        seg.export(wav_path, format="wav")
        return wav_path
    except Exception:
        # try ffmpeg CLI fallback
        cmd = ["ffmpeg", "-y", "-i", mp3_path, "-ar", str(sample_rate), "-ac", "1", wav_path]
        import subprocess
        subprocess.run(cmd, check=True)
        return wav_path

# Placeholder for future cloud fallback (e.g., Google Cloud TTS, Azure)
def run_cloud_tts_placeholder(text: str, lang: str = "en", out_dir: str = "tts_output", out_name: str = "cloud_out.wav"):
    """
    Placeholder to integrate cloud TTS later. Currently raises NotImplementedError.
    """
    raise NotImplementedError("Cloud TTS fallback not implemented. Use run_gtts instead for now.")
