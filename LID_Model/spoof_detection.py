# spoof_detection.py
import torchaudio
import subprocess
import tempfile
import os
import soundfile as sf  # Ensure soundfile is available

# ✅ Force compatible backend
torchaudio.set_audio_backend("soundfile")

def reencode_audio(audio_path):
    """Ensure file is in 16kHz mono WAV format."""
    reencoded = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ar", "16000",
        "-ac", "1",
        reencoded
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return reencoded

def is_spoofed_audio(audio_path, threshold=1e-4):
    try:
        fixed_path = reencode_audio(audio_path)
        waveform, sr = torchaudio.load(fixed_path)

        if waveform.shape[0] == 0 or waveform.abs().mean().item() < threshold:
            print("⚠️ Detected low energy or empty waveform.")
            return True  # Possibly spoofed or silent

        return False  # Valid, high-energy waveform
    except Exception as e:
        print(f"⚠️ Spoof detection failed (backend error): {e}")
        return True  # Treat uncertain as spoofed
