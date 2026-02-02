import whisper
import torch
import argparse
import tempfile
import subprocess
import yt_dlp
import sounddevice as sd
import numpy as np
import os
from scipy.io.wavfile import write
from langid import classify
import torchaudio
from .spoof_detection import is_spoofed_audio

# ✅ Updated to include all IndicConformer-supported languages
TARGET_LANGS = {
    'en': 'English', 'as': 'Assamese', 'bn': 'Bengali', 'brx': 'Bodo',
    'doi': 'Dogri', 'gu': 'Gujarati', 'hi': 'Hindi', 'kn': 'Kannada',
    'kok': 'Konkani', 'ks': 'Kashmiri', 'mai': 'Maithili', 'ml': 'Malayalam',
    'mni': 'Manipuri', 'mr': 'Marathi', 'ne': 'Nepali', 'or': 'Odia',
    'pa': 'Punjabi', 'sa': 'Sanskrit', 'sat': 'Santali', 'sd': 'Sindhi',
    'ta': 'Tamil', 'te': 'Telugu', 'ur': 'Urdu', 'zh': 'Mandarin Chinese',
    'ar': 'Arabic', 'ru': 'Russian', 'it': 'Italian', 'ko': 'Korean', 'ja': 'Japanese',
    'es': 'Spanish'
}

class LanguageIdentifier:
    def __init__(self, model_size="small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📦 Loading Whisper model '{model_size}' on {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)

    def detect(self, audio_path, duration_limit=None):
        print(f"🧠 Detecting language using Whisper transcription (Limit: {duration_limit}s)...")
        try:
            # Load audio using Whisper's internal utility to get 16kHz mono array
            audio = whisper.load_audio(audio_path)
            
            # Trim if needed (Whisper expects 16kHz)
            if duration_limit:
                samples = int(duration_limit * 16000)
                if len(audio) > samples:
                    audio = audio[:samples]
            
            # Pad or Trim to 30s is handled by log_mel_spectrogram usually, 
            # but transcribe handles raw audio directly.
            result = self.model.transcribe(audio, task="transcribe", language=None, fp16=False)
            
            detected_lang = result.get("language")  # Code like 'bn'
            
            # Attempt to extract confidence if available in segments or result
            # Whisper 'transcribe' result usually just has 'language'. 
            # To get confidence we might need to look at segments or internal logits if we used detect_language()
            # But standard transcribe results don't always expose language confidence top-level easily.
            # However, for this task, we will assume 1.0 or try to parse.
            # Actually, using model.detect_language() is more appropriate for LID!
            
            # Let's switch to model.detect_language() which is efficient for LID (Encoder only)
            # It requires computing Mel spectrogram first.
            
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # detect the spoken language
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            
            print(f"✅ Whisper LID: {detected_lang} (Confidence: {confidence:.2f})")
            return detected_lang, {detected_lang: confidence}

        except Exception as e:
            print(f"❌ Language detection error: {e}")
            return None, {}

def detect_dialect(audio_path):
    try:
        print("🌐 Detecting dialect...")
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        lang_code, _ = classify(result['text'])
        return lang_code
    except Exception as e:
        print(f"⚠️ Dialect detection failed: {e}")
        return "unknown"

def extract_audio_ffmpeg(video_path):
    print("🎬 Extracting audio from video...")
    temp_audio = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", temp_audio]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_audio

def download_youtube_audio(url):
    """
    ⬇️ Robust YouTube audio downloader:
    - Requests best available audio (any codec/bitrate)
    - Converts to WAV using ffmpeg (via yt-dlp postprocessor)
    - Retries on transient errors / format availability issues
    - Returns the actual .wav path
    """
    print("⬇️ Downloading YouTube audio...")
    # Put artifacts in a unique temp directory (so we can locate the converted WAV reliably)
    tmpdir = tempfile.mkdtemp(prefix="yt_")
    outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")

    # yt-dlp options to be resilient against format unavailability and 403s
    ydl_opts = {
        "format": "bestaudio/best",          # ✅ Let yt-dlp pick the best available audio
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": False,                      # keep some logging (helps debugging)
        "ignoreerrors": True,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "geo_bypass": True,
        "nocheckcertificate": True,
        "http_headers": {                    # helps with some CDN/User-Agent picky endpoints
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
        # Convert to WAV after download
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",   # bitrate for source; WAV is PCM anyway
            }
        ],
        # Ensure ffmpeg writes a standard PCM wav (mono downmix & 16k can be done by your ASR later)
        # If you prefer to force 16k mono right here, uncomment:
        # "postprocessor_args": ["-ar", "16000", "-ac", "1"],
    }

    # Try a couple of times with slight format variations if needed
    attempted_formats = ["bestaudio/best", "ba* / b* / best"]  # second is a very relaxed fallback
    last_err = None

    for fmt in attempted_formats:
        ydl_opts["format"] = fmt
        for attempt in range(1, 4):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if info is None:
                        raise RuntimeError("yt-dlp failed to extract info")

                    # If it's a playlist/redirect, pick the first entry
                    if "entries" in info and info["entries"]:
                        info = next((e for e in info["entries"] if e), None)
                        if info is None:
                            raise RuntimeError("yt-dlp returned empty playlist/entries")

                    # After postprocessing, output should be .wav in tmpdir with the video id basename
                    video_id = info.get("id")
                    if not video_id:
                        # Fallback: find the largest .wav file in tmpdir
                        wavs = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith(".wav")]
                        if not wavs:
                            raise RuntimeError("No WAV produced by yt-dlp/ffmpeg")
                        # Pick the largest (safest bet if multiple)
                        wavs.sort(key=lambda p: os.path.getsize(p), reverse=True)
                        return wavs[0]
                    else:
                        wav_path = os.path.join(tmpdir, f"{video_id}.wav")
                        if os.path.exists(wav_path):
                            return wav_path
                        # Fallback scan if naming differed
                        wavs = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith(".wav")]
                        if wavs:
                            wavs.sort(key=lambda p: os.path.getsize(p), reverse=True)
                            return wavs[0]
                        raise RuntimeError("WAV file not found after conversion.")

            except Exception as e:
                last_err = e
                print(f"⚠️ Attempt {attempt}/3 with format '{fmt}' failed: {e}")

    # If all attempts failed, re-raise the last error so caller can handle/log it
    raise RuntimeError(f"YouTube download failed: {last_err}")

def record_live_audio(duration=5, sample_rate=16000):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording complete.")
    out_path = os.path.join(os.getcwd(), "recorded.wav")
    write(out_path, sample_rate, (audio * 32767).astype(np.int16))
    print(f"💾 Audio saved to {out_path}")
    return out_path

def print_results(lang, probs, dialect, spoofed):
    print("\n🧾 Analysis Results:")
    if lang:
        print(f"✅ Detected Language: {TARGET_LANGS.get(lang, lang)} ({lang})")
        print("📊 Language Probabilities:")
        for code, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {TARGET_LANGS.get(code, code)} ({code}): {prob:.2f}")
    else:
        print("⚠️ Could not detect a supported language.")

    print(f"🗣 Dialect (langid): {dialect}")
    print(f"🔒 Spoofing: {'❌ FAKE audio detected' if spoofed else '✅ Real Human Speech'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🌍 Whisper Language ID + Dialect + Spoof Detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', help="Audio/video file")
    group.add_argument('--youtube', help="YouTube URL")
    group.add_argument('--mic', action='store_true', help="Mic input")
    parser.add_argument('--duration', type=int, default=5, help="Mic duration (seconds)")
    args = parser.parse_args()

    if args.file:
        audio_path = extract_audio_ffmpeg(args.file) if args.file.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')) else args.file
    elif args.youtube:
        audio_path = download_youtube_audio(args.youtube)
    elif args.mic:
        audio_path = record_live_audio(duration=args.duration)
    else:
        print("❗ No valid input source.")
        exit(1)

    lid = LanguageIdentifier()
    lang, probs = lid.detect(audio_path)
    dialect = detect_dialect(audio_path)
    spoofed = is_spoofed_audio(audio_path)
    print_results(lang, probs, dialect, spoofed)
