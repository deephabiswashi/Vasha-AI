import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
# recommended to run this on a gpu with flash_attn installed
# don't set attn_implemetation if you don't have flash_attn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

src_lang, tgt_lang = "hin_Deva", "ben_Beng"
model_name = "ai4bharat/indictrans2-indic-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, # performance might slightly vary for bfloat16
    attn_implementation="flash_attention_2"
).to(DEVICE)

ip = IndicProcessor(inference=True)

input_sentences = [
    "जब मैं छोटा था, मैं हर रोज़ पार्क जाता था।",
    "हमने पिछले सप्ताह एक नई फिल्म देखी जो कि बहुत प्रेरणादायक थी।",
    "अगर तुम मुझे उस समय पास मिलते, तो हम बाहर खाना खाने चलते।",
    "मेरे मित्र ने मुझे उसके जन्मदिन की पार्टी में बुलाया है, और मैं उसे एक तोहफा दूंगा।",
]

batch = ip.preprocess_batch(
    input_sentences,
    src_lang=src_lang,
    tgt_lang=tgt_lang,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenize the sentences and generate input encodings
inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True,
).to(DEVICE)

# Generate translations using the model
with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        use_cache=True,
        min_length=0,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
    )

# Decode the generated tokens into text
generated_tokens = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

# Postprocess the translations, including entity replacement
translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

for input_sentence, translation in zip(input_sentences, translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")



import argparse
import os
import re

# Import ASR + MT modules
from asr_pipeline import run_asr
from mt_pipeline import translate_batch


# 🔹 Complete LID → IndicTrans mapping
LID2INDICTRANS = {
    # Indo-Aryan
    "hi": "hin_Deva",  # Hindi
    "mr": "mar_Deva",  # Marathi
    "gu": "guj_Gujr",  # Gujarati
    "bn": "ben_Beng",  # Bengali
    "pa": "pan_Guru",  # Punjabi (Gurmukhi)
    "as": "asm_Beng",  # Assamese
    "or": "ory_Orya",  # Odia
    "ne": "npi_Deva",  # Nepali
    "sd": "snd_Arab",  # Sindhi (Arabic script)
    "sa": "san_Deva",  # Sanskrit
    "ur": "urd_Arab",  # Urdu
    # Dravidian
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "kn": "kan_Knda",  # Kannada
    "ml": "mal_Mlym",  # Malayalam
    # Fallback: Whisper sometimes returns Latn code
    "hi_Latn": "hin_Deva",
    "bn_Latn": "ben_Beng",
    "pa_Latn": "pan_Guru",
    "ta_Latn": "tam_Taml",
    "te_Latn": "tel_Telu",
    "kn_Latn": "kan_Knda",
    "ml_Latn": "mal_Mlym",
    # English
    "en": "eng_Latn",
}


def split_into_sentences(text):
    """
    Simple Indic-friendly sentence splitter.
    Splits on । ! ? . followed by whitespace.
    """
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def batch_translate(text, src_lang, tgt_lang, save_path):
    """
    Batch translation with sentence splitting + joining.
    """
    # Split text into sentences
    sentences = split_into_sentences(text)
    print(f"📝 Splitting into {len(sentences)} sentences for batch translation...")

    # Translate batch
    translated_sentences = translate_batch(
        sentences,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        save_path=save_path,
    )

    # Join into one final translation
    return " ".join(translated_sentences)


def asr_mt_pipeline(audio=None, mic=False, youtube=None, duration=10, tgt_lang=None,
                    asr_model="whisper", whisper_size="large", decoding="ctc"):
    """
    Combined ASR + MT pipeline.
    """
    # Step 1: Run ASR
    transcription, lang_code, transcription_file = run_asr(
        audio_path=audio,
        mic=mic,
        youtube=youtube,
        duration=duration,
        asr_model=asr_model,
        whisper_size=whisper_size,
        decoding=decoding,
    )

    print("\n📝 Raw Transcription:", transcription)
    print("🌐 Detected Source Language:", lang_code)

    # Save ASR transcription to asrfiles/
    os.makedirs("asrfiles", exist_ok=True)
    asr_save_path = f"asrfiles/{lang_code}_transcription.txt"
    with open(asr_save_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"💾 ASR saved to: {asr_save_path}")

    # Step 2: Run MT (if target language is given)
    if tgt_lang:
        print(f"\n🔄 Translating {lang_code} → {tgt_lang} ...")

        # ✅ Normalize source language code
        src_lang = LID2INDICTRANS.get(lang_code, lang_code)
        print(f"✅ Normalized Source Lang: {src_lang}")

        # ✅ Ensure mt_transcript folder exists
        os.makedirs("mt_transcript", exist_ok=True)

        # ✅ Save MT outputs in mt_transcript/
        base_name = os.path.basename(transcription_file).replace(
            "_transcription.txt", "_translation.txt"
        )
        save_path = os.path.join("mt_transcript", base_name)

        # 🔹 Perform batch translation
        final_translation = batch_translate(
            transcription,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            save_path=save_path,
        )

        print(f"\n💾 Translation saved to: {save_path}")
        print(f"✅ Final Translation: {final_translation}")
        return transcription, final_translation

    return transcription, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR + MT Pipeline")
    parser.add_argument("--audio", help="Path to input audio file (wav/mp3)")
    parser.add_argument("--mic", action="store_true", help="Use microphone input instead of file")
    parser.add_argument("--youtube", help="YouTube video URL")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration (if using mic)")
    parser.add_argument("--tgt_lang", help="Target language code for translation (e.g., hin_Deva, eng_Latn)")
    parser.add_argument("--asr_model", choices=["whisper", "ai4bharat"], default="whisper")
    parser.add_argument("--whisper_size", default="large")
    parser.add_argument("--decoding", choices=["ctc", "rnnt"], default="ctc")

    args = parser.parse_args()

    asr_mt_pipeline(
        audio=args.audio,
        mic=args.mic,
        youtube=args.youtube,
        duration=args.duration,
        tgt_lang=args.tgt_lang,
        asr_model=args.asr_model,
        whisper_size=args.whisper_size,
        decoding=args.decoding,
    )

#lid.py
import whisper
import torch
import tempfile
import subprocess
import yt_dlp
import sounddevice as sd
from scipy.io.wavfile import write
import spacy

# Load spaCy English large model for proper noun filtering
nlp = spacy.load("en_core_web_lg")

# Supported target languages for LID filtering
TARGET_LANGS = {
    'as': 'Assamese',
    'bn': 'Bengali',
    'brx': 'Bodo',
    'doi': 'Dogri',
    'gu': 'Gujarati',
    'hi': 'Hindi',
    'kn': 'Kannada',
    'kok': 'Konkani',
    'ks': 'Kashmiri',
    'mai': 'Maithili',
    'ml': 'Malayalam',
    'mni': 'Manipuri',
    'mr': 'Marathi',
    'ne': 'Nepali',
    'or': 'Odia',
    'pa': 'Punjabi',
    'sa': 'Sanskrit',
    'sat': 'Santali',
    'sd': 'Sindhi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ur': 'Urdu',
    'en': 'English',
    'es': 'Spanish',
    'zh': 'Mandarin Chinese',
    'ar': 'Arabic'
}

class LanguageIdentifier:
    def __init__(self, model_size="small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_size, device=self.device)

    def filter_proper_nouns(self, text):
        doc = nlp(text)
        tokens = [token.text for token in doc if token.pos_ != "PROPN"]
        return " ".join(tokens)

    def detect(self, audio_path):
        # Step 1: Transcribe audio to get raw text
        result = self.model.transcribe(audio_path, task="transcribe", language=None)
        raw_text = result["text"].strip()

        if not raw_text:
            print("⚠️ No transcription available for language detection.")
            return None, {}

        # Step 2: Try filtering proper nouns
        filtered_text = self.filter_proper_nouns(raw_text)

        # Step 3: Decide whether to fallback
        if not filtered_text.strip():
            print("⚠️ Not enough non-proper noun content; falling back to full text for detection.")
            text_for_detection = raw_text
        else:
            text_for_detection = filtered_text

        # Step 4: Load and prepare audio for language detection
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # Step 5: Detect language using Whisper
        _, probs = self.model.detect_language(mel)

        # Step 6: Filter only supported languages
        filtered_probs = {lang: prob for lang, prob in probs.items() if lang in TARGET_LANGS}

        if not filtered_probs:
            print("⚠️ Could not find any supported language in detected results.")
            return None, {}

        detected_lang = max(filtered_probs, key=filtered_probs.get)
        return detected_lang, filtered_probs

# --- Utility functions for audio sources ---

def extract_audio_ffmpeg(video_path):
    temp_audio = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",      # Mono
        "-vn",           # No video
        temp_audio
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_audio

def download_youtube_audio(url):
    out_path = tempfile.mktemp(suffix=".wav")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_path.replace('.wav', '.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return out_path.replace('.wav', '.wav')

def record_live_audio(duration=5, sample_rate=16000):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("✅ Recording complete.")
    out_path = tempfile.mktemp(suffix=".wav")
    write(out_path, sample_rate, audio)
    return out_path


#asr_mt_pipeline.py
import argparse
import os
import re

# Import ASR + MT modules
from asr_pipeline import run_asr
from mt_pipeline import translate_batch as indic_translate
from mt_google import translate_google


# 🔹 Complete LID → IndicTrans mapping
LID2INDICTRANS = {
    # Indo-Aryan
    "hi": "hin_Deva",  # Hindi
    "mr": "mar_Deva",  # Marathi
    "gu": "guj_Gujr",  # Gujarati
    "bn": "ben_Beng",  # Bengali
    "pa": "pan_Guru",  # Punjabi (Gurmukhi)
    "as": "asm_Beng",  # Assamese
    "or": "ory_Orya",  # Odia
    "ne": "npi_Deva",  # Nepali
    "sd": "snd_Arab",  # Sindhi (Arabic script)
    "sa": "san_Deva",  # Sanskrit
    "ur": "urd_Arab",  # Urdu
    # Dravidian
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "kn": "kan_Knda",  # Kannada
    "ml": "mal_Mlym",  # Malayalam
    # Fallback: Whisper sometimes returns Latn code
    "hi_Latn": "hin_Deva",
    "bn_Latn": "ben_Beng",
    "pa_Latn": "pan_Guru",
    "ta_Latn": "tam_Taml",
    "te_Latn": "tel_Telu",
    "kn_Latn": "kan_Knda",
    "ml_Latn": "mal_Mlym",
    # English
    "en": "eng_Latn",
}


def split_into_sentences(text):
    """
    Simple Indic-friendly sentence splitter.
    Splits on । ! ? . followed by whitespace.
    """
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def batch_translate(text, src_lang, tgt_lang, save_path, backend="indic"):
    """
    Batch translation with sentence splitting + joining.
    backend = "indic" (AI4Bharat) or "google"
    """
    sentences = split_into_sentences(text)
    print(f"📝 Splitting into {len(sentences)} sentences for batch translation...")

    if backend == "google":
        translated_sentences = translate_google(
            sentences,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            save_path=save_path,
        )
    else:
        translated_sentences = indic_translate(
            sentences,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            save_path=save_path,
        )

    return " ".join(translated_sentences)


def asr_mt_pipeline(audio=None, mic=False, youtube=None, duration=10,
                    tgt_lang=None, asr_model="whisper",
                    whisper_size="large", decoding="ctc",
                    mt_backend="indic"):
    """
    Combined ASR + MT pipeline.
    """
    # Step 1: Run ASR
    transcription, lang_code, transcription_file = run_asr(
        audio_path=audio,
        mic=mic,
        youtube=youtube,
        duration=duration,
        asr_model=asr_model,
        whisper_size=whisper_size,
        decoding=decoding,
    )

    print("\n📝 Raw Transcription:", transcription)
    print("🌐 Detected Source Language:", lang_code)

    # Save ASR transcription to asrfiles/
    os.makedirs("asrfiles", exist_ok=True)
    asr_save_path = f"asrfiles/{lang_code}_transcription.txt"
    with open(asr_save_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"💾 ASR saved to: {asr_save_path}")

    # Step 2: Run MT (if target language is given)
    if tgt_lang:
        print(f"\n🔄 Translating {lang_code} → {tgt_lang} ...")

        # ✅ Normalize source language code
        src_lang = LID2INDICTRANS.get(lang_code, lang_code)
        print(f"✅ Normalized Source Lang: {src_lang}")

        # ✅ Ensure mt_transcript folder exists
        os.makedirs("mt_transcript", exist_ok=True)

        # ✅ Save MT outputs in mt_transcript/
        base_name = os.path.basename(transcription_file).replace(
            "_transcription.txt", "_translation.txt"
        )
        save_path = os.path.join("mt_transcript", base_name)

        # 🔹 Perform batch translation
        final_translation = batch_translate(
            transcription,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            save_path=save_path,
            backend=mt_backend,
        )

        print(f"\n💾 Translation saved to: {save_path}")
        print(f"✅ Final Translation: {final_translation}")
        return transcription, final_translation

    return transcription, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR + MT Pipeline")
    parser.add_argument("--audio", help="Path to input audio file (wav/mp3)")
    parser.add_argument("--mic", action="store_true", help="Use microphone input instead of file")
    parser.add_argument("--youtube", help="YouTube video URL")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration (if using mic)")
    parser.add_argument("--tgt_lang", help="Target language code for translation (e.g., hin_Deva, eng_Latn)")
    parser.add_argument("--asr_model", choices=["whisper", "ai4bharat"], default="whisper")
    parser.add_argument("--whisper_size", default="large")
    parser.add_argument("--decoding", choices=["ctc", "rnnt"], default="ctc")
    parser.add_argument("--mt_backend", choices=["indic", "google"], default="indic",
                        help="Choose translation backend: indic (AI4Bharat) or google (Google Translate)")

    args = parser.parse_args()

    asr_mt_pipeline(
        audio=args.audio,
        mic=args.mic,
        youtube=args.youtube,
        duration=args.duration,
        tgt_lang=args.tgt_lang,
        asr_model=args.asr_model,
        whisper_size=args.whisper_size,
        decoding=args.decoding,
        mt_backend=args.mt_backend,
    )

#asr_pipeline.py
import os
import time
import whisper
import spacy
import torchaudio
import torch
from transformers import AutoModel
from jiwer import wer
from lid import (
    LanguageIdentifier,
    extract_audio_ffmpeg,
    download_youtube_audio,
    record_live_audio,
    TARGET_LANGS
)

# Load spaCy English model (used for proper-noun highlighting)
nlp = spacy.load("en_core_web_lg")

# Map LID ISO codes to IndicTrans2 language tags
LID2INDICTRANS = {
    "hi": "hin_Deva", "mr": "mar_Deva", "gu": "guj_Gujr", "bn": "ben_Beng",
    "pa": "pan_Guru", "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda",
    "ml": "mal_Mlym", "ne": "npi_Deva", "as": "asm_Beng", "or": "ory_Orya",
    "sd": "snd_Arab", "sa": "san_Deva", "ur": "urd_Arab", "en": "eng_Latn",
}

# Lazy load AI4Bharat ASR
ai4bharat_model = None
def load_ai4bharat_model():
    global ai4bharat_model
    if ai4bharat_model is None:
        ai4bharat_model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True
        )
    return ai4bharat_model

def get_proper_nouns(text):
    doc = nlp(text)
    return set(token.text for token in doc if token.pos_ == "PROPN")

def transcribe_whisper(audio_path, language_code=None, model_size="large"):
    print("\n🔠 Running Whisper ASR...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language=language_code)
    return result['text']

def transcribe_ai4bharat(audio_path, language_code, decoding_strategy="ctc"):
    print(f"\n🔠 Running AI4Bharat ASR with {decoding_strategy.upper()} decoding...")
    wav, sr = torchaudio.load(audio_path)
    wav = torch.mean(wav, dim=0, keepdim=True)  # Convert to mono

    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)

    model = load_ai4bharat_model()
    transcription = model(wav, language_code, decoding_strategy)
    return transcription

def process_transcription(audio_path, text):
    proper_nouns = get_proper_nouns(text)
    final_output = " ".join(text.split())

    output_txt_path = os.path.splitext(audio_path)[0] + "_transcription.txt"
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"\n📝 Transcription: {final_output}")
    print(f"🔍 Proper Nouns: {', '.join(proper_nouns) if proper_nouns else 'None'}")
    print(f"💾 Saved to: {output_txt_path}")
    return final_output, output_txt_path

# ✅ This is the function we’ll import in asr_mt_pipeline.py
def run_asr(audio_path, asr_model="whisper", whisper_size="large", decoding="ctc", duration=5, mic=False, youtube=None):
    # Handle input
    if youtube:
        audio_path = download_youtube_audio(youtube)
    elif mic:
        print("\n🎤 Speak now...")
        time.sleep(1)
        audio_path = record_live_audio(duration=duration)
    elif audio_path.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
        audio_path = extract_audio_ffmpeg(audio_path)

    # Language Identification
    lid = LanguageIdentifier()
    lang, probs = lid.detect(audio_path)
    src_lang_tag = LID2INDICTRANS.get(lang, "eng_Latn")

    print(f"\n✅ Detected Language: {TARGET_LANGS.get(lang, lang)} ({lang})")

    # Run ASR
    if asr_model == "whisper":
        text = transcribe_whisper(audio_path, language_code=lang, model_size=whisper_size)
    else:
        text = transcribe_ai4bharat(audio_path, language_code=lang, decoding_strategy=decoding)

    # Process + save
    final_text, output_file = process_transcription(audio_path, text)

    return final_text, src_lang_tag, output_file

# ----------------
# Standalone mode
# ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone ASR pipeline")
    parser.add_argument("--file", help="Audio file path")
    parser.add_argument("--youtube", help="YouTube video URL")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--asr_model", choices=["whisper", "ai4bharat"], default="whisper")
    parser.add_argument("--whisper_size", default="large")
    parser.add_argument("--decoding", choices=["ctc", "rnnt"], default="ctc")
    args = parser.parse_args()

    run_asr(
        audio_path=args.file,
        youtube=args.youtube,
        mic=args.mic,
        duration=args.duration,
        asr_model=args.asr_model,
        whisper_size=args.whisper_size,
        decoding=args.decoding,
    )
