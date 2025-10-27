# 🎧 Vasha-AI — Real-Time AI Speech Translation System

> 🧠 **Vasha-AI** is an intelligent multilingual pipeline that performs **real-time speech-to-speech translation** across 200+ global and Indic languages.
> It integrates **Automatic Speech Recognition (ASR)**, **Language Identification (LID)**, and **Machine Translation (MT)** with features like **NER-preservation**, **code-mixed handling**, **transliteration**, and **spoof detection**.

---

## 🌍 Key Features

✅ **Real-Time Language Identification (LID)**
✅ **Automatic Speech Recognition (ASR)** with:

* OpenAI **Whisper**
* AI4Bharat **IndicConformer**
* **Faster-Whisper** (batched inference)

✅ **Machine Translation (MT)** with:

* Meta **NLLB (No Language Left Behind)**
* AI4Bharat **IndicTrans2**
* **Google Translate** API wrapper

✅ **Smart Preprocessing**:

* Named Entity Preservation (NER)
* Script Transliteration
* Code-Mixed Text Handling (e.g., Hinglish)

✅ **Advanced Debugging**:

* Back-Translation Consistency Check
* NER Preservation Mode
* Transliteration-only Mode

✅ **Security**:

* Spoof Detection for fake audio
* Dialect tagging (Hindi, Tamil, Bengali, etc.)

✅ **UX Improvements**:

* Real-time progress bars (`tqdm`)
* Session-wise results saved locally

---

## 🧩 Project Directory Structure

```
Vasha-AI/
│
├── ASR_Model/
│   ├── faster-whisper/              # Faster-Whisper backend
│   ├── indic_conformer/             # AI4Bharat IndicConformer ASR
│   │   ├── conformer_asr.py
│   │   └── __init__.py
│   ├── whisper/                     # Whisper wrapper
│   └── __init__.py
│
├── LID_Model/
│   ├── lid.py                       # Language ID + dialect detection
│   ├── spoof_detection.py           # Spoof detection
│   ├── requirements.txt
│   └── __init__.py
│
├── MT_Model/
│   ├── IndicTrans2/                 # AI4Bharat IndicTrans2 model
│   ├── nllb-3.3B/                   # Meta NLLB 3.3B model weights
│   ├── Open-NLLB/                   # Optional fine-tuned NLLB variant
│   ├── mt_model.py                  # Unified translation model loader
│   ├── mt_helper.py                 # Menu + progress bar integration
│   ├── mt_google.py                 # Google Translate API
│   ├── mt_preprocessor.py           # NER, transliteration, code-mix logic
│   ├── mt_debug.py                  # Back-translation utilities
│   └── __init__.py
│
├── Audio_TestWAV/                   # Sample WAVs
├── sessions/                        # Saved transcriptions & translations
├── youtube_cache/                   # Cached YouTube WAVs
│
├── transcribe_pipeline.py           # Main entry-point script
├── gpusage.py                       # GPU usage tracker
├── recorded.wav                     # Sample audio
├── requirements.txt                 # Global dependencies
└── README.md                        # You're reading this file
```

---

## 🧠 System Architecture

```
🎤 Audio Input
   │
   ├──► Language Identification (LID_Model)
   │      ├── Language & Dialect Detection
   │      └── Spoof Detection (Fake vs Real)
   │
   ├──► Automatic Speech Recognition (ASR_Model)
   │      ├── Whisper / Faster-Whisper / IndicConformer
   │      └── Converts Speech → Text
   │
   ├──► Machine Translation (MT_Model)
   │      ├── NLLB / IndicTrans2 / Google
   │      ├── Transliteration / Code-mixed / NER-Preserve
   │      └── Progress bar + batching (tqdm)
   │
   └──► Output
          ├── Translated Text
          ├── Saved Transcription Files
          └── Optional Back-Translation Debug
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/Vasha-AI.git
cd Vasha-AI
```

### 2️⃣ Create a new environment

```bash
conda create -n lid-env python=3.10 -y
conda activate lid-env
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ (Optional) Install extra packages for IndicTrans2 & NLLB

```bash
pip install torch torchvision torchaudio
pip install transformers sentencepiece sacremoses accelerate
```

---

## 🎤 Running the Pipeline

### ▶️ Real-Time Microphone Translation

```bash
python transcribe_pipeline.py --mic --duration 10
```

### ▶️ Translate YouTube Videos

```bash
python transcribe_pipeline.py --youtube https://youtu.be/<video_id>
```

### ▶️ Process a Local File

```bash
python transcribe_pipeline.py --file sample_video.mp4
```

---

## 🧪 Debugging Options

| Option                   | Description                              |
| ------------------------ | ---------------------------------------- |
| **1**                    | Normal Translation                       |
| **2**                    | Batch Translation (with progress bar)    |
| **3**                    | Back-Translation Debug                   |
| **4**                    | NER-Preservation Mode                    |
| **Transliteration Mode** | Converts script without changing meaning |
| **Code-Mixed Mode**      | Handles bilingual runs like Hinglish     |

---

## 🧾 Example Output

```
🕣 Transcribed Text:
 नमस्ते मेरा नाम दीप है...

💬 Translation (Hindi → English):
 Hello, my name is Deep...

💾 Saved to:
 sessions/session_20251027_142512/translation_hi_to_eng_Latn.txt
```

---

## 💡 Advanced Features

| Feature              | Description                                        |
| -------------------- | -------------------------------------------------- |
| **Progress Bar**     | `tqdm` integrated into translation chunks          |
| **NER Preservation** | Keeps named entities (people, places) untranslated |
| **Back-Translation** | Validates translation consistency                  |
| **Spoof Detection**  | Flags fake/AI-generated voices                     |
| **GPU Monitoring**   | Optional `gpusage.py` shows CUDA stats             |

---

## 🦉 Example Session Folder

```
sessions/session_20251027_142512/
├── output_hi_whisper.txt
├── translation_hi_to_eng_Latn.txt
└── debug_log.txt
```

---

## 🧑‍💻 Development Notes

* Each module is fully independent (LID, ASR, MT).
* Uses **Whisper** or **IndicConformer** dynamically based on detected language.
* **Meta NLLB** is default global MT model; falls back to **IndicTrans2** for Indic pairs.
* Integrated **tqdm progress bar** for smoother user experience during long translations.

---

## 🔋 Requirements

```
torch
torchaudio
transformers
tqdm
sentencepiece
sacremoses
faster-whisper
indic-nlp-library
langid
spacy
flask
pydub
openai-whisper
```

---

## 🧱 Future Enhancements

* Real-time subtitle overlay for YouTube/Twitch streams
* Flask/FastAPI dashboard
* Speaker diarization
* Multi-GPU batch processing
* Offline IndicTrans2 quantization

---

## 👨‍💻 Author

**Deep Habiswashi**
**Soumyadeep Dutta**


---

## 🪄 License

```
MIT License © 2025 Deep Habiswashi
```


