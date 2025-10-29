import argparse
import os
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Try importing IndicProcessor (optional dependency)
try:
    from IndicTransToolkit.processor import IndicProcessor
    HAS_INDIC_PROCESSOR = True
except ImportError:
    HAS_INDIC_PROCESSOR = False

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model selection mapping
MODEL_MAP = {
    ("eng_Latn", "indic"): "ai4bharat/indictrans2-en-indic-1B",
    ("indic", "eng_Latn"): "ai4bharat/indictrans2-indic-en-1B",
    ("indic", "indic"): "ai4bharat/indictrans2-indic-indic-1B",
}

def split_into_sentences(text: str):
    """
    Very simple sentence splitter using regex.
    Helps MT handle long paragraphs better.
    """
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def detect_model(src_lang: str, tgt_lang: str) -> str:
    """Pick correct IndicTrans2 model based on src/tgt language."""
    if src_lang == "eng_Latn" and tgt_lang != "eng_Latn":
        return MODEL_MAP[("eng_Latn", "indic")]
    elif src_lang != "eng_Latn" and tgt_lang == "eng_Latn":
        return MODEL_MAP[("indic", "eng_Latn")]
    else:
        return MODEL_MAP[("indic", "indic")]

def translate_batch(sentences, src_lang, tgt_lang, save_path=None):
    """
    Translate a batch of sentences with IndicTrans2.
    Uses IndicProcessor if available, else falls back to manual tagging.
    """
    model_name = detect_model(src_lang, tgt_lang)
    print(f"\n🔄 Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

    if HAS_INDIC_PROCESSOR:
        print("✅ Using IndicProcessor for preprocessing/postprocessing")
        ip = IndicProcessor(inference=True)

        # Preprocess sentences
        batch = ip.preprocess_batch(
            sentences,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=1024,
                num_beams=5,
                num_return_sequences=1,
            )

        decoded = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Postprocess with IndicProcessor
        translations = ip.postprocess_batch(decoded, lang=tgt_lang)

    else:
        print("⚠️ IndicProcessor not found. Falling back to source/target tagging.")
        tagged_sentences = [f"<{src_lang}><{tgt_lang}> {s}" for s in sentences]

        inputs = tokenizer(
            tagged_sentences,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1024,   # increased for longer texts
                num_beams=5,
                use_cache=True,
            )

        translations = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Optionally save translations
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for line in translations:
                f.write(line + "\n")
        print(f"\n💾 Translation saved to: {save_path}")

    return translations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndicTrans2 Translation Pipeline")
    parser.add_argument("--src_lang", required=True, help="Source language code (e.g., eng_Latn, hin_Deva)")
    parser.add_argument("--tgt_lang", required=True, help="Target language code (e.g., eng_Latn, tam_Taml)")
    parser.add_argument("--text", nargs="+", help="Text to translate")
    parser.add_argument("--file", help="Path to a .txt file with text to translate")
    parser.add_argument("--save", help="Optional path to save translations to .txt")

    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
        sentences = split_into_sentences(raw_text)
    elif args.text:
        raw_text = " ".join(args.text)
        sentences = split_into_sentences(raw_text)
    else:
        raise ValueError("You must provide either --text or --file input")

    print(f"\n📝 Splitting into {len(sentences)} sentences for batch translation...")

    translated_texts = translate_batch(
        sentences,
        args.src_lang,
        args.tgt_lang,
        save_path=args.save,
    )

    for t in translated_texts:
        print(f"\n✅ Translation: {t}")
