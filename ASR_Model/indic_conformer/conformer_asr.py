# ASR_Model/indic_conformer/conformer_asr.py
import torch
import torchaudio
from transformers import AutoModel
import onnxruntime as ort


class IndicConformerASR:
    def __init__(self, model_name="ai4bharat/indic-conformer-600m-multilingual"):
        print("🔊 Loading IndicConformer ASR model...")

        # Detect available ONNX Runtime providers
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            provider = "CUDAExecutionProvider"
            print("✅ Using GPU (CUDAExecutionProvider) for IndicConformer")
        else:
            provider = "CPUExecutionProvider"
            print("⚠️ CUDAExecutionProvider not available. Falling back to CPU.")

        # Load model with chosen provider
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            provider=provider
        )
        self.sample_rate = 16000

    def load_audio(self, audio_path):
        wav, sr = torchaudio.load(audio_path)
        wav = torch.mean(wav, dim=0, keepdim=True)  # Convert to mono
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(wav)
        return wav

    def transcribe(self, audio_path, language_code="hi", decoder_type="ctc"):
        wav = self.load_audio(audio_path)
        result = self.model(wav, language_code, decoder_type)
        return result
