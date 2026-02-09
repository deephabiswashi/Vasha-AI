from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "My name is Deep. I study Btevh at KIIT University"
inputs = tokenizer(text, return_tensors="pt")

# FIX: use convert_tokens_to_ids
tgt_lang = "hin_Deva"
forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=forced_bos_token_id, max_length=50
)
translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

print("Original:", text)
print("Translated:", translation)
