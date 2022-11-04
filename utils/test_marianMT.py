from transformers import MarianTokenizer, MarianMTModel
from typing import List

src = "en"  # source language
trg = "fr"  # target language

sample_text = ["This is a sample text"]
model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

batch = tokenizer(sample_text, return_tensors="pt")
gen = model.generate(**batch)

sample_text_translated = tokenizer.batch_decode(gen, skip_special_tokens=True)
print(sample_text_translated)
back_model_name = f"Helsinki-NLP/opus-mt-{trg}-{src}"
back_model = MarianMTModel.from_pretrained(back_model_name)
back_tokenizer = MarianTokenizer.from_pretrained(back_model_name)

batch = back_tokenizer(sample_text_translated, return_tensors="pt")
gen = back_model.generate(**batch)

output = back_tokenizer.batch_decode(gen, skip_special_tokens=True)
print(output)
