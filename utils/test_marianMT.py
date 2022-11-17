from transformers import MarianTokenizer, MarianMTModel
from typing import List

src = "en"  # source language
trg = "fr"  # target language

sample_text = [
"<HomeTeam> achieved a great victory at their home on <DateTime>. They beat <AwayTeam> <FTHG>-<FTAG>. <HomeTeam> made a good start by leading the opponent <HTHG>-<HTAG> at 1st half. <AwayTeam> was not competitive enough to win the losing game after conceding another goal."
]

print(sample_text)

model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

tokenizer.add_tokens(["<HOME>, <DATE>, <AWAY>, <FT>, <HT>"], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

# This doesn't work
# special_tokens_dict = {'hometeam_token': '<HOME>', 'awayteam_token': '<AWAY>', 'date_token': '<DATE>', 'halftime_token': '<HT>', 'fulltime_token': '<FT>'}
# tokenizer.add_special_tokens(special_tokens_dict)
# model.resize_token_embeddings(len(tokenizer))

batch = tokenizer(sample_text, return_tensors="pt")
gen = model.generate(**batch)

sample_text_translated = tokenizer.batch_decode(gen, skip_special_tokens=True)
print(sample_text_translated)
back_model_name = f"Helsinki-NLP/opus-mt-{trg}-{src}"
back_model = MarianMTModel.from_pretrained(back_model_name)
back_tokenizer = MarianTokenizer.from_pretrained(back_model_name)

back_tokenizer.add_tokens(["<HOME>, <DATE>, <AWAY>, <FT>, <HT>"], special_tokens=True)
model.resize_token_embeddings(len(back_tokenizer))

batch = back_tokenizer(sample_text_translated, return_tensors="pt")
gen = back_model.generate(do_sample=False, num_beams=10, num_return_sequences=10, no_repeat_ngram_size=3, num_beam_groups=2, **batch)

output = back_tokenizer.batch_decode(gen, skip_special_tokens=True)
for i in range(len(output)):
    print(output[i]+'\n')
