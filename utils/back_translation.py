from transformers import MarianTokenizer, MarianMTModel
from typing import List


def back_translate(sample, src, trg):
    input_ids = tokenizer(sample, return_tensors="pt")
    gen = model.generate(**input_ids)
    translated = tokenizer.batch_decode(gen, skip_special_tokens=True)
    back_input_ids = back_tokenizer(translated, return_tensors="pt")
    back_gen = back_model.generate(do_sample=False, num_beams=10, num_return_sequences=10, no_repeat_ngram_size=4, num_beam_groups=1, **back_input_ids)

    output = back_tokenizer.batch_decode(back_gen, skip_special_tokens=True)
    
    return output


if __name__ == '__main__':
    # sample_text = "Watford achieved a great victory at their home on November 21st. They beat Chelsea 3-2. Watford made a good start by leading the opponent 1-0 at 1st half. Chelsea was not competitive enough to win the losing game after conceding another goal."
    with open("data/templates.txt", "r") as f:
        samples = [t.strip() for t in f.readlines()]
    print(samples)
    outputs = samples

    src = "en"  # source language
    trgs = ["fr", "es", "de"]  # target language

    for trg in trgs:
        print(f"Back-translate using {trg}")
        model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        back_model_name = f"Helsinki-NLP/opus-mt-{trg}-{src}"
        back_model = MarianMTModel.from_pretrained(back_model_name)
        back_tokenizer = MarianTokenizer.from_pretrained(back_model_name)

        output_transl = back_translate(samples, src, trg)

        for i in range(len(output_transl)):
            output_transl2 = back_translate(output_transl[i], src, trg)
            outputs = outputs + output_transl2

    output_set = set(outputs)
    for output in output_set:
        print(output)

    print(len(output_set))

