import argparse

from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel
from typing import List


DATA_TOKEN_MAP = {
    '[HomeTeam]': 'Watford',
    '[AwayTeam]': 'Chelsea',
    '[DateTime]': 'November',
    '[FTHG]': '6',
    '[FTAG]': '5',
    '[HTHG]': '4',
    '[HTAG]': '3'
}

INV_DATA_TOKEN_MAP = {
    'Watford': '[HomeTeam]',
    'Chelsea': '[AwayTeam]',
    'November': '[Datetime]',
    '6': '[FTHG]',
    '5': '[FTAG]',
    '4': '[HTHG]',
    '3': '[HTAG]'
}


def back_translate(sample, src, trg, b_size=32):
    steps = int(len(sample) / b_size + 1) if len(sample) % b_size != 0 else int(len(sample) / b_size)
    for i in tqdm(range(steps)):
        input_ids = tokenizer(sample[b_size*i:b_size*(i+1)], padding=True, return_tensors="pt")
        gen = model.generate(**input_ids)
        translated = tokenizer.batch_decode(gen, skip_special_tokens=True)
        back_input_ids = back_tokenizer(translated, padding=True, return_tensors="pt")
        back_gen = back_model.generate(do_sample=False, num_beams=5, num_return_sequences=5, no_repeat_ngram_size=4, num_beam_groups=1, **back_input_ids)

    output = back_tokenizer.batch_decode(back_gen, skip_special_tokens=True)
    
    return output


def replace_data_tokens(sample):
    sample_r = sample
    for key in DATA_TOKEN_MAP:
        sample_r = sample_r.replace(key, DATA_TOKEN_MAP[key])
    return sample_r


def replace_data_tokens_inverse(sample):
    sample_r = sample
    for key in INV_DATA_TOKEN_MAP:
        sample_r = sample_r.replace(key, INV_DATA_TOKEN_MAP[key])
    return sample_r


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    samples = list()

    # sample_text = "Watford achieved a great victory at their home on November 21st. They beat Chelsea 3-2. Watford made a good start by leading the opponent 1-0 at 1st half. Chelsea was not competitive enough to win the losing game after conceding another goal."
    with open(args.filename, "r") as f:
        for t in f.readlines():
            sample = replace_data_tokens(t.strip())
            samples.append(sample)
    # print(samples)

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
        outputs = outputs + output_transl
        # for i in range(len(output_transl)):
        #     output_transl2 = back_translate(output_transl[i], src, trg)
        #     outputs = outputs + output_transl2

    output_set = set(outputs)
    with open(f"{args.filename.split('.')[0]}_augmented.txt", "w") as fw:
        for output in output_set:
            fw.write(replace_data_tokens_inverse(output) + '\n')

    print(len(output_set))

