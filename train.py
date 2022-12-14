import argparse
import logging
import os
import sys

import datasets
from datasets import load_dataset, load_metric
import evaluate
import numpy as np
import nltk
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', type=bool, default=False)
    parser.add_argument('--do_eval', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--model_name', type=str, default="facebook/bart-base", help="a hf model name")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_source_length', type=int, default=1024)
    parser.add_argument('--max_target_length', type=int, default=512)
    parser.add_argument('--num_beams', type=int, default=None)
    parser.add_argument('--label_smoothing_factor', type=float, default=0.0)
    parser.add_argument('--no_repeat_ngram', type=int, default=3)
    parser.add_argument('--logging_strategy', type=str, default='steps')
    parser.add_argument('--logging_steps', type=int, default=500)
    parser.add_argument('--save_strategy', type=str, default='epoch')
    parser.add_argument('--output_dir', type=str, default='./results')
    # parser.add_argument('--save_steps', type=int, default=10000)
    return parser.parse_args()


def create_dataset(data_path):
    data_files = {"train": "train.tsv", "validation": "val.tsv", "test": "test.tsv"}
    dataset = load_dataset(data_path, data_files=data_files)

    return dataset


def main():
    # Set logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    dataset = create_dataset(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["data"], max_length=args.max_source_length, padding=False, truncation=True)
        labels = tokenizer(examples["text"], max_length=args.max_target_length, padding=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1', 'rouge2', 'rougeL'])
        rouge = {k: round(v * 100, 4) for k, v in rouge.items()}
        bert_s = bertscore_metric.compute(predictions=decoded_preds, references=decoded_labels, model_type="bert-base-uncased")
        bert_s["precision"] = np.mean(bert_s["precision"])
        bert_s["recall"] = np.mean(bert_s["recall"])
        bert_s["f1"] = round(np.mean(bert_s["f1"]) * 100, 4)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

        result = rouge
        result["bertscore_f1"] = bert_s["f1"]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    if args.do_train:
        train_dataset = dataset["train"]
        train_dataset = train_dataset.map(preprocess_function, batched=True)
    if args.do_eval:
        eval_dataset = dataset["validation"]
        eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    if args.do_predict:
        predict_dataset = dataset["test"]
        predict_dataset = predict_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        predict_with_generate=True,
        label_smoothing_factor=args.label_smoothing_factor,
        generation_num_beams=args.num_beams,
        # save_steps=args.save_steps
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None
    )

    # Training
    if args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else args.max_target_length
    )
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_length, num_beams=training_args.generation_num_beams, metric_key_prefix="eval", no_repeat_ngram_size=args.no_repeat_ngram
        )
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=training_args.generation_num_beams, no_repeat_ngram_size=args.no_repeat_ngram
        )
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(predict_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(training_args.output_dir, "predictions.txt")
            with open(output_prediction_file, "w") as w:
                w.write("\n".join(predictions))


if __name__ == '__main__':
    args = args_parse()
    main()

