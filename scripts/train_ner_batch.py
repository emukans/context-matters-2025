import csv
import sys
from random import shuffle, seed

import torch
import wandb
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, EarlyStoppingCallback

import os

import numpy as np
from time import time

from datasets import Dataset, load_from_disk, load_metric
import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification



print_example = False


def main(dataset_name, iteration):
    # checkpoint = "Clinical-AI-Apollo/Medical-NER"
    checkpoint = "google-bert/bert-base-uncased"
    # checkpoint = "google-bert/bert-large-uncased"


    MAX_LEN = 70
    # batch_size = 2
    batch_size = 64
    epoch_count = 50
    learning_rate = 1e-5
    downsample = 1
    save_total_limit = 1
    random_size = 0
    decay = 0.001

    classified_only_train = False
    classified_only_dev = False

    # extra_tag_replacement = 3
    extra_tag_replacement = -100
    # extra_tag_replacement = 0

    # os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_NAME"] = f"{checkpoint}/{dataset_name}/classified_only-{classified_only_train}-lr-{learning_rate}-decay-{decay}-count-{epoch_count}-max_len-{MAX_LEN}-downsample-{downsample}-ade_tag-{extra_tag_replacement}-01-{iteration}-{os.environ['CUDA_VISIBLE_DEVICES']}"

    dataset = load_from_disk(f'data/{dataset_name}')


    if extra_tag_replacement in (-100, 0):
        id2label = {0: "O", 1: "B-ADE", 2: "I-ADE"}
        label2id = {"O": 0, "B-ADE": 1, "I-ADE": 2}
    else:
        id2label = {0: "O", 1: "B-ADE", 2: "I-ADE", 3: "EXTRA"}
        label2id = {"O": 0, "B-ADE": 1, "I-ADE": 2, "EXTRA": 3}


    label_list = dataset["train"].features[f"ner_tags"].feature.names
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")


    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l not in (-100, 3)]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l not in (-100, 3)]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        # return results
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


    train_dataset = dataset['train']
    dev_dataset = dataset['dev']

    log_params = {
        'train_size': len(dataset['train']),
        'dev_size': len(dataset['dev']),
        'model_size': model.num_parameters(),
        'max_len': MAX_LEN,
    }

    if classified_only_train:
        classified_ids = []
        with open('data/train/classified.csv', 'r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                classified_ids.append(line['tweet_id'])

        train_dataset = train_dataset.filter(lambda item: item['idx'] in classified_ids)

    if classified_only_dev:
        classified_ids = []
        with open('data/dev/classified.csv', 'r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                classified_ids.append(line['tweet_id'])

        dev_dataset = dev_dataset.filter(lambda item: item['idx'] in classified_ids)

    if downsample != 1:
        has_ade_ids = []
        for i, item in enumerate(train_dataset):
            if 1 in item['ner_tags']:
                has_ade_ids.append(item['idx'])

        no_ade_ids = list(set(train_dataset['idx']) - set(has_ade_ids))
        no_ade_ids = no_ade_ids[:round(len(no_ade_ids) * downsample)]
        ids_to_keep = no_ade_ids + has_ade_ids
        train_dataset = train_dataset.filter(lambda item: item['idx'] in ids_to_keep)

        log_params['train_size'] = len(train_dataset)
        log_params['positive_sample_proportion_train'] = len(has_ade_ids) / len(train_dataset)
        log_params['downsample'] = downsample

        has_ade_ids = []
        for i, item in enumerate(dev_dataset):
            if 1 in item['ner_tags']:
                has_ade_ids.append(item['idx'])
        log_params['positive_sample_proportion_dev'] = len(has_ade_ids) / len(dev_dataset)


    wandb.init()
    wandb.log(log_params)


    def tokenize_and_align_labels(examples):
        """
        After tokenization, a word is split into multiple tokens. This function assigns the same POS tag for every token of the word.
        """
        global print_example

        tokenized_inputs = tokenizer(examples["tokens"], max_length=MAX_LEN, truncation=True, padding='max_length', is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    # 3 is EXTRA token
                    if label[word_idx] == 3:
                        label_ids.append(extra_tag_replacement)
                    else:
                        label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if label[word_idx] == 3:
                        label_ids.append(extra_tag_replacement)
                    else:
                        label_ids.append(label[word_idx])
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels

        if print_example:
            print_example = False
            print('Before: ', examples['ner_tags'][:5])
            print('Before: ', examples['tokens'][:5])
            print('After: ', tokenized_inputs['labels'][:5])
            print('After: ', tokenized_inputs['input_ids'][:5])
            exit()

        return tokenized_inputs


    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True).shuffle(seed=42)
    dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True).shuffle(seed=42)

    training_args = TrainingArguments(
        output_dir="model/" + os.environ["WANDB_NAME"],
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch_count,
        weight_decay=decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    result_metrics = trainer.evaluate()


    return result_metrics


if __name__ == "__main__":
    it = int(sys.argv[1])
    iteration = int(sys.argv[2])

    rank_id = os.environ['CUDA_VISIBLE_DEVICES']

    dataset_list = [
        ['2024_gpt4o_context', '2024_gpt4o_span', '2024_gpt4o_span_with_context', '2024_gpt4o_mini_context', '2024_gpt4o_mini_span', '2024_gpt4o_mini_span_with_context'],
        ['2024_tweet_only', '2024_gpt35_context', '2024_gpt35_span', '2024_gpt35_span_with_context', '2024_gpt4o_context_with_drug_dataset', '2024_gpt4o_context_with_symptom_dataset'],
        ['2024_gpt4o_context_with_drug_with_symptom_dataset', '2024_gpt4o_context_tweet_only_with_drug_dataset', '2024_gpt4o_context_tweet_only_with_symptom_dataset', '2024_gpt4o_context_tweet_only_with_drug_with_symptom_dataset', '2024_gpt4o_span_with_drug_dataset'],
        ['2024_gpt4o_span_with_symptom_dataset', '2024_gpt4o_span_with_drug_with_symptom_dataset', '2024_gpt4o_span_tweet_only_with_drug_dataset', '2024_gpt4o_span_tweet_only_with_symptom_dataset', '2024_gpt4o_span_tweet_only_with_drug_with_symptom_dataset'],
    ]

    dataset_data_map = {
        '2024_gpt4o_context': {'type': 'context', 'gpt_model': '4o', 'drugs_context': False, 'symptom_context': False},
        '2024_gpt4o_span': {'type': 'span', 'gpt_model': '4o', 'drugs_context': False, 'symptom_context': False},
        '2024_gpt4o_span_with_context': {'type': 'span/context', 'gpt_model': '4o', 'drugs_context': False, 'symptom_context': False},
        '2024_gpt4o_mini_context': {'type': 'context', 'gpt_model': '4o-mini', 'drugs_context': False, 'symptom_context': False},
        '2024_gpt4o_mini_span': {'type': 'span', 'gpt_model': '4o-mini', 'drugs_context': False, 'symptom_context': False},
        '2024_gpt4o_mini_span_with_context': {'type': 'span/context', 'gpt_model': '4o-mini', 'drugs_context': False, 'symptom_context': False},
        '2024_gpt35_context': {'type': 'context', 'gpt_model': '3.5', 'drugs_context': False, 'symptom_context': False},
        '2024_gpt35_span': {'type': 'span', 'gpt_model': '3.5', 'drugs_context': False, 'symptom_context': False},
        '2024_gpt35_span_with_context': {'type': 'span/context', 'gpt_model': '3.5', 'drugs_context': False, 'symptom_context': False},
        '2024_tweet_only': {'type': 'None', 'gpt_model': 'No', 'drugs_context': False, 'symptom_context': False},
        '2024_gpt4o_context_with_drug_dataset': {'type': 'context', 'gpt_model': '4o', 'drugs_context': True, 'symptom_context': False},
        '2024_gpt4o_context_with_symptom_dataset': {'type': 'context', 'gpt_model': '4o', 'drugs_context': False, 'symptom_context': True},
        '2024_gpt4o_context_with_drug_with_symptom_dataset': {'type': 'context', 'gpt_model': '4o', 'drugs_context': True, 'symptom_context': True},
        '2024_gpt4o_context_tweet_only_with_drug_dataset': {'type': 'context', 'gpt_model': 'No', 'drugs_context': True, 'symptom_context': False},
        '2024_gpt4o_context_tweet_only_with_symptom_dataset': {'type': 'context', 'gpt_model': 'No', 'drugs_context': False, 'symptom_context': True},
        '2024_gpt4o_context_tweet_only_with_drug_with_symptom_dataset': {'type': 'context', 'gpt_model': 'No', 'drugs_context': True, 'symptom_context': True},
        '2024_gpt4o_span_with_drug_dataset': {'type': 'span', 'gpt_model': '4o', 'drugs_context': True, 'symptom_context': False},
        '2024_gpt4o_span_with_symptom_dataset': {'type': 'span', 'gpt_model': '4o', 'drugs_context': False, 'symptom_context': True},
        '2024_gpt4o_span_with_drug_with_symptom_dataset': {'type': 'span', 'gpt_model': '4o', 'drugs_context': True, 'symptom_context': True},
        '2024_gpt4o_span_tweet_only_with_drug_dataset': {'type': 'span', 'gpt_model': 'No', 'drugs_context': True, 'symptom_context': False},
        '2024_gpt4o_span_tweet_only_with_symptom_dataset': {'type': 'span', 'gpt_model': 'No', 'drugs_context': False, 'symptom_context': True},
        '2024_gpt4o_span_tweet_only_with_drug_with_symptom_dataset': {'type': 'span', 'gpt_model': 'No', 'drugs_context': True, 'symptom_context': True},
    }

    fieldnames = ['dataset', 'iteration', 'type', 'gpt_model', 'drugs_context', 'symptom_context', 'eval_loss', 'eval_precision', 'eval_recall', 'eval_f1', 'eval_accuracy', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch']

    if it > len(dataset_list[int(rank_id)]) - 1:
        exit()
    dataset_name = dataset_list[int(rank_id)][it]

    result_path = os.path.join('result_csv_ner_bert_base', f'{dataset_name}.csv')

    if not os.path.exists(result_path):
        with open(result_path, 'w') as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()

    print('Run: ', dataset_name)
    seed(int(time()))
    np.random.seed(int(time()))
    result_metrics = main(dataset_name, iteration)

    with open(result_path, 'a') as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writerow({'dataset': dataset_name, 'iteration': iteration, **dataset_data_map[dataset_name], **result_metrics})
