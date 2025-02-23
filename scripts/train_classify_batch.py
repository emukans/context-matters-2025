import csv
import sys
from random import shuffle, seed

from collections import Counter

import torch
import wandb
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, EarlyStoppingCallback

import os

import numpy as np
from time import time

from datasets import Dataset, load_from_disk, load_metric
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding



print_example = False


def main(dataset_name, iteration):
    # checkpoint = "cardiffnlp/twitter-roberta-large-topic-sentiment-latest"
    # checkpoint = "google-bert/bert-base-uncased"
    checkpoint = "google-bert/bert-large-uncased"

    MAX_LEN = 70
    # batch_size = 2
    batch_size = 64
    epoch_count = 30
    learning_rate = 5e-7
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
    os.environ["WANDB_NAME"] = f"{checkpoint}/{dataset_name}/classified_only-{classified_only_train}-lr-{learning_rate}-decay-{decay}-count-{epoch_count}-max_len-{MAX_LEN}-downsample-{downsample}-ade_tag-{extra_tag_replacement}-01-{iteration}"

    dataset = load_from_disk(f'data/{dataset_name}')

    id2label = {0: "no_symptom", 1: "has_symptom"}
    label2id = {"no_symptom": 0, "has_symptom": 1}

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)


    train_dataset = dataset['train']
    dev_dataset = dataset['dev']

    log_params = {
        'train_size': len(dataset['train']),
        'dev_size': len(dataset['dev']),
        'model_size': model.num_parameters(),
        'max_len': MAX_LEN,
    }

    train_classified_ids = []
    for i, item in enumerate(train_dataset):
        if 1 in item['ner_tags']:
            train_classified_ids.append(item['idx'])

    dev_classified_ids = []
    for i, item in enumerate(dev_dataset):
        if 1 in item['ner_tags']:
            dev_classified_ids.append(item['idx'])

    if classified_only_train:
        train_dataset = train_dataset.filter(lambda item: item['idx'] in train_classified_ids)

    if classified_only_dev:
        dev_dataset = dev_dataset.filter(lambda item: item['idx'] in dev_classified_ids)

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

    def preprocess_function(examples):
        return tokenizer(examples["text_tokens"], max_length=MAX_LEN, truncation=True, padding='max_length', is_split_into_words=True)

    train_dataset = Dataset.from_dict({'text_tokens': train_dataset['tokens'], 'idx': train_dataset['idx'], 'label': [1 if sample in train_classified_ids else 0 for sample in train_dataset['idx']]})
    dev_dataset = Dataset.from_dict({'text_tokens': dev_dataset['tokens'], 'idx': dev_dataset['idx'], 'label': [1 if sample in dev_classified_ids else 0 for sample in dev_dataset['idx']]})
    train_dataset = train_dataset.map(preprocess_function, batched=True).shuffle()
    dev_dataset = dev_dataset.map(preprocess_function, batched=True).shuffle()

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

    result_path = os.path.join('result_csv_classify_bert_large', f'{dataset_name}.csv')

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
