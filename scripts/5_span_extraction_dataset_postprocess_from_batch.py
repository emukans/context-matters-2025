import csv
import json
import os
from glob import glob

from script.classification_dataset_prepare import normalize_string


def build_tweet_dict(data_path):
    result_json = {}

    with open(data_path, 'r') as f:
        reader = f.readlines()

        for line in reader:
            line = line.strip().split('\t')
            if line[0].strip() in result_json:
                raise Exception(f"Duplicate tweet: {line[0].strip()}")

            result_json[line[0].strip()] = normalize_string(line[1].strip())

    return result_json


if __name__ == '__main__':
    for env in ('dev', 'train'):
        if env == 'train':
            span_data_path = '../data/Train_2024/ade_span_extraction/span_extraction_gpt35_full'
            positive_data_path = '../data/Train_2024/train_spans_norm_downcast.tsv'
            tweet_data_path = '../data/Train_2024/tweets.tsv'
        else:
            span_data_path = '../data/Dev_2024/ade_span_extraction/span_extraction_gpt35_full'
            positive_data_path = '../data/Dev_2024/norms_downcast.tsv'
            tweet_data_path = '../data/Dev_2024/tweets.tsv'

        with open(positive_data_path, 'r') as f:
            reader = f.readlines()

            has_symptom_list = []
            for line in reader:
                line = line.strip().split('\t')
                has_symptom_list.append(line[0])

            has_symptom_list = set(has_symptom_list)

        tweet_positive_list = []
        tweet_negative_list = []

        total_entries = 0
        has_span_correctly_classified = 0
        has_span_incorrectly_classified = 0
        no_span_correctly_classified = 0
        no_span_incorrectly_classified = 0

        tweet_span_map = {}

        with open(os.path.join(span_data_path, 'ade_response.jsonl'), 'r') as f:
            for line in f.readlines():
                extracted_span = json.loads(line)
                tweet_id = extracted_span['custom_id']
                tweet_text = normalize_string(extracted_span['response']['body']['choices'][0]['message']['content'].lower().strip())

                tweet_span_map[tweet_id] = tweet_text

        tweet_json = dict()
        span_map = {}

        with open(tweet_data_path, 'r') as f:
            reader = f.readlines()

            for line in reader:
                total_entries += 1

                line = line.strip().split('\t')
                tweet_id = line[0]
                tweet_text = normalize_string(line[1].lower().strip())
                has_extracted_span = False

                span_map[tweet_id] = []

                if tweet_id in tweet_span_map:
                    response = tweet_span_map[tweet_id].splitlines()
                    for r in response:
                        span = r.lower().lstrip('span:').strip()
                        if 'null' in span and span != 'null':
                            raise Exception(f"Tweet {tweet_id}")

                        if span == 'null' or not len(span):
                            continue

                        has_extracted_span = True
                        tweet_text += f' [sep] {span}'

                        span_map[tweet_id].append(span)

                tweet_json[tweet_id] = tweet_text
                if line[0] in has_symptom_list:
                    if has_extracted_span:
                        has_span_correctly_classified += 1
                    else:
                        has_span_incorrectly_classified += 1
                    tweet_positive_list.append(tweet_text)
                else:
                    if has_extracted_span:
                        no_span_incorrectly_classified += 1
                    else:
                        no_span_correctly_classified += 1
                    tweet_negative_list.append(tweet_text)

        if sum([has_span_incorrectly_classified, has_span_correctly_classified, no_span_correctly_classified, no_span_incorrectly_classified]) != total_entries:
            raise Exception(f"Calculation is wrong")

        accuracy = (has_span_correctly_classified + no_span_correctly_classified) / total_entries
        precision = has_span_correctly_classified / (has_span_correctly_classified + no_span_incorrectly_classified)
        recall = has_span_correctly_classified / (has_span_correctly_classified + has_span_incorrectly_classified)
        f1 = 2 * precision * recall / (precision + recall)

        print(f'ENV: {env}')
        print(f'Total entries: {total_entries}')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1: {f1:.2f}')
        print()

        os.makedirs(span_data_path, exist_ok=True)

        with open(os.path.join(span_data_path, 'has_symptom.txt'), 'w') as f:
            f.write('\n'.join(set(tweet_positive_list)))

        with open(os.path.join(span_data_path, 'no_symptom.txt'), 'w') as f:
            f.write('\n'.join(set(tweet_negative_list)))

        with open(os.path.join(span_data_path, 'tweet_with_ade.json'), 'w') as f:
            json.dump(tweet_json, f)

        with open(os.path.join(span_data_path, 'span.json'), 'w') as f:
            json.dump(tweet_json, f)

        with open(os.path.join(span_data_path, 'metrics.txt'), 'w') as f:
            f.write(f'Total entries: {total_entries}\n')
            f.write(f'Accuracy: {accuracy:.2f}\n')
            f.write(f'Precision: {precision:.2f}\n')
            f.write(f'Recall: {recall:.2f}\n')
            f.write(f'F1: {f1:.2f}\n')
