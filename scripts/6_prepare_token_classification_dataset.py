import csv
import json
import re
from contextlib import suppress

import jellyfish
from datasets import load_dataset, ClassLabel, DatasetDict
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk import word_tokenize
from random import randint, random, seed

from tqdm import tqdm

from script.classification_dataset_prepare import normalize_string

# import nltk
# nltk.download('punkt_tab')

seed(42)


detokenizer = TreebankWordDetokenizer()


def find_best_match(sentence_tokens, phrase, span_to_exclude, threshold=None):
    """Find the substring in the sentence with the highest Jaccard similarity to the phrase."""
    best_similarity = 0
    best_span = (0, 0)  # Placeholder for the best span

    # Sliding window of length equal to phrase length (in tokens)
    phrase_length = len(word_tokenize(phrase))

    for k in range(5):
        span_to_check = phrase_length + k
        for i in range(len(sentence_tokens) - span_to_check + 1):
            # if any([from_ <= i <= to_ or from_ <= i + span_to_check <= to_ for from_, to_ in span_to_exclude]):
            if (i, i + span_to_check) in span_to_exclude:
                continue
            window_tokens = detokenizer.detokenize(sentence_tokens[i:i + span_to_check])
            similarity = jellyfish.jaro_winkler_similarity(phrase.strip(), window_tokens.strip())
            if threshold and threshold > similarity:
                continue

            if similarity > best_similarity:
                best_similarity = similarity
                best_span = (i, i + span_to_check)

        span_to_check = phrase_length - k - 1
        if span_to_check < 1:
            continue
        for i in range(len(sentence_tokens) - span_to_check + 1):
            if (i, i + span_to_check) in span_to_exclude:
            # if any([from_ <= i <= to_ or from_ <= i + span_to_check <= to_ for from_, to_ in span_to_exclude]):
                continue
            window_tokens = detokenizer.detokenize(sentence_tokens[i:i + span_to_check])
            similarity = jellyfish.jaro_winkler_similarity(phrase.strip(), window_tokens.strip())

            if threshold and threshold < similarity:
                continue

            if similarity > best_similarity:
                best_similarity = similarity
                best_span = (i, i + span_to_check)

    for from_, to_ in span_to_exclude:
        if from_ < best_span[0] < to_:
            best_span = (from_, best_span[1])

        if from_ < best_span[1] < to_:
            best_span = (best_span[0], to_)

    best_match = detokenizer.detokenize(sentence_tokens[best_span[0]:best_span[1]])

    return best_span, best_similarity, best_match


def build_tweet_dict(data_path):
    result_json = {}

    with open(data_path, 'r') as f:
        reader = f.readlines()

        for line in reader:
            line = line.strip().split('\t')
            if line[0].strip() in result_json:
                raise Exception(f"Duplicate tweet: {line[0].strip()}")

            result_json[line[0].strip()] = line[1].strip()

    return result_json


def prepare_dataset(split):
    if split == 'train':
        spans_data_path = '../data/Train_2024/train_spans_norm_downcast.tsv'
        tweet_data_path = '../data/Train_2024/tweets.tsv'
        symptom_data_path = '../data/Train_2024/symptoms.json'
        drug_data_path = '../data/Train_2024/drug.json'
        json_data_path = '../data/Train_2024/ade_span_extraction/span_extraction_gpt4o_full/span.json'
    else:
        spans_data_path = '../data/Dev_2024/norms_downcast.tsv'
        tweet_data_path = '../data/Dev_2024/tweets.tsv'
        symptom_data_path = '../data/Dev_2024/symptoms.json'
        drug_data_path = '../data/Dev_2024/drug.json'
        json_data_path = '../data/Dev_2024/ade_span_extraction/span_extraction_gpt4o_full/span.json'

    tweet_dict = build_tweet_dict(tweet_data_path)
    ner_dict = tweet_dict.copy()

    tokenizer = TreebankWordTokenizer()

    with open(spans_data_path, 'r') as f:
        reader = f.readlines()

    for line in tqdm(reader):
        line = line.strip().split('\t')
        tweet_id = line[0]
        if tweet_id not in ner_dict:
            raise Exception('Not found')

        span_from = int(line[2])
        span_to = int(line[3])
        span = ner_dict[tweet_id][span_from:span_to]
        shift_left = 0
        shift_right = 0
        space_left = ''
        space_right = ''
        if span_to < len(tweet_dict[tweet_id]) - 1:
            shift_right = 1
            space_right = ' '
        if span_from != 0:
            shift_left = 1
            space_left = ' '

        expected_token_length = len(span)
        token_replacement = ['±'] * len(tokenizer.tokenize(span))
        token_replacement = ' '.join(token_replacement)
        token_replacement += '±' * (expected_token_length - len(token_replacement))

        ner_dict[tweet_id] = ner_dict[tweet_id][:span_from - shift_left] + space_left + token_replacement + space_right + ner_dict[tweet_id][span_to + shift_right:]
        tweet_dict[tweet_id] = tweet_dict[tweet_id][:span_from - shift_left] + space_left + tweet_dict[tweet_id][span_from:span_to] + space_right + tweet_dict[tweet_id][span_to + shift_right:]


    with open(json_data_path, 'r') as f:
        span_dict = json.load(f)

    with open(drug_data_path, 'r') as f:
        drug_dict = json.load(f)

    with open(symptom_data_path, 'r') as f:
        symptom_dict = json.load(f)

    max_tweet_length = 0
    dataset = []
    ade_dataset = []
    summary_dataset = []
    context_dataset = []
    context_tweet_only_with_drug_dataset = []
    context_tweet_only_with_symptom_dataset = []
    context_tweet_only_with_drug_with_symptom_dataset = []
    context_with_drug_dataset = []
    context_with_symptom_dataset = []
    context_with_drug_with_symptom_dataset = []
    span_dataset = []
    span_with_context_dataset = []
    span_tweet_only_with_drug_dataset = []
    span_tweet_only_with_drug_with_symptom_dataset = []
    span_tweet_only_with_symptom_dataset = []
    span_with_drug_dataset = []
    span_with_symptom_dataset = []
    span_with_drug_with_symptom_dataset = []
    for tweet_id in tqdm(ner_dict.keys()):
        tokens = tokenizer.tokenize(normalize_string(tweet_dict[tweet_id].strip().lstrip('"').rstrip('"')))
        ner_tags = tokenizer.tokenize(re.sub(r'±+', 'ADE', normalize_string(ner_dict[tweet_id].strip().lstrip('"').rstrip('"'))))

        for i, tag in enumerate(ner_tags):
            if 'ADE' in tag and tag != 'ADE':
                raise Exception(f'Check {tag}, tweet_id: {tweet_id}')
                
        ner_tags_raw = [tag if tag == 'ADE' else 'O' for tag in ner_tags]
        is_inside = False
        ner_tags = []
        for tag in ner_tags_raw:
            if is_inside:
                if tag == 'ADE':
                    ner_tags.append('I-ADE')
                else:
                    is_inside = False
                    ner_tags.append('O')
            else:
                if tag == 'ADE':
                    ner_tags.append('B-ADE')
                    is_inside = True
                else:
                    ner_tags.append('O')

        if len(ner_tags) != len(tokens):
            raise Exception(f'Check {tweet_id}, {len(tokens)}, {len(ner_tags)}, {tokens}, {" ".join(tokens)} {" ".join(ner_tags)}')
        dataset.append({
            'idx': tweet_id,
            'tokens': tokens,
            'ner_tags': ner_tags
        })

        drug_span_to_exclude = []
        for s1 in drug_dict[tweet_id]['spans']:
            for s2 in drug_dict[tweet_id]['spans']:
                if s1 == s2:
                    continue
                if s1[0] <= s2[0] <= s1[1] and s1[0] <= s2[1] <= s1[1]:
                    drug_span_to_exclude.append(s2)

        found_drugs_spans = [s for s in drug_dict[tweet_id]['spans'] if s not in drug_span_to_exclude]

        symptom_span_to_exclude = []
        for s1 in symptom_dict[tweet_id]['spans']:
            for s2 in symptom_dict[tweet_id]['spans']:
                if s1 == s2:
                    continue
                if s1[0] <= s2[0] <= s1[1] and s1[0] <= s2[1] <= s1[1]:
                    symptom_span_to_exclude.append(s2)
        found_symptoms_spans = [s for s in symptom_dict[tweet_id]['spans'] if s not in symptom_span_to_exclude]

        span_tokenized = []
        drug_tokenized = []
        symptom_tokenized = []
        updated_tokens = tokens.copy()
        updated_tags = ner_tags.copy()
        if tweet_id in span_dict:
            if random() < 0.0:
                start = 1
                for _ in range(randint(1, 3)):
                    if start >= len(tokens) - 2:
                        break
                    span_start = randint(start, len(tokens) - 2)
                    span_end = min(span_start + randint(2, 5), len(tokens) - 1)
                    span_tokenized += tokenizer.tokenize('[sep]') + tokens[span_start:span_end]
                    start = span_end
                    if span_end == len(tokens):
                        break
            else:
                span_text = ' [sep] ' + ' [sep] '.join(span_dict[tweet_id])
                span_text = normalize_string(span_text.strip().lstrip('"').rstrip('"'))
                span_tokenized = tokenizer.tokenize(span_text)

                drug_text = ' [drug] ' + ' [drug] '.join(drug_dict[tweet_id]['matches'])
                drug_text = normalize_string(drug_text.strip().lstrip('"').rstrip('"'))
                drug_tokenized = tokenizer.tokenize(drug_text)

                symptom_text = ' [symptom] ' + ' [symptom] '.join(symptom_dict[tweet_id]['matches'])
                symptom_text = normalize_string(symptom_text.strip().lstrip('"').rstrip('"'))
                symptom_tokenized = tokenizer.tokenize(symptom_text)

                span_to_exclude = []
                for span in span_dict[tweet_id]:
                    span = normalize_string(span.strip().lstrip('"').rstrip('"'))
                    best_span, best_similarity, best_match = find_best_match(tokens, span.strip(), span_to_exclude)
                    span_to_exclude.append(best_span)

                # todo: test this
                span_to_exclude = sorted(span_to_exclude, key=lambda x: x[0], reverse=True)
                drug_span_with_gpt = found_drugs_spans
                symptom_span_with_gpt = found_symptoms_spans

                for span in span_to_exclude:
                    drug_span_with_gpt_updated = []
                    for drug_span in drug_span_with_gpt:
                        if span[1] <= drug_span[0]:
                            drug_span_with_gpt_updated.append((drug_span[0] + 6, drug_span[1] + 6))
                        elif span[0] <= drug_span[0]:
                            drug_span_with_gpt_updated.append((drug_span[0] + 3, drug_span[1] + 3))
                        else:
                            drug_span_with_gpt_updated.append((drug_span[0], drug_span[1]))
                    drug_span_with_gpt = drug_span_with_gpt_updated

                    symptom_span_with_gpt_updated = []
                    for symptom_span in symptom_span_with_gpt:
                        if span[1] <= symptom_span[0]:
                            symptom_span_with_gpt_updated.append((symptom_span[0] + 6, symptom_span[1] + 6))
                        elif span[0] <= symptom_span[0]:
                            symptom_span_with_gpt_updated.append((symptom_span[0] + 3, symptom_span[1] + 3))
                        else:
                            symptom_span_with_gpt_updated.append((symptom_span[0], symptom_span[1]))

                    symptom_span_with_gpt = symptom_span_with_gpt_updated
                    updated_tokens = updated_tokens[:span[0]] + ['[', r'ade', ']'] + updated_tokens[span[0]:span[1]] + ['[', r'\ade', ']'] + updated_tokens[span[1]:]
                    updated_tags = updated_tags[:span[0]] + ['3', r'3', '3'] + updated_tags[span[0]:span[1]] + ['3', r'3', '3'] + updated_tags[span[1]:]

                drug_span_no_gpt = found_drugs_spans
                symptom_span_no_gpt = found_symptoms_spans

                tokens_with_drugs = tokens.copy()
                tags_with_drugs = ner_tags.copy()
                for drug_span in found_drugs_spans:
                    symptom_span_no_gpt_updated = []
                    for symptom_span in symptom_span_no_gpt:
                        if drug_span[1] <= symptom_span[0]:
                            symptom_span_no_gpt_updated.append((symptom_span[0] + 6, symptom_span[1] + 6))
                        elif drug_span[0] <= symptom_span[0]:
                            symptom_span_no_gpt_updated.append((symptom_span[0] + 3, symptom_span[1] + 3))
                        else:
                            symptom_span_no_gpt_updated.append((symptom_span[0], symptom_span[1]))

                    symptom_span_no_gpt = symptom_span_no_gpt_updated
                    tokens_with_drugs = tokens_with_drugs[:drug_span[0]] + ['[', r'drug', ']'] + tokens_with_drugs[drug_span[0]:drug_span[1]] + ['[', r'\drug', ']'] + tokens_with_drugs[drug_span[1]:]
                    tags_with_drugs = tags_with_drugs[:drug_span[0]] + ['3', r'3', '3'] + tags_with_drugs[drug_span[0]:drug_span[1]] + ['3', r'3', '3'] + tags_with_drugs[drug_span[1]:]

                tokens_with_drugs_and_symptoms = tokens.copy()
                tags_with_drugs_and_symptoms = ner_tags.copy()

                for symptom_span in symptom_span_no_gpt:
                    tokens_with_drugs_and_symptoms = tokens_with_drugs_and_symptoms[:symptom_span[0]] + ['[', r'symptom', ']'] + tokens_with_drugs_and_symptoms[symptom_span[0]:symptom_span[1]] + ['[', r'\symptom', ']'] + tokens_with_drugs_and_symptoms[symptom_span[1]:]
                    tags_with_drugs_and_symptoms = tags_with_drugs_and_symptoms[:symptom_span[0]] + ['3', r'3', '3'] + tags_with_drugs_and_symptoms[symptom_span[0]:symptom_span[1]] + ['3', r'3', '3'] + tags_with_drugs_and_symptoms[symptom_span[1]:]

                tokens_with_symptoms = tokens.copy()
                tags_with_symptoms = ner_tags.copy()
                for symptom_span in found_symptoms_spans:
                    tokens_with_symptoms = tokens_with_symptoms[:symptom_span[0]] + ['[', r'symptom', ']'] + tokens_with_symptoms[symptom_span[0]:symptom_span[1]] + ['[', r'\symptom', ']'] + tokens_with_symptoms[symptom_span[1]:]
                    tags_with_symptoms = tags_with_symptoms[:symptom_span[0]] + ['3', r'3', '3'] + tags_with_symptoms[symptom_span[0]:symptom_span[1]] + ['3', r'3', '3'] + tags_with_symptoms[symptom_span[1]:]


                symptom_span_with_gpt_and_drug = symptom_span_with_gpt

                updated_tokens_with_drugs = updated_tokens.copy()
                updated_tags_with_drugs = updated_tags.copy()
                for drug_span in drug_span_with_gpt:
                    symptom_span_with_gpt_and_drug_updated = []
                    for symptom_span in symptom_span_with_gpt_and_drug:
                        if drug_span[1] <= symptom_span[0]:
                            symptom_span_with_gpt_and_drug_updated.append((symptom_span[0] + 6, symptom_span[1] + 6))
                        elif drug_span[0] <= symptom_span[0]:
                            symptom_span_with_gpt_and_drug_updated.append((symptom_span[0] + 3, symptom_span[1] + 3))
                        else:
                            symptom_span_with_gpt_and_drug_updated.append((symptom_span[0], symptom_span[1]))

                    symptom_span_with_gpt_and_drug = symptom_span_with_gpt_and_drug_updated

                    updated_tokens_with_drugs = updated_tokens_with_drugs[:drug_span[0]] + ['[', r'drug', ']'] + updated_tokens_with_drugs[drug_span[0]:drug_span[1]] + ['[', r'\drug', ']'] + updated_tokens_with_drugs[drug_span[1]:]
                    updated_tags_with_drugs = updated_tags_with_drugs[:drug_span[0]] + ['3', r'3', '3'] + updated_tags_with_drugs[drug_span[0]:drug_span[1]] + ['3', r'3', '3'] + updated_tags_with_drugs[drug_span[1]:]


                updated_tokens_with_symptoms = updated_tokens.copy()
                updated_tags_with_symptoms = updated_tags.copy()
                for symptom_span in symptom_span_with_gpt:
                    updated_tokens_with_symptoms = updated_tokens_with_symptoms[:symptom_span[0]] + ['[', r'symptom', ']'] + updated_tokens_with_symptoms[symptom_span[0]:symptom_span[1]] + ['[', r'\symptom', ']'] + updated_tokens_with_symptoms[symptom_span[1]:]
                    updated_tags_with_symptoms = updated_tags_with_symptoms[:symptom_span[0]] + ['3', r'3', '3'] + updated_tags_with_symptoms[symptom_span[0]:symptom_span[1]] + ['3', r'3', '3'] + updated_tags_with_symptoms[symptom_span[1]:]

                updated_tokens_with_drugs_and_symptoms = updated_tokens_with_drugs.copy()
                updated_tags_with_drugs_and_symptoms = updated_tags_with_drugs.copy()
                for symptom_span in symptom_span_with_gpt_and_drug:
                    updated_tokens_with_drugs_and_symptoms = updated_tokens_with_drugs_and_symptoms[:symptom_span[0]] + ['[', r'symptom', ']'] + updated_tokens_with_drugs_and_symptoms[symptom_span[0]:symptom_span[1]] + ['[', r'\symptom', ']'] + updated_tokens_with_drugs_and_symptoms[symptom_span[1]:]
                    updated_tags_with_drugs_and_symptoms = updated_tags_with_drugs_and_symptoms[:symptom_span[0]] + ['3', r'3', '3'] + updated_tags_with_drugs_and_symptoms[symptom_span[0]:symptom_span[1]] + ['3', r'3', '3'] + updated_tags_with_drugs_and_symptoms[symptom_span[1]:]
        else:
            span_text = ' [sep] ' + ' [sep] '.join(span_dict[tweet_id])
            span_text = normalize_string(span_text.strip().lstrip('"').rstrip('"'))
            span_tokenized = tokenizer.tokenize(span_text)

            span_to_exclude = []
            for span in span_dict[tweet_id]:
                span = normalize_string(span.strip().lstrip('"').rstrip('"'))
                best_span, best_similarity, best_match = find_best_match(tokens, span.strip(), span_to_exclude)
                span_to_exclude.append(best_span)

            span_to_exclude = sorted(span_to_exclude, key=lambda x: x[0], reverse=True)

            for _from, _to in span_to_exclude:
                pass

        if len(updated_tokens) != len(updated_tags):
            raise Exception(f'1 Check: {tweet_id}')

        span_dataset.append({
            'idx': tweet_id,
            'tokens': updated_tokens,
            'ner_tags': updated_tags
        })

        if len(tokens + span_tokenized) != len(ner_tags + ['EXTRA'] * len(span_tokenized)):
            raise Exception(f'2 Check: {tweet_id}')

        context_dataset.append({
            'idx': tweet_id,
            'tokens': tokens + span_tokenized,
            'ner_tags': ner_tags + ['EXTRA'] * len(span_tokenized)
        })

        if len(updated_tokens + span_tokenized) != len(updated_tags + ['EXTRA'] * len(span_tokenized)):
            raise Exception(f'3 Check: {tweet_id}')
        span_with_context_dataset.append({
            'idx': tweet_id,
            'tokens': updated_tokens + span_tokenized,
            'ner_tags': updated_tags + ['EXTRA'] * len(span_tokenized)
        })

        context_tweet_only_with_drug_dataset.append({
            'idx': tweet_id,
            'tokens': tokens + drug_tokenized,
            'ner_tags': ner_tags + ['EXTRA'] * len(drug_tokenized)
        })

        context_tweet_only_with_symptom_dataset.append({
            'idx': tweet_id,
            'tokens': tokens + symptom_tokenized,
            'ner_tags': ner_tags + ['EXTRA'] * len(symptom_tokenized)
        })

        context_tweet_only_with_drug_with_symptom_dataset.append({
            'idx': tweet_id,
            'tokens': tokens + drug_tokenized + symptom_tokenized,
            'ner_tags': ner_tags + ['EXTRA'] * len(drug_tokenized) + ['EXTRA'] * len(symptom_tokenized)
        })

        context_with_drug_dataset.append({
            'idx': tweet_id,
            'tokens': tokens + span_tokenized + drug_tokenized,
            'ner_tags': ner_tags + ['EXTRA'] * len(span_tokenized) + ['EXTRA'] * len(drug_tokenized)
        })

        context_with_symptom_dataset.append({
            'idx': tweet_id,
            'tokens': tokens + span_tokenized + symptom_tokenized,
            'ner_tags': ner_tags + ['EXTRA'] * len(span_tokenized) + ['EXTRA'] * len(symptom_tokenized)
        })

        context_with_drug_with_symptom_dataset.append({
            'idx': tweet_id,
            'tokens': tokens + span_tokenized + drug_tokenized + symptom_tokenized,
            'ner_tags': ner_tags + ['EXTRA'] * len(span_tokenized) + ['EXTRA'] * len(drug_tokenized) + ['EXTRA'] * len(symptom_tokenized)
        })

        span_tweet_only_with_drug_dataset.append({
            'idx': tweet_id,
            'tokens': tokens_with_drugs,
            'ner_tags': tags_with_drugs
        })

        span_tweet_only_with_symptom_dataset.append({
            'idx': tweet_id,
            'tokens': tokens_with_symptoms,
            'ner_tags': tags_with_symptoms
        })

        span_with_drug_dataset.append({
            'idx': tweet_id,
            'tokens': updated_tokens_with_drugs,
            'ner_tags': updated_tags_with_drugs
        })

        span_with_symptom_dataset.append({
            'idx': tweet_id,
            'tokens': updated_tokens_with_symptoms,
            'ner_tags': updated_tags_with_symptoms
        })

        span_with_drug_with_symptom_dataset.append({
            'idx': tweet_id,
            'tokens': updated_tokens_with_drugs_and_symptoms,
            'ner_tags': updated_tags_with_drugs_and_symptoms
        })

        span_tweet_only_with_drug_with_symptom_dataset.append({
            'idx': tweet_id,
            'tokens': tokens_with_drugs_and_symptoms,
            'ner_tags': tags_with_drugs_and_symptoms
        })

    print(f'Max tweet length: {max_tweet_length}')

    with open(json_data_path.replace('span.json', 'tweet_only_dataset.json'), 'w') as f:
        json.dump(dataset, f)
    #
    # with open(json_data_path.replace('ner', 'ade_ner'), 'w') as f:
    #     json.dump(ade_dataset, f)
    #
    # with open(json_data_path.replace('ner', 'summary_ner'), 'w') as f:
    #     json.dump(summary_dataset, f)

    with open(json_data_path.replace('span.json', 'span_dataset.json'), 'w') as f:
        json.dump(span_dataset, f)

    with open(json_data_path.replace('span.json', 'context_dataset.json'), 'w') as f:
        json.dump(context_dataset, f)

    with open(json_data_path.replace('span.json', 'span_with_context_dataset.json'), 'w') as f:
        json.dump(span_with_context_dataset, f)

    with open(json_data_path.replace('span.json', 'context_with_drug_dataset.json'), 'w') as f:
        json.dump(context_with_drug_dataset, f)

    with open(json_data_path.replace('span.json', 'context_with_symptom_dataset.json'), 'w') as f:
        json.dump(context_with_symptom_dataset, f)

    with open(json_data_path.replace('span.json', 'context_with_drug_with_symptom_dataset.json'), 'w') as f:
        json.dump(context_with_drug_with_symptom_dataset, f)

    with open(json_data_path.replace('span.json', 'context_tweet_only_with_drug_dataset.json'), 'w') as f:
        json.dump(context_tweet_only_with_drug_dataset, f)

    with open(json_data_path.replace('span.json', 'context_tweet_only_with_symptom_dataset.json'), 'w') as f:
        json.dump(context_tweet_only_with_symptom_dataset, f)

    with open(json_data_path.replace('span.json', 'context_tweet_only_with_drug_with_symptom_dataset.json'), 'w') as f:
        json.dump(context_tweet_only_with_drug_with_symptom_dataset, f)

    with open(json_data_path.replace('span.json', 'span_with_drug_dataset.json'), 'w') as f:
        json.dump(span_with_drug_dataset, f)

    with open(json_data_path.replace('span.json', 'span_with_symptom_dataset.json'), 'w') as f:
        json.dump(span_with_symptom_dataset, f)

    with open(json_data_path.replace('span.json', 'span_with_drug_with_symptom_dataset.json'), 'w') as f:
        json.dump(span_with_drug_with_symptom_dataset, f)

    with open(json_data_path.replace('span.json', 'span_tweet_only_with_drug_dataset.json'), 'w') as f:
        json.dump(span_tweet_only_with_drug_dataset, f)

    with open(json_data_path.replace('span.json', 'span_tweet_only_with_symptom_dataset.json'), 'w') as f:
        json.dump(span_tweet_only_with_symptom_dataset, f)

    with open(json_data_path.replace('span.json', 'span_tweet_only_with_drug_with_symptom_dataset.json'), 'w') as f:
        json.dump(span_tweet_only_with_drug_with_symptom_dataset, f)

    tag_mapping = {
        'O': 0,
        'B-ADE': 1,
        'I-ADE': 2,
        'EXTRA': -100
    }

    dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'tweet_only_dataset.json'), split="train")
    new_features = dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    dataset = dataset.cast(new_features)
    #
    # ade_dataset = load_dataset('json', data_files=json_data_path.replace('ner', 'ade_ner'), split="train")
    # new_features = ade_dataset.features.copy()
    # new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    # ade_dataset = ade_dataset.cast(new_features)
    #
    # summary_dataset = load_dataset('json', data_files=json_data_path.replace('ner', 'summary_ner'), split="train")
    # new_features = summary_dataset.features.copy()
    # new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    # summary_dataset = summary_dataset.cast(new_features)

    span_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'span_dataset.json'), split="train")
    new_features = span_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    span_dataset = span_dataset.cast(new_features)

    context_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'context_dataset.json'), split="train")
    new_features = context_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    context_dataset = context_dataset.cast(new_features)

    span_with_context_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'span_with_context_dataset.json'), split="train")
    new_features = span_with_context_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    span_with_context_dataset = span_with_context_dataset.cast(new_features)

    context_with_drug_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'context_with_drug_dataset.json'), split="train")
    new_features = context_with_drug_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    context_with_drug_dataset = context_with_drug_dataset.cast(new_features)

    context_with_symptom_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'context_with_symptom_dataset.json'), split="train")
    new_features = context_with_symptom_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    context_with_symptom_dataset = context_with_symptom_dataset.cast(new_features)

    context_with_drug_with_symptom_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'context_with_drug_with_symptom_dataset.json'), split="train")
    new_features = context_with_drug_with_symptom_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    context_with_drug_with_symptom_dataset = context_with_drug_with_symptom_dataset.cast(new_features)

    context_tweet_only_with_drug_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'context_tweet_only_with_drug_dataset.json'), split="train")
    new_features = context_tweet_only_with_drug_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    context_tweet_only_with_drug_dataset = context_tweet_only_with_drug_dataset.cast(new_features)

    context_tweet_only_with_symptom_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'context_tweet_only_with_symptom_dataset.json'), split="train")
    new_features = context_tweet_only_with_symptom_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    context_tweet_only_with_symptom_dataset = context_tweet_only_with_symptom_dataset.cast(new_features)

    context_tweet_only_with_drug_with_symptom_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'context_tweet_only_with_drug_with_symptom_dataset.json'), split="train")
    new_features = context_tweet_only_with_drug_with_symptom_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    context_tweet_only_with_drug_with_symptom_dataset = context_tweet_only_with_drug_with_symptom_dataset.cast(new_features)

    span_with_drug_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'span_with_drug_dataset.json'), split="train")
    new_features = span_with_drug_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    span_with_drug_dataset = span_with_drug_dataset.cast(new_features)

    span_with_symptom_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'span_with_symptom_dataset.json'), split="train")
    new_features = span_with_symptom_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    span_with_symptom_dataset = span_with_symptom_dataset.cast(new_features)

    span_with_drug_with_symptom_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'span_with_drug_with_symptom_dataset.json'), split="train")
    new_features = span_with_drug_with_symptom_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    span_with_drug_with_symptom_dataset = span_with_drug_with_symptom_dataset.cast(new_features)

    span_tweet_only_with_drug_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'span_tweet_only_with_drug_dataset.json'), split="train")
    new_features = span_tweet_only_with_drug_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    span_tweet_only_with_drug_dataset = span_tweet_only_with_drug_dataset.cast(new_features)

    span_tweet_only_with_symptom_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'span_tweet_only_with_symptom_dataset.json'), split="train")
    new_features = span_tweet_only_with_symptom_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    span_tweet_only_with_symptom_dataset = span_tweet_only_with_symptom_dataset.cast(new_features)

    span_tweet_only_with_drug_with_symptom_dataset = load_dataset('json', data_files=json_data_path.replace('span.json', 'span_tweet_only_with_drug_with_symptom_dataset.json'), split="train")
    new_features = span_tweet_only_with_drug_with_symptom_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    span_tweet_only_with_drug_with_symptom_dataset = span_tweet_only_with_drug_with_symptom_dataset.cast(new_features)

    return dataset, ade_dataset, summary_dataset, context_dataset, span_dataset, span_with_context_dataset, context_with_drug_dataset, context_with_symptom_dataset, context_with_drug_with_symptom_dataset, context_tweet_only_with_drug_dataset, context_tweet_only_with_symptom_dataset, context_tweet_only_with_drug_with_symptom_dataset, span_with_drug_dataset, span_with_symptom_dataset, span_with_drug_with_symptom_dataset, span_tweet_only_with_drug_dataset, span_tweet_only_with_symptom_dataset, span_tweet_only_with_drug_with_symptom_dataset


if __name__ == '__main__':
    dev_dataset, dev_ade_dataset, dev_summary_dataset, dev_context_dataset, dev_span_dataset, dev_span_with_context_dataset, dev_context_with_drug_dataset, dev_context_with_symptom_dataset, dev_context_with_drug_with_symptom_dataset, dev_context_tweet_only_with_drug_dataset, dev_context_tweet_only_with_symptom_dataset, dev_context_tweet_only_with_drug_with_symptom_dataset, dev_span_with_drug_dataset, dev_span_with_symptom_dataset, dev_span_with_drug_with_symptom_dataset, dev_span_tweet_only_with_drug_dataset, dev_span_tweet_only_with_symptom_dataset, dev_span_tweet_only_with_drug_with_symptom_dataset = prepare_dataset('dev')
    train_dataset, train_ade_dataset, train_summary_dataset, train_context_dataset, train_span_dataset, train_span_with_context_dataset, train_context_with_drug_dataset, train_context_with_symptom_dataset, train_context_with_drug_with_symptom_dataset, train_context_tweet_only_with_drug_dataset, train_context_tweet_only_with_symptom_dataset, train_context_tweet_only_with_drug_with_symptom_dataset, train_span_with_drug_dataset, train_span_with_symptom_dataset, train_span_with_drug_with_symptom_dataset, train_span_tweet_only_with_drug_dataset, train_span_tweet_only_with_symptom_dataset, train_span_tweet_only_with_drug_with_symptom_dataset = prepare_dataset('train')


    dataset_path = '../data/2024_ner_gpt4'

    dataset = DatasetDict({'train': train_dataset, 'dev': dev_dataset})
    dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_tweet_only'))

    # dataset = DatasetDict({'train': train_dataset, 'dev': dev_dataset})
    # dataset.save_to_disk(dataset_path)
    #
    # ade_dataset = DatasetDict({'train': train_ade_dataset, 'dev': dev_ade_dataset})
    # ade_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_ade_ner_gpt4'))
    #
    # summary_dataset = DatasetDict({'train': train_summary_dataset, 'dev': dev_summary_dataset})
    # summary_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_summary_ner_gpt4'))

    # context_dataset = DatasetDict({'train': train_context_dataset, 'dev': dev_context_dataset})
    # context_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_context'))
    #
    # span_dataset = DatasetDict({'train': train_span_dataset, 'dev': dev_span_dataset})
    # span_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_span'))
    #
    # span_with_context_dataset = DatasetDict({'train': train_span_with_context_dataset, 'dev': dev_span_with_context_dataset})
    # span_with_context_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_span_with_context'))

    context_with_drug_dataset = DatasetDict({'train': train_context_with_drug_dataset, 'dev': dev_context_with_drug_dataset})
    context_with_drug_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_context_with_drug_dataset'))

    context_with_symptom_dataset = DatasetDict({'train': train_context_with_symptom_dataset, 'dev': dev_context_with_symptom_dataset})
    context_with_symptom_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_context_with_symptom_dataset'))

    context_with_drug_with_symptom_dataset = DatasetDict({'train': train_context_with_drug_with_symptom_dataset, 'dev': dev_context_with_drug_with_symptom_dataset})
    context_with_drug_with_symptom_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_context_with_drug_with_symptom_dataset'))

    context_tweet_only_with_drug_dataset = DatasetDict({'train': train_context_tweet_only_with_drug_dataset, 'dev': dev_context_tweet_only_with_drug_dataset})
    context_tweet_only_with_drug_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_context_tweet_only_with_drug_dataset'))

    context_tweet_only_with_symptom_dataset = DatasetDict({'train': train_context_tweet_only_with_symptom_dataset, 'dev': dev_context_tweet_only_with_symptom_dataset})
    context_tweet_only_with_symptom_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_context_tweet_only_with_symptom_dataset'))

    context_tweet_only_with_drug_with_symptom_dataset = DatasetDict({'train': train_context_tweet_only_with_drug_with_symptom_dataset, 'dev': dev_context_tweet_only_with_drug_with_symptom_dataset})
    context_tweet_only_with_drug_with_symptom_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_context_tweet_only_with_drug_with_symptom_dataset'))

    span_with_drug_dataset = DatasetDict({'train': train_span_with_drug_dataset, 'dev': dev_span_with_drug_dataset})
    span_with_drug_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_span_with_drug_dataset'))

    span_with_symptom_dataset = DatasetDict({'train': train_span_with_symptom_dataset, 'dev': dev_span_with_symptom_dataset})
    span_with_symptom_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_span_with_symptom_dataset'))

    span_with_drug_with_symptom_dataset = DatasetDict({'train': train_span_with_drug_with_symptom_dataset, 'dev': dev_span_with_drug_with_symptom_dataset})
    span_with_drug_with_symptom_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_span_with_drug_with_symptom_dataset'))

    span_tweet_only_with_drug_dataset = DatasetDict({'train': train_span_tweet_only_with_drug_dataset, 'dev': dev_span_tweet_only_with_drug_dataset})
    span_tweet_only_with_drug_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_span_tweet_only_with_drug_dataset'))

    span_tweet_only_with_symptom_dataset = DatasetDict({'train': train_span_tweet_only_with_symptom_dataset, 'dev': dev_span_tweet_only_with_symptom_dataset})
    span_tweet_only_with_symptom_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_span_tweet_only_with_symptom_dataset'))

    span_tweet_only_with_drug_with_symptom_dataset = DatasetDict({'train': train_span_tweet_only_with_drug_with_symptom_dataset, 'dev': dev_span_tweet_only_with_drug_with_symptom_dataset})
    span_tweet_only_with_drug_with_symptom_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_gpt4o_span_tweet_only_with_drug_with_symptom_dataset'))
