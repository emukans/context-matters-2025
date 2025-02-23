import json
import re

import jellyfish
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer, word_tokenize
from tqdm import tqdm


with open('../data/corpora/data/medicine/drugs.json', 'r') as f:
    drugs = json.load(f)

drugs = [d.lower() for d in drugs['drugs'] if len(d) > 4]

with open('../data/corpora/data/medicine/symptoms.json', 'r') as f:
    symptoms = json.load(f)

symptoms = [d.lower() for d in symptoms['symptoms'] if len(d) > 4]

print('Drugs:', len(drugs))
print('Symptoms:', len(symptoms))



def normalize_string(s):
    s = s.strip().lstrip('"').rstrip('"')
    s = re.sub(r'@user_*', '[user]', s, flags=re.I)
    s = re.sub(r'httpurl_*', '[url]', s, flags=re.I)
    s = re.sub(r'(\[user]\s)+', r'[user] ', s)
    s = re.sub(r'(\[url]\s)+', r'[url] ', s)

    return s


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


if __name__ == '__main__':
    for split in ('train', 'dev'):
        if split == 'train':
            # !NB paths on gengo
            symptom_data_path = 'data/train/symptoms.json'
            drug_data_path = 'data/train/drug.json'
            tweet_data_path = 'data/train/tweets.tsv'
        else:
            symptom_data_path = 'data/dev/symptoms.json'
            drug_data_path = 'data/dev/drug.json'
            tweet_data_path = 'data/dev/tweets.tsv'

        drug_map = dict()
        symptom_map = dict()

        tweet_dict = build_tweet_dict(tweet_data_path)
        tokenizer = TreebankWordTokenizer()

        tweet_json = dict()
        for tweet_id, tweet_text in tqdm(tweet_dict.items()):
            tokens = tokenizer.tokenize(normalize_string(tweet_text.lower().strip().lstrip('"').rstrip('"')))

            found_drugs = []
            found_drugs_spans = []
            for drug in drugs:
                best_span, best_similarity, best_match = find_best_match(tokens, drug, found_drugs_spans, threshold=0.9)
                if best_similarity > 0.9:
                    found_drugs.append(drug)
                    found_drugs_spans.append(best_span)

            found_drugs_spans = sorted(found_drugs_spans, key=lambda x: x[0], reverse=True)
            drug_map[tweet_id] = {'spans': found_drugs_spans, 'texts': found_drugs}

            found_symptoms = []
            found_symptoms_spans = []
            for symptom in symptoms:
                best_span, best_similarity, best_match = find_best_match(tokens, symptom, found_symptoms_spans,
                                                                         threshold=0.9)
                if best_similarity > 0.9:
                    found_symptoms.append(symptom)
                    found_symptoms_spans.append(best_span)

            found_symptoms_spans = sorted(found_symptoms_spans, key=lambda x: x[0], reverse=True)
            symptom_map[tweet_id] = {'spans': found_symptoms_spans, 'texts': found_symptoms}

        with open(symptom_data_path, 'w') as f:
            json.dump(symptom_map, f)

        with open(drug_data_path, 'w') as f:
            json.dump(drug_map, f)
