import csv
import json
import os
from glob import glob
import dotenv

import openai
from tqdm import tqdm


dotenv.load_dotenv()


prompt_span = """TASK: Extract spans from tweets related to adverse drug events.

RESTRICTIONS:
1. Each span length should be 5 words or less.
2. Multiple span selections are allowed.
3. Place different drug events types into a separate spans.
4. If no span related to adverse drug events is present in the tweet, mark it as "null".

---
INPUT: "trying to distract with masterchef, twitter, texting &amp; candy crush but i'm anxious &amp; have stomach &amp; head pain #venlafaxine #withdrawal"
OUTPUT:
SPAN: anxious
SPAN: stomach pain
SPAN: head pain
SPAN: withdrawal

---
INPUT: [user] yes ma'am. MTX of 20mg weekly too. Tried Enbrel before this, and will do Remicade if this doesn't help. #rheum
OUTPUT:
SPAN: null

---
INPUT: So glad I'm off #effexor, so sad it ruined my teeth. #tip Please be careful w/ taking #antidepressants and read about it 1st! #venlafaxine
OUTPUT:
SPAN: ruined my teeth

---
INPUT: Bananas contain a natural chemical which can make a person happy. This same chemical is also found in Prozac.
OUTPUT:
SPAN: null

---
INPUT: "[user] i found the humira to fix all my crohn's issues, but cause other issues. i went off it due to issues w nerves/muscle spasms"
OUTPUT:
SPAN: nerves
SPAN: muscle spasms

---
INPUT: {tweet}
OUTPUT:
"""

prompt_summary = '''
You will be provided with a tweet. Summarise it into a brief sentence and highlight already happened adverse drug events (ADE) if there are any related to drugs.
Format:
Summary: text
ADE: text or null
---
Tweet:
"""
{tweet}
"""
'''

initial_ade_mining_prompt = '''
You will be provided with a tweet. Highlight already happened adverse drug events (ADE) if there are any related to drugs. Omit the ADE context, output just ADE. Write the span exactly as in the tweet. If there are many different ADEs by context within the same span, then split it. If ADE repeats, then put each mention on a new line. If there are multiple spans then put each on a new line.
---
Format:
SPAN: text or null
---
Samples:
Tweet:
"""
[user] if #avelox has hurt your liver, avoid tylenol always, as it further damages liver, eat grapefruit unless taking cardiac drugs
"""
SPAN: hurt your liver
---
Tweet:
"""
losing it. could not remember the word power strip. wonder which drug is doing this memory lapse thing. my guess the cymbalta. #helps
"""
SPAN: not remember
SPAN: memory lapse
Tweet:
"""
is adderall a performance enhancing drug for mathletes?
"""
SPAN: null
---
Tweet:
"""
[user] i found the humira to fix all my crohn's issues, but cause other issues. i went off it due to issues w nerves/muscle spasms
"""
'''

ade_mining = '''
You will be provided with a tweet. Your task is to identify and highlight any adverse drug events (ADEs) mentioned in relation to drug use. Only the exact phrases describing the ADEs should be outputted, without including any additional context. Each ADE should be listed on a new line. If the same ADE is mentioned multiple times, each occurrence should be listed separately. If multiple different ADEs are identified within the same tweet, they should be listed on separate lines. If no ADEs are found, output "null".
---
Format:
SPAN: text or null
---
Samples:
Tweet:
"""
[user] if #avelox has hurt your liver, avoid tylenol always, as it further damages liver, eat grapefruit unless taking cardiac drugs
"""
SPAN: hurt your liver
---
Tweet:
"""
losing it. could not remember the word power strip. wonder which drug is doing this memory lapse thing. my guess the cymbalta. #helps
"""
SPAN: not remember
SPAN: memory lapse
Tweet:
"""
is adderall a performance enhancing drug for mathletes?
"""
SPAN: null
---
Tweet:
"""
{tweet}
"""
'''

if __name__ == '__main__':
    dev_dataset_path = '../data/Dev_2024/ade_span_extraction/span_extraction_gpt35_full'
    train_dataset_path = '../data/Train_2024/ade_span_extraction/span_extraction_gpt4o_mini'
    test_dataset_path = '../data/test/ade_extraction_gpt4'

    dataset_path = train_dataset_path
    # span_from = int(sys.argv[1])
    # span_to = int(sys.argv[2])

    os.makedirs(os.path.join(dataset_path, 'response'), exist_ok=True)
    with open(os.path.join(dataset_path, 'tweets.json'), 'r') as f:
        dataset = json.load(f)

    extracted_span_list = glob(os.path.join(dataset_path, 'response', '*.txt'))
    extracted_span_list = [os.path.splitext(os.path.basename(name))[0] for name in extracted_span_list]

    with open(os.path.join(dataset_path, 'payload.jsonl'), 'w') as f:
        for tweet_id, tweet in tqdm(list(dataset.items())):
            if tweet_id in extracted_span_list:
                continue

            json.dump({"custom_id": tweet_id, "method": "POST", "url": "/v1/chat/completions",
                       "body": {
                           "model": "gpt-4o-mini",
                           # "model": "gpt-4o",
                           # "model": "gpt-3.5-turbo-0125",
                           "messages": [
                               {
                                   "role": "user",
                                   "content": ade_mining.format(tweet=tweet)
                               }
                           ],
                           "max_tokens": 64,
                           "temperature": 0,
                           "top_p": 1,
                           "frequency_penalty": 0,
                           "presence_penalty": 0}}, f)
            f.write('\n')

    client = openai.OpenAI()

    batch_input_file = client.files.create(
        file=open(os.path.join(dataset_path, 'payload.jsonl'), "rb"),
        purpose="batch"
    )

    print(batch_input_file.id)
    response = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "ADE discovery"
        }
    )

    print(response)

    with open(os.path.join(dataset_path, 'response.json'), 'w') as f:
        f.write(response.json())
