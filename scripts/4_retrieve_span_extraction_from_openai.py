import os
import dotenv
import openai


dotenv.load_dotenv()


if __name__ == '__main__':
    dev_dataset_path = '../data/Dev_2024/ade_span_extraction/span_extraction_gpt4o_full'
    train_dataset_path = '../data/Train_2024/ade_span_extraction/span_extraction_gpt35_full'

    dataset_path = train_dataset_path

    dev_batch_id = 'file-xxx'

    batch_id = dev_batch_id

    client = openai.OpenAI()
    file_response = client.files.content(batch_id)

    with open(os.path.join(dataset_path, 'ade_response.jsonl'), 'w') as f:
        f.write(file_response.text)
