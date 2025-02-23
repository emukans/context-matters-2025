import os

import openai
import dotenv


dotenv.load_dotenv()


if __name__ == '__main__':
    train_batch_id = 'batch_xxx'

    batch_id = train_batch_id

    client = openai.OpenAI()
    # response = client.batches.list()
    response = client.batches.retrieve(batch_id)

    print(response.json())
    print('Status: ', response.status)
    print('Completed: ', response.request_counts.completed / response.request_counts.total)
    print('Output file: ', response.output_file_id)
