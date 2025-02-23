import pandas as pd
from glob import glob


def main():
    source_dir = 'result_csv_ner'
    update_dir = 'results/result_csv_ner_bert_large'

    for file_path in glob(f'../{source_dir}/*.csv'):
        with open(file_path, 'r') as f:
            source = pd.read_csv(f)

        with open(file_path.replace(source_dir, update_dir), 'r') as f:
            update = pd.read_csv(f)

        if len(source['dataset'].unique()) != 1 and len(update['dataset'].unique()) != 1:
            raise Exception(file_path)

        name_map = {update['dataset'].unique()[0]: source['dataset'].unique()[0]}
        update.loc[:, 'dataset'] = update['dataset'].replace(name_map)

        update.to_csv(file_path.replace(source_dir, update_dir), header=True, na_rep='None', index=False)



if __name__ == '__main__':
    main()
