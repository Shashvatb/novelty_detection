import pandas as pd
import numpy as np
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.getcwd())

from time import time
from modelling.model import MemNet
from data_processing.data_preprocess import get_data_path
from modelling.train import get_sentences
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from config import text_column, novel_flag


def get_validation_data(path):
    df = pd.read_parquet(os.path.join(path, 'validate.parquet')).fillna('')
    df = df[df[text_column] != '']
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='trec')
    args = parser.parse_args()

    dataset_name = args.dataset

    path = get_data_path(dataset_name)
    df = get_validation_data(path)
    sentences = get_sentences(df)

    model = MemNet(path)

    s = time()
    vectors = model.input_layer(sentences)
    print('input vector created')
    print(time()-s)
    print()

    s = time()
    similarities = model.memory_unit(vectors)
    print('similarities calculated')
    print(time() - s)
    print()

    metric_df = pd.DataFrame()
    score = []
    thresh = []

    for i in np.arange(0.2, 1, 0.001):
        s = time()
        vals = model.inference(similarities, i)
        assert len(vals) == len(df)
        thresh.append(i)
        score.append([f1_score(df[novel_flag], vals), precision_score(df[novel_flag], vals),
                      recall_score(df[novel_flag], vals), accuracy_score(df[novel_flag], vals)])

    metric_df['cosine_sim'] = thresh
    metric_df['f1_score'] = [i[0] for i in score]
    metric_df['precision_score'] = [i[1] for i in score]
    metric_df['recall_score'] = [i[2] for i in score]
    metric_df['accuracy_score'] = [i[3] for i in score]

    metric_df.to_excel(os.path.join(path, 'hyperparam.xlsx'), engine='xlsxwriter', index=False)
