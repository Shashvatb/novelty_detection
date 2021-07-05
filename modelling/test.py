import pandas as pd
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
from sklearn.metrics import f1_score

from config import text_column, novel_flag


def get_test_data(path):
    df = pd.read_parquet(os.path.join(path, 'test.parquet')).fillna('')
    df = df[df[text_column] != '']
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='trec')
    args = parser.parse_args()

    dataset_name = args.dataset

    path = get_data_path(dataset_name)
    df = get_test_data(path)
    sentences = get_sentences(df)

    model = MemNet(path)

    s = time()
    vectors = model.input_layer(sentences)
    print('input vector created')
    print(time() - s)
    print()

    s = time()
    similarities = model.memory_unit(vectors)
    print('similarities calculated')
    print(time() - s)
    print()

    vals = model.inference(similarities, )
    assert len(vals) == len(df)
    print('test F1 score: ', f1_score(df[novel_flag], vals))
