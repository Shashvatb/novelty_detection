import pandas as pd
import argparse
import os
import sys
import warnings
import pickle
warnings.filterwarnings("ignore")

sys.path.insert(0, os.getcwd())

from time import time
from modelling.model import MemNet
from data_processing.data_preprocess import get_data_path
from modelling.train import get_sentences
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
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
    if os.path.exists(os.path.join(path, 'test_similarities.pkl')):
        with open(os.path.join(path, 'test_similarities.pkl'), 'rb') as f:
            similarities = pickle.load(f)
    else:
        similarities = model.memory_unit(vectors)
        with open(os.path.join(path, 'test_similarities.pkl'), 'wb') as f:
            pickle.dump(similarities, f)
    print('similarities calculated')
    print(time() - s)
    print()

    vals = model.inference(similarities, 0.3)
    assert len(vals) == len(df)
    print('test F1 score: ', f1_score(df[novel_flag], vals))
    print('test precision score: ', precision_score(df[novel_flag], vals))
    print('test recall score: ', recall_score(df[novel_flag], vals))
    print('test accuracy score: ', accuracy_score(df[novel_flag], vals))
    print('Count of True: ', vals.count(True))
    print('Count of False: ', vals.count(False))
