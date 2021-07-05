import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.getcwd())

from modelling.model import MemNet
from data_processing.data_preprocess import get_data_path
from config import text_column, novel_flag


def get_sentences(df, col=None, val=None):
    if col:
        df = df[df[col] == val]
    sentences = df[text_column].tolist()
    return sentences


def get_train_data(path):
    df = pd.read_parquet(os.path.join(path, 'train.parquet'))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='trec')
    args = parser.parse_args()

    dataset_name = args.dataset

    path = get_data_path(dataset_name)

    train_data = get_train_data(path)
    sentences = get_sentences(train_data)
    print("number of sentences for training the vectorizer: ", len(sentences))

    model = MemNet(path, train=True)
    model.input_layer_train(sentences)

    sentences = get_sentences(train_data, novel_flag, val=True)
    model.memory_unit_train(sentences)
    print("number of sentences for memory unit: ", len(sentences))
