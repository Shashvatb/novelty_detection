import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd())
from modelling.model import MemNet
from data_processing.data_preprocess import load_data
from config import text_column


def get_sentences(df):
    sentences = df[text_column].tolist()
    return sentences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', default='2004')
    parser.add_argument('--dataset', default='trec')
    args = parser.parse_args()

    year = args.year + '_preprocessed'
    dataset_name = args.dataset

    df = load_data(dataset_name, year)
    sentences = get_sentences(df)

    model = MemNet()
    model.input_layer_train(sentences)
    print(model.input_layer(np.array(sentences[:2])))