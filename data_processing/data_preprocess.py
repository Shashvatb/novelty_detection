import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd
import nltk

nltk.download('popular', quiet=True)

from nltk.stem import WordNetLemmatizer, porter
from nltk.corpus import stopwords
from data_processing.data_extract import get_data_path
from config import text_column, relevant_flag, novel_flag


def load_data(dataset_name, year):
    path = get_data_path(dataset_name)
    file = os.path.join(path, f'{year}.parquet')
    df = pd.read_parquet(file)
    df = df.dropna(subset=[text_column])
    df[text_column] = df[text_column].apply(str)
    return df


def remove_punctuation(df):
    df[text_column] = df[text_column].replace(r'[^\w\s]', '', regex=True)
    df[text_column] = df[text_column].replace(r'_', ' ', regex=True)
    df[text_column] = df[text_column].replace(r'\s+', ' ', regex=True)
    df[text_column] = df[text_column].apply(lambda x: x.strip())
    return df


def remove_stopwords_helper(x, words):
    x = x.split()
    x = [i for i in x if i not in words]
    x = ' '.join(x)
    return x


def remove_stopwords(df):
    words = stopwords.words('english')
    words.remove('it')
    df[text_column] = df[text_column].apply(lambda x: remove_stopwords_helper(x, words))
    return df


def lemmatize_text_helper(x, lemmatizer):
    x = x.split()
    x = [lemmatizer.lemmatize(i) for i in x]
    x = ' '.join(x)
    return x


def lemmatize_text(df):
    lemmatizer = WordNetLemmatizer()
    df[text_column] = df[text_column].apply(lambda x: lemmatize_text_helper(x, lemmatizer))
    return df


def stemmer_text_helper(x, stemmer):
    x = x.split()
    x = [stemmer.stem(i) for i in x]
    x = ' '.join(x)
    return x


def stem_text(df):
    stemmer = porter.PorterStemmer()
    df[text_column] = df[text_column].apply(lambda x: stemmer_text_helper(x, stemmer))
    return df


def final_cleanup(df):
    print('Data size before cleanup: ', len(df))
    df = df.fillna('')
    df = df[df[text_column] != '']
    return df


def export_data(df, dataset_name, year):
    path = get_data_path(dataset_name)
    file = os.path.join(path, f'{year}_preprocessed.parquet')
    df.to_parquet(file)
    file = os.path.join(path, f'{year}_preprocessed.xlsx')
    df.sample(frac=1).head(50).to_excel(file, engine='xlsxwriter', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', default='2003')
    parser.add_argument('--dataset', default='trec')
    parser.add_argument('-lemmatize', action='store_true', default=True)
    parser.add_argument('-stem', action='store_true', default=False)
    parser.add_argument('-stopword', action='store_true', default=True)
    parser.add_argument('-punctuation', action='store_true', default=True)
    parser.add_argument('-lower', action='store_true', default=True)
    parser.add_argument('-keep_relevant', action='store_true', default=True)
    parser.add_argument('-keep_novel', action='store_true', default=False)

    args = parser.parse_args()

    year = args.year
    dataset_name = args.dataset

    df = load_data(dataset_name, year)

    if args.lower:
        print('lower')
        df[text_column] = df[text_column].apply(lambda x: x.lower())

    if args.punctuation:
        print('punctuation')
        df = remove_punctuation(df)

    if args.stopword:
        print('stopwords')
        df = remove_stopwords(df)

    if args.lemmatize:
        print('lemmatize')
        df = lemmatize_text(df)

    if args.stem:
        print('stem')
        df = stem_text(df)

    if args.keep_relevant:
        print('keep relevant')
        df = df[df[relevant_flag] == True]

    if args.keep_novel:
        print('keep novel')
        df = df[df[novel_flag] == True]

    df = final_cleanup(df)
    print('length of the dataframe: ', len(df))
    export_data(df, dataset_name, year)
