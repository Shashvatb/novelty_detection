import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.getcwd())

from data_processing.data_extract import get_data_path


def get_preprocessed_file_list(dataset_name):
    if dataset_name == 'trec':
        files = ['2003_preprocessed.parquet',
                 '2004_preprocessed.parquet']
    # TODO RTE
    else:
        files = []
    return files


def read_data(dataset_name):
    file_list = get_preprocessed_file_list(dataset_name)
    data_path = get_data_path(dataset_name)

    df = pd.DataFrame()
    for i in file_list:
        temp = pd.read_parquet(os.path.join(data_path, i))
        df = df.append(temp)
        del temp
    df = df.drop_duplicates().reset_index().drop('index', axis=1)
    print(df.columns)
    print(df.shape)

    return df


def segregate_data(df, flag, rel_flag):
    assert len(df[rel_flag].unique()) == 1
    df_n = df[df[flag] == True].drop_duplicates()
    df_nn = df[df[flag] == False].drop_duplicates()
    print('len of df_n: ', len(df_n))
    print('len of df_nn: ', len(df_nn))
    del df
    return df_n, df_nn


def create_test_train_data(df_novel, df_non_novel, fraction):
    l = len(df_novel)
    train = df_novel.sample(frac=fraction)
    test = df_novel.drop(train.index)
    test_length = len(test)
    assert l-test_length == len(train)

    validate = test.sample(frac=0.5)
    test = test.drop(validate.index)

    validate_temp = df_non_novel.sample(n=test_length // 2)
    validate = validate.append(validate_temp)
    df_non_novel = df_non_novel.drop(validate_temp.index)
    test = test.append(df_non_novel.sample(n=test_length//2))

    print('Size of Train data: ', len(train))
    print('Size of validate data: ', len(validate))
    print('Size of Test data: ', len(test))
    return train, validate, test


def export_test_train_df(df_train, df_validate, df_test, dataset):
    path = get_data_path(dataset)
    df_train.to_parquet(os.path.join(path, 'train.parquet'))
    df_validate.to_parquet(os.path.join(path, 'validate.parquet'))
    df_test.to_parquet(os.path.join(path, 'test.parquet'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='trec')
    parser.add_argument('--train_fraction', default='0.9')

    args = parser.parse_args()
    dataset_name = args.dataset
    fraction = float(args.train_fraction)

    df = read_data(dataset_name)
    df_novel, df_non_novel = segregate_data(df, 'novel_flag', 'relevant_flag')
    del df

    df_train, df_validate, df_test = create_test_train_data(df_novel, df_non_novel, fraction)
    del df_novel, df_non_novel
    export_test_train_df(df_train, df_validate, df_test, dataset_name)
