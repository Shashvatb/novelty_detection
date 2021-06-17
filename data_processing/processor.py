import argparse
import os
from bs4 import BeautifulSoup
import pandas as pd


def get_data_path(dataset_name):
    if dataset_name.lower() == 'trec':
        return os.path.join(
            os.getcwd(), '..', '..', 'data', 'trec'
        )
    # TODO RTE


def get_file_list(data_path, year, dataset_name):
    file_list = os.listdir(data_path)
    files = []
    if dataset_name == 'trec':
        if year == '2004':
            indices = list(range(51, 101))
            for i in indices:
                temp = [j for j in file_list if j.startswith(str(i))]
                assert len(temp) == 1
                files.append(temp[0])

        elif year == '2003':
            indices = list(range(1, 51))
            for i in indices:
                temp = [j for j in file_list if j.startswith('N' + str(i) + '.')]
                assert len(temp) == 1
                files.append(temp[0])
        else:
            raise Exception('Invalid Year')
        files = [os.path.join(data_path, i) for i in files]
    # TODO RTE data
    return files


def get_metadata(data_path, dataset_name, year, novel=False):
    if dataset_name == 'trec':
        if year == '2004':
            if novel:
                file_name = os.path.join(data_path, 'novel_sentences_2004.txt')
            else:
                file_name = os.path.join(data_path, 'relevant_sentences_2004.txt')
        if year == '2003':
            if novel:
                file_name = os.path.join(data_path, 'novel_sentences_2003.txt')
            else:
                file_name = os.path.join(data_path, 'relevant_sentences_2003.txt')
        with open(file_name, 'r') as f:
            content = f.readlines()
        metadata = []
        for i in content:
            temp = i.strip().split()
            metadata.append([temp[0], temp[1].split(':')[0], temp[1].split(':')[1]])

        return metadata
    # TODO RTE


def convert_data_to_df(file_list, dataset, year):
    df = pd.DataFrame(columns=['text', 'docid', 'line_number', 'file'])
    for file in file_list:
        with open(file, 'r') as f:
            data = f.read()

        if dataset == 'trec':
            if year == '2004':
                file = 'N' + file.split('\\')[-1].split('.')[0]
            elif year == '2003':
                file = file.split('\\')[-1].split('.')[0]
        # TODO RTE

        data = BeautifulSoup(data, "lxml")
        data = data.find_all('s')
        for i in data:
            df2 = {'text': i.text, 'docid': i.get('docid'), 'line_number': i.get('num'), 'file': file}
            df = df.append(df2, ignore_index=True)
    return df


def add_flag(df, metadata, flag_column):
    df[flag_column] = False
    for i in metadata:
        df.loc[(df["file"] == i[0]) & (df["docid"] == i[1]) & (df["line_number"] == i[2]), flag_column] = True
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', default='2003')
    parser.add_argument('--dataset', default='trec')

    args = parser.parse_args()

    year = args.year
    dataset_name = args.dataset

    # Create the path for the data
    data_path = get_data_path(dataset_name)

    # Get the list of all the required files
    file_list = get_file_list(data_path, year, dataset_name)

    # Get metadata
    relevant_data_meta = get_metadata(data_path, dataset_name, year)
    novel_data_meta = get_metadata(data_path, dataset_name, year, novel=True)

    # Convert data to pandas dataframe
    df = convert_data_to_df(file_list, dataset_name, year)

    # Add flags for relevant and novel data
    df = add_flag(df, relevant_data_meta, 'relevant_flag')
    df = add_flag(df, novel_data_meta, 'novel_flag')

    # Export data
    df = df.drop_duplicates()
    df.to_parquet(os.path.join(data_path, year + '.parquet'))
