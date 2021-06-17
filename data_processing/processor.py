import argparse
import os


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
                temp = [j for j in file_list if j.startswith('N' + str(i))]
                assert len(temp) == 1
                files.append(temp[0])
        else:
            raise Exception('Invalid Year')
    # TODO RTE data
    return files


def get_metadata(data_path, dataset_name, year, novel=False):
    if dataset_name == 'trec':
        if year == '2004':
            if novel:
                file_name = os.path.join(data_path, 'novel_sentences_2004.txt')
            else:
                file_name = os.path.join(data_path, 'relavant_sentences_2004.txt')
        if year == '2003':
            if novel:
                file_name = os.path.join(data_path, 'novel_sentences_2003.txt')
            else:
                file_name = os.path.join(data_path, 'relavant_sentences_2003.txt')
        with open(file_name, 'r') as f:
            content = f.readlines()
        metadata = []
        for i in content:
            temp = i.strip().split()
            metadata.append([temp[0], temp[1].split(':')[0], temp[1].split(':')[1]])

        return metadata
    # TODO RTE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', default='2004')
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

