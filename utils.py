import pandas as pd
import os
from config import novel_flag, unique_ids, text_column
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from theano.tensor import _shared
import torch
import numpy as np

device = torch.device('cuda')


def load_featurizer():
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news").to(device)
    return tokenizer, model


def gen_observations(data, tokenizer, model):
    # print(model)
    labels = data[novel_flag].tolist()
    labels = [torch.from_numpy(np.array(int(i))).to(device) for i in labels]
    ids = data[unique_ids].tolist()
    data = data[text_column].tolist()
    result = []
    for i in range(len(data)):
        inputs = tokenizer(data[i], return_tensors="pt").to(device)
        outputs = model(**inputs, labels=labels[i])
        result += _shared(outputs[0].cpu().detach().numpy())
        if i == 0:
            print(result[0].shape)
            print(result)

    assert len(result) == len(labels)
    assert len(result) == len(ids)
    return result, labels, ids


def load_data(path):
    df = pd.DataFrame()
    print(path)
    for i in path:
        print(os.path.exists(i))
        df_temp = pd.read_parquet(i)
        print('length of data: ', len(df_temp))
        df = df.append(df_temp)
    print('Length of final df: ', len(df))

    return df
