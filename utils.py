import pandas as pd
import os
from config import novel_flag, unique_ids, text_column
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from theano.tensor import _shared


def load_featurizer():
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
    return tokenizer, model


def gen_observations(data, tokenizer, model):
    print('list of columns: ', list(data.columns))
    labels = data[novel_flag].tolist()
    ids = data[unique_ids].tolist()
    data = data[text_column].tolist()
    result = []
    for i in range(len(data)):
        inputs = tokenizer(data[i], return_tensors="pt")
        outputs = model(**inputs, labels=labels[i])
        result += _shared(outputs[0].detach().numpy())

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
    print('Length of final df: ',len(df))

    return df
