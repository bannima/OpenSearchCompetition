import os
import pandas as pd
from common.utils import load_data

dataset = load_data()

def analysis_query_doc(dataset):
    merge_data = pd.merge(dataset["qrels.train"],dataset["train.query"],on="query_id")
    merge_data = pd.merge(merge_data, dataset["corpus"], on="doc_id")
    return merge_data


if __name__ == '__main__':
    data = analysis_query_doc(dataset)
    print(dataset)