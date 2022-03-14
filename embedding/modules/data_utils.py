from torch.utils.data import DataLoader,TensorDataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import RandomSampler,SequentialSampler
from tqdm import tqdm
import csv
import numpy as np

#将数据转换为bert的输入
def convert_to_bert_inputs(data, tokenizer, max_length):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for row in data:
        encoded_dict = tokenizer.encode_plus(
            row,max_length=max_length,pad_to_max_length=True,\
            return_attention_mask=True,return_tensors='pt',truncation=True)

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        #对于bert输入有token_type_ids,其他模型没有
        try:
            token_type_ids.append(encoded_dict['token_type_ids'])
        except:
            pass

    #convert lists to tensor
    input_ids = torch.cat(input_ids,dim=0)
    attention_masks = torch.cat(attention_masks,dim=0)

    if len(token_type_ids)!=0:
        token_type_ids = torch.cat(token_type_ids,dim=0)

    #训练集和测试集，有labels
    #labels = torch.tensor(labels.values)
    if len(token_type_ids)==0:
        return (input_ids,attention_masks)
    return (input_ids,attention_masks,token_type_ids)
