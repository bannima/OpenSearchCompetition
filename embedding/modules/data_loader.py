import os
import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from abc import ABCMeta,ABC
import numpy as np
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from common.utils import load_data
from config import random_state,logger

class OpenSearchDataset(Dataset):
    def __init__(self,data,target):
        super(OpenSearchDataset, self).__init__()
        self.query = data['query'].tolist()
        self.title = data['title'].tolist()
        self.target = target.tolist()

    def __getitem__(self, idx):
        #x = Bunch()
        x = self.query[idx] + "[SEP]" +self.title[idx]
        return x, self.target[idx]

    def __len__(self):
        if self.target is not None:
            return len(self.target)
        elif self.query is not None:
            return len(self.query)
        else:
            raise ValueError("dataset length not recognized")

class BaseCorpus(ABC):
    """ base corpus template """
    def __init__(self,batch_size):
        super(BaseCorpus, self).__init__()
        self.batch_size = batch_size
    @property
    def train_loader(self):
        return None
    @property
    def valid_loader(self):
        return None
    @property
    def test_loader(self):
        return None

class Corpus(BaseCorpus):
    """ corpus implementation of Open Search Dataset """
    def __init__(self,batch_size=64):
        super(Corpus, self).__init__(batch_size)
        logger.info(" start preprocess open search dataset ")
        opensearch_dataset = load_data()
        # positive samples
        opensearch_dataset["qrels.train"]['label']=1
        # construct random negative samples randomly
        faked_doc_id = opensearch_dataset["qrels.train"]['doc_id'].sample(frac=1.0,random_state=random_state,ignore_index=True)
        faked_train= pd.concat([opensearch_dataset["qrels.train"]['query_id'],faked_doc_id],axis=1)
        faked_train['label'] = 0
        dataset = pd.concat([opensearch_dataset["qrels.train"],faked_train],axis=0)
        # random reorder
        dataset = dataset.sample(frac=1.0)
        dataset = pd.merge(dataset, opensearch_dataset["train.query"], on="query_id")
        self.dataset = pd.merge(dataset, opensearch_dataset["corpus"], on="doc_id").reset_index()

        logger.info(" construct negative samples done")

        labels = self.dataset['label']
        data = self.dataset.drop(labels=['label'],axis=1)
        # train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data,
            labels,
            test_size=0.1,
            random_state=random_state,
            shuffle=True
        )
        logger.info(" train & test split")

        self._train_loader, self._validation_loader, self._test_loader = None, None, None

    @property
    def train_loader(self):
        if self._train_loader is None:
            self._train_loader = DataLoader(
                dataset=OpenSearchDataset(data =self.X_train,target=self.y_train),
                batch_size=self.batch_size,
                num_workers=1,
                shuffle=True
            )
        return self._train_loader

    @property
    def test_loader(self):
        if self._test_loader is None:
            self._test_loader = DataLoader(
                dataset=OpenSearchDataset(data = self.X_test, target=self.y_test),
                batch_size=self.batch_size,
                num_workers=1,
                shuffle=True
            )
        return self._test_loader



