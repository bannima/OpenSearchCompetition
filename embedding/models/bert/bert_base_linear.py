import torch.nn as nn
from embedding.modules.pretrian_models import load_pretrain_model

class BertBaseLinear(nn.Module):
    """ Implementation of Bert-base-chinese with linear output model """

    def __init__(self,output_size,model_name="bert-base-chinese"):
        super(BertBaseLinear, self).__init__()
        self.bert_base = load_pretrain_model(model_name)
        self.linear = nn.Linear(768,output_size)

    def forward(self,x):
        cls_embs = self.bert_base(*x)[0][:, 0, :].squeeze(1)
        return self.linear(cls_embs)
