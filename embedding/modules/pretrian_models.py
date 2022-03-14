import os
from transformers import BertTokenizer,BertModel
from transformers import RobertaModel,RobertaTokenizer
from config import pretrain_model_dir

# pretrain models and tokenizers
__registered_pretrain_models = {
    'bert-base-chinese':{
        'class':BertModel,
        'tokenizer':BertTokenizer,
        'path':'bert-base-chinese/',
    },
    'roberta-base':{
        'class':RobertaModel,
        'tokenizer':RobertaTokenizer,
        'path':'roberta-base/'
    }
}

def load_pretrain_model(model_name):
    if model_name not in __registered_pretrain_models:
        raise ValueError(" pretrain model {} not registered, must in {} ".format(model_name,list(__registered_pretrain_models.keys())))
    model_config = __registered_pretrain_models[model_name]
    model = model_config["class"].from_pretrained(os.path.join(pretrain_model_dir,model_config["path"]))
    return model

def load_pretrian_tokenizer(model_name):
    if model_name not in __registered_pretrain_models:
        raise ValueError(" pretrain model {} not registered, must in {} ".format(model_name,list(__registered_pretrain_models.keys())))
    model_config = __registered_pretrain_models[model_name]
    tokenizer = model_config["tokenizer"].from_pretrained(os.path.join(pretrain_model_dir,model_config["path"]))
    return tokenizer