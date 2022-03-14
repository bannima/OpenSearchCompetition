import os
from functools import partial
from embedding.modules.data_loader import Corpus
from config import logger
from common.trainer import Trainer
from common.utils import parse_parmas
from embedding.models import BertBaseLinear
from embedding.modules.pretrian_models import load_pretrian_tokenizer
from embedding.modules.data_utils import convert_to_bert_inputs


def bert_linear_experiment(HYPERS):
    logger.info(" try bert-base-chinese with linear output ")
    pretrain_model_name = "bert-base-chinese"
    tokenizer = load_pretrian_tokenizer(pretrain_model_name)

    #1. load the open search dataset
    corpus = Corpus(batch_size=HYPERS['Batch'])

    # 2. prepare BertBaseLinear model
    bert_linear = BertBaseLinear(
        model_name=pretrain_model_name,
        output_size=2
    )
    logger.info(" BertBaseLinear initialized ")

    # 3. train mode use Single Task Trainer
    result_path = os.path.join(os.path.dirname(__file__), 'results')
    trainer = Trainer(
        model=bert_linear,
        dataloaders=(corpus.train_loader,corpus.valid_loader,corpus.test_loader),
        data_converter=partial(convert_to_bert_inputs, tokenizer=tokenizer, max_length=100),
        result_path=result_path,
        HYPERS=HYPERS
    )
    epoch_stats_file = trainer.fit()


if __name__ == '__main__':
    HYPERS = parse_parmas()
    HYPERS['Epochs'] = 10
    HYPERS['LearningRate'] = 2e-3
    HYPERS['Batch'] = 16
    HYPERS['Save_Model'] = False
    HYPERS['Criterion'] = 'CrossEntropy'

    bert_linear_experiment(HYPERS)



