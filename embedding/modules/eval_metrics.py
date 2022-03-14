from functools import partial
from sklearn.metrics import f1_score, precision_score, accuracy_score, roc_auc_score
from sklearn.metrics import ndcg_score
from sklearn.metrics import average_precision_score

__registered_metrics = {
    'multi_label': {
        'Macro F1': partial(f1_score, average='macro'),
        'Micro F1': partial(f1_score, average='micro'),
        'Weighted F1': partial(f1_score, average='weighted'),
        'Samples F1': partial(f1_score, average='samples')
    },
    'rank': {
        'ndcg@1': partial(ndcg_score,k=1),
        'ndcg@3': partial(ndcg_score, k=3),
        'ndcg@5': partial(ndcg_score, k=5),
        'ndcg@10': partial(ndcg_score, k=10),
        'ndcg@20': partial(ndcg_score, k=20),
        'ndcg@30': partial(ndcg_score, k=30),
        'ndcg@50': partial(ndcg_score, k=50)
    },
    'classification': {
        'f1': f1_score,
        'precision': precision_score,
        'accuracy': accuracy_score,
        'roc_auc_score': roc_auc_score
    }
}


def create_metrics(metrics_type):
    if metrics_type not in __registered_metrics:
        raise ValueError("{} not registered, must in {}".format(metrics_type, list(__registered_metrics.keys())))
    return __registered_metrics[metrics_type]
