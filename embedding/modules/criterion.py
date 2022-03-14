import torch.nn as nn

__registered_loss = {
    'BCEWithLogitsLoss': {
        "cls": nn.BCEWithLogitsLoss,
        'intro': ''
    },
    'KLDivLoss': {
        'cls': nn.KLDivLoss,
        'intro': ''
    },
    'MultiLabelMarginLoss': {
        "cls": nn.MultiMarginLoss,
        'intro': ''
    },
    'CrossEntropy': {
        'cls': nn.CrossEntropyLoss,
        'intro': ''
    },
    'BCELoss': {
        'cls': nn.BCELoss,
        'intro': ''
    }
}


def create_loss(loss_type):
    if loss_type not in __registered_loss:
        raise ValueError("{} not registered, must in {}".format(loss_type, list(__registered_loss.keys())))
    return __registered_loss[loss_type]['cls']

