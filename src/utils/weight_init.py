from torch import nn


def init_weight(model: nn.Module, std=0.01):
    model.weight.data.normal_(std=std)

    try:
        model.bias.data.normal_(std=std)
    except:
        return model

    return model


def freeze_weight(model: nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False
