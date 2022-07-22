from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, config, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model,config, skip_list=(), skip_keywords=()):
    no_decay = []
    lr_1x, lr_1x_nodecay = model.get_1x_lr_params_NOscale()
    lr_10x, lr_10x_nodecay = model.get_10x_lr_params()

    return [{'params': lr_1x_nodecay, 'lr':config.TRAIN.BASE_LR, 'weight_decay': 0.},
            {'params': lr_10x_nodecay, 'lr':2 * config.TRAIN.BASE_LR, 'weight_decay': 0.},
            {'params': lr_1x, 'lr':config.TRAIN.BASE_LR},
            {'params': lr_10x, 'lr':2 * config.TRAIN.BASE_LR}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
    
