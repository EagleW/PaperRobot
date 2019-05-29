import torch.optim as optim


def get_optimizer(model, lr_method, lr_rate):
    """
    parse optimization method parameters, and initialize optimizer function
    """
    lr_method_name = lr_method

    # initialize optimizer function
    if lr_method_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif lr_method_name == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr_rate)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif lr_method_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr_rate)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.001)
    else:
        raise Exception('unknown optimization method.')

    return optimizer # , scheduler
