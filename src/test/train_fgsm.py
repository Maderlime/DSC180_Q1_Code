import argparse
import copy
import logging
import os
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/model')
sys.path.insert(0, 'src/test')
from preact_resnet_orig import PreActResNet18
from utils_orig import upper_limit, lower_limit, std, clamp, get_loaders, attack_pgd, evaluate_pgd, evaluate_standard

logger = logging.getLogger(__name__)


def main():
    
    batch_size = 1
    data_dir = '../../test/cxr'
    epochs = 15
    lr_schedule = 'cyclic'
    lr_min = float(0.)
    lr_max = 0.04
    weight_decay = 5e-4
    momentum = 0.9
    epsilon = 8
    alpha = 10
    delta_init = 'random'
    out_dir = 'train_fgsm_output'
    seed = 0
    early_stop = 'store_true'
    opt_level = 'O2'
    loss_scale = '1.0'
    master_weights = 'store_true'
    
    args = "    batch_size = 1; \
    data_dir = '../../cifar-data'; \
    epochs = 15; \
    lr_schedule = 'cyclic'; \
    lr_min = float(0.); \
    lr_max = 0.04; \
    weight_decay = 5e-4; \
    momentum = 0.9; \
    epsilon = 8; \
    alpha = 10; \
    delta_init = 'random'; \
    out_dir = 'train_fgsm_output'; \
    seed = 0; \
    early_stop = 'store_true'; \
    opt_level = 'O2'; \
    loss_scale = '1.0'; \
    master_weights = 'store_true'"
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    logfile = os.path.join(out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(out_dir, 'output.log'))
    logger.info(args)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_loader, test_loader = get_loaders(data_dir, batch_size)

    epsilon = (epsilon / 255.) / std
    alpha = (alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    model = PreActResNet18().cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay)
    amp_args = dict(opt_level=opt_level, loss_scale=loss_scale, verbosity=False)
    if opt_level == 'O2':
        amp_args['master_weights'] = master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    if delta_init == 'previous':
#         delta = torch.zeros(batch_size, 3, 32, 32).cuda()
        delta = torch.zeros(batch_size, 3, 224, 224).cuda()

    lr_steps = epochs * len(train_loader)
    if lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_min, max_lr=lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)
            if delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            if delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            output = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)
            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        if early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
    train_time = time.time()
    if not early_stop:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, os.path.join(out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)

def train_model():
    main()
    print("FGSM model trained")
if __name__ == "__main__":
    main()
