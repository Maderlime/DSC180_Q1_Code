
import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import random

# dr_mean = (0.4914, 0.4822, 0.4465)
# dr_std = (0.2471, 0.2435, 0.2616)

# WILL NEED TO COMMENT OUT BASED ON THE DATASET USED

dr_mean = (0.4175977, 0.29157516, 0.20427993)
dr_std = (0.39969242, 0.39826953, 0.3930208)

# dr_mean = (0.9764, 0.9729, 0.9738)
# dr_std = (0.0121, 0.01076, 0.01035)
mu = torch.tensor(dr_mean).view(3,1,1).cuda()
std = torch.tensor(dr_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

def npy_loader(path):
    return torch.from_numpy(np.load(path))

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.Normalize(dr_mean, dr_std),
    ])
    test_transform = transforms.Compose([
        transforms.Normalize(dr_mean, dr_std),
    ])
    num_workers = 2
#     train_dataset = datasets.CIFAR10(
#         dir_, train=True, transform=train_transform, download=True)
#     test_dataset = datasets.CIFAR10(
#         dir_, train=False, transform=test_transform, download=True)
    train_dataset = datasets.DatasetFolder(root=dir_, loader = npy_loader, transform=train_transform, extensions = ('.npy'))
    test_dataset = datasets.DatasetFolder(root=dir_+"_test", transform=test_transform, loader = npy_loader, extensions = ('.npy'))
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
#         pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
#         pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    # Edit here to run different attack epsilons
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
#     t = list(enumerate(test_loader))
#     random.shuffle(t)         
    print ("STARTING PGD EVALUATION on epsilon: ", epsilon)
    for i, (X, y) in enumerate(test_loader):
        print("i: ", i)
        try:
            X, y = X.cuda(), y.cuda()
            
            print("X shape: ", X.shape)
            
            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
#             print("pgd delta: ", pgd_delta)
            with torch.no_grad():
                output = model(X + pgd_delta)
                
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                print("Output: ", output.max(1)[1])
                print("y shape: ", y)
                n += y.size(0)
        except Exception as e:
            print ("Error: ", e)
            print ("Stopped at: ", n)
            return pgd_loss/n, pgd_acc/n
        print ("Current PGD accuracy: ", pgd_acc/n)
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    print ("STARTING STANDARD EVALUATION")
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            print("i: ", i)
            X, y = X.cuda(), y.cuda()
            print("X shape: ", X.shape)
            print("y shape: ", y.shape)
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            print ("Current standard accuracy: ", test_acc/n)
    return test_loss/n, test_acc/n