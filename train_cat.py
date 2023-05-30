import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
from models import *
from utils import *
from tensorboardX import SummaryWriter
from trades import *
import shutil


upper_limit, lower_limit = 1, 0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, BNeval=False):
    max_delta = torch.zeros_like(X).cuda()

    if BNeval:
        model.eval()

    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError

        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True

        iter_count = torch.zeros(y.shape[0])

        # craft adversarial examples
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()

            d = delta
            g = grad
            x = X
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g),
                                min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(
                    g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0), -
                                              1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data = d
            delta.grad.zero_()

        max_delta = delta.detach()

    if BNeval:
        model.train()

    return max_delta, iter_count


def softEntropy(inputs, targets):
    target_prob = F.softmax(targets, dim=1).detach()
    inputs_logprob = F.log_softmax(inputs, dim=1)
    loss = -1.0 * (target_prob * inputs_logprob).mean(0).sum()
    return loss


def main():
    args = get_args()
    if args.fname == 'auto':
        names = get_auto_fname(args)
        args.fname = f'result_{args.prefix}/' + names
    else:
        args.fname = f'trained_{args.prefix}/' + args.fname

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    writer = SummaryWriter(os.path.join(args.fname, 'runs'))

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Prepare data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.data == "cifar10":
        print("use cifar10")
        trainset = torchvision.datasets.CIFAR10(
            root='~/datasets/cifar10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(
            root='~/datasets/cifar10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    elif args.data == "cifar100":
        print("use cifar100")
        trainset = torchvision.datasets.CIFAR100(
            root='~/datasets/cifar100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR100(
            root='~/datasets/cifar100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Set perturbations
    epsilon = (args.epsilon / 255.)
    test_epsilon = (args.test_epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    test_pgd_alpha = (args.test_pgd_alpha / 255.)

    # Set models
    if args.model == "ResNet34":
        num_class = 10 if args.data == "cifar10" else 100
        print(f"use resnet34  {num_class}")

        model1 = ResNet34(num_classes=num_class)
        model2 = ResNet34(num_classes=num_class)

    elif args.model == 'ResNet18':
        num_class = 10 if args.data == "cifar10" else 100
        print(f"use resnet18  {num_class}")

        model1 = ResNet18(num_classes=num_class)
        model2 = ResNet18(num_classes=num_class)
    elif args.model == 'WideResNet':
        num_class = 10 if args.data == "cifar10" else 100
        print(f"use wide resnet {num_class}")

        model1 = WideResNet(depth=34, num_classes=num_class)
        model2 = WideResNet(depth=34, num_classes=num_class)
    else:
        raise ValueError("Unknown model")

    model1 = model1.cuda()
    model1.train()

    model2 = model2.cuda()
    model2.train()
    params1 = model1.parameters()
    params2 = model2.parameters()

    if args.optimizer == 'momentum':
        opt1 = torch.optim.SGD(params1, lr=args.lr_max,
                               momentum=0.9, weight_decay=args.weight_decay)
        opt2 = torch.optim.SGD(params2, lr=args.lr_max,
                               momentum=0.9, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    # Set lr schedulea
    if args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t < 100:
                return args.lr_max
            if args.lrdecay == 'base':
                if t < 150:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.

    best_test_robust_acc1 = 0
    best_test_robust_acc2 = 0
    if args.resume:
        print("-"*100)
        start_epoch = args.resume
        print("load from 99")
        model1.load_state_dict(torch.load("./resume_path/model2_99.pth"))
        model2.load_state_dict(torch.load("./resume_path/model2_99.pth"))
    else:
        start_epoch = 1

    logger.info(
        'Epoch \t Train Acc \t Train Robust Acc \t Test Acc \t Test Robust Acc')

    epochs = args.epochs
    criterion_kl = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(start_epoch, epochs+1):
        model1.train()
        model2.train()
        start_time = time.time()

        train_loss1 = 0
        train_acc1 = 0
        train_robust_loss1 = 0
        train_robust_acc1 = 0
        train_n = 0

        record_iter = torch.tensor([])

        for i, (X, y) in enumerate(trainloader):
            X, y = X.cuda(), y.cuda()

            lr = lr_schedule(epoch)
            opt1.param_groups[0].update(lr=lr)
            opt2.param_groups[0].update(lr=lr)

            ######################### traing processure###########################
            if args.attack == 'pgd':
                x_adv1 = trades_loss(model1, epoch, X, y,
                                     pgd_alpha, epsilon, args.attack_iters)

                delta2, iter_counts2 = attack_pgd(
                    model2, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, BNeval=args.BNeval)
                record_iter2 = torch.cat((record_iter, iter_counts2))
                delta2 = delta2.detach()
                x_adv2 = torch.clamp(
                    X + delta2, min=lower_limit, max=upper_limit)

            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)

            clean_logit1 = model1(X)
            adv_logit1 = model1(x_adv1)
            loss_natural = F.cross_entropy(clean_logit1, y)
            loss_robust = criterion_kl(F.log_softmax(adv_logit1, dim=1),
                                       F.softmax(clean_logit1, dim=1))
            loss1 = loss_natural + 6.0*loss_robust

            adv_logit2 = model2(x_adv2)
            clean_logit2 = model2(X)
            loss2 = 0.5*(criterion(adv_logit2, y) + criterion(clean_logit2, y)
                         ) + F.mse_loss(adv_logit2, clean_logit2)

            # soft loss
            adv_logit12 = model1(x_adv2)
            adv_logit21 = model2(x_adv1)

            loss3 = criterion_kl(F.log_softmax(adv_logit1, dim=1), F.softmax(adv_logit21, dim=1).detach()) +  \
                criterion_kl(F.log_softmax(adv_logit2, dim=1),
                             F.softmax(adv_logit12, dim=1).detach())

            robust_loss = 1.0/20*(loss1 + loss2) + 19.0/20*loss3
            opt1.zero_grad()
            opt2.zero_grad()
            robust_loss.backward()
            opt1.step()
            opt2.step()

            ###############################################################
            output1 = model1(X)
            loss11 = criterion(output1, y)

            # Record the statstic values
            train_robust_loss1 += loss1.item() * y.size(0)
            robust_output1 = adv_logit1
            train_robust_acc1 += (robust_output1.max(1)[1] == y).sum().item()
            train_loss1 += loss11.item() * y.size(0)
            train_acc1 += (output1.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        train_time = time.time()
        print('Learning rate: ', lr)

        model1.eval()
        model2.eval()

        test_loss1 = 0
        test_acc1 = 0
        test_robust_loss1 = 0
        test_robust_acc1 = 0
        test_n = 0

        test_loss2 = 0
        test_acc2 = 0
        test_robust_loss2 = 0
        test_robust_acc2 = 0
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta1, _ = attack_pgd(
                    model1, X, y, test_epsilon, test_pgd_alpha, args.attack_iters, args.restarts, args.norm)
                delta2, _ = attack_pgd(
                    model2, X, y, test_epsilon, test_pgd_alpha, args.attack_iters, args.restarts, args.norm)
            delta1 = delta1.detach()
            delta2 = delta2.detach()

            adv_input1 = torch.clamp(
                X + delta1, min=lower_limit, max=upper_limit)
            adv_input2 = torch.clamp(
                X + delta2, min=lower_limit, max=upper_limit)

            adv_input1.requires_grad = True
            robust_output1 = model1(adv_input1)

            adv_input2.requires_grad = True
            robust_output2 = model2(adv_input2)

            robust_loss1 = criterion(robust_output1, y)
            robust_loss2 = criterion(robust_output2, y)

            clean_input = X
            clean_input.requires_grad = True
            output1 = model1(clean_input)
            output2 = model2(clean_input)

            loss1 = criterion(output1, y)
            loss2 = criterion(output2, y)

            # Get the gradient norm values

            test_robust_loss1 += robust_loss1.item() * y.size(0)
            test_robust_acc1 += (robust_output1.max(1)[1] == y).sum().item()
            test_loss1 += loss1.item() * y.size(0)
            test_acc1 += (output1.max(1)[1] == y).sum().item()
            test_n += y.size(0)

            test_robust_loss2 += robust_loss2.item() * y.size(0)
            test_robust_acc2 += (robust_output2.max(1)[1] == y).sum().item()
            test_loss2 += loss2.item() * y.size(0)
            test_acc2 += (output2.max(1)[1] == y).sum().item()

        test_time = time.time()

        if epoch >= 0:
            logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                        epoch, lr, train_acc1/train_n, train_robust_acc1/train_n, test_acc1/test_n, test_robust_acc1/test_n)

            # save tesnsorboard
            writer.add_scalar(f'train/nat_loss', train_loss1 /
                              train_n, global_step=epoch)
            writer.add_scalar(f'train/nat_acc', train_acc1 /
                              train_n*100, global_step=epoch)
            writer.add_scalar(f'train/robust_loss',
                              train_robust_loss1/train_n, global_step=epoch)
            writer.add_scalar(f'train/robust_acc',
                              train_robust_acc1/train_n*100, global_step=epoch)

            writer.add_scalar(f'test/nat_loss1', test_loss1 /
                              test_n, global_step=epoch)
            writer.add_scalar(f'test/nat_acc1', test_acc1 /
                              test_n*100, global_step=epoch)
            writer.add_scalar(f'test/robust_loss1',
                              test_robust_loss1/test_n, global_step=epoch)
            writer.add_scalar(f'test/robust_acc1',
                              test_robust_acc1/test_n*100, global_step=epoch)

            writer.add_scalar(f'test/nat_loss2', test_loss2 /
                              test_n, global_step=epoch)
            writer.add_scalar(f'test/nat_acc2', test_acc2 /
                              test_n*100, global_step=epoch)
            writer.add_scalar(f'test/robust_loss2',
                              test_robust_loss2/test_n, global_step=epoch)
            writer.add_scalar(f'test/robust_acc2',
                              test_robust_acc2/test_n*100, global_step=epoch)

            torch.save(model1.state_dict(), os.path.join(
                args.fname, f'model1_last.pth'))
            torch.save(model2.state_dict(), os.path.join(
                args.fname, f'model2_last.pth'))

            if epoch == 99 or epoch == 149:
                torch.save(model1.state_dict(), os.path.join(
                    args.fname, f'model1_{epoch}.pth'))
                torch.save(model2.state_dict(), os.path.join(
                    args.fname, f'model2_{epoch}.pth'))
            # save best

            if test_robust_acc1/test_n > best_test_robust_acc1:
                torch.save({
                    'state_dict': model1.state_dict(),
                    'test_robust_acc': test_robust_acc1/test_n,
                    'test_robust_loss': test_robust_loss1/test_n,
                    'test_loss': test_loss1/test_n,
                    'test_acc': test_acc1/test_n,
                }, os.path.join(args.fname, f'model1_best.pth'))
                best_test_robust_acc1 = test_robust_acc1/test_n

            if test_robust_acc2/test_n > best_test_robust_acc2:
                torch.save({
                    'state_dict': model2.state_dict(),
                    'test_robust_acc': test_robust_acc2/test_n,
                    'test_robust_loss': test_robust_loss2/test_n,
                    'test_loss': test_loss2/test_n,
                    'test_acc': test_acc2/test_n,
                }, os.path.join(args.fname, f'model2_best.pth'))
                best_test_robust_acc2 = test_robust_acc2/test_n
        else:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                        epoch, train_time - start_time, test_time - train_time, -1,
                        -1, -1, -1, -1,
                        test_loss1/test_n, test_acc1/test_n, test_robust_loss1/test_n, test_robust_acc1/test_n)
            return
    writer.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--affix', default=None, type=str)
    parser.add_argument('--prefix', default=None, type=str)
    parser.add_argument('--data', default='cifar10', type=str)
    parser.add_argument(
        '--data-dir', default='/home/lxb/datasets/cifar10', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=[
                        'superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--attack', default='pgd', type=str,
                        choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--test_epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--test-pgd-alpha', default=2, type=float)
    parser.add_argument('--norm', default='l_inf',
                        type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    # Group 1

    parser.add_argument('--weight_decay', default=5e-4,
                        type=float)  # weight decay

    parser.add_argument('--batch-size', default=128, type=int)  # batch size
    parser.add_argument('--lrdecay', default='base', type=str,
                        choices=['intenselr', 'base', 'looselr', 'lineardecay'])

    # Group 2
    # whether use eval mode for BN when crafting adversarial examples
    parser.add_argument('--BNeval', action='store_true')
    parser.add_argument('--optimizer', default='momentum',
                        choices=['momentum', 'SGD_GC', 'SGD_GCC', 'Adam', 'AdamW'])

    return parser.parse_args()


def get_auto_fname(args):
    names = args.model + '_eps' + \
        str(args.epsilon) + '_bs' + str(args.batch_size) + \
        '_maxlr' + str(args.lr_max)
    # Group 1
    if args.weight_decay != 5e-4:
        names = names + '_wd' + str(args.weight_decay)

    # Group 2
    if args.lrdecay != 'base':
        names = names + '_' + args.lrdecay
    if args.BNeval:
        names = names + '_BNeval'
    if args.optimizer != 'momentum':
        names = names + '_' + args.optimizer
    if args.attack != 'pgd':
        names = names + '_' + args.attack
    names = names + f"_{args.epochs}" + f"_{args.data}"
    if args.affix:
        names = names + '_' + args.affix

    print('File name: ', names)
    return names


if __name__ == "__main__":
    main()
