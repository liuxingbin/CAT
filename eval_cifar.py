import argparse
import copy
import logging
import os
import time
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import advertorch
from advertorch.attacks import LinfSPSAAttack, spsa

from models import *
from utils_plus import (upper_limit, lower_limit, clamp, get_loaders,
                        attack_pgd, evaluate_pgd, evaluate_standard)
from autoattack import AutoAttack
# installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--head', type=int, default=0)
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--batch-size', default=500, type=int)
    parser.add_argument("--type", default="best", type=str)
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--data-dir', default="~/datasets/", type=str)
    parser.add_argument('--epsilon', default=8, type=float)
    parser.add_argument('--out-dir', default='train_fgsm_output',
                        type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler()
        ])

    logger.info(args)

    _, test_loader = get_loaders(
        args.data_dir, args.batch_size, args.data)

    path = os.path.join(args.out_dir, f'model_{args.type}.pth')
    best_state_dict = torch.load(path)
    print(os.path.join(path))
    print(args.head)
    print("epsilon: ", args.epsilon)

    if args.model == 'ResNet18':
        model = ResNet18(num_classes=10 if args.data == "cifar10" else 100)
    elif args.model == 'WideResNet':
        print("use wide resnet")
        model = WideResNet(
            depth=34, num_classes=10 if args.data == "cifar10" else 100)
    elif args.model == "vgg":
        num_class = 10 if args.data == "cifar10" else 100
        print(f"use wide vgg {num_class}")
        model = VGG('VGG16')

    elif args.model == "mobilenet":
        num_class = 10 if args.data == "cifar10" else 100
        print(f"use wide mobilenet {num_class}")

        model = MobileNet()
    else:
        raise NotImplementedError

    model_test = model.cuda()
    if 'state_dict' in best_state_dict.keys():
        state_dict = {}
        for k, v in best_state_dict['state_dict'].items():
            if "module." in k:
                state_dict[k[len('module.'):]] = v
            else:
                state_dict[k] = v
        model_test.load_state_dict(state_dict)
    else:
        model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    black_model = None

    ## Evaluate clean acc ###
    _, test_acc, clean_list = evaluate_standard(test_loader, model_test)
    print('Clean acc: ', test_acc)

    #### evaluate FGSM (CE loss) acc####
    _, fgsm_acc_CE, fgsm_list = evaluate_pgd(
        test_loader, model_test, attack_iters=1, restarts=1, step=args.epsilon, use_CWloss=False, random_init=False, black_model=black_model)
    print('FGSM acc: ', fgsm_acc_CE)

    ### Evaluate PGD 20 (CE loss) acc ###
    _, pgd_acc_CE, pgd_list = evaluate_pgd(
        test_loader, model_test, attack_iters=20, restarts=1, eps=args.epsilon, step=2, use_CWloss=False, black_model=black_model)
    print('PGD-20 (1 restarts, step 2, CE loss) acc: ', pgd_acc_CE)

    #  Evaluate PGD 40(CE loss) acc ###
    _, pgd_acc_CE, pgd_list = evaluate_pgd(
        test_loader, model_test, attack_iters=40, restarts=1, eps=args.epsilon, step=2, use_CWloss=False, black_model=black_model)
    print('PGD-40 (1 restarts, step 2, CE loss) acc: ', pgd_acc_CE)

    # Evaluate PGD (CW loss) acc ###
    _, pgd_acc_CW, cw_list = evaluate_pgd(
        test_loader, model_test, attack_iters=20, restarts=1,
        eps=args.epsilon, step=2, use_CWloss=True, black_model=black_model)
    print('PGD-20 (1 restarts, step 2, CW loss) acc: ', pgd_acc_CW)

    ###Evaluate AutoAttack ###
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    epsilon = 8 / 255.
    adversary = AutoAttack(model_test, norm='Linf',
                           eps=epsilon, version='standard')
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=500)

    X_adv = X_adv.cuda()
    y_test = y_test.cuda()
    attack_list = []
    for index in range(X_adv.size(0)//256 + 1):
        if 256*(index+1) > X_adv.size(0):
            x = X_adv[index*256:]
            y = y_test[index*256:]
        else:
            x = X_adv[index*256:256*(index+1)]
            y = y_test[index*256:256*(index+1)]
        output = model(x)
        attack_list.extend((output.max(1)[1] == y).cpu().numpy())

    print(np.mean(attack_list))

    print("test over")


if __name__ == "__main__":
    main()
