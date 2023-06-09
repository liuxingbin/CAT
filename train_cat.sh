CUDA_VISIBLE_DEVICES=0 python train_cat.py \
    --model ResNet18 \
    --lr-max 0.1 \
    --data cifar10 \
    --attack pgd \
    --lr-schedule piecewise \
    --norm l_inf \
    --epsilon 8 \
    --test_epsilon 8 \
    --epochs 200 \
    --attack-iters 10 \
    --pgd-alpha 2 \
    --fname auto \
    --optimizer 'momentum' \
    --weight_decay 5e-4 \
    --batch-size 128 \
    --BNeval