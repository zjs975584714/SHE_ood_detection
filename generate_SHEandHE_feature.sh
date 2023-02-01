#!/bin/bash  

for dataset in 'cifar10' 'cifar100'
do
    for model in 'resnet18' 'resnet34' 'wrn'
    do
        python ./generate_stored_pattern.py --model $model --dataset $dataset --score 'HE'
    done
done

for dataset in 'cifar10' 'cifar100'
do
    for model in 'resnet18' 'resnet34' 'wrn'
    do
        python ./generate_stored_pattern.py --model $model --dataset $dataset --score 'SHE'
    done
done