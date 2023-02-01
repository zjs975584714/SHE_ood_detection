# coding=utf-8
from __future__ import print_function
import math
from random import random
from random import seed
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import models.ResNet as ResNet
from models.autoaugment import CIFAR10Policy
from models.wrn import WideResNet

def train(model,train_loader, optimizer,scheduler,epoch):
    total_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, total=len(train_loader))
    for data,target in loop:
        data, target = data.cuda(), target.cuda()
        model = model.cuda()
        optimizer.zero_grad()
        prediction = model(data)
        critetion = nn.CrossEntropyLoss()
        loss = critetion(prediction,target)
        pred = prediction.detach().argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_description(f'Epoch [{epoch}/{args.training_epochs}]')
        total_loss /= len(train_loader.dataset)
        accuracy = 100. * correct / total
        loop.set_postfix(loss=loss.item(), acc=accuracy)
    scheduler.step(total_loss)



def valid(model, valid_loader):
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.cuda(), target.cuda()
            model = model.cuda()
            prediction = model(data)
            critetion = nn.CrossEntropyLoss()
            loss = critetion(prediction,target)
            valid_loss += loss.item()
            pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    valid_loss /= len(valid_loader.dataset)
    accuracy = 100. * correct / len(valid_loader.dataset)
    return valid_loss, correct, accuracy

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():

    # Set random seed
    random_seed = args.random_seed
    seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set training config
    batch_size = args.batch_size
    training_epochs = args.training_epochs
    learning_rate = args.learning_rate
 
    transform_train = transforms.Compose([
        transforms.Resize((args.resize_val,args.resize_val)),
        CIFAR10Policy(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.resize_val,args.resize_val)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if args.dataset=='cifar10':
        trainset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform_train)
        validset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4)
        num_class = 10
    elif args.dataset=='cifar100':
        trainset = torchvision.datasets.CIFAR100(root=args.dataset_path, train=True, download=True, transform=transform_train)
        validset = torchvision.datasets.CIFAR100(root=args.dataset_path, train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4)
        num_class = 100

    if args.model == 'resnet18':
        model = ResNet.ResNet18(num_classes=num_class)
        args.resize_val = 112
    elif args.model == 'resnet34':
        model = ResNet.ResNet34(num_classes=num_class)
        args.resize_val = 112
    elif args.model=='wrn':
        model = WideResNet(args.layers, num_class, args.widen_factor, dropRate=args.droprate)
        args.resize_val = 64
        
    if len(args.parallel_list)>0:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94,verbose=True,patience = 1,min_lr = 0.000001)

    best_acc = 0


    for epoch in range(0, training_epochs):
        if epoch in [training_epochs*2//4,training_epochs*3//4]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        train(model, train_loader, optimizer,scheduler,epoch=epoch)

        valid_loss, valid_correct, valid_accuracy = valid(model, valid_loader)
        if valid_accuracy>best_acc:
            best_acc = valid_accuracy
            torch.save(model.state_dict(),os.path.join(save_path,'test_useresize_{}_size_{}.pth'.format(args.model,args.resize_val)))
        print('valid_epoch:{}, valid_loss:{:.4f},valid_accuracy:{:.4f},Best_acc:{:.4f}'.format(epoch, valid_loss, valid_accuracy,best_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='input learning rate for training (default: 0.2)')
    parser.add_argument('--training_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--init_model_with_bias', type=int, default=1,
                        help='init model with bias as 0 or constant positive value')
    parser.add_argument('--random_seed', type=int, default=12,
                        help='input random seed for training (default: 1)')
    parser.add_argument('--model', type=str, default='wrn')
    parser.add_argument('--resize_val', type=int, default=64)
    parser.add_argument('--dataset_path', type=str, default='/data/ood_detection/data/')
    parser.add_argument('--dataset', type=str, default='cifar10',help='ID dataset')
    parser.add_argument('--parallel_list', type=str, default='0',help='give number if want parallel')
    
    #for wrn
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.parallel_list
        
    save_path = './checkpoints/{}/'.format(args.dataset)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    main()