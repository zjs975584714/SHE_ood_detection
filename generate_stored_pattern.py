from random import seed
from torch.autograd import Variable
import numpy as np
import os
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
from models import ResNet 
from models.wrn import WideResNet
parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset name.')
parser.add_argument('--resize_val', default=112, type=int, help='transform resize length')
parser.add_argument('--model', type=str, default='resnet34')
parser.add_argument('--parallel_list', type=str, default='0',help='give number if want parallel')
parser.add_argument('--stored_data_path', type=str, default='/data/ood_detection/data/', help='the path for storing data.')
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--score', default='SHE', type=str, help='score options:MSP|Energy|ReAct|HE|SHE')

args = parser.parse_args()
print(args)



if args.model == 'resnet18' or 'resnet34':
    args.resize_val = 112
else:
    args.resize_val = 64
    
dataset_path = args.stored_data_path
random_seed = 12
if args.dataset == 'cifar10':
    args.num_class = 10 
elif args.dataset == 'cifar100':
    args.num_class = 100 


np.set_printoptions(threshold=np.inf)
os.environ['CUDA_VISIBLE_DEVICES'] = args.parallel_list
cudnn.benchmark = True  # fire on all cylinders

# Set random seed
seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

print('Size of sample is {}*{}'.format(args.resize_val,args.resize_val))
transform_all = trn.Compose([
    trn.Resize((args.resize_val,args.resize_val)),
    trn.ToTensor(),
    trn.Normalize(mean, std),
])



if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root=args.stored_data_path, train=True, download=True, transform=transform_all)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_data = torchvision.datasets.CIFAR10(root=args.stored_data_path, train=False, download=True, transform=transform_all)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root=args.stored_data_path, train=True, download=True, transform=transform_all)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_data = torchvision.datasets.CIFAR100(root=args.stored_data_path, train=False, download=True, transform=transform_all)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)



if args.model == 'resnet18':
    net = ResNet.ResNet18(num_classes=args.num_class)
elif args.model == 'resnet34':
    net = ResNet.ResNet34(num_classes=args.num_class)
elif args.model=='wrn':
    net = WideResNet(args.layers, args.num_class, args.widen_factor, dropRate=args.droprate)



net = nn.DataParallel(net).cuda()

PATH = './checkpoints/{}/test_useresize_{}_size_{}.pth'.format(args.dataset,args.model,args.resize_val)
net.load_state_dict(torch.load(PATH,map_location=None))
net.eval()





def generate_avg_feature(path):
    # SHE: use the penultimate layer output as the stored
    feature_list = [None for i in range(args.num_class)]
    
    with torch.no_grad():
        for data,target in train_loader:
            data, target = data.cuda(), target.cuda()
            prediction,penultimate = net(data,need_penultimate=4)
            pred_val,pred = torch.max(prediction,dim=1)
            correct_index =  pred.eq(target.view_as(pred))
            for i in range(args.num_class):
                each_label_tensor = torch.tensor([i for _ in range(target.size(0))]).cuda()
                target_index = pred.eq(each_label_tensor.view_as(pred))
                combine_index = correct_index * target_index
                each_label_feature = penultimate[combine_index]
                if each_label_feature.size(0) == 0: continue
                if feature_list[i] is None:
                    feature_list[i] = each_label_feature
                else:
                    feature_list[i] = torch.cat((feature_list[i],each_label_feature),dim=0)
    for i in range(args.num_class):
        feature_list[i] = torch.mean(feature_list[i],dim=0,keepdim=True)
        torch.save(feature_list[i],os.path.join(path,'stored_avg_class_{}.pth'.format(i)))

def generate_all_feature(path):
    # HE: use all stored pattern to calculate the HE score
    feature_list = [None for i in range(args.num_class)]
    
    with torch.no_grad():
        for data,target in train_loader:
            data, target = data.cuda(), target.cuda()
            prediction,penultimate = net(data)
            pred_val,pred = torch.max(prediction,dim=1)
            correct_index =  pred.eq(target.view_as(pred))
            for i in range(args.num_class):
                each_label_tensor = torch.tensor([i for _ in range(target.size(0))]).cuda()
                target_index = pred.eq(each_label_tensor.view_as(pred))
                combine_index = correct_index * target_index
                each_label_feature = penultimate[combine_index]
                if each_label_feature.size(0) == 0: continue
                if feature_list[i] is None:
                    feature_list[i] = each_label_feature
                else:
                    feature_list[i] = torch.cat((feature_list[i],each_label_feature),dim=0)
    for i in range(args.num_class):
        torch.save(feature_list[i],os.path.join(path,'stored_all_class_{}.pth'.format(i)))




if __name__ == '__main__':

    if args.score == 'HE':
        path = './stored_pattern/all_stored_pattern/size_{}/{}/{}/'.format(args.resize_val,args.dataset,args.model)
    else:
        path = './stored_pattern/avg_stored_pattern/size_{}/{}/{}/'.format(args.resize_val,args.dataset,args.model)
        
    if not os.path.exists(path):
        os.makedirs(path)

    if args.score == 'HE':
        generate_all_feature(path)
        print('HE patterns have been done!')
    else:
        generate_avg_feature(path)
        print('SHE patterns have been done!')