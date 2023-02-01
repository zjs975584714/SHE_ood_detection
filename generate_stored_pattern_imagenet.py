from random import seed
import numpy as np
import os
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models.ResNet_with_pretrained import resnet18,resnet34,resnet50
from models.wrn import WideResNet
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--batch_size', type=int, default=1280)
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name.')
parser.add_argument('--resize_val', default=224, type=int, help='transform resize length')
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--parallel_list', type=str, default='0,1,2,3',help='give number if want parallel')
parser.add_argument('--stored_data_path', type=str, default='/data/ood_detection/data/', help='the path for storing data.')
parser.add_argument('--num_class', type=int, default=1000)
parser.add_argument('--score', default='SHE', type=str, help='score options:MSP|Energy|ReAct|SHE')

args = parser.parse_args()
print(args)
random_seed = 12

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

train_dataset_path = '/data/imagenet/train/'
val_dataset_path = './data/val/'

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.opt = opt

        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    print(k,type(k))
                    self.batch[k] = self.batch[k].to(device="cuda:0", non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            if args.fp16:
                self.next_input = self.next_input.half()
            else:
                self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

train_set = torchvision.datasets.ImageFolder(
    train_dataset_path,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),         
        transforms.ToTensor(),                          
        transforms.Normalize(mean=[0.485, 0.456, 0.406],# x = (x - mean(x))/std(x)
                             std=[0.229, 0.224, 0.225]),
    ]))

valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])


valid_set = torchvision.datasets.ImageFolder(
    val_dataset_path,
    valid_transform
)
train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loader = DataLoaderX(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=8)


if args.model == 'resnet50':
    model = resnet50(num_classes=1000)


# url: https://download.pytorch.org/models/resnet50-0676ba61.pth (you can download it into the corresponding path)
pre_path = './checkpoints/imagenet/{}.pth'.format(args.model)
model.load_state_dict(torch.load(pre_path))

net = nn.DataParallel(model).cuda()
net.eval()


def generate_avg_feature(path):
    final_feature_list = [None for i in range(args.num_class)]
    
    with torch.no_grad():
        cur_num = 0
        for data,target in train_loader:
            cur_num += data.size(0)
            feature_list = [None for i in range(args.num_class)]
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
                if feature_list[i] is None:
                    continue
                elif final_feature_list[i] is None:
                    final_feature_list[i] = torch.mean(feature_list[i],dim=0,keepdim=True)
                else:
                    feature_list[i] = torch.mean(feature_list[i],dim=0,keepdim=True)
                    final_feature_list[i] = torch.cat((final_feature_list[i],feature_list[i]),dim=0)
                    final_feature_list[i] = torch.mean(final_feature_list[i],dim=0,keepdim=True)

    for i in range(args.num_class):
        torch.save(final_feature_list[i],os.path.join(path,'_stored_avg_class_{}.pth'.format(i)))


if __name__ == '__main__':

    path = './stored_pattern/avg_stored_pattern/size_{}/{}/{}/'.format(args.resize_val,args.dataset,args.model)
    
    if not os.path.exists(path):
        os.makedirs(path)

    generate_avg_feature(path)

