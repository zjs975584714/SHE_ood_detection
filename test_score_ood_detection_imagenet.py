from random import seed
import numpy as np
import os
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.ResNet_with_pretrained import resnet50
from Utils.display_results import get_measures, print_measures, print_measures_with_std
import Utils.score_calculation as lib
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--batch_size', type=int, default=1280)
parser.add_argument('--num_class', type=int, default=1000)
parser.add_argument('--num_to_avg', type=int, default=10, help='Average measures across num_to_avg runs.')
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name.')
parser.add_argument('--stored_data_path', type=str, default='/data/ood_detection/data/', help='the path for storing data.')
parser.add_argument('--score', default='SHE', type=str, help='score options: MSP|SHE|Energy||ReAct')
parser.add_argument('--parallel_list', type=str, default='0,1,2,3',help='give number if want parallel')
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--metric', type=str, default='inner_product')
parser.add_argument('--resize_val', default=224, type=int, help='transform resize length')
parser.add_argument('--noise', type=float, default=0.0014, help='pertubation')
parser.add_argument('--threshold', type=float, default=1.0)
parser.add_argument('--T', default=1.0, type=float)
#parameters for wrn
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--need_penultimate', default=4, type=int)

args = parser.parse_args()
print(args)


    
random_seed = 12


os.environ['CUDA_VISIBLE_DEVICES'] = args.parallel_list
cudnn.benchmark = True

# Set random seed
seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

train_dataset_path = '/data/imagenet/train/'
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
        # transforms.Resize(224),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])
val_dataset_path = './data/val/'
valid_set = torchvision.datasets.ImageFolder(
    val_dataset_path,
    valid_transform
)
train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loader = DataLoaderX(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=8)


model = resnet50(num_classes=1000)



pre_path = './checkpoints/imagenet/{}.pth'.format(args.model)
model.load_state_dict(torch.load(pre_path))

net = nn.DataParallel(model).cuda()
net.eval()


# Used for the ReAct method
def get_threshold(p=0.9):
    cur_num = 0
    tempres = []
    with torch.no_grad():
        for data, target in train_loader:
            cur_num += data.size(0)
            data, target = data.cuda(), target.cuda()
            _,penultimate = net(data,need_penultimate=args.need_penultimate)
            for i in range(penultimate.size(0)):
                cur_feature = penultimate[i].detach().tolist()
                tempres.extend(cur_feature)
    tempres.sort()
    index = int(len(tempres)*p)
    threshold = tempres[index]
    print('threshold is :',threshold)
    return threshold

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(valid_set) // 10
expected_ap = ood_num_examples / (ood_num_examples + len(valid_set))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.batch_size and in_dist is False:
                break

            data = data.cuda()


            if args.score == 'SHE':
                output,penultimate = net(data,need_penultimate=args.need_penultimate)
                _score.extend(simple_compute_score_HE(prediction=output,penultimate=penultimate))
            elif args.score == 'MSP':
                output,penultimate = net(data,need_penultimate=args.need_penultimate)
                smax = to_np(F.softmax(output, dim=1))
                _score.append(-np.max(smax, axis=1))
            elif args.score == 'Energy':
                output,penultimate = net(data,need_penultimate=args.need_penultimate)
                _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))
            elif args.score == 'ReAct':
                output,penultimate = net(data,threshold=args.threshold,need_penultimate=args.need_penultimate)
                _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))
        if in_dist:
            return concat(_score).copy()
        else:
            return concat(_score)[:ood_num_examples].copy()


def simple_compute_score_HE(prediction,penultimate):

    numclass = args.num_class
    #----------------------------------------Step 1: classifier the test feature-----------------------------------
    pred = prediction.argmax(dim=1, keepdim=False)
    pred = pred.cpu().tolist()
    
    #----------------------------------------Step 2: get the stored pattern------------------------------------

    total_stored_feature = None
    for i in range(numclass):
        path = './stored_pattern/avg_stored_pattern/size_{}/{}/{}/add_clip_stored_avg_class_{}.pth'.format(args.resize_val,args.dataset,args.model,i)
        stored_tensor = torch.load(path)
        if total_stored_feature is None:
            total_stored_feature = stored_tensor
        else:
            total_stored_feature = torch.cat((total_stored_feature,stored_tensor),dim=0)
    #--------------------------------------------------------------------------------------

    target = total_stored_feature[pred,:]
    res = []
    if args.metric == 'inner_product':
        res_energy_score = torch.sum(torch.mul(penultimate,target),dim=1) #inner product
    elif args.metric == 'euclidean_distance':
        res_energy_score = -torch.sqrt(torch.sum((penultimate-target)**2, dim=1))
    elif args.metric == 'cos_similarity':
        res_energy_score = torch.cosine_similarity(penultimate,target, dim=1)

    lse_res = -to_np(res_energy_score)
    res.append(lse_res)
    return res







in_score = get_ood_scores(test_loader, in_dist=True)


# /////////////// OOD Detection ///////////////

def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)

        measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])


    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, method_name='method:{}\tsize:{}\tdataset:{}\tmodel:{}'.format(args.score,args.resize_val,args.dataset,args.model))
    else:
        print_measures(auroc, aupr, fpr, method_name='method:{}_dataset:{}'.format(args.score,args.dataset))
    return 100*np.mean(fprs), 100*np.mean(aurocs)




fprlist,auclist = [],[]


# # /////////////// Places365 ///////////////

ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'Places'),
                            transform=valid_transform)                                                   
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nPlaces365 Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)


# # /////////////// Textures ///////////////

ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'dtd/images'),
                            transform=valid_transform)                                                   
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nTexture Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)


# /////////////// SUN /////////////// # cropped and no sampling of the test set
ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'SUN'),
                            transform=valid_transform)                                                   
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nSUN Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)

# # /////////////// iNaturalist /////////////// # cropped and no sampling of the test set
ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'iNaturalist/'),
                            transform=valid_transform)                                                   
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\niNaturalist  Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)

print('avg:',sum(fprlist)/len(fprlist),sum(auclist)/len(auclist))