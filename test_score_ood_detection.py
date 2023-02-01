from random import seed
import numpy as np
import os
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
from models import ResNet 
from Utils.display_results import get_measures, print_measures, print_measures_with_std
import Utils.score_calculation as lib

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--num_to_avg', type=int, default=10, help='Average measures across num_to_avg runs.')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name.')
parser.add_argument('--stored_data_path', type=str, default='/data/ood_detection/data/', help='the path for storing data.')
parser.add_argument('--score', default='SHE', type=str, help='score options: MSP|Energy|ReAct|HE|SHE|SHE_react|SHE_with_perturbation')
parser.add_argument('--parallel_list', type=str, default='0',help='give number if want parallel')
parser.add_argument('--model', type=str, default='resnet18')

parser.add_argument('--resize_val', default=112, type=int, help='transform resize length')
parser.add_argument('--beita', default=0.01, type=float, help='for HE')
parser.add_argument('--noise', type=float, default=0.0014, help='pertubation')
parser.add_argument('--threshold', type=float, default=1.0)
parser.add_argument('--T', default=1.0, type=float)
parser.add_argument('--k', default=0.8, type=float)
parser.add_argument('--metric', type=str, default='inner_product',help='ablation: choose which metric for the SHE')

#parameters for wrn
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

parser.add_argument('--need_penultimate', default=4, type=int,help='choose which layer as the pattern')

args = parser.parse_args()
print(args)

if args.model == 'wrn':
    args.resize_val = 64
else:
    args.resize_val = 112
    
random_seed = 12

args.beita = 0.2 if args.model == 'wrn' else 0.01

os.environ['CUDA_VISIBLE_DEVICES'] = args.parallel_list
cudnn.benchmark = True

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
    args.num_class = 10
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root=args.stored_data_path, train=True, download=True, transform=transform_all)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_data = torchvision.datasets.CIFAR100(root=args.stored_data_path, train=False, download=True, transform=transform_all)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    args.num_class = 100
else:
    print('The dataset is not provided.')

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

# ---------If you want to test the accuracy of this model, you can use the code below:------

# def valid(model, valid_loader,numclass):
#     valid_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in valid_loader:
#             data, target = data.cuda(), target.cuda()
#             model = model.cuda()
#             prediction,_ = model(data)
#             critetion = nn.CrossEntropyLoss()
#             loss = critetion(prediction,target)
#             valid_loss += loss.item()
#             pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     valid_loss /= len(valid_loader.dataset)
#     accuracy = 100. * correct / len(valid_loader.dataset)
#     return valid_loss, correct, accuracy

# print('valid initialization')
# valid_loss, valid_correct, valid_accuracy = valid(net, test_loader,args.num_class)
# print('validing set: Average loss: {:.4f}, Accuracy: ({:.4f}%)'.format(valid_loss, valid_accuracy))


# Used for the ReAct method
def get_threshold(p=0.9):
    tempres = []
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            _,penultimate = net(data,need_penultimate=args.need_penultimate)
            for i in range(penultimate.size(0)):
                cur_feature = penultimate[i].detach().tolist()
                tempres.extend(cur_feature)
    tempres.sort()
    index = int(len(tempres)*p)
    threshold = tempres[index]
    return threshold

if args.score == 'ReAct':
    args.threshold = get_threshold(p=0.9) 
elif args.score == 'SHE_react':
    args.threshold = get_threshold(p=0.95) 


# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

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
            elif args.score == 'SHE_react':
                output,penultimate = net(data,threshold=args.threshold,need_penultimate=args.need_penultimate)
                _score.extend(simple_compute_score_HE(prediction=output,penultimate=penultimate))
            elif args.score == 'HE':
                output,penultimate = net(data,need_penultimate=args.need_penultimate)
                _score.extend(compute_score_HE(prediction=output,penultimate=penultimate))
            elif args.score == 'MSP':
                output,penultimate = net(data,need_penultimate=args.need_penultimate)
                smax = to_np(F.softmax(output, dim=1))
                _score.append(-np.max(smax, axis=1))
            elif args.score == 'Energy':
                output,penultimate = net(data,need_penultimate=args.need_penultimate)
                _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))
            elif args.score == 'ReAct':
                output,penultimate = net(data,threshold=args.threshold)
                _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))
        if in_dist:
            return concat(_score).copy()
        else:
            return concat(_score)[:ood_num_examples].copy()



def compute_score_HE(prediction,penultimate):
    #----------------------------------------Step 1: classifier the test feature-----------------------------------
    numclass = args.num_class
    feature_list = [None for i in range(numclass)]
    pred = prediction.argmax(dim=1, keepdim=True)
    # get each class tensor
    for i in range(numclass):
        each_label_tensor = torch.tensor([i for _ in range(prediction.size(0))]).cuda()
        target_index = pred.eq(each_label_tensor.view_as(pred))

    # get the penultimate layer
        each_label_feature = penultimate[target_index.squeeze(1)]
        if each_label_feature is None: continue
        if feature_list[i] is None:
            feature_list[i] = each_label_feature
        else:
            feature_list[i] = torch.cat((feature_list[i],each_label_feature),dim=0)
    

    #----------------------------------------Step 2: get the stored pattern------------------------------------
    stored_feature_list = []
    for i in range(numclass):
        path = './stored_pattern/all_stored_pattern/size_{}/{}/{}/stored_all_class_{}.pth'.format(args.resize_val,args.dataset,args.model,i)
        stored_tensor = torch.load(path)
        stored_feature_list.append(stored_tensor) #Here we get all the stored pattestr(i) +'.pth'rns

    res = []
    #----------------------------------------Step 3: compute energy--------------------------------------------------------------------
    for i in range(numclass):

        test_feature = feature_list[i].transpose(0,1) #[dim,B_test]
        stored_feature = stored_feature_list[i] #[B_stored,dim]
        

        if test_feature is None: continue
        res_energy_score = torch.mm(stored_feature,test_feature) #[B_stored,B_test]
        lse_res = -to_np(torch.logsumexp(res_energy_score*args.beita, dim=0)) #[1,B_test]
        res.append(lse_res)
    return res  




def simple_compute_score_HE(prediction,penultimate,need_mask=False):

    numclass = args.num_class
    #----------------------------------------Step 1: classifier the test feature-----------------------------------
    pred = prediction.argmax(dim=1, keepdim=False)
    pred = pred.cpu().tolist()
    
    #----------------------------------------Step 2: get the stored pattern------------------------------------

    total_stored_feature = None
    for i in range(numclass):
        path = './stored_pattern/avg_stored_pattern/size_{}/{}/{}/stored_avg_class_{}.pth'.format(args.resize_val,args.dataset,args.model,i)
        stored_tensor = torch.load(path)
        if total_stored_feature is None:
            total_stored_feature = stored_tensor
        else:
            total_stored_feature = torch.cat((total_stored_feature,stored_tensor),dim=0)
    #--------------------------------------------------------------------------------------

    target = total_stored_feature[pred,:]
    res = []

    # for ablation exp: different metric for SHE
    if args.metric == 'inner_product':
        res_energy_score = torch.sum(torch.mul(penultimate,target),dim=1) #inner product
    elif args.metric == 'euclidean_distance':
        res_energy_score = -torch.sqrt(torch.sum((penultimate-target)**2, dim=1))
    elif args.metric == 'cos_similarity':
        res_energy_score = torch.cosine_similarity(penultimate,target, dim=1)
    lse_res = -to_np(res_energy_score)
    res.append(lse_res)
    return res







if args.score == 'SHE_with_perturbation':
    in_score = lib.get_ood_scores_perturbation(args,test_loader, net, args.batch_size, ood_num_examples, args.T, args.noise, in_dist=True)
else:
    in_score = get_ood_scores(test_loader, in_dist=True)


# /////////////// OOD Detection ///////////////
# auroc_list, aupr_list, fpr_list = [], [], []

def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        if args.score == 'SHE_with_perturbation':
            out_score = lib.get_ood_scores_perturbation(args,ood_loader, net, args.batch_size, ood_num_examples, args.T, args.noise)
        else:
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
# # # /////////////// SVHN /////////////// 
ood_data = torchvision.datasets.SVHN(root=os.path.join(args.stored_data_path,'svhn'), split="test",download=True,
                     transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nSVHN Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)

# # /////////////// LSUN-C ///////////////
ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'LSUN_C'),
                            transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nLSUN_C Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)
# # # /////////////// LSUN-R ///////////////
ood_data = dset.ImageFolder(os.path.join(args.stored_data_path,'LSUN_resize'),
                            transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nLSUN_Resize Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)

# # /////////////// iSUN ///////////////
ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'iSUN'),
                            transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\niSUN Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)

# # /////////////// Places365 ///////////////

ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'Places'),
                            transform=transform_all)                                                   
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nPlaces365 Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)




# # /////////////// Textures ///////////////

ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'dtd/images'),
                            transform=transform_all)                                                   
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nTexture Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)

# # # /////////////// Tiny Imagenet /////////////// # cropped and no sampling of the test set
ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'Imagenet_resize'),
                            transform=transform_all)                                                   
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nTiny Imagenet Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)

# /////////////// SUN /////////////// # cropped and no sampling of the test set
ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'SUN'),
                            transform=transform_all)                                                   
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nSUN Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)

# # /////////////// iNaturalist /////////////// # cropped and no sampling of the test set
ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'iNaturalist/'),
                            transform=transform_all)                                                   
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\niNaturalist  Detection')
fpr,auc = get_and_print_results(ood_loader)
fprlist.append(fpr)
auclist.append(auc)

print('avg:',sum(fprlist)/len(fprlist),sum(auclist)/len(auclist))