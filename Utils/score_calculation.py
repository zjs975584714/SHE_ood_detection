from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from scipy import misc
to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'


def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, volatile=True)
        output, out_features = model.module.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision



def get_ood_scores_perturbation(args,loader, net, bs, ood_num_examples, T, noise, in_dist=False):
    _score = []
    net.eval()
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // bs and in_dist is False:
            break
        data = data.cuda()
        data = Variable(data, requires_grad = True)

        output,_ = net(data)

        outputs,penultimate = Add_perturbation(data, output,net, T, noise)
        _score.extend(simple_compute_score_HN(prediction=outputs,penultimate=penultimate,args=args,net=net))
        # break # !!!!!!!!!
    if in_dist:
        return concat(_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def Add_perturbation(inputs, outputs, model, temper, noiseMagnitude1):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    std = [0.2023, 0.1994, 0.2010]
    gradient[:,0] = (gradient[:,0] )/std[0]
    gradient[:,1] = (gradient[:,1] )/std[1]
    gradient[:,2] = (gradient[:,2] )/std[2]

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)#data' = data - noise*grad
    outputs,pernulminate = model(Variable(tempInputs))
    return outputs,pernulminate

def simple_compute_score_HN(prediction,penultimate,args,net):
    #----------------------------------------Step 1: classifier the test feature-----------------------------------
    feature_list = [None for i in range(args.num_class)]
    pred = prediction.argmax(dim=1, keepdim=True)
    # get each class tensor
    for i in range(args.num_class):
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
    for i in range(args.num_class):
        path = './stored_pattern/avg_stored_pattern/{}/{}/stored_avg_class_{}.pth'.format(args.dataset,args.model,i)
        stored_tensor = torch.load(path).detach()
        stored_feature_list.append(stored_tensor) 
    res = []
    #----------------------------------------Step 3: compute energy--------------------------------------------------------------------
    for i in range(args.num_class):

        test_feature = feature_list[i].transpose(0,1) #[dim,B_test]
        stored_feature = stored_feature_list[i] #[B_stored,dim]

        if test_feature is None: continue
        res_energy_score = torch.mm(stored_feature,test_feature) #[B_stored,B_test]
        lse_res = -to_np(torch.logsumexp(res_energy_score, dim=0))
        
        res.append(lse_res) 
    return res 


def get_avg_stored_pattern(path,args):
    feature_list = []
    for i in range(args.num_class):
        cur_feature = torch.load(os.path.join(path,'stored_avg_class_{}.pth'.format(i)))
        cur_feature = cur_feature.detach()
        feature_list.append(cur_feature)
    return feature_list