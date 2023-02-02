### OUT-OF-DISTRIBUTION DETECTION BASED ON IN-DISTRIBUTION DATA PATTERNS MEMORIZATION WITH MODERN HOPFIELD ENERGY

The document is guidance for reproducing our paper, some codes are from [energy-ood](https://github.com/wetliu/energy_ood).

#### Preliminaries
Our code is tested under Ubuntu Linux 18.04.1 and Python 3.6 environment. The environment can be accomplished by the following command:
```
pip install -r requirement.txt
```

#### Download the Out-of-distribution (OOD) Dataset
In our paper, we use nine OOD datasets and two In-distribution (ID) datasets.
 For the ID datasets CIFAR10 and CIFAR100 and one of the OOD datasets SVHN, it is easy to use them directly by the ```torchvision``` as follows (```data_path``` refers to your specified dataset path):
```
dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
dataset = torchvision.datasets.CIFAR100(root=data_path, train=False=True)
dataset = torchvision.datasets.SVHN(root=data_path, train=False, download=True)
```
However, another eight OOD datasets need to download and we provide the link to them.
[LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz),[LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz), [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz), [Places](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/),[DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/),[Tiny Imagenet](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz),[SUN](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/),[iNaturalist](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/)
Please place them into your dataset path, and use them as follows:
```
dataset =  torchvision.datasets.ImageFolder(root=data_path)
```
### Pretrained Model
We use ResNet18, ResNet34, and WRN40-2 as our backbone networks. And the pre-trained models are all trained on CIFAR10 and CIFAR100 respectively.  As mentioned in our paper, to get better performance, we use data augmentation (e.g., flip, rotate) and resize the image during the training/testing process. We use size = 112 and 64 for ResNet and WRN respectively.

### Evaluation process
##### (1) Prepare the model
All the model weights have been saved at ```./checkpoints/cifar10/``` and ```./checkpoints/cifar100/```, you can use them directly if you do not want to train them again.
Otherwise, you can train the model by the following command:
```
python train_model.py --model xxx(resnet18/resnet34/wrn)
```
##### (2) Prepare the stored pattern we need
You can directly run the bash and the stored pattern used for **HE** and **SHE** will be generated automatically.
```
bash generate_SHEandHE_feature.sh
```
After several time, the stored pattern is stored at ```./stored_pattern/all_stored_pattern``` and ```./stored_pattern/avg_stored/pattern/```.

##### (3) Calculate the Hopfield energy score
You can evaluate our method by run the following command:
```
python test_score_ood_detection.py --dataset xxx(cifar10/cifar100) --model xxx(resnet18/resnet34/wrn) --score SHE
```
### Evaluation of ImageNet-1k
We use ResNet-50 as the backbone network to evaluate the OOD detection performance. And the command is as following:
```
python generate_stored_pattern_imagenet.py
python test_score_ood_detection_imagenet.py
```
