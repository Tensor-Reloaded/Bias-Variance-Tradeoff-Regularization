import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import *
from collections import OrderedDict

"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""
num_classes=10

model = PreResNet(56)
if True:
    model = nn.DataParallel(model).cuda()
    
#Loading Trained Model
baseline= 'runs/PreResNet56/Baseline/69/model_179_92.67.pth'
robust_model= 'multirun/2020-02-21/15-07-30/4/runs/PreResNet(56)/Lipschitz-Block-lvl/mse/69/model_198_92.86999999999999.pth'
# robust_model= 'runs/PreResNet101 K=6 full gradual cos/model_300_0.pth'


state_dict = torch.load(robust_model)
new_state_dict = OrderedDict()

for key, value in state_dict.items():
    new_key = "module."+key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loading Test Data (Un-normalized)
transform_test = transforms.Compose([transforms.ToTensor(),])
train_set = torchvision.datasets.CIFAR10(root='../storage', train=True, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=512, shuffle=True)


testset = torchvision.datasets.CIFAR10(root='../storage', train=False,
                                         download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=512, pin_memory=True,
                                          shuffle=False, num_workers=4)

# Mean and Standard Deiation of the Dataset
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t
def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t

# Attacking Images batch-wise
def attack(model, criterion, img, label, eps, attack_type, iters):
    adv = img.detach()
    adv.requires_grad = True

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations
        
        noise = 0
        
    for j in range(iterations):
        out_adv = model(normalize(adv.clone()))
        loss = criterion(out_adv, label)
        loss.backward()

        if attack_type == 'mim':
            adv_mean= torch.mean(torch.abs(adv.grad), dim=1,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=2,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=3,  keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
            noise = adv.grad

        # Optimization step
        adv.data = adv.data + step * noise.sign()
#        adv.data = adv.data + step * adv.grad.sign()

        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach()

# Loss Criteria
criterion = nn.CrossEntropyLoss()
adv_acc = 0
clean_acc = 0
eps =8/255 # Epsilon for Adversarial Attack



clean_clean_img, _ = next(iter(train_loader))  
clean_clean_img = normalize(clean_clean_img.clone().detach()).to(device)

aug_test=None
aug_test_lambda = 0.5

#Clean accuracy:91.710%   Adversarial accuracy:16.220%
for idx, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)
    if aug_test != None:
        clean_img = normalize(img.clone().detach())
        outputs = []
        for i in range(aug_test):
            aug_data = clean_img * (1 - aug_test_lambda) + aug_test_lambda * clean_clean_img[torch.randperm(label.size(0))]
            outputs.append(model(aug_data).detach())
        output = torch.stack(outputs, dim=0).mean(0)
        clean_acc += torch.sum(output.argmax(dim=-1) == label).item()
        
        adv= attack(model, criterion, img, label, eps=eps, attack_type= 'fgsm', iters= 10 )
        adv_img = normalize(adv.clone().detach())
        outputs = []
        for i in range(aug_test):
            aug_data = adv_img * (1 - aug_test_lambda) + aug_test_lambda * clean_clean_img[torch.randperm(label.size(0))]
            outputs.append(model(aug_data).detach())
        output = torch.stack(outputs, dim=0).mean(0)
        adv_acc += torch.sum(output.argmax(dim=-1) == label).item()
    else:
        clean_acc += torch.sum(model(normalize(img.clone().detach())).argmax(dim=-1) == label).item()
        adv= attack(model, criterion, img, label, eps=eps, attack_type= 'fgsm', iters= 10 )
        adv_acc += torch.sum(model(normalize(adv.clone().detach())).argmax(dim=-1) == label).item()
    print('Batch: {0}'.format(idx))
print('Clean accuracy:{0:.3%}\t Adversarial accuracy:{1:.3%}'.format(clean_acc / len(testset), adv_acc / len(testset)))

# FGSM:
# Baseline: Clean 92.0, Adv 15.24
# Lipchnitz Block lvl: Clean 89.84, Adv 9.01

# BIM
# Baseline: Clean 92.0, Adv 
# Lipchnitz Block lvl: Clean 89.84, Adv 0.0

# MIN:
# Baseline: Clean 92.0, Adv 
# Lipchnitz Block lvl: Clean 89.84, Adv 0.0

# PGD 
# Baseline: Clean 92.0, Adv 
# Lipchnitz Block lvl: Clean 89.84, Adv 0.0
