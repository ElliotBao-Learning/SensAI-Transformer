import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.cifar.vit import ViT, channel_selection
from models.cifar.vit_slim import vit_slim
from datasets import cifar
from vit_record import ViTRecord

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

model = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,                  # 512
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
    )
model = model.to(device)

model_path = "./warm_up_models/vit-4-ckpt.t7"
print("=> loading checkpoint '{}'".format(model_path))
checkpoint = torch.load(model_path)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
start_epoch = checkpoint['epoch']
best_prec1 = checkpoint['acc']
model.load_state_dict(new_state_dict)
print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(model_path, checkpoint['epoch'], best_prec1))
total = 0
for m in model.modules():
    if isinstance(m, channel_selection):
        total += m.indexes.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, channel_selection):
        size = m.indexes.data.shape[0]
        bn[index:(index+size)] = m.indexes.data.abs().clone()
        index += size

percent = 0.3
y, i = torch.sort(bn)
thre_index = int(total * percent)
thre = y[thre_index]

def prune_output_linear_layer_(linear_layer, class_indices, use_bce=False):
    if use_bce:
        assert len(class_indices) == 1
    else:
        # use 0 as the placeholder of the negative class
        class_indices = [0] + list(class_indices)
    linear_layer.bias.data = linear_layer.bias.data[class_indices]
    linear_layer.weight.data = linear_layer.weight.data[class_indices, :]
    if not use_bce:
        # reinitialize the negative sample class
        linear_layer.weight.data[0].normal_(0, 0.01)
    linear_layer.out_features = len(class_indices)


def test(model):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

test(model)

print(thre)

dataset = cifar.CIFAR10TrainingSetWrapper([0,2,4,6,8], False)
num_classes = 10
pruning_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        num_workers=2,
        pin_memory=False)

print('\nMake a test run to generate activations. \n Using training set.\n')
with ViTRecord(model, 'vit') as recorder:
    # collect pruning data
    #bar = tqdm(total=len(pruning_loader))
    for batch_idx, (inputs, _) in enumerate(pruning_loader):
        #bar.update(1)
        inputs = inputs.to(device)
        recorder.record_batch(inputs)

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, channel_selection):
        # print(k)
        # print(m)
        if k in [16,40,64,88,112,136]:
            weight_copy = torch.from_numpy(recorder.scores[(k-16)//12]).abs().cuda()
            # weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            thre_ = thre.clone()
            while (torch.sum(mask)%8 !=0):                       # heads
                thre_ = thre_ - 0.0001
                mask = weight_copy.gt(thre_).float().cuda()
        else:
            weight_copy = torch.from_numpy(recorder.scores[(k-16)//12]).abs().cuda()
            # weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.indexes.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total
print('Pre-processing Successful!')
print(cfg)



# dataset = cifar.CIFAR10TrainingSetWrapper([3,7], False)
# num_classes = 10
# pruning_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=1000,
#         num_workers=2,
#         pin_memory=False)

# print('\nMake a test run to generate activations. \n Using training set.\n')
# with ViTRecord(model, 'vit') as recorder:
#     # collect pruning data
#     #bar = tqdm(total=len(pruning_loader))
#     for batch_idx, (inputs, _) in enumerate(pruning_loader):
#         #bar.update(1)
#         inputs = inputs.to(device)
#         recorder.record_batch(inputs)


cfg_prune = []
for i in range(len(cfg)):
    if i%2!=0:
        cfg_prune.append([cfg[i-1],cfg[i]])

newmodel = vit_slim(image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    cfg=cfg_prune)

newmodel.to(device)
# num_parameters = sum([param.nelement() for param in newmodel.parameters()])

newmodel_dict = newmodel.state_dict().copy()

i = 0
newdict = {}
for k,v in model.state_dict().items():
    if 'net1.0.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'net1.0.bias' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'to_q' in k or 'to_k' in k or 'to_v' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'net2.0.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[:,idx.tolist()].clone()
        i = i + 1
    elif 'to_out.0.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[:,idx.tolist()].clone()
        i = i + 1

    elif k in newmodel.state_dict():
        newdict[k] = v

newmodel_dict.update(newdict)
newmodel.load_state_dict(newmodel_dict)

torch.save(newmodel.state_dict(), 'pruned.pth')
print('after pruning: ', end=' ')
test(newmodel)

prune_output_linear_layer_(newmodel.mlp_head[4], [0,2,4,6,8])
# torch.save(newmodel, './pruned_models/vit_slim/vit_1_pruned_model.pth')

