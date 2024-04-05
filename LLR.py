import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST, SVHN
import torch.distributions as dist

import numpy as np
from tqdm import tqdm

import os
import argparse
import pickle
import time
import json
import pprint
from functools import partial


cifar_model = {'model': 'pixelcnnpp', 'n_channels': 160, 'n_res_layers': 5, 'n_logistic_mix': 10, 'train': True, 'evaluate': False, 'generate': False, 'restore_file': None, 'seed': 0, 'cuda': 0, 'mini_data': False, 'dataset': 'cifar10', 'n_cond_classes': None, 'n_bits': 8, 'image_dims': (3, 28, 28), 'mutation_rate': 0, 'data_path': './data', 'lr': 0.0001, 'lr_decay': 0.999995, 'weight_decay': 0.001, 'polyak': 0.9995, 'batch_size': 32, 'n_epochs': 40, 'step': 0, 'start_epoch': 0, 'log_interval': 50, 'eval_interval': 1, 'n_samples': 9, 'output_dir': '/root/tf-logs/pixelcnnpp/2024-03-29_03-57-05'}
cifar_model['device'] = 'cuda'

bk_cifar_model = {'model': 'pixelcnnpp', 'n_channels': 160, 'n_res_layers': 5, 'n_logistic_mix': 10, 'train': True, 'evaluate': False, 'generate': False, 'restore_file': None, 'seed': 0, 'cuda': 0, 'mini_data': False, 'dataset': 'cifar10', 'n_cond_classes': None, 'n_bits': 8, 'image_dims': (3, 28, 28), 'mutation_rate': 0.1, 'data_path': './data', 'lr': 0.0001, 'lr_decay': 0.999995, 'weight_decay': 0.001, 'polyak': 0.9995, 'batch_size': 32, 'n_epochs': 40, 'step': 0, 'start_epoch': 0, 'log_interval': 50, 'eval_interval': 1, 'n_samples': 9, 'output_dir': '/root/tf-logs/pixelcnnpp/2024-03-29_03-57-05'}
bk_cifar_model['device'] = 'cuda'

mnist = {'model': 'pixelcnnpp', 'n_channels': 32, 'n_res_layers': 5, 'n_logistic_mix': 1, 'train': True, 'evaluate': False, 'generate': False, 'restore_file': None, 'seed': 0, 'cuda': 0, 'mini_data': False, 'dataset': 'mnist', 'n_cond_classes': None, 'n_bits': 4, 'image_dims': (3, 28, 28), 'mutation_rate': 0, 'data_path': './data', 'lr': 0.0001, 'lr_decay': 0.999995, 'weight_decay': 0, 'polyak': 0.9995, 'batch_size': 32, 'n_epochs': 20, 'step': 0, 'start_epoch': 0, 'log_interval': 50, 'eval_interval': 1, 'n_samples': 9, 'output_dir': '/root/tf-logs/pixelcnnpp/2024-03-29_06-45-49'}
mnist['device'] = 'cuda'

bk_mnist = {'model': 'pixelcnnpp', 'n_channels': 32, 'n_res_layers': 5, 'n_logistic_mix': 1, 'train': True, 'evaluate': False, 'generate': False, 'restore_file': None, 'seed': 0, 'cuda': 0, 'mini_data': False, 'dataset': 'mnist', 'n_cond_classes': None, 'n_bits': 4, 'image_dims': (3, 28, 28), 'mutation_rate': 0.1, 'data_path': './data', 'lr': 0.0001, 'lr_decay': 0.999995, 'weight_decay': 0, 'polyak': 0.9995, 'batch_size': 32, 'n_epochs': 20, 'step': 0, 'start_epoch': 0, 'log_interval': 50, 'eval_interval': 1, 'n_samples': 9, 'output_dir': '/root/tf-logs/pixelcnnpp/2024-03-29_06-45-49'}
bk_mnist['device'] = 'cuda'
# --------------------
# Data
# --------------------
def rescale(x, n_bits ):
    return x.mul(255).div(2**(8-n_bits)).floor()

def fetch_dataloaders(model_type, dataset, batch_size):
    if model_type['mutation_rate']:
        print('***************************')
        transform = T.Compose([T.ToTensor(), 
                        lambda x: mutate_x(x, model_type['mutation_rate']),                       # add noise                                                                          # tensor in [0,1]
                        lambda x: rescale(x, model_type['n_bits']),                              # lower bits
                        partial(preprocess, n_bits=model_type['n_bits'])])                       # to model space [-1,1]
        target_transform =  None
    else:
        transform = T.Compose([T.ToTensor(),                                            # tensor in [0,1]
                        lambda x: rescale(x, model_type['n_bits']),    # lower bits
                        partial(preprocess, n_bits=model_type['n_bits'])])              # to model space [-1,1]
        target_transform =  None

    if dataset == 'mnist':
        model_type['image_dims'] = (1,28,28)
        valid_dataset = MNIST('/root/pixelcnn/data', download=True, train=False, transform=transform, target_transform=target_transform)
    elif dataset == 'fashionmnist':
        model_type['image_dims'] = (1,28,28)
        valid_dataset = FashionMNIST('/root/pixelcnn/data', download=True, train=False, transform=transform, target_transform=target_transform)
    elif dataset == 'cifar10':
        model_type['image_dims'] = (3,32,32)
        valid_dataset = CIFAR10('/root/pixelcnn/data', download=True, train=False, transform=transform, target_transform=target_transform)
    elif dataset == 'svhn':
        model_type['image_dims'] = (3,32,32)
        valid_dataset = SVHN('/root/pixelcnn/data', download=True, split='test', transform=transform, target_transform=target_transform)
    elif dataset == 'cifar100':
        model_type['image_dims'] = (3,32,32)
        valid_dataset = CIFAR100('/root/pixelcnn/data', download=True, train=False, transform=transform, target_transform=target_transform)        
    else:
        raise RuntimeError('Dataset not recognized')


    if dataset == 'svhn':
        val_indices = list(range(len(valid_dataset)))
        np.random.shuffle(val_indices)
        val_subset_indices = val_indices[:10000]
        val_sampler = SubsetRandomSampler(val_subset_indices)
        valid_dataloader = DataLoader(valid_dataset, batch_size, sampler=val_sampler, shuffle=False, pin_memory=True, num_workers=0)
        print('Dataset {}\n\tvalid len: {}\n\tshape: {}'.format(
        dataset, len(val_sampler), valid_dataset[0][0].shape))
    else:
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=0)

        print('Dataset {}\n\tvalid len: {}\n\tshape: {}'.format(
        dataset, len(valid_dataset), valid_dataset[0][0].shape))
    # save a sample
    data_sample = next(iter(valid_dataloader))[0]
    # save_image(data_sample, os.path.join('./out', 'data_sample.png'), normalize=True, scale_each=True)

    return data_sample, valid_dataloader


def preprocess(x, n_bits):
    # 1. convert data to float
    # 2. normalize to [0,1] given quantization
    # 3. shift to [-1,1]
    return x.float().div(2**n_bits - 1).mul(2).add(-1)

def deprocess(x, n_bits):
    # 1. shift to [0,1]
    # 2. quantize to n_bits
    # 3. convert data to long
    return x.add(1).div(2).mul(2**n_bits - 1).long()

def mutate_x(x, mutation_rate):
    """Add mutations to input.

    Args:
      x: input image tensor of size batch*width*height*channel
      mutation_rate: mutation rate

    Returns:
      mutated input
    """
    c, h, w = x.size()
    mask = dist.Categorical(torch.tensor([1.0 - mutation_rate, mutation_rate])).sample((c * h * w,))
    mask = mask.view(c, h, w)

    possible_mutations = torch.randint(0, 256, (c, h, w), dtype=torch.int32) / 255

    x = (x + mask * possible_mutations) % 1
    return x.float()

model_type = cifar_model
bk_model_type = bk_cifar_model

# model_type = mnist
# bk_model_type = bk_mnist  # change ID dataset ( cifar_model or mnist )

batch = 40

if model_type == mnist:

    #min
    restore_file = '/root/tf-logs/pixelcnnpp/2024-03-29_06-45-49/checkpoint.pt'    # path to pretrained model
    bk_restore_file = '/root/tf-logs/pixelcnnpp/2024-03-29_18-28-40/checkpoint.pt'  # path to pretrained background model

    dataset = 'mnist'
    data_sample, valid_dataloader= fetch_dataloaders(model_type, dataset, batch)
    data_sample = data_sample.to(model_type['device'])  
    save_image(data_sample, os.path.join('/root/pixelcnn/image', 'mnist_sample.png'), normalize=True, scale_each=True)
    bk_data_sample, bk_valid_dataloader= fetch_dataloaders(bk_model_type, dataset, batch)
    bk_data_sample = bk_data_sample.to(bk_model_type['device'])  
    save_image(bk_data_sample, os.path.join('/root/pixelcnn/image', 'bk_mnist_sample.png'), normalize=True, scale_each=True)

    ood_dataset = 'fashionmnist'
    ood_data_sample, ood_valid_dataloader, = fetch_dataloaders(model_type, ood_dataset, batch)
    ood_data_sample = ood_data_sample.to(model_type['device'])  
    save_image(ood_data_sample, os.path.join('/root/pixelcnn/image', 'fsmnist_sample.png'), normalize=True, scale_each=True)
    bk_ood_data_sample, bk_ood_valid_dataloader, = fetch_dataloaders(bk_model_type, ood_dataset, batch)
    bk_ood_data_sample = bk_ood_data_sample.to(bk_model_type['device'])  
    save_image(bk_ood_data_sample, os.path.join('/root/pixelcnn/image', 'bk_fsmnist_sample.png'), normalize=True, scale_each=True)

elif model_type == cifar_model:
    # cifar
    restore_file = '/root/tf-logs/pixelcnnpp/2024-03-30_01-13-27/checkpoint.pt'
    bk_restore_file = '/root/tf-logs/pixelcnnpp/2024-03-30_01-13-33/checkpoint.pt'

    dataset = 'cifar10'
    data_sample, valid_dataloader = fetch_dataloaders(model_type, dataset, batch)
    data_sample = data_sample.to(model_type['device'])  
    save_image(data_sample, os.path.join('/root/pixelcnn/image', 'cifar10_sample.png'), normalize=True, scale_each=True)
    bk_data_sample, bk_valid_dataloader= fetch_dataloaders(bk_model_type, dataset, batch)
    bk_data_sample = bk_data_sample.to(bk_model_type['device'])  
    save_image(bk_data_sample, os.path.join('/root/pixelcnn/image', 'bk_cifar10_sample.png'), normalize=True, scale_each=True)

    ood_dataset = 'svhn' #'svhn' 'cifar100'  # change ood dataset
    ood_data_sample, ood_valid_dataloader = fetch_dataloaders(model_type, ood_dataset, batch)
    ood_data_sample = ood_data_sample.to(model_type['device'])  
    save_image(ood_data_sample, os.path.join('/root/pixelcnn/image', '{}_sample.png'.format(ood_dataset)), normalize=True, scale_each=True)
    bk_ood_data_sample, bk_ood_valid_dataloader, = fetch_dataloaders(bk_model_type, ood_dataset, batch)
    bk_ood_data_sample = bk_ood_data_sample.to(bk_model_type['device'])  
    save_image(bk_ood_data_sample, os.path.join('/root/pixelcnn/image', 'bk_{}_sample.png'.format(ood_dataset)), normalize=True, scale_each=True)

else:
    raise RuntimeError('Dataset not recognized')

import pixelcnnpp

model = pixelcnnpp.PixelCNNpp(model_type['image_dims'], model_type['n_channels'], model_type['n_res_layers'], model_type['n_logistic_mix'],).to(model_type['device'])
loss_fn = pixelcnnpp.loss_fn

model_checkpoint = torch.load(restore_file, map_location=model_type['device'])
model.load_state_dict(model_checkpoint['state_dict'])

#pred
with torch.no_grad():
    model.eval()
    model = model.to(model_type['device']) 

    log_stacked = []

    for x,y in tqdm(valid_dataloader, desc='Evaluate'):
        x = x.to(model_type['device'])
        logits = model(x)
        loss = loss_fn(logits, x, model_type['n_bits'])
        log_stacked.append(loss)
        # torch.cuda.empty_cache()
    log = torch.cat(log_stacked, dim=0)


    ood_log_stacked = []
    for ood_x,y in tqdm(ood_valid_dataloader, desc='OOD Evaluate'):
        ood_x = ood_x.to(model_type['device'])
        ood_logits = model(ood_x)
        ood_loss = loss_fn(ood_logits, ood_x, model_type['n_bits'])
        ood_log_stacked.append(ood_loss)
    ood_log = torch.cat(ood_log_stacked, dim=0)


    print('fg size: ',log.shape, 'fg_ood size: ',ood_log.shape)

    bk_model = pixelcnnpp.PixelCNNpp(model_type['image_dims'], model_type['n_channels'], model_type['n_res_layers'], model_type['n_logistic_mix'],).to(model_type['device'])
    loss_fn =  pixelcnnpp.loss_fn

    bk_model_checkpoint = torch.load(bk_restore_file, map_location=model_type['device'])
    bk_model.load_state_dict(bk_model_checkpoint['state_dict'])

    bk_model.eval()
    bk_model = bk_model.to(model_type['device']) 

    bk_log_stacked = []
    for x,y in tqdm(bk_valid_dataloader, desc='bk Evaluate'):
        x = x.to(model_type['device'])
        bk_logits = bk_model(x)
        loss = loss_fn(bk_logits, x, model_type['n_bits'])
        bk_log_stacked.append(loss)

    bk_log = torch.cat(bk_log_stacked, dim=0)


    ood_bk_log_stacked = []
    for x,y in tqdm(ood_valid_dataloader, desc='OOD bk Evaluate'):
        x = x.to(model_type['device'])
        bk_ood_logits = bk_model(x)
        loss = loss_fn(bk_ood_logits, x, model_type['n_bits'])
        ood_bk_log_stacked.append(loss)
    bk_ood_log = torch.cat(ood_bk_log_stacked, dim=0)

    print('bk size: ', bk_log.shape, 'bk_ood size: ',bk_ood_log.shape)

    llr = log - bk_log
    ood_llr = ood_log - bk_ood_log
    print('llr mean: ', llr.mean(0))
    print('ood llr mean: ', ood_llr.mean(0))


llr = llr.cpu().detach().numpy()
ood_llr = ood_llr.cpu().detach().numpy()
log = log.cpu().detach().numpy()
bk_log = bk_log.cpu().detach().numpy()
ood_log = ood_log.cpu().detach().numpy()
bk_ood_log = bk_ood_log.cpu().detach().numpy()

from scipy.stats import pearsonr
r, p_value = pearsonr(llr, ood_llr) 

print(f"Correlation coefficient: {r}, P-value: {p_value}")

from sklearn import metrics
from sklearn.metrics import roc_curve

def compute_auc(neg, pos, pos_label=0):
  ys = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))), axis=0)
  neg = np.array(neg)[np.logical_not(np.isnan(neg))]
  pos = np.array(pos)[np.logical_not(np.isnan(pos))]
  scores = np.concatenate((neg, pos), axis=0)
  fpr, tpr, thresholds = roc_curve(ys, scores)
  auc = metrics.roc_auc_score(ys, scores)

  if pos_label == 1:
    return auc, fpr, tpr, thresholds
  else:
    return 1 - auc, fpr, tpr, thresholds

def compute_auc_llr(preds_in, preds_ood, preds0_in, preds0_ood):
  """Compute AUC for LLR."""
  # check if samples are in the same order
#   assert np.array_equal(preds_in['labels'], preds0_in['labels'])
#   assert np.array_equal(preds_ood['labels'], preds0_ood['labels'])

  # evaluate AUROC for OOD detection
  auc, fpr, tpr, thresholds = compute_auc(preds_in, preds_ood, pos_label=0)
  llr_in = preds_in - preds0_in
  llr_ood = preds_ood - preds0_ood
  auc_llr, fpr_llr, tpr_ll, thresholds_ll = compute_auc(llr_in, llr_ood, pos_label=1)
  return auc, auc_llr, fpr, tpr, thresholds, fpr_llr, tpr_ll, thresholds_ll

auc, auc_llr, fpr, tpr, thresholds, fpr_llr, tpr_llr, thresholds_llr = compute_auc_llr(log, ood_log, bk_log, bk_ood_log)
print('auc: ', auc,'auc_llr: ', auc_llr)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(thresholds, fpr, marker='.', label='False Positive Rate')
plt.plot(thresholds, tpr, label='True Positive Rate', marker='.', color='RED') 
plt.xlabel('Thresholds')
plt.ylabel('False Positive Rate')
plt.title('LL False Positive Rate across Different Thresholds')
plt.legend()
plt.grid(True)
plt.savefig('FTP_{}_vs_{}.png'.format(dataset,ood_dataset))
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(thresholds_llr, fpr_llr, marker='.', label='False Positive Rate')
plt.plot(thresholds_llr, tpr_llr, label='True Positive Rate', marker='.', color='RED') 
plt.xlabel('Thresholds')
plt.ylabel('False Positive Rate')
plt.title('LLR False Positive Rate across Different Thresholds')
plt.legend()
plt.grid(True)
plt.savefig('FTP_LLR_{}_vs_{}.png'.format(dataset,ood_dataset))
plt.show()


import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10, 6))

# Plotting the in-distribution LLRs
plt.hist(llr, bins=30, alpha=0.6, label='ID_LLR_{} '.format(dataset), color='blue', density=True)

# Plotting the OOD LLRs
plt.hist(ood_llr, bins=30, alpha=0.6, label='OOD_LLR_{}'.format(ood_dataset), color='red', density=True)


# Adding labels and title
plt.xlabel('Log-Likelihood Ratio (LLR)')
plt.ylabel('Density')
plt.title('Distribution of LLRs for In-Distribution and OOD Data')
plt.legend(loc='upper left')
plt.savefig('LLR_Distribution_{}_vs_{}.png'.format(dataset, ood_dataset))
# Show the plot
plt.show()


plt.figure(figsize=(10, 6))


# Plotting the in-distribution LLRs
plt.hist(log, bins=30, alpha=0.6, label='ID_LL_{}'.format(dataset), color='blue', density=True)

# Plotting the OOD LLRs
plt.hist(ood_log, bins=30, alpha=0.6, label='OOD_LL_{}'.format(ood_dataset), color='red', density=True)


# Adding labels and title
plt.xlabel('Log-Likelihood (LL)')
plt.ylabel('Density')
plt.title('Distribution of LL for In-Distribution and OOD Data')
plt.legend(loc='upper left')
plt.savefig('LL_Distribution_{}_vs_{}.png'.format(dataset, ood_dataset))
# Show the plot
plt.show()


