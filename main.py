import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST, CIFAR10
import torch.distributions as dist

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

import os
import argparse
import pickle
import time
import json
import pprint
from functools import partial

from optim import Adam

parser = argparse.ArgumentParser()

# subparsers = parser.add_subparsers(dest='model', help='Select model architecture.', required=True)
parser = argparse.ArgumentParser(description='PixelCNN Configuration')
# pixelcnn++ args
# parser_b = subparsers.add_parser('pixelcnnpp')
parser.add_argument('--model', default='pixelcnnpp', type=str)
parser.add_argument('--n_channels', default=160, type=int, help='Number of channels for residual blocks.')
parser.add_argument('--n_res_layers', default=5, type=int, help='Number of residual blocks at each stage.')
parser.add_argument('--n_logistic_mix', default=10, type=int, help='Number of of mixture components for logistics output.')

# action
parser.add_argument('--train', action='store_true', help='Train model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate model.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
parser.add_argument('--cuda', type=int, default=0, help='Which cuda device to use.')
parser.add_argument('--mini_data', action='store_true', help='Truncate dataset to mini_data number of examples.')

# data params
parser.add_argument('--dataset', choices=['mnist', 'cifar10'])
parser.add_argument('--n_cond_classes', type=int, help='Number of classes for class conditional model.')
parser.add_argument('--n_bits', type=int, default=8, help='Number of bits of input data.')
parser.add_argument('--image_dims', type=int, nargs='+', default=(3,28,28), help='Dimensions of the input data.')
parser.add_argument('--mutation_rate', type=float, default=0, help='Mutation rate.')
parser.add_argument('--data_path', default='./data', help='Location of datasets.')
# training param
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization.')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization weight.')
parser.add_argument('--polyak', type=float, default=0.9995, help='Polyak decay for parameter exponential moving average.')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--log_interval', type=int, default=50, help='How often to show loss statistics and save samples.')
parser.add_argument('--eval_interval', type=int, default=1, help='How often to evaluate and save samples.')
# generation param
parser.add_argument('--n_samples', type=int, default=9, help='Number of samples to generate.')


# --------------------
# Data
# --------------------
def rescale(x, n_bits ):
    return x.mul(255).div(2**(8-n_bits)).floor()


def fetch_dataloaders(args):
    # preprocessing transforms
    if args.mutation_rate:
        print('***************************')
        transform = T.Compose([T.ToTensor(), 
                        lambda x: mutate_x(x, args.mutation_rate),                       # add noise                                                                          # tensor in [0,1]
                        lambda x: rescale(x, args.n_bits),                              # lower bits
                        partial(preprocess, n_bits=args.n_bits)])                       # to model space [-1,1]
        target_transform = (lambda y: torch.eye(args.n_cond_classes)[y]) if args.n_cond_classes else None
    else:
        transform = T.Compose([T.ToTensor(),                                            # tensor in [0,1]
                            lambda x: rescale(x, args.n_bits),    # lower bits
                            partial(preprocess, n_bits=args.n_bits)])                # to model space [-1,1]
        target_transform = (lambda y: torch.eye(args.n_cond_classes)[y]) if args.n_cond_classes else None

    if args.dataset=='mnist':
        args.image_dims = (1,28,28)
        train_dataset = MNIST(args.data_path, download=True, train=True, transform=transform, target_transform=target_transform)
        valid_dataset = MNIST(args.data_path, download=True, train=False, transform=transform, target_transform=target_transform)
    elif args.dataset=='cifar10':
        args.image_dims = (3,32,32)
        train_dataset = CIFAR10(args.data_path, download=True, train=True, transform=transform, target_transform=target_transform)
        valid_dataset = CIFAR10(args.data_path, download=True, train=False, transform=transform, target_transform=target_transform)
    else:
        raise RuntimeError('Dataset not recognized')

    if args.mini_data:  # dataset to a single batch
        if args.dataset=='colored-mnist':
            train_dataset = train_dataset.tensors[0][:args.batch_size]
        else:
            train_dataset.data = train_dataset.data[:args.batch_size]
            train_dataset.targets = train_dataset.targets[:args.batch_size]
        valid_dataset = train_dataset

    print('Dataset {}\n\ttrain len: {}\n\tvalid len: {}\n\tshape: {}\n\troot: {}'.format(
        args.dataset, len(train_dataset), len(valid_dataset), train_dataset[0][0].shape, args.data_path))

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=(args.device.type=='cuda'), num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=(args.device.type=='cuda'), num_workers=0)

    # save a sample
    data_sample = next(iter(train_dataloader))[0]
    writer.add_image('data_sample', make_grid(data_sample, normalize=True, scale_each=True), args.step)
    save_image(data_sample, os.path.join(args.output_dir, 'data_sample.png'), normalize=True, scale_each=True)

    return train_dataloader, valid_dataloader

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

def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# --------------------
# Train, evaluate, generate
# --------------------

def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, epoch, writer, args):
    model.train()

    with tqdm(total=len(dataloader), desc='epoch {}/{}'.format(epoch, args.start_epoch + args.n_epochs)) as pbar:
        for x,y in dataloader:
            args.step += 1

            x = x.to(args.device)
            logits = model(x, y.to(args.device) if args.n_cond_classes else None)
            loss = loss_fn(logits, x, args.n_bits).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if scheduler: scheduler.step()

            pbar.set_postfix(bits_per_dim='{:.4f}'.format(loss.item() / (np.log(2) * np.prod(args.image_dims))))
            pbar.update()

            # record
            if args.step % args.log_interval == 0:
                writer.add_scalar('train_bits_per_dim', loss.item() / (np.log(2) * np.prod(args.image_dims)), args.step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], args.step)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, args):
    model.eval()

    losses = 0
    for x,y in tqdm(dataloader, desc='Evaluate'):
        x = x.to(args.device)
        logits = model(x, y.to(args.device) if args.n_cond_classes else None)
        losses += loss_fn(logits, x, args.n_bits).mean(0).item()
    return losses / len(dataloader)

@torch.no_grad()
def generate(model, generate_fn, args):
    model.eval()
    if args.n_cond_classes:
        samples = []
        for h in range(args.n_cond_classes):
            h = torch.eye(args.n_cond_classes)[h,None].to(args.device)
            samples += [generate_fn(model, args.n_samples, args.image_dims, args.device, h=h)]
        samples = torch.cat(samples)
    else:
        samples = generate_fn(model, args.n_samples, args.image_dims, args.device)
    return make_grid(samples.cpu(), normalize=True, scale_each=True, nrow=args.n_samples)

def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, generate_fn, writer, args):
    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        # train
        train_epoch(model, train_dataloader, optimizer, scheduler, loss_fn, epoch, writer, args)

        if (epoch+1) % args.eval_interval == 0:
            # save model
            torch.save({'epoch': epoch,
                        'global_step': args.step,
                        'state_dict': model.state_dict()},
                        os.path.join(args.output_dir, 'checkpoint.pt'))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint.pt'))
            if scheduler: torch.save(scheduler.state_dict(), os.path.join(args.output_dir, 'sched_checkpoint.pt'))

            # # swap params to ema values
            # optimizer.swap_ema()
            scheduler.step()

            # evaluate
            eval_loss = evaluate(model, test_dataloader, loss_fn, args)
            print('Evaluate bits per dim: {:.3f}'.format(eval_loss / (np.log(2) * np.prod(args.image_dims))))
            writer.add_scalar('eval_bits_per_dim', eval_loss / (np.log(2) * np.prod(args.image_dims)), args.step)

            # generate
            samples = generate(model, generate_fn, args)
            writer.add_image('samples', samples, args.step)
            save_image(samples, os.path.join(args.output_dir, 'generation_sample_step_{}.png'.format(args.step)))

            # # restore params to gradient optimized
            # optimizer.swap_ema()

            
# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()
    args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else \
                        os.path.join('/root/tf-logs', args.model, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    writer = SummaryWriter(log_dir = args.output_dir)

    # save config
    if not os.path.exists(os.path.join(args.output_dir, 'config.json')): save_json(args.__dict__, 'config', args)
    writer.add_text('config', str(args.__dict__))
    pprint.pprint(args.__dict__)

    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_dataloader, test_dataloader = fetch_dataloaders(args)


    if args.model=='pixelcnnpp':
        import pixelcnnpp
        model = pixelcnnpp.PixelCNNpp(args.image_dims, args.n_channels, args.n_res_layers, args.n_logistic_mix,
                                      args.n_cond_classes).to(args.device)
        loss_fn = pixelcnnpp.loss_fn
        generate_fn = pixelcnnpp.generate_fn
        # optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995), polyak=args.polyak)

        # optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995), weight_decay=args.weight_decay, polyak=args.polyak)        
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995), weight_decay = args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)  

#    print(model)
    print('Model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

    if args.restore_file:
        model_checkpoint = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        optimizer.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/optim_checkpoint.pt', map_location=args.device))
        if scheduler:
            scheduler.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/sched_checkpoint.pt', map_location=args.device))
        args.start_epoch = model_checkpoint['epoch'] + 1
        args.step = model_checkpoint['global_step']

    if args.train:
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, generate_fn, writer, args)

    if args.evaluate:
        # if args.step > 0: optimizer.swap_ema()
        eval_loss = evaluate(model, test_dataloader, loss_fn, args)
        print('Evaluate bits per dim: {:.3f}'.format(eval_loss / (np.log(2) * np.prod(args.image_dims))))
        #if args.step > 0: optimizer.swap_ema()

    if args.generate:
        #if args.step > 0: optimizer.swap_ema()
        samples = generate(model, generate_fn, args)
        writer.add_image('samples', samples, args.step)
        save_image(samples, os.path.join(args.output_dir, 'generation_sample_step_{}.png'.format(args.step)), nrow = 8)
        #if args.step > 0: optimizer.swap_ema()