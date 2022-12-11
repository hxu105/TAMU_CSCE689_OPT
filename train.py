#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This codebase is modified based on moco implementations: https://github.com/facebookresearch/moco

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp

import sogclr.builder
import sogclr.loader
import sogclr.optimizer
import sogclr.cifar  # cifar

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = torchvision_model_names

parser = argparse.ArgumentParser(description='SogCLR Pre-Training')
parser.add_argument('--data', metavar='DIR', default='/data/xht/data/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1.0, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

# moco specific configs:
parser.add_argument('--dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--mlp-dim', default=2048, type=int,
                    help='hidden dimension in MLPs (default: 2048)')
parser.add_argument('--t', default=0.3, type=float,
                    help='softmax temperature (default: 0.3)')
parser.add_argument('--num_proj_layers', default=2, type=int,
                    help='number of non-linear projection heads')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

# dataset 
parser.add_argument('--data_name', default='cifar10', type=str) 
parser.add_argument('--save_dir', default='./saved_models/', type=str) 


# sogclr
parser.add_argument('--loss_type', default='dcl', type=str,
                    choices=['dcl', 'cl', 'moco_cl'],
                    help='learing rate scaling (default: linear)')
parser.add_argument('--gamma', default=0.9, type=float,
                    help='for updating u')
parser.add_argument('--learning-rate-scaling', default='linear', type=str,
                    choices=['sqrt', 'linear'],
                    help='learing rate scaling (default: linear)')

# distributed
parser.add_argument('--local_rank', default=0, type=int)

# SWA
parser.add_argument('--swa', action='store_true', default=False, help='Self-ensemble learning')

# SAM
parser.add_argument('--norm_type', type=str, default='l2')
parser.add_argument('--adv_steps', type=int, default=2)
parser.add_argument(
        '--adv_init_mag',
        type=float,
        default=0.1,
        help=""
    )
parser.add_argument(
        '--adv_max_norm',
        type=float,
        default=3,
        help="set to 0 to be unlimited"
    )
parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help="set to 0 to be unlimited"
    )


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    else:
        args.gpu = 0
        print("Use GPU: {} for training".format(args.gpu))
        
    # data sizes
    if args.data_name == 'cifar10' or args.data_name == 'cifar100' : 
        data_size = 50000
    else:
        data_size = 1000000 
    print ('pretraining on %s'%args.data_name)
    # global_rank = int(os.environ['SLURM_PROCID'])
    # loca_rank = os.environ['LOCAL_RANK']
    # world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend='nccl', world_size=int(os.environ["WORLD_SIZE"]))

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = sogclr.builder.SimCLR_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True), 
            args.dim, args.mlp_dim, args.t, cifar_head=('cifar' in args.data_name), loss_type=args.loss_type, N=data_size, num_proj_layers=args.num_proj_layers)

    # infer learning rate before changing batch size
    if args.learning_rate_scaling == 'linear':
        # linear scaling
        args.lr = args.lr * args.batch_size / 256
    else:
        # sqrt scaling  
        args.lr = args.lr * math.sqrt(args.batch_size)
        
    print ('initial learning rate:', args.lr)      
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.to(f'cuda:{local_rank}')
        # torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank])
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[local_rank],
                                                            output_device=local_rank)

    # optimizers
    if args.optimizer == 'lars':
        optimizer = sogclr.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
        
    scaler = amp.GradScaler()
   
    # log_dir 
    save_root_path = args.save_dir
    global_batch_size = args.batch_size
    method_name = {'dcl': 'sogclr', 'cl': 'simclr', 'moco_cl':'moco_sog'}[args.loss_type]
    logdir = 'basline_%s_%s_%s-%s-%s_bz_%s_E%s_WR%s_lr_%.3f_%s_wd_%s_t_%s_g_%s_%s'%(args.data_name, args.arch, method_name, args.dim, args.mlp_dim, global_batch_size, args.epochs, args.warmup_epochs, args.lr, args.learning_rate_scaling, args.weight_decay, args.t, args.gamma, args.optimizer )
    summary_writer = SummaryWriter(log_dir=os.path.join(save_root_path, logdir))
    print (logdir)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    

    # Data loading code
    mean = {'cifar10':      [0.4914, 0.4822, 0.4465],
            'cifar100':     [0.4914, 0.4822, 0.4465] 
            }[args.data_name]
    std = {'cifar10':      [0.2470, 0.2435, 0.2616],
            'cifar100':     [0.2470, 0.2435, 0.2616]
            }[args.data_name]

    image_size = {'cifar10':32, 'cifar100':32}[args.data_name]
    normalize = transforms.Normalize(mean=mean, std=std)

    # simclr augmentations
    augmentation1 = [
        transforms.RandomResizedCrop(image_size, scale=(args.crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        # AddGaussianNoise(0, 2/255)
    ]
    
    if args.data_name == 'cifar10':
        DATA_ROOT = args.data
        train_dataset = sogclr.cifar.CIFAR10(root=DATA_ROOT, train=True, download=True, 
                                           transform=sogclr.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                                                                                   transforms.Compose(augmentation1)))
    elif args.data_name == 'cifar100':
        DATA_ROOT = args.data
        train_dataset = sogclr.cifar.CIFAR100(root=DATA_ROOT, train=True, download=True, 
                                           transform=sogclr.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                                                                                   transforms.Compose(augmentation1)))
    else:
        raise ValueError

 
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)

    if args.swa:
        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        swa_start = 300
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    else:
        swa_model = None
        scheduler = None
        swa_scheduler = None
    
    # train_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        loss = train(train_loader, model, optimizer, scaler, summary_writer, epoch, args, local_rank, scheduler, swa_model, swa_scheduler)
        if epoch % 10 == 0 or args.epochs - epoch < 3:
            if torch.cuda.device_count() > 1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=False, filename=os.path.join(save_root_path, logdir, 'checkpoint_%04d.pth.tar' % epoch))
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=False, filename=os.path.join(save_root_path, logdir, 'checkpoint_%04d.pth.tar' % epoch))
                # train_loss = loss
    summary_writer.close()
    

def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args, local_rank=None, scheduler=None, swa_model=None, swa_scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)

    noise_sampler = torch.distributions.uniform.Uniform(
                low=-args.eps, high=args.eps
            )

    average_loss = 0.0

    for i, (images, _, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)

        if args.gpu is not None:
            # images[0] = images[0].to(device)
            # images[1] = images[1].to(device)
            # images[0] = images[0].cuda(args.gpu, non_blocking=True)
            # images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images[0] = images[0].cuda(local_rank, non_blocking=True)
            images[1] = images[1].cuda(local_rank, non_blocking=True)
        # print(images[0].shape)
        # print(images[1].shape)
        
        #############################################################################################
        # if isinstance(model, torch.nn.DataParallel):
        #     weights0 = model.module.base_encoder.fc[0].weight.data.detach().clone()
        #     weights1 = model.module.base_encoder.fc[1].weight.data.detach().clone()
        # else:
        #     weights0 = model.base_encoder.fc[0].weight.detach().clone()
        #     weights1 = model.base_encoder.fc[1].weight.detach().clone()
        # noise0 = noise_sampler.sample(sample_shape=weights0.shape).to(
        #                     weights0
        #                     )
        # noise1 = noise_sampler.sample(sample_shape=weights1.shape).to(
        #                     weights0
        #                     )
        # if isinstance(model, torch.nn.DataParallel):
        #     model.module.base_encoder.fc[0].weight.data.add(noise0)
        #     model.module.base_encoder.fc[1].weight.data.add(noise1)
        # else:
        #     model.base_encoder.fc[0].weight.data.add(noise0)
        #     model.base_encoder.fc[1].weight.data.add(noise1)
        # with torch.cuda.amp.autocast(True):
        #     # logits = model.moco_contrast(images[0], images[1])
        #     loss = model(images[0], images[1], index, args.gamma, local_rank=local_rank)
        # if isinstance(model, torch.nn.DataParallel):
        #     model.module.base_encoder.fc[0].weight.data = weights0
        #     model.module.base_encoder.fc[1].weight.data = weights1
        # else:
        #     model.base_encoder.fc[0].weight.data = weights0
        #     model.base_encoder.fc[1].weight.data = weights1
        #############################################################################################

        # compute output
        with torch.cuda.amp.autocast(True):
            # logits = model.moco_contrast(images[0], images[1])
            loss = model(images[0], images[1], index, args.gamma, local_rank=local_rank)
        ##change loss
        # exit()
        average_loss += loss.item() / len(train_loader)
        losses.update(loss.item(), images[0].size(0))

        summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            if epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return average_loss


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
