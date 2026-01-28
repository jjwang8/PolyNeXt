import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR, LinearLR
from torchvision.transforms import v2
    
from torch.autograd import Variable
from model import *
from lamb import Lamb
from sys import platform
import operations

from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=4e-3, help='init learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=12, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')


parser.add_argument('--debug', default=False, action='store_true', help='keep only 5 steps per epoch')
parser.add_argument('--smooth', type=float, default=0, help='amount of label smoothing')
parser.add_argument('--dropout', type=float, default=0, help='dropout prob')
parser.add_argument('--auto_aug', default=False, action='store_true', help='use AutoAugment')
parser.add_argument('--warmup_length', type=int, default=0, help='epochs to do warm up for')
parser.add_argument('--cutmix', default=False, action='store_true', help='use cut mix and mix up')
parser.add_argument('--lr_min', type=float, default=0, help='min learning rate')
parser.add_argument('--opt', type=str, default="SGD", help='optimizer to use')
parser.add_argument('--eps', type=float, default=1e-8, help='opt eps')
parser.add_argument('--sch', type=str, default="cos", help='lr schedular to use')
parser.add_argument('--accumulate', type=int, default=1, help='gradient accumulation')
parser.add_argument('--config', type=str, default="CPolyNeXt_T", help='config to use for model')
parser.add_argument('--workers', type=int, default=1, help='number of workers to load dataset')
parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint')
parser.add_argument('--chk_path', type=str, default="", help='path to store checkpoints')
parser.add_argument('--rampout', type=int, default=-1, help='ramp dropout')
parser.add_argument('--wandb_off', default=False, action='store_true', help='turn off wandb')

import wandb
from dotenv import load_dotenv
load_dotenv()
args = parser.parse_args()

args.start_epoch = 0

if args.debug:
  args.save += "-debug"

if __name__ == '__main__':
  wandb_name = args.save[:-1][1:] if args.save[0] == "'" else args.save
  args.save = 'logs/eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

args.save = './'
CIFAR_CLASSES = 10

if args.set=='cifar100' or args.set == "imagenet100":
    CIFAR_CLASSES = 100
elif args.set == "imagenet1000":
    CIFAR_CLASSES = 1000

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  run = wandb.init(config=args, name=wandb_name, settings=wandb.Settings(init_timeout=120), mode="disabled" if args.wandb_off else "online")

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  if args.config != "LowRes":
    model = NetworkPolyImageNet(args.init_channels, CIFAR_CLASSES, args.layers, eval(args.config), args.dropout, args.drop_path_prob)
  else:
    model = NetworkPoly(args.init_channels, CIFAR_CLASSES, args.layers, eval(args.config), args.dropout, args.drop_path_prob)
  model = model.cuda(args.gpu)

  img_size = 32 
  if args.set == "imagenet100" or args.set == "imagenet1000":
    img_size = 224
  flops, params = get_model_complexity_info(model, (3, img_size, img_size), as_strings=False, backend='aten', print_per_layer_stat=False, verbose=False)
  run_str = time.strftime('%Y%m%d-%H%M')
  logging.info("param size traditional = %fMB", utils.count_parameters_in_MB(model))
  logging.info('flops = %fM', flops / 1e6)
  logging.info('param size = %fM', params / 1e6)
  logging.info(run_str)
  if not args.debug:
    model.compile()

  table = wandb.Table(columns=["flops", "params", "config", "path"])
  table.add_data(str(flops / 1e6), str(params / 1e6), str(eval(args.config)), 
                 os.path.join(args.chk_path, args.save, f'{wandb_name+"_" + run_str}-weights.pt'))
  wandb.log({"Model stats": table})

  criterion = nn.CrossEntropyLoss(label_smoothing=args.smooth)
  criterion = criterion.cuda(args.gpu)
  params = model.parameters()
  if args.opt == "adamW":
    optimizer = torch.optim.AdamW(
        params,
        args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True
        )
  elif args.opt == "lamb": #Match amp defaults
    optimizer = Lamb(
        params,
        args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        decoupled_decay=True,
        eps=args.eps
        )
  if args.set == "imagenet100":
    train_transform, valid_transform = utils.transforms_imagenet(args)
  elif args.set == "imagenet1000":
    train_transform, valid_transform = utils.transforms_imagenet(args)
  else:
    train_transform, valid_transform = utils._data_transforms_cifar10(args, img_size)
  print(train_transform)
  if args.set=='cifar100' or (args.set in ["imagenet100", "imagenet1000"] and args.debug):
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.set == "imagenet100":
      root_dir = ""
      train_data = dset.ImageFolder(root_dir+"/train", transform=train_transform)
      valid_data = dset.ImageFolder(root_dir+"/val", transform=valid_transform)
  elif args.set == "imagenet1000":
      root_dir = ""
      train_data = dset.ImageFolder(args.data+root_dir+"/train", transform=train_transform)
      valid_data = dset.ImageFolder(args.data+root_dir+"/val", transform=valid_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  print("Train dataset:")
  print(f"  Total images: {len(train_data)}")
  print(f"  Classes used: {len(set(train_data.targets))}")
  collate_fn = None
  if args.cutmix:
    if args.set == "imagenet1000":
        cutmix = v2.CutMix(num_classes=CIFAR_CLASSES)
        mixup = v2.MixUp(num_classes=CIFAR_CLASSES, alpha=0.8)
    else:
        cutmix = v2.CutMix(num_classes=CIFAR_CLASSES)
        mixup = v2.MixUp(num_classes=CIFAR_CLASSES)
    cm_func = v2.RandomChoice([cutmix, mixup])
    collate_fn = lambda x: cm_func(*torch.utils.data.default_collate(x))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers if platform != "win32" else 0, collate_fn=collate_fn)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers if platform != "win32" else 0)

  scheduler = CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min)
  if args.sch == "linear":
    scheduler = LinearLR(optimizer, 1, args.lr_min/args.learning_rate, float(args.epochs))
  if args.warmup_length > 0:
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor= 1/10,
        end_factor=1.0,
        total_iters=args.warmup_length
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_length,
        eta_min=args.lr_min
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_length]
    )
  best_acc = 0.0
  best_acc5 = 0.0

  if args.resume:
    if os.path.isfile(args.resume):
        logging.info(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
        best_acc5 = checkpoint.get('best_acc5', 0.0)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logging.info(f"Loaded checkpoint '{args.resume}' (epoch {args.start_epoch})")
        scheduler.step()
    else:
        logging.warning(f"No checkpoint found at '{args.resume}'")

  last_time = time.time()
  for epoch in range(args.start_epoch, args.epochs):
    if args.rampout > 0:
      model.set_head_dropout(min(args.dropout, args.dropout * epoch/args.rampout))
      print(f"Head dropout is now: {model.head[2].p.item()}")

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_acc5, valid_obj = infer(valid_queue, model, criterion)
    if valid_acc > best_acc:
        if not args.debug:
          utils.save(model, os.path.join(args.chk_path, args.save, f'{wandb_name+"_" + run_str}-weights.pt'))
        best_acc = valid_acc
    if valid_acc5 > best_acc5:
        best_acc5 = valid_acc5
    
    if not args.debug:
      utils.save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'best_acc': best_acc,
          'best_acc5': best_acc5,
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict()
      }, valid_acc > best_acc, os.path.join(args.chk_path, args.save), wandb_name)
    
    logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
    logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)
    scheduler.step()

    wandb.log({"epoch": epoch, "learning rate": scheduler.get_last_lr()[0], "runtime": time.time()-last_time, "val/best_top1": best_acc, "val/best_top5": best_acc5})
    last_time = time.time()

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    if args.debug and step > 5:
        break
    input = input.cuda(args.gpu, non_blocking=True)
    target = target.cuda(args.gpu, non_blocking=True)

    if step%args.accumulate == 0:
      optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
      logits = model(input)
      loss = criterion(logits, target)
      if args.accumulate > 1:
        loss /= args.accumulate
    loss.backward()
    if (step + 1) % args.accumulate == 0 or (step + 1) == len(train_queue):
      grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 99999999 if args.opt == "lamb" else args.grad_clip)
      optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target.argmax(dim=1) if args.cutmix else target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if (step + 1) % (args.report_freq*args.accumulate) == 0 or (step + 1) == len(train_queue) or args.debug:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      wandb.log({"train/loss": loss.item()*args.accumulate, "train/top1_acc": prec1, "train/top5_acc": prec5, "train/grad_norm": grad_norm})
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      if args.debug and step > 5:
        break

      input = input.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)
      logits = model(input)
      loss = criterion(logits.nan_to_num(), target).clamp(max=10)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)
      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    wandb.log({"val/loss": objs.avg, "val/top1_acc": top1.avg, "val/top5_acc": top5.avg})
    return top1.avg, top5.avg, objs.avg

if __name__ == '__main__':
  main() 

