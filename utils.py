import os
import tempfile
import numpy as np
import torch
import shutil
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import AutoAugment, AutoAugmentPolicy, InterpolationMode, RandAugment, RandomErasing
from torch.autograd import Variable

from operations import ScalePerChannel

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  """Compute the top1 and top5 accuracy

  """
  maxk = max(topk)
  batch_size = target.size(0)

  # Return the k largest elements of the given input tensor
  # along a given dimension -> N * k
  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].sum().float()
    res.append(correct_k.mul_(100.0/batch_size))
  return res

from rand_augment import rand_augment_transform
from resize import RandomResizedCropAndInterpolation
def transforms_imagenet(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            # RandomResizedCropAndInterpolation(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            RandomErasing(p=0.25, value='random')
        ])
    valid_transforms = transforms.Compose([
            transforms.Resize(232, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    if args.auto_aug:
        train_transforms.transforms.insert(2, rand_augment_transform("rand-m9-mstd0.5-inc1", {}))
    if args.rand_interp:
       train_transforms.transforms[0] = RandomResizedCropAndInterpolation(224)
    return train_transforms, valid_transforms

# Normalization statistics (CIFAR-10 values, used for all low-resolution datasets).
LOWRES_MEAN = [0.49139968, 0.48215827, 0.44653124]
LOWRES_STD = [0.24703233, 0.24348505, 0.26158768]

def transforms_lowres(size, hflip=True, auto_aug=False):
    """Augmentation for the low-resolution datasets (CIFAR-10, SVHN, Tiny-ImageNet).

    Training is at native resolution, so size is 32 (CIFAR-10, SVHN) or 64
    (Tiny-ImageNet). Horizontal flip is disabled for SVHN, whose house-number
    digits are not flip-invariant.
    """
    normalize = transforms.Normalize(mean=LOWRES_MEAN, std=LOWRES_STD)

    train_list = [RandomResizedCropAndInterpolation(size)]
    if hflip:
        train_list.append(transforms.RandomHorizontalFlip())
    train_list += [transforms.ToTensor(), normalize, RandomErasing(p=0.25, value='random')]
    if auto_aug:
        train_list.insert(1, rand_augment_transform("rand-m9-mstd0.5-inc1", {}))

    train_transforms = transforms.Compose(train_list)
    valid_transforms = transforms.Compose([
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, valid_transforms

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def atomic_torch_save(state, filename):
    """Safely save to `filename` by writing to a temp file first, then
    atomically replacing the target."""
    dir_ = os.path.dirname(filename) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filename)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

def save_checkpoint(state, is_best, save, name):
    """Saves model + optimizer + scheduler state to '*_checkpoint.pth.tar' in save/,
       and if is_best is True, also copies it to 'model_best.pth.tar'."""
    filename = os.path.join(save, name + "_checkpoint.pth.tar")

    try:
        atomic_torch_save(state, filename)
    except Exception:
        print("Unable to save checkpoint to", filename)
        fallback = "./" + name + "_checkpoint.pth.tar"
        try:
            atomic_torch_save(state, fallback)
        except Exception:
            print("Fallback save also failed")

    if is_best:
        best_f = os.path.join(save, "model_best.pth.tar")
        dir_ = os.path.dirname(best_f) or "."
        fd, tmp_best = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        os.close(fd)
        try:
            shutil.copyfile(filename, tmp_best)
            os.replace(tmp_best, best_f)
        except Exception:
            try:
                os.unlink(tmp_best)
            except OSError:
                pass
            raise


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

def drop_paths(rate, depth):
   return [x.tolist() for x in torch.linspace(0, rate, depth)]

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists("./logs"):
    os.mkdir("./logs")
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)