# ---------------------------------------------------------------
# This file has been modified from following sources: 
# Source:
# 1. https://github.com/NVlabs/LSGM/blob/main/util/ema.py (NVIDIA License)
# 2. https://github.com/NVlabs/denoising-diffusion-gan/blob/main/train_ddgan.py (NVIDIA License)
# ---------------------------------------------------------------

import os
import random
from functools import partial
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import EMNIST, MNIST, CIFAR10, ImageFolder
import io
from torchvision.datasets.vision import VisionDataset
import os.path
from collections.abc import Iterable
import pickle
from torchvision.datasets.utils import verify_str_arg, iterable_to_str
from PIL import Image
import torch.distributions as td

ROOT_PATH = "../../data/"

def get_data_dim(problem_name):
    return {
        '8gaussian': [2],
        '8gaussian-half': [2],
        'checkerboard': [2],
        'spiral': [2],
        'moon': [2],
        '25gaussian': [2],
        'twocircles' : [2],
        'threecircles': [2],
        'moon2spiral': [2],
        'spiral2moon': [2],
        'mnist':       [1,32,32],
        'celeba64':    [3,64,64],
        'cifar10':     [3,32,32],
        'celeba_256':    [3,256,256],
        'lsun':     [3,256,256],
        'emnist2mnist': [1,32,32]
    }.get(problem_name)


def build_boundary_distribution(args):
    args.nu_dim = args.mu_dim = get_data_dim(args.dataset)
    if args.model_name == 'otm':
        args.mu_dim = [192]
    
    prior = build_prior_sampler(args)
    if len(args.nu_dim) == 1: toy = True
    else: toy = False

    data = build_data_sampler(args, toy)

    return data, prior

    
# Prior sampler
def build_prior_sampler(args):
    if args.dataset not in ['spiral2moon', 'moon2spiral', 'emnist2mnist']:
        prior = td.MultivariateNormal(torch.zeros(args.mu_dim), 1*torch.eye(args.mu_dim[-1]))
        return PriorSampler(args, prior)
    
    elif args.dataset == 'moon2spiral':
        prior = Moon(args.batch_size)
        return PriorSampler(args, prior)
    
    elif args.dataset == 'spiral2moon':
        prior = Spiral(args.batch_size)
        return PriorSampler(args, prior)
    
    else:
        args.dataset = 'emnist'
        args.test = False
        sampler = DataSampler(args)
        args.dataset = 'emnist2mnist'
        return sampler

    

def build_prior_test_sampler(args):
    if args.dataset == 'emnist2mnist':
        args.dataset = 'emnist'
        args.test = True
        sampler = DataSampler(args)
        args.dataset = 'emnist2mnist'
        return sampler
    else:
        return build_prior_sampler(args)
        


class PriorSampler: # a dump prior sampler to align with DataSampler
    def __init__(self, args, prior):
        self.prior = prior
        self.batch_size = args.batch_size

        if args.dataset in ['spiral2moon', 'moon2spiral', 'emnist2mnist']:
            self.noise = False
        else:
            self.noise = True

    def sample(self):
        if self.noise:
            return self.prior.sample([self.batch_size])
        else:
            return self.prior.sample()


# Data sampler
def build_data_sampler(args, toy):
    if not toy:
        if args.dataset == 'emnist2mnist': 
            args.dataset = 'mnist'
        sampler = DataSampler(args)
        if args.dataset == 'mnist': 
            args.dataset = 'emnist2mnist'
        
        return sampler
    
    else:
        return {
            '8gaussian': MixMultiVariateNormal,
            '8gaussian-half': MixMultiVariateNormal2,
            'moon': Moon,
            'checkerboard': CheckerBoard,
            'spiral': Spiral,
            'moon2spiral': Spiral,
            '25gaussian': SquareGaussian,
            'spiral2moon': Moon,
            'twocircles': partial(Circles, centers=[[0,0], [0,0]], radius=[4,8], sigmas=[0.2, 0.2]),
            'threecircles': partial(Circles, centers=[[-8,0], [0,0], [8,0]], radius=[4,4,4], sigmas=[0.1, 0.1]),
        }.get(args.dataset)(args.batch_size)


def get_dataloader(args):
    num_workers = 4

    ## Image dataset
    if args.dataset == 'emnist':
        if not args.test:
            dataset = EMNIST(ROOT_PATH, split='letters', train=True, transform=transforms.Compose([
                            transforms.Resize(args.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5), (0.5))]), download=True)
        else:
            dataset = EMNIST(ROOT_PATH, split='letters', train=False, transform=transforms.Compose([
                            transforms.Resize(args.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5), (0.5))]), download=True)

    if args.dataset == 'mnist':
        dataset = MNIST(ROOT_PATH, train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)
    
    elif args.dataset == 'cifar10':
        dataset = CIFAR10(ROOT_PATH, train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                        ]), download=False)
    
    elif args.dataset == 'cifar10+mnist':
        normal_dataset = CIFAR10(ROOT_PATH, train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        
        anomaly_dataset = MNIST(ROOT_PATH, train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset)
    
    elif args.dataset == 'celeba64':
        num_workers = 4
        train_transform = transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = ImageFolder(os.path.join(ROOT_PATH, 'celeba/aligned_celeba'), transform=train_transform)
    
    elif args.dataset == 'celeba_256':
        num_workers = 8
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        dataset = CelebA_HQ(
            root=os.path.join(ROOT_PATH, 'celeba-hq/celeba-256'),
            partition_path=os.path.join(ROOT_PATH,'celeba-hq/list_eval_partition_celeba.txt'),
            mode='train', # 'train', 'val', 'test'
            transform=train_transform,
        )
    
    elif args.dataset == 'lsun':
        train_transform = transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.CenterCrop(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ])
        train_data = LSUN(root=ROOT_PATH, classes=['church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return data_loader


# ------------------------
# Image Datasets
# ------------------------
class DataSampler: # a dump data sampler
    def __init__(self, args):
        self.dataloader = get_dataloader(args)
        self.batch_size = args.batch_size

    def sample(self):
        try: 
            data = next(self.iterloader)
        except:
            self.iterloader = iter(self.dataloader)
            data = next(self.iterloader)
        
        try: data,_=data
        except: pass
        
        return data.float()


class CelebA_HQ(data.Dataset):
    '''Note: CelebA (about 200000 images) vs CelebA-HQ (30000 images)'''
    def __init__(self, root, partition_path, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        # Split train/val/test 
        self.partition_dict = {}
        self.get_partition_label(partition_path)
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.save_img_path()
        print('[Celeba-HQ Dataset]')
        print(f'Train {len(self.train_dataset)} | Val {len(self.val_dataset)} | Test {len(self.test_dataset)}')

        if mode == 'train':
            self.dataset = self.train_dataset
        elif mode == 'val':
            self.dataset = self.val_dataset
        elif mode == 'test':
            self.dataset = self.test_dataset
        else:
            raise ValueError

    def get_partition_label(self, list_eval_partition_celeba_path):
        '''Get partition labels (Train 0, Valid 1, Test 2) from CelebA
        See "celeba/Eval/list_eval_partition.txt"
        '''
        with open(list_eval_partition_celeba_path, 'r') as f:
            for line in f.readlines():
                filenum = line.split(' ')[0].split('.')[0] # Use 6-digit 'str' instead of int type
                partition_label = int(line.split(' ')[1]) # 0 (train), 1 (val), 2 (test)
                self.partition_dict[filenum] = partition_label

    def save_img_path(self):
        for filename in os.listdir(self.root):
            assert os.path.isfile(os.path.join(self.root, filename))
            filenum = filename.split('.')[0]
            label = self.partition_dict[filenum]
            if label == 0:
                self.train_dataset.append(os.path.join(self.root, filename))
            elif label == 1:
                self.val_dataset.append(os.path.join(self.root, filename))
            elif label == 2:
                self.test_dataset.append(os.path.join(self.root, filename))
            else:
                raise ValueError

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.dataset)


class LSUNClass(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        import lmdb
        super(LSUNClass, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        # cache_file = '_cache_' + ''.join(c for c in root if c in string.ascii_letters)
        # av begin
        # We only modified the location of cache_file.
        cache_file = os.path.join(self.root, '_cache_')
        # av end
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        img, target = None, -1
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length


class LSUN(VisionDataset):
    def __init__(self, root, classes='train', transform=None, target_transform=None):
        super(LSUN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.classes = self._verify_classes(classes)

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(
                root=root + '/' + c + '_lmdb',
                transform=transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes):
        categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                      'conference_room', 'dining_room', 'kitchen',
                      'living_room', 'restaurant', 'tower', 'cat']
        dset_opts = ['train', 'val', 'test']

        try:
            verify_str_arg(classes, "classes", dset_opts)
            if classes == 'test':
                classes = [classes]
            else:
                classes = [c + '_' + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = ("Expected type str or Iterable for argument classes, "
                       "but got type {}.")
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr = ("Expected type str for elements in argument classes, "
                          "but got type {}.")
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr.format(type(c)))
                c_short = c.split('_')
                category, dset_opt = '_'.join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class",
                                        iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target

    def __len__(self):
        return self.length

    def extra_repr(self):
        return "Classes: {classes}".format(**self.__dict__)


class AnomalyDataset(data.Dataset):
    def __init__(self, dataset, anomaly_dataset, frac=0.01):
        '''
        dataset : target dataset (CIFAR10)
        anomaly_dataset : anomaly dataset (MNIST)
        frac : fraction of anomaly dataset (p=0.01)
        '''
        try: normal_sample, _ = dataset[0]
        except: normal_sample = dataset[0]
        c, size, _ = normal_sample.shape # [c, w, h]
        
        self.dataset = dataset
        self.anomaly_dataset = anomaly_dataset

        self.num_normal = dataset.__len__()
        self.num_anomaly = int(frac * self.num_normal)
        
        self.ANOMALIES = []
        for i in range(self.num_anomaly):
            # get samples
            x = anomaly_dataset[i]
            try: x, _ = x
            except: pass
            # check if image size is same
            if i==0: assert x.shape[1] == size
            # match the number of channels
            if x.shape[0]==1 and c==3:
                x = x.repeat(3,1,1)
            # append to self.ANOMALIES
            self.ANOMALIES.append(x)
    
    def __getitem__(self, index):
        if index < self.num_normal:
            x = self.dataset[index]
            try: x, _ = x
            except: pass
        else:
            x = self.ANOMALIES[index-self.num_normal]
        
        return x

    def __len__(self):
        return self.num_normal + self.num_anomaly


# ------------------------
# Toy Datasets
# ------------------------
class CheckerBoard:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        n = self.batch_size
        n_points = 3*n
        n_classes = 2
        freq = 5
        x = np.random.uniform(-(freq//2)*np.pi, (freq//2)*np.pi, size=(n_points, n_classes))
        mask = np.logical_or(np.logical_and(np.sin(x[:,0]) > 0.0, np.sin(x[:,1]) > 0.0), \
        np.logical_and(np.sin(x[:,0]) < 0.0, np.sin(x[:,1]) < 0.0))
        y = np.eye(n_classes)[1*mask]
        x0=x[:,0]*y[:,0]
        x1=x[:,1]*y[:,0]
        sample=np.concatenate([x0[...,None],x1[...,None]],axis=-1)
        sqr=np.sum(np.square(sample),axis=-1)
        idxs=np.where(sqr==0)
        sample=np.delete(sample,idxs,axis=0)
        sample=torch.Tensor(sample)
        sample=sample[0:n,:]
        return sample / 3.


class Spiral:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        n = self.batch_size
        theta = np.sqrt(np.random.rand(n))*3*np.pi-0.5*np.pi # np.linspace(0,2*pi,100)

        r_a = theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + 0.25*np.random.randn(n,2)
        samples = np.append(x_a, np.zeros((n,1)), axis=1)
        samples = samples[:,0:2]
        return torch.Tensor(samples)


class Moon:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        n = self.batch_size
        x = np.linspace(0, np.pi, n // 2)
        u = np.stack([np.cos(x) + .5, -np.sin(x) + .2], axis=1) * 10.
        u += 0.5*np.random.normal(size=u.shape)
        v = np.stack([np.cos(x) - .5, np.sin(x) - .2], axis=1) * 10.
        v += 0.5*np.random.normal(size=v.shape)
        x = np.concatenate([u, v], axis=0)
        return torch.Tensor(x)


class MixMultiVariateNormal:
    def __init__(self, batch_size, radius=12, num=8, sigma=0.4):

        # build mu's and sigma's
        arc = 2*np.pi/num
        xs = [np.cos(arc*idx)*radius for idx in range(num)]
        ys = [np.sin(arc*idx)*radius for idx in range(num)]
        mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]
        dim = len(mus[0])
        sigmas = [sigma*torch.eye(dim) for _ in range(num)] 

        if batch_size%num!=0:
            raise ValueError('batch size must be devided by number of gaussian')
        self.num = num
        self.batch_size = batch_size
        self.dists=[
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def sample(self):
        ind_sample = self.batch_size/self.num
        samples=[dist.sample([int(ind_sample)]) for dist in self.dists]
        samples=torch.cat(samples,dim=0)
        return samples


class MixMultiVariateNormal2:
    def __init__(self, batch_size, radius=12, num=8, sigma=0.4):

        # build mu's and sigma's
        arc = 2*np.pi/num
        xs = [np.cos(arc*idx)*radius for idx in range(num//2)]
        ys = [np.sin(arc*idx)*radius for idx in range(num//2)]
        mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]
        dim = len(mus[0])
        sigmas = [sigma*torch.eye(dim) for _ in range(num//2)] 

        if batch_size%num!=0:
            raise ValueError('batch size must be devided by number of gaussian')
        self.num = num
        self.batch_size = batch_size
        self.dists=[
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def sample(self):
        ind_sample = 2*self.batch_size/self.num
        samples=[dist.sample([int(ind_sample)]) for dist in self.dists]
        samples=torch.cat(samples,dim=0)
        return samples


class SquareGaussian:
    def __init__(self, batch_size, num=25, sigma=0.005):

        # build mu's and sigma's
        x1,x2,x3,x4,x5 = -6,-3,0,3,6
        xs = [x1]*5+[x2]*5+[x3]*5+[x4]*5+[x5]*5
        ys = [x1,x2,x3,x4,x5]*5
        mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]
        dim = len(mus[0])
        sigmas = [sigma*torch.eye(dim) for _ in range(num)] 

        if batch_size%num!=0:
            raise ValueError('batch size must be devided by number of gaussian')
        self.num = num
        self.batch_size = batch_size
        self.dists=[
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def sample(self):
        ind_sample = self.batch_size/self.num
        samples=[dist.sample([int(ind_sample)]) for dist in self.dists]
        samples=torch.cat(samples,dim=0)
        return samples
    

class Circles:
    def __init__(self, batch_size, centers, radius, sigmas):
        assert len(centers) == len(radius)
        assert  len(radius) == len(sigmas)
        assert batch_size % len(centers) == 0
        
        self.batch_size = batch_size
        self.num_circles = len(centers)
        self.ind_sample =  self.batch_size // self.num_circles
        
        self.centers = torch.tensor(centers * self.ind_sample, dtype=torch.float32)
        self.radius = torch.tensor(radius * self.ind_sample, dtype=torch.float32)[:,None]
        self.sigmas = torch.tensor(sigmas * self.ind_sample, dtype=torch.float32)[:,None]
    
    def sample(self):
        noise = torch.randn(size=(self.batch_size, 2))
        z = torch.randn(size=(self.batch_size, 2))
        z = z/torch.norm(z, dim=1, keepdim=True)
        return self.centers + self.radius* z + self.sigmas * noise
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('UOTM parameters')
    
    # Experiment description
    parser.add_argument('--dataset', default='cifar10', choices=['checkerboard', '8gaussian', '25gaussian', 'mnist', 'cifar10', 'celeba64', 'lsun', 'celeba_256'], help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32, help='size of image (or data)')
    parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_name', type=str, default='ncsnpp')
    args = parser.parse_args()

    datasampler, priorsampler = build_boundary_distribution(args)

    for i in range(200):
        x = datasampler.sample()
        x = priorsampler.sample()
        print(x.shape)
        print('-------------------------')

    print('Succesfully sampled')
