import os
import torch
import scipy.io
import h5py
import os.path as osp
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image
import boto3
import gzip 
import pickle

import torch.utils.data as data_utils
from .DownsampledImageNet import ImageNet16
from .SearchDatasetWrap import SearchDataset
from config_utils import load_config
from .download_data import download_from_s3

Dataset2Class = {'cifar10': 10,
                 'cifar100': 100,
                 'imagenet-1k-s': 1000,
                 'imagenet-1k': 1000,
                 'ImageNet16' : 1000,
                 'ImageNet16-150': 150,
                 'ImageNet16-120': 120,
                 'ImageNet16-200': 200,
                 'ninapro': 18,
                 'scifar100': 100
                 }

s3_bucket = "pde-xd"

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class CUTOUT(object):

    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return ('{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


imagenet_pca = {
        'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
        'eigvec': np.asarray([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
        ])
}


class Lighting(object):
    def __init__(self, alphastd,
                  eigval=imagenet_pca['eigval'],
                  eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

def create_grid(sub):
    '''construct a grid for pde data'''
    s = int(((421 - 1) / sub) + 1)
    grids = []
    grids.append(np.linspace(0, 1, s))
    grids.append(np.linspace(0, 1, s))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1, s, s, 2)
    grid = torch.tensor(grid, dtype=torch.float)

    return grid, s

'''sEMG ninapro data'''
def load_ninapro_data(path, train=True):

    trainset = load_ninapro(path, 'train')
    valset = load_ninapro(path, 'val')
    testset = load_ninapro(path, 'test')

    if train:
        return trainset, valset, testset

    else:
        trainset = data_utils.ConcatDataset([trainset, valset])

    return trainset, None, testset

def load_ninapro(path, whichset):
    data_str = 'ninapro_' + whichset + '.npy'
    label_str = 'label_' + whichset + '.npy'

    data = np.load(os.path.join(path, data_str),
                             encoding="bytes", allow_pickle=True)
    labels = np.load(os.path.join(path, label_str), encoding="bytes", allow_pickle=True)

    data = np.transpose(data, (0, 2, 1))
    data = data[:, None, :, :]
    data = torch.from_numpy(data.astype(np.float32))
    labels = torch.from_numpy(labels.astype(np.int64))

    all_data = data_utils.TensorDataset(data, labels)
    return all_data

def load_scifar100_data(path, val_split=0.2, train=True):

    data_file = os.path.join(path, 's2_cifar100.gz')
    with gzip.open(data_file, 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, :, :, :].astype(np.float32))
    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64))


    all_train_dataset = data_utils.TensorDataset(train_data, train_labels)
    print(len(all_train_dataset))
    if val_split == 0.0 or not train:
        val_dataset = None
        train_dataset = all_train_dataset
    else:
        ntrain = int((1-val_split) * len(all_train_dataset))
        train_dataset = data_utils.TensorDataset(train_data[:ntrain], train_labels[:ntrain])
        val_dataset = data_utils.TensorDataset(train_data[ntrain:], train_labels[ntrain:])

    print(len(train_dataset))
    test_data = torch.from_numpy(
        dataset["test"]["images"][:, :, :, :].astype(np.float32))
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    test_dataset = data_utils.TensorDataset(test_data, test_labels)

    return train_dataset, val_dataset, test_dataset

def load_darcyflow_data(path):
    TRAIN_PATH = os.path.join(path, 'piececonst_r421_N1024_smooth1.mat')
    reader = MatReader(TRAIN_PATH)
    r = 5
    grid, s = create_grid(r)
    ntrain = 1000
    ntest = 100

    x_train = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]
    y_train = reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), grid.repeat(ntrain, 1, 1, 1)], dim=3)
    train_data = torch.utils.data.TensorDataset(x_train, y_train)

    TEST_PATH = os.path.join(path, 'piececonst_r421_N1024_smooth2.mat')
    reader = MatReader(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
    y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

    x_test = x_normalizer.encode(x_test)
    x_test = torch.cat([x_test.reshape(ntest, s, s, 1), grid.repeat(ntest, 1, 1, 1)], dim=3)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)

    return train_data, test_data

def get_datasets(name, root, cutout):

    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std  = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std  = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith('imagenet-1k'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith('ImageNet16'):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
    else:
        pass
    # Data Argumentation
    if name == 'cifar10' or name == 'cifar100':
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
    elif name.startswith('ImageNet16'):
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 16, 16)
    elif name == 'tiered':
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(80, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.CenterCrop(80), transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
    elif name.startswith('imagenet-1k'):
        lists = [
                    transforms.Resize((32, 32), interpolation=2),
                    transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)
                ]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32) 
    elif name == 'ninapro':
        xshape = (1, 1, 16, 52)
    elif name == 'scifar100':
        xshape = (1, 3, 60, 60)
    elif name == 'darcyflow':
        xshape = (1, 3, 85, 85) 

    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == 'cifar10':
        train_data = dset.CIFAR10 (root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR10 (root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = dset.CIFAR100(root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR100(root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith('imagenet-1k'):
        train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
        test_data  = dset.ImageFolder(osp.join(root, 'val'),   test_transform)
    elif name == 'ImageNet16':
        train_data = ImageNet16(root, True , train_transform)
        test_data  = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == 'ImageNet16-120':
        train_data = ImageNet16(root, True , train_transform, 120)
        test_data  = ImageNet16(root, False, test_transform , 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == 'ImageNet16-150':
        train_data = ImageNet16(root, True , train_transform, 150)
        test_data  = ImageNet16(root, False, test_transform , 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == 'ImageNet16-200':
        train_data = ImageNet16(root, True , train_transform, 200)
        test_data  = ImageNet16(root, False, test_transform , 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    elif name == "ninapro":
        s3 = boto3.client("s3")
        path = os.path.join(root, 'ninapro_data')
        os.makedirs(path, exist_ok=True)
        download_from_s3(s3_bucket, name, path)
        train_data, _, test_data = load_ninapro_data(path, train=False)
        assert len(train_data) == 3297 and len(test_data) == 659  
    elif name == "scifar100":
        s3 = boto3.client("s3")
        path = os.path.join(root, 'scifar100_data')
        os.makedirs(path, exist_ok=True)
        download_from_s3(s3_bucket, name, path)
        train_data, _, test_data = load_scifar100_data(path, train=False)
        assert len(train_data) == 50000 and len(test_data) == 10000 
    elif name == "darcyflow":
        s3 = boto3.client("s3")
        path = os.path.join(root, 'darcyflow_data')
        os.makedirs(path, exist_ok=True)
        download_from_s3(s3_bucket, name, path)
        train_data, test_data = load_darcyflow_data(path)
        assert len(train_data) == 1000 and len(test_data) == 100 

    else: raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num


def get_nas_search_loaders(train_data, valid_data, dataset, config_root, batch_size, workers):
    if isinstance(batch_size, (list,tuple)):
        batch, test_batch = batch_size
    else:
        batch, test_batch = batch_size, batch_size
    if dataset == 'cifar10':
        cifar_split = load_config('{:}/cifar-split.txt'.format(config_root), None, None)
        train_split, valid_split = cifar_split.train, cifar_split.valid # search over the proposed training and validation set
        # To split data
        xvalid_data  = deepcopy(train_data)
        if hasattr(xvalid_data, 'transforms'): # to avoid a print issue
            xvalid_data.transforms = valid_data.transform
        xvalid_data.transform  = deepcopy( valid_data.transform )
        search_data   = SearchDataset(dataset, train_data, train_split, valid_split)
        # data loader
        search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
        train_loader  = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
        valid_loader  = torch.utils.data.DataLoader(xvalid_data, batch_size=test_batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split), num_workers=workers, pin_memory=True)
    elif dataset == 'cifar100':
        cifar100_test_split = load_config('{:}/cifar100-test-split.txt'.format(config_root), None, None)
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data) ; search_valid_data.transform = train_data.transform
        search_data   = SearchDataset(dataset, [search_train_data,search_valid_data], list(range(len(search_train_data))), cifar100_test_split.xvalid)
        search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
        train_loader  = torch.utils.data.DataLoader(train_data , batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
        valid_loader  = torch.utils.data.DataLoader(valid_data , batch_size=test_batch, shuffle=False, num_workers=workers, pin_memory=True)
    elif dataset == 'ImageNet16-120':
        imagenet_test_split = load_config('{:}/imagenet-16-120-test-split.txt'.format(config_root), None, None)
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data) ; search_valid_data.transform = train_data.transform
        search_data   = SearchDataset(dataset, [search_train_data,search_valid_data], list(range(len(search_train_data))), imagenet_test_split.xvalid)
        search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
        train_loader  = torch.utils.data.DataLoader(train_data , batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
        valid_loader  = torch.utils.data.DataLoader(valid_data , batch_size=test_batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(imagenet_test_split.xvalid), num_workers=workers, pin_memory=True)
    else:
        raise ValueError('invalid dataset : {:}'.format(dataset))
    return search_loader, train_loader, valid_loader
