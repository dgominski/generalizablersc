import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
torchvision.set_image_backend('accimage')
import torchvision.transforms as transforms
from data import CustomDatasetDataLoader
from data.datahelpers import pil_loader
from data.base_dataset import BaseDataset, BaseTestDataset
from data.templates import DatasetFromDataframe
from data import aid_dataset, braziliancoffee_dataset, patternnet_dataset, resisc45_dataset, rsicb_dataset, \
    rsscn7_dataset, siriwhu_dataset, ucm_dataset, whurs19_dataset
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = 1000000000     # to avoid error "https://github.com/zimeon/iiif/issues/11"
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # to avoid error "https://github.com/python-pillow/Pillow/issues/1510"
import random
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate
import os
from scipy.special import softmax


def collate(dictlist):
    returned_dict = {}
    returned_dict['support'] = torch.stack([elem['support'] for elem in dictlist])
    returned_dict['query'] = torch.stack([elem['query'] for elem in dictlist])
    return returned_dict


class CustomBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, dataset_names, dataset_sizes, class_idxs, batch_size, n_samples, epoch_size):
        self.datasets = np.arange(len(dataset_sizes))
        self.datasets_idxs = np.repeat(self.datasets, dataset_sizes)
        self.class_idxs = class_idxs
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.rng = np.random.default_rng()
        self.epoch_size = epoch_size
        self.dataset_name_to_id = {d:i for i,d in enumerate(dataset_names)}

    def __iter__(self):
        self.count = 0
        while self.count < len(self):
            picked_datasets = np.random.choice(self.datasets, self.batch_size, replace=True)
            idxs = [] 
            for d in picked_datasets:
                selected_idxs = np.argwhere(self.datasets_idxs==d)
                classlist = self.class_idxs[selected_idxs]
                unique_classes = np.unique(classlist)
                c = self.rng.choice(unique_classes, 1)
                sameclass = selected_idxs[classlist == c]
                idxs.extend(self.rng.choice(sameclass, self.n_samples, replace=False))
            yield idxs
            self.count += 1

    def __len__(self):
        return self.epoch_size
    

class MultiDataset(torch.utils.data.Dataset):
    """
    Helper class to load corresponding support and query sets.
    __getitem__ returns a support set of 5 random images and a query set of 5 images from the same class
    """
    def __init__(self, opt, trainingdatasets):
        super().__init__()
        self.opt = opt

        self.k = opt.k  # Number of examples per class
        
        self.transform = transforms.Compose((
            transforms.Resize((opt.imsize, opt.imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean,
                                 std=opt.std)
        ))
        self.loader = pil_loader

        self.all_datasets = [
            aid_dataset.AIDDataset(opt=opt),
            patternnet_dataset.PatternNetDataset(opt=opt),
            resisc45_dataset.Resisc45Dataset(opt=opt),
            rsicb_dataset.RSICBDataset(opt=opt),
            rsscn7_dataset.RSSCN7Dataset(opt=opt),
            siriwhu_dataset.SIRIWHUDataset(opt=opt),
            ucm_dataset.UCMDataset(opt=opt),
            whurs19_dataset.WHURS19Dataset(opt=opt),
        ]
        self.all_dataset_names = [d.name for d in self.all_datasets]
        self.datasets = [d for d in self.all_datasets if d.name.lower() in trainingdatasets]
        self.dataset_names = [d.name for d in self.datasets]
        
        self.val_datasets = [d for d in self.all_datasets if d.name.lower() not in trainingdatasets]

        # Split each training dataset with train/val
        self.test_datasets = self.val_datasets
        self.val_datasets = [d.sample_n_samples_per_class(samples_per_class=15, remove=True) for d in self.val_datasets]
        
        print("Multidataset created with datasets {}".format(self.dataset_names))
        for d in self.datasets:
            print("Train split for dataset {} contains {} samples".format(d.name, len(d.df.index)))
        for d in self.val_datasets:
            print("Val split for dataset {} contains {} samples".format(d.name, len(d.df.index)))
        for d in self.test_datasets:
            print("Test split for dataset {} contains {} samples".format(d.name, len(d.df.index)))
        self.qdatasets = list(range(len(self.datasets)))

        # Merging classes
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("airportrunway", 'runway'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("runwaymarking", 'runway'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("baseballfield", 'baseball'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("baseballdiamond", 'baseball'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("commercialarea", 'commercial'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("circularfarmland", 'farmland'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("rectangularfarmland", 'farmland'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("greenfarmland", 'farmland'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("dryfarm", 'farmland'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("parkinglot", 'parking'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("parkingspace", 'parking'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("residents", 'residential'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("denseresidential", 'residential'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("sparseresidential", 'residential'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("mediumresidential", 'residential'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("sandbeach", 'beach'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("snowberg", 'snowmountain'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("industrialarea", 'industrial'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("riverprotectionforest", 'river'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("storagetanks", 'storagetank'))
        #self.df['classname'] = self.df['classname'].apply(lambda x: x.replace("sparseforest", 'forest'))

        #self.df = BaseDataset.quantize(self.df, 'classname', 'qclass')
        #self.df.reset_index(inplace=True, drop=True)
        #self.unique_classes = pd.unique(self.df['qclass'])
        #
        #self.qdatasets = list(range(5))
    
        dataset_sizes = [len(d) for d in self.datasets]
        self.dataset_idxs = np.repeat(np.arange(len(dataset_sizes)), dataset_sizes)
        df = pd.concat([d.df for d in self.datasets], ignore_index=True)
        self.paths = df['path'].to_numpy()
        self.classes = df['qclass'].to_numpy()
         
        # Preprocess: delete all images belonging to a class with less than k examples
        to_delete = []
        for d in range(len(self.datasets)):
            idxs = np.argwhere(self.dataset_idxs==d)
            classes = self.classes[idxs]
            val, counts = np.unique(classes, return_counts=True)
            classes_to_delete = val[counts<5]
            for i,cd in enumerate(classes_to_delete):
                to_delete.extend(np.argwhere(np.logical_and(self.dataset_idxs==d, self.classes==cd)).squeeze().tolist())
        
        np.save("to_delete_idxs.npy", np.array(to_delete))
        self.dataset_idxs = np.delete(self.dataset_idxs, to_delete)
        _, self.dataset_sizes = np.unique(self.dataset_idxs, return_counts=True)
        self.paths = np.delete(self.paths, to_delete)
        self.classes = np.delete(self.classes, to_delete)
        np.save("dataset_idxs.npy", self.dataset_idxs)
        np.save("dataset_names.npy", self.dataset_names)
        np.save("paths.npy", self.paths)
        np.save("classes.npy", self.classes)

    def load_from_disk(self, path):
        self.dataset_idxs = np.load(os.path.join(path, "dataset_idxs.npy"))
        self.paths = np.load(os.path.join(path, "paths.npy"), allow_pickle=True)
        self.classes = np.load(os.path.join(path, "classes.npy"), allow_pickle=True)
        _, self.dataset_sizes = np.unique(self.dataset_idxs, return_counts=True)
        self.dataset_names = np.load("dataset_names.npy")
        return
    
    def __getitem__(self, index):
        path = self.paths[index]
        inputimage = self.loader(path)
        inputimage = self.transform(inputimage) if self.transform else inputimage

        sampleclass = self.classes[index]

        sampledict = {'input': inputimage, 'class': sampleclass, 'path': path, 'dataset': self.dataset_names[self.dataset_idxs[index]]}

        return sampledict 
    
    def update_val_split(self):
        pass

    def set_net(self, net):
        pass
    
    def prepare_epoch(self):
        batch_sampler = CustomBatchSampler(self.dataset_names, self.dataset_sizes, self.classes, self.opt.batch_size, n_samples=int(self.k), epoch_size=int(50000/self.opt.batch_size))
        loader = DataLoader(self, batch_sampler=batch_sampler, num_workers=self.opt.num_threads)
        return loader