import torch
import torchvision
import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
torchvision.set_image_backend('accimage')
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, BaseTestDataset
from data.templates import DatasetFromDataframe
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = 1000000000     # to avoid error "https://github.com/zimeon/iiif/issues/11"
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # to avoid error "https://github.com/python-pillow/Pillow/issues/1510"
import glob
import os
import os.path


class UCMDataset(BaseDataset):
    def __init__(self, opt=None):
        self.dataroot = "DATAROOT"
        super().__init__(opt)
        self.transform = transforms.Compose((
            transforms.Resize((self.opt.imsize, self.opt.imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean,
                                 std=opt.std)
        ))
    
    def __getitem__(self, index):
        query = self.df.iloc[index]
        path = query.at['path']
        output = self.transform(self.loader(path))

        sampledict = {'input': output, 'path': path}
        return sampledict

    def load_data(self):
        """ Loads dataset from disk
        """
        if not os.path.exists(os.path.join(self.dataroot, "data.pkl")) or self.opt.no_cache:
            self.df = pd.DataFrame(data=None, columns=['path', 'classname', 'class'])
            classes = glob.glob(os.path.join(self.dataroot, "*"))
            for i, classname in enumerate(classes):
                tempdf = pd.DataFrame({'classname': os.path.basename(classname), 'class': i, 'path': glob.glob(os.path.join(classname, "*.tif"))})
                self.df = pd.concat((self.df, tempdf))
            self.df.reset_index(drop=True, inplace=True)
            if not self.opt.no_cache:
                self.df.to_pickle(os.path.join(self.dataroot, "data.pkl"))
        else:
            self.df = pd.read_pickle(os.path.join(self.dataroot, "data.pkl"))
        self.df['qclass'] = self.df['class']
        return


class UCMTestDataset(BaseTestDataset):
    def __init__(self, opt=None):
        self.dataroot = "DATAROOT"
        super().__init__(opt)
        self.transform = transforms.Compose((
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean,
                                 std=opt.std)
        ))

    def load_data(self):
        """ Loads dataset from disk
        """
        if not os.path.exists(os.path.join(self.dataroot, "data.pkl")) or self.opt.no_cache:
            self.df = pd.DataFrame(data=None, columns=['path', 'classname', 'class'])
            classes = glob.glob(os.path.join(self.dataroot, "*"))
            for i, classname in enumerate(classes):
                tempdf = pd.DataFrame({'classname': os.path.basename(classname), 'class': i, 'path': glob.glob(os.path.join(classname, "*.tif"))})
                self.df = pd.concat((self.df, tempdf))
            self.df.reset_index(drop=True, inplace=True)
            if not self.opt.no_cache:
                self.df.to_pickle(os.path.join(self.dataroot, "data.pkl"))
        else:
            self.df = pd.read_pickle(os.path.join(self.dataroot, "data.pkl"))
        return


if __name__ == '__main__':
    from options.options import Options

    opt = Options().parse()
    refdataset = UCMTestDataset(opt)
