import torchvision
import pandas as pd
torchvision.set_image_backend('accimage')
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, BaseTestDataset
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = 1000000000     # to avoid error "https://github.com/zimeon/iiif/issues/11"
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # to avoid error "https://github.com/python-pillow/Pillow/issues/1510"
import os
import os.path
import glob


class RSSCN7Dataset(BaseDataset):
    def __init__(self, opt=None):
        self.dataroot = "DATAROOT"
        super().__init__(opt)
        self.transform = transforms.Compose((
            transforms.Resize((self.opt.imsize, self.opt.imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean,
                                 std=opt.std)
        ))

    def load_data(self):
        """ Loads dataset from disk. This method will save a .pkl dataset file to the DATAROOT folder the first time
        the dataset is used and directly read it next time, to avoid systematic parsing.
        """
        if not os.path.exists(os.path.join(self.dataroot, "data.pkl")):
            self.df = pd.DataFrame(data=None, columns=['path', 'classname', 'class'])
            classes = glob.glob(os.path.join(self.dataroot, "*"))
            classes = [c for c in classes if os.path.isdir(c)]
            for i, classname in enumerate(classes):
                tempdf = pd.DataFrame({'classname': os.path.basename(classname), 'class': i, 'path': glob.glob(os.path.join(classname, "*"))})
                self.df = self.df.append(tempdf)
            self.df.reset_index(drop=True, inplace=True)
            self.df.to_pickle(os.path.join(self.dataroot, "data.pkl"))
        else:
            self.df = pd.read_pickle(os.path.join(self.dataroot, "data.pkl"))
        self.df['qclass'] = self.df['class']
        return
    
    def __getitem__(self, index):
        query = self.df.iloc[index]
        path = query.at['path']
        output = self.transform(self.loader(path))

        sampledict = {'input': output, 'path': path}
        return sampledict


class RSSCN7TestDataset(BaseTestDataset):
    def __init__(self, opt=None):
        self.dataroot = "DATAROOT"
        super().__init__(opt)
        self.transform = transforms.Compose((
            #transforms.Resize((self.opt.imsize, self.opt.imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean,
                                 std=opt.std)
        ))

    def load_data(self):
        """ Loads dataset from disk. This method will save a .pkl dataset file to the DATAROOT folder the first time
        the dataset is used and directly read it next time, to avoid systematic parsing.
        """
        if not os.path.exists(os.path.join(self.dataroot, "data.pkl")):
            self.df = pd.DataFrame(data=None, columns=['path', 'classname', 'class'])
            classes = glob.glob(os.path.join(self.dataroot, "*"))
            classes = [c for c in classes if os.path.isdir(c)]
            for i, classname in enumerate(classes):
                tempdf = pd.DataFrame({'classname': os.path.basename(classname), 'class': i, 'path': glob.glob(os.path.join(classname, "*"))})
                self.df = self.df.append(tempdf)
            self.df.reset_index(drop=True, inplace=True)
            self.df.to_pickle(os.path.join(self.dataroot, "data.pkl"))
        else:
            self.df = pd.read_pickle(os.path.join(self.dataroot, "data.pkl"))
        return
