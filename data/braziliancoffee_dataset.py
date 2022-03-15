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


class BrazilianCoffeeTestDataset(BaseTestDataset):
    def __init__(self, opt=None):
        self.opt = opt
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
        """ Loads dataset from disk. This method will save a .pkl dataset file to the DATAROOT folder the first time
        the dataset is used and directly read it next time, to avoid systematic parsing.
        """
        if not os.path.exists(os.path.join(self.dataroot, "data.pkl")):
            classes = glob.glob(os.path.join(self.dataroot, "*"))
            images = []
            txtfiles = []
            for i, classname in enumerate(classes):
                if ".txt" in classname:
                    txtfiles.append(pd.read_csv(classname, sep=" ", header=None))
                else:
                    images.extend(glob.glob(os.path.join(classname, "*")))
            gt = pd.concat(txtfiles)
            gt.columns = ['fullname']
            gt['classname'] = gt.fullname.str.split(".", expand=True)[0]
            gt['filename'] = gt.fullname.str.split("coffee.", expand=True)[1].apply(lambda x: x+'.jpg')
            gt.drop(columns='fullname', inplace=True)
            gt.reset_index(drop=True, inplace=True)
            images = pd.DataFrame(data=images, columns=['path'])
            images['filename'] = images.path.str.rsplit("/", n=2).str[-1].apply(str)
            self.df = pd.merge(images, gt, on='filename')
            self.df['class'] = (self.df['classname'] == 'coffee').astype(int)
            self.df.reset_index(drop=True, inplace=True)
            self.df.to_pickle(os.path.join(self.dataroot, "data.pkl"))
        else:
            self.df = pd.read_pickle(os.path.join(self.dataroot, "data.pkl"))
        return
