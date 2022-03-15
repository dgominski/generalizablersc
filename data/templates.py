"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
# from data.image_folder import make_dataset
# from PIL import Image
import torch.utils.data as data
from data.datahelpers import pil_loader, imresize, cid2filename, glandmarks_loader, make_dataset, default_loader, IMG_EXTENSIONS
import os


class DatasetFromDataframe(data.Dataset):
    """Creates a dataset to load images from a pandas Dataframe linking to all the images.
    Must have a "path" column"""

    def __init__(self, dataframe, label='class', labelname='class', transform=None):
        """
        Args:
            dataframe : dataframe
            label : optional additional label to get from dataframe
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.label = label
        self.labelname = labelname
        self.dataframe = dataframe
        self.dataframe.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):
        """Returns a tuple containing the image path and the set of parameters. """

        samplepath = self.dataframe.at[idx, 'path']

        sample = pil_loader(samplepath)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.label:
            samplelabel = self.dataframe.at[idx, self.label]
            if samplelabel is None:
                samplelabel = 0

            return {'input': sample, self.labelname: samplelabel, 'path': samplepath}
        else:
            return {'input': sample, 'path': samplepath}


class DatasetFromNPArrray(data.Dataset):
    """Creates a dataset to load images from a numpy array linking to all the images.
    First column should be the path to the image, second column the class"""

    def __init__(self, nparray, transform=None):
        """
        Args:
            nparray : dataframe
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.nparray = nparray

    def __len__(self):
        return self.nparray.shape[0]

    def __getitem__(self, idx):
        """Returns a dict containing the loaded image path and the label. """

        samplepath = self.nparray[idx, 0]

        sample = pil_loader(samplepath)
        if self.transform is not None:
            sample = self.transform(sample)

        samplelabel = self.nparray.at[idx, 1]

        return {'input': sample, 'class': samplelabel}


class CachedDatasetFromTensor(data.Dataset):
    """Creates a dataset to load images from a tensor containing all preprocessed images.
    Classes should be given with a second tensor giving class labels."""

    def __init__(self, datatensor, classtensor):
        """
        Args:
            datatensor : tensor with image data
            classtensor : class tensor
        """
        self.datatensor = datatensor
        self.classtensor = classtensor

    def __len__(self):
        return self.datatensor.shape[0]

    def __getitem__(self, idx):
        """Returns a dict containing the loaded image and the label. """

        sample = self.datatensor[idx]
        sampleclass = self.classtensor[idx]
        return {'input': sample, 'class': sampleclass}


class ImagesFromList(data.Dataset):
    """A generic data loader that loads images from a list
        (Based on ImageFolder from pytorch)

    Args:
        root (string): Root directory path.
        images (list): Relative image paths as strings.
        imsize (int, Default: None): Defines the maximum size of longer image side
        bbxs (list): List of (x1,y1,x2,y2) tuples to crop the query images
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        images_fn (list): List of full image filename
    """

    def __init__(self, root, images, imsize=None, bbxs=None, transform=None, loader=pil_loader):

        images_fn = [os.path.join(root,images[i]) for i in range(len(images))]

        if len(images_fn) == 0:
            raise(RuntimeError("Dataset contains 0 images!"))

        self.root = root
        self.images = images
        self.imsize = imsize
        self.images_fn = images_fn
        self.bbxs = bbxs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            image (PIL): Loaded image
        """
        path = self.images_fn[index]
        img = self.loader(path)

        if self.bbxs:
            img = img.crop(self.bbxs[index])

        if self.imsize is not None:
            img = imresize(img, self.imsize)

        if self.transform is not None:
            img = self.transform(img)

        return {'input': img}

    def __len__(self):
        return len(self.images_fn)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
