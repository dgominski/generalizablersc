"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import copy
import pandas as pd
from data import CustomDatasetDataLoader
from data.datahelpers import pil_loader
from data.templates import DatasetFromDataframe
import os
import torch
import tqdm


class BaseDataset(data.Dataset, ABC):
    """Base class for datasets. Each dataset must have a Pandas Dataframe with columns ['qclass, 'class', 'path'] for every item.
    'qclass' is mandatory for training datasets, it will be generated if not present.
    Data loading is performed here, children classes must therefore define dataset files before calling super().__init__
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.df = pd.DataFrame(columns=['class', 'path', 'qclass'])
        self.df_val = pd.DataFrame(columns=['class', 'path', 'qclass'])
        self.transform = transforms.Compose((
            transforms.Resize(opt.imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean,
                                 std=opt.std)
        ))
        self.val_transform = transforms.Compose((
            transforms.Resize((opt.imsize, opt.imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean,
                                 std=opt.std)
        ))
        self.opt = opt
        self.net = None
        self.name = type(self).__name__.split('Dataset')[0]
        self.unique_classes = None
        self.loader = pil_loader
        self.queries = pd.DataFrame(columns=self.df.columns)
        self.positives = pd.DataFrame(columns=self.df.columns)
        self.negatives = pd.DataFrame(columns=self.df.columns)

        self.load_data()

        if not 'qclass' in self.df.columns:
            self.df = self.quantize(self.df, 'class')
        self.unique_classes = np.unique(self.df['qclass'].values)

        print("dataset {} was created, {} samples, {} classes".format(type(self).__name__, len(self), self.get_num_classes()))

    @abstractmethod
    def load_data(self):
        """ Loads dataset from disk
        """
        return

    def __len__(self):
        return len(self.df.index)

    def __add__(self, other):
        if not isinstance(other, BaseDataset):
            raise TypeError
        self.df = pd.concat([self.df, other.df])
        self.df_val = pd.concat([self.df_val, other.df_val])
        self.unique_classes = np.unique(self.df['class'].values)
        return self

    def __getitem__(self, index):
        pass

    def sample_n_samples_per_class(self, samples_per_class=0, inplace=False, remove=False):
        #print('>> Selecting subset with {} samples per class (0: all samples picked)' .format(samples_per_class))
        to_select = []
        for c in self.unique_classes:
            chosen_class = self.df[self.df['qclass'] == c]
            if len(chosen_class.index) < samples_per_class:
                continue
            if samples_per_class > 0:
                chosen_class = chosen_class.sample(n=samples_per_class)
            to_select.extend(chosen_class.index.to_list())
        newdf = self.df.iloc[to_select]
        if remove:
            self.df = self.df.drop(to_select)
        return self.copy(newdf, inplace=inplace)

    @staticmethod
    def quantize(df, column_to_quantize='class', newname=None):
        new_unique_classes = np.unique(df[column_to_quantize].values)
        reassign_class_dict = {uclass: i for i, uclass in enumerate(new_unique_classes)}
        if newname is None:
            df['q' + column_to_quantize] = df[column_to_quantize].replace(reassign_class_dict)
        else:
            df[newname] = df[column_to_quantize].replace(reassign_class_dict)
        return df

    def prepare_epoch(self):
        return CustomDatasetDataLoader(self, self.opt)

    def get_val_loader(self):
        return torch.utils.data.DataLoader(
                DatasetFromDataframe(dataframe=self.df_val, transform=self.val_transform, label="qclass"),
                batch_size=self.opt.batch_size, shuffle=True, num_workers=int(self.opt.num_threads), pin_memory=True
            )

    def copy(self, newdf, inplace):
        newdf.reset_index(inplace=True, drop=True)
        if not inplace:
            newdataset = copy.deepcopy(self)
            newdataset.df = newdf
            newdataset.unique_classes = np.unique(newdataset.df['qclass'].values)
            return newdataset
        else:
            self.df = newdf
            self.unique_classes = np.unique(self.df['qclass'].values)
            return self

    def get_num_classes(self):
        n_classes = max(self.df['qclass'].max(), self.df_val['qclass'].max()) + 1
        return n_classes

    def check_data(self, fast=False):
        print("Verifying data integrity")
        failed = []
        for i in tqdm.tqdm(range(len(self)), total=len(self)):
            if not fast:
                try:
                   self.__getitem__(i)
                except:
                   failed.append(i)
            else:
                if not os.path.exists(self.df.iloc[i]['path']):
                    failed.append(i)
        print(failed)

    def get_protos_and_loader(self, nprotos):
        """
        Builds class prototypes by randomly selecting opt.nprotos images per class
        Returns prototype classes, prototypes, database loader with protos removed, and ground truth labels for database without protos
        """
        self.paths = self.df['path'].to_numpy()
        self.classes = self.df['class'].to_numpy()
        ps = []
        paths_without_protos = self.paths
        classes_without_protos = self.classes
        protoidxs = []
        for uclass in self.unique_classes:
            idxs = np.random.choice(np.argwhere(classes_without_protos == uclass)[:,0], nprotos, replace=False).tolist()
            protoidxs.extend(idxs)
        paths_without_protos = np.delete(paths_without_protos, protoidxs)
        classes_without_protos = np.delete(classes_without_protos, protoidxs)
        df_without_protos = pd.DataFrame({'path': paths_without_protos, 'class': classes_without_protos})
        prototypes = protoidxs
        
        for i in prototypes:
            ps.append(self.__getitem__(i)['input'])
        ps = torch.stack(ps)
        print("Selected {} class prototypes in {} classes leaving {} images in the set".format(ps.shape[0], len(self.unique_classes), len(df_without_protos.index)))
        return np.repeat(self.unique_classes, nprotos), ps, torch.utils.data.DataLoader(
            DatasetFromDataframe(dataframe=df_without_protos, transform=self.transform),
            batch_size=1 if not self.opt.batch_test else self.opt.batch_test, shuffle=False,
            num_workers=int(self.opt.num_threads), pin_memory=False, drop_last=False
        ), df_without_protos['class'].to_numpy()

    def get_database_loader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=1 if not self.opt.batch_test else self.opt.batch_test, shuffle=False,
            num_workers=int(self.opt.num_threads), pin_memory=False, drop_last=False
            )

    def get_queries_loader(self):
        return torch.utils.data.DataLoader(
            DatasetFromDataframe(dataframe=self.queries, transform=self.transform),
            batch_size=1 if not self.opt.batch_test else self.opt.batch_test, shuffle=False,
            num_workers=int(self.opt.num_threads), pin_memory=False, drop_last=False
        )

    def get_ground_truth(self):
        dic = self.queries.to_dict(orient='index')
        return dic


class BaseTestDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.name = type(self).__name__.split('TestDataset')[0]
        self.unique_classes = np.unique(self.df[self.df['qclass'].notnull()]['qclass'].values)
        if self.opt.batch_test > 1:
            self.transform = transforms.Compose((
                transforms.Resize((self.opt.imsize, self.opt.imsize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=opt.mean,
                                     std=opt.std)
            ))
        else:
            self.transform = transforms.Compose((
                transforms.ToTensor(),
                transforms.Normalize(mean=opt.mean,
                                     std=opt.std)
            ))
        self.load_ground_truth(None)

        self.paths = self.df['path'].to_numpy()
        self.classes = self.df['class'].to_numpy()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            query image (Tensor): Loaded query tensor at inde
        """
        if self.__len__() == 0:
            raise (RuntimeError(
                "List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        path = self.paths[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return{'input': sample, 'path': path}

    def __len__(self):
        return len(self.df.index)

    def __add__(self, other):
        if not isinstance(other, BaseTestDataset):
            raise TypeError
        self.df = pd.concat([self.df, other.df])
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        return self

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(len(self.df.index))
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @abstractmethod
    def load_data(self):
        """ Loads dataset from disk
        """
        return

    def copy(self, newdf, inplace):
        newdf.reset_index(inplace=True, drop=True)
        if not inplace:
            newdataset = copy.deepcopy(self)
            newdataset.df = newdf
            newdataset.unique_classes = np.unique(newdataset.df['qclass'].values)
            return newdataset
        else:
            self.df = newdf
            self.unique_classes = np.unique(self.df['qclass'].values)
            return self

    def load_ground_truth(self, n_queries=None):
        """Loads self.queries: for each query column 'positives' contains the list of indices of positive images.
        Will generate and save the queries.pkl file when calling for the first time.
        :param n_queries:
        """

        if os.path.exists(os.path.join(self.dataroot, "queries.pkl")):
            self.queries = pd.read_pickle(os.path.join(self.dataroot, "queries.pkl"))
            print("-- loading ground truth with {} queries from disk".format(len(self.queries.index)))
            return

        print("Ground truth not found on disk at {}, generating with {}".format(os.path.join(self.dataroot, "queries.pkl"),
                                                                                "n_queries="+str(n_queries) if n_queries else "all images as queries"))
        self.queries = self.generate_ground_truth(self.df, n_queries=n_queries)

        print("Saving generated queries.pkl to {}".format(self.dataroot))
        self.queries.to_pickle(os.path.join(self.dataroot, "queries.pkl"))

    @staticmethod
    def generate_ground_truth(df, n_queries=None):
        """Parses the val part of the dataset and generates the ground_truth"""

        queries = df[df['qclass'].notnull()].sample(n=n_queries, replace=False) if n_queries else df
        queries['positive'] = None
        queries['ignore'] = None

        for idx, query in queries.iterrows():
            # Get all images from the same class in the subset
            positives = df[df['qclass'] == query['qclass']]
            positives = positives.index.tolist()
            positives.remove(idx)
            queries.at[idx, 'positive'] = positives
            queries.at[idx, 'ignore'] = [idx]

        return queries
 
    def get_database_loader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=1 if not self.opt.batch_test else self.opt.batch_test, shuffle=False,
            num_workers=int(self.opt.num_threads), pin_memory=False, drop_last=False
            )

    def get_queries_loader(self):
        return torch.utils.data.DataLoader(
            DatasetFromDataframe(dataframe=self.queries, transform=self.transform),
            batch_size=1 if not self.opt.batch_test else self.opt.batch_test, shuffle=False,
            num_workers=int(self.opt.num_threads), pin_memory=False, drop_last=False
        )

    def get_ground_truth(self):
        dic = self.queries.to_dict(orient='index')
        return dic

    def ranks_to_paths(self, ranks):
        np_paths = self.df['path'].to_numpy()
        return np_paths[ranks]

    def get_protos_and_loader(self, nprotos):
        """
        Builds class prototypes by randomly selecting opt.nprotos images per class
        Returns prototype classes, prototypes, database loader with protos removed, and ground truth labels for database without protos
        """
        ps = []
        paths_without_protos = self.paths
        classes_without_protos = self.classes
        protoidxs = []
        for uclass in self.unique_classes:
            idxs = np.random.choice(np.argwhere(classes_without_protos == uclass)[:,0], nprotos, replace=False).tolist()
            protoidxs.extend(idxs)
        paths_without_protos = np.delete(paths_without_protos, protoidxs)
        classes_without_protos = np.delete(classes_without_protos, protoidxs)
        df_without_protos = pd.DataFrame({'path': paths_without_protos, 'class': classes_without_protos})
        prototypes = protoidxs
        
        for i in prototypes:
            ps.append(self.__getitem__(i)['input'])
        ps = torch.stack(ps)
        print("Selected {} class prototypes in {} classes leaving {} images in test set".format(ps.shape[0], len(self.unique_classes), len(df_without_protos.index)))
        return np.repeat(self.unique_classes, nprotos), ps, torch.utils.data.DataLoader(
            DatasetFromDataframe(dataframe=df_without_protos, transform=self.transform),
            batch_size=1 if not self.opt.batch_test else self.opt.batch_test, shuffle=False,
            num_workers=int(self.opt.num_threads), pin_memory=False, drop_last=False
        ), df_without_protos['class'].to_numpy()
