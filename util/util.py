"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import re
from torch.nn import functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def sdmkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def tensor2im(input_image, imtype=np.uint8,std=[0.5,0.5,0.5],mean=[0.5,0.5,0.5]):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = np.transpose(image_numpy,(1,2,0))
        for k in range(0,3):
            image_numpy[:,:,k] = (image_numpy[:,:,k] * std[k] + mean[k])*255.0 
#        image_numpy = inv_transforms(image_numpy)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_floats_from_string(string):
    return re.findall("\d+\.\d+", string)

def string_3x3array_to_numpy(string):
    floats = get_floats_from_string(string)
    nb_floats = len(floats)
    if nb_floats != 9:
        print("Cannot build a 3x3 array")
    else:
        return np.array(floats).reshape((3,3)).astype(np.float)

def string_vector_to_numpy(string):
    floats = get_floats_from_string(string)
    floats = [float(s) for s in floats]
    return np.array(floats)

def display(img, denormalize=True, hold=False):
    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img[0, :, :, :]
        img = img.detach().cpu().numpy()
    if isinstance(img, Image.Image):
        img = np.array(img)
    # if isinstance(img, accimage.Image):
    #     tmpimg = img
    #     img = np.zeros([img.channels, img.height, img.width], dtype=np.float32)
    #     tmpimg.copyto(img)
    if isinstance(img, np.ndarray):
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[0] == 1:
            img = img.squeeze(axis=0)
        if denormalize:
            img = (img - np.min(img)) / np.ptp(img)
    else:
        raise ValueError("Unknown image type")
    plt.figure()
    plt.imshow(img)
    if not hold:
        plt.show()
    return

class PrintPILImageSize(object):
    """Prints the size of the input PIL Image for debugging
    """

    def __call__(self, img):
        print(img.size)
        return img

    def __repr__(self):
        return self.__class__.__name__

class PrintTensorSize(object):
    """Prints the size of the input PIL Image for debugging
    """

    def __call__(self, tensor):
        print(tensor.shape)
        return tensor

    def __repr__(self):
        return self.__class__.__name__


def get_2d_gaussian(center, device, size=(256, 256), mu=0.0, sigma=2.0, minimum=0.5):
    batch_size = center[0].shape[0]
    x, y = torch.meshgrid(torch.linspace(-1, 1, size[0]), torch.linspace(-1, 1, size[1]))
    x = torch.cat(batch_size * [x[None, :, :]], dim=0).float().to(device)
    y = torch.cat(batch_size * [y[None, :, :]], dim=0).float().to(device)
    x -= (2*center[0].float().unsqueeze(-1).unsqueeze(-1)/(size[0]-1)) - 1
    y -= (2*center[1].float().unsqueeze(-1).unsqueeze(-1)/(size[1]-1)) - 1
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    g[g <= minimum] = minimum
    return g


def denormalize_image(image):
    "denormalizes the input image for displaying"
    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image - np.min(image))/np.ptp(image)
    return image


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def random_bbox(config, batch_size):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including img

    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width, _ = config['image_shape']
    h, w = config['mask_shape']
    margin_height, margin_width = config['margin']
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []
    if config['mask_batch_same']:
        t = np.random.randint(margin_height, maxt)
        l = np.random.randint(margin_width, maxl)
        bbox_list.append((t, l, h, w))
        bbox_list = bbox_list * batch_size
    else:
        for i in range(batch_size):
            t = np.random.randint(margin_height, maxt)
            l = np.random.randint(margin_width, maxl)
            bbox_list.append((t, l, h, w))

    return torch.tensor(bbox_list, dtype=torch.int64)


def test_random_bbox():
    image_shape = [256, 256, 3]
    mask_shape = [128, 128]
    margin = [0, 0]
    bbox = random_bbox(image_shape)
    return bbox


def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
    batch_size = bboxes.size(0)
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    for i in range(batch_size):
        bbox = bboxes[i]
        delta_h = np.random.randint(max_delta_h // 2 + 1)
        delta_w = np.random.randint(max_delta_w // 2 + 1)
        mask[i, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.
    return mask


def test_bbox2mask():
    image_shape = [256, 256, 3]
    mask_shape = [128, 128]
    margin = [0, 0]
    max_delta_shape = [32, 32]
    bbox = random_bbox(image_shape)
    mask = bbox2mask(bbox, image_shape[0], image_shape[1], max_delta_shape[0], max_delta_shape[1])
    return mask


def local_patch(x, bbox_list):
    assert len(x.size()) == 4
    patches = []
    for i, bbox in enumerate(bbox_list):
        t, l, h, w = bbox
        patches.append(x[i, :, t:t + h, l:l + w])
    return torch.stack(patches, dim=0)

def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x

def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

def getrecursivefilelist(dirName, extensions=None):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()

    if extensions and not sum(['.' in ext for ext in extensions]):
        raise Warning("getListOfFiles method - Extensions should be given with a dot")

    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getrecursivefilelist(fullPath, extensions=extensions)
        else:
            if extensions:
                filename, ext = os.path.splitext(fullPath)
                if ext in extensions:
                    allFiles.append(fullPath)
            else:
                allFiles.append(fullPath)
    return allFiles


def scale_feature(feature, min, max):
    return (feature - min) / (max - min)


def load_state_dict(net, state_dict):
    stlist = list(net.state_dict())
    if not stlist:
        net.load_state_dict(state_dict)
        return
    if "module" in list(net.state_dict())[0]:
        if "module" in list(state_dict)[0]:
            net.load_state_dict(state_dict)
        else:
            state_dict = {"module."+x: y for x, y in state_dict.items()}
            net.load_state_dict(state_dict)
    else:
        if "module" in list(state_dict)[0]:
            state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
            net.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)
    return

def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()