import os
from PIL import Image

import torch


def cid2filename(cid, prefix):
    """
    Creates a training image path out of its CID name
    
    Arguments
    ---------
    cid      : name of the image
    prefix   : root directory where images are saved
    
    Returns
    -------
    filename : full image filename
    """
    return os.path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
            img = img.convert('RGB')
        except:
            img = Image.new('RGB', (10, 10))
        return img


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def glandmarks_loader(path):
    from torchvision import get_image_backend
    fullpath = os.path.join('/'.join(path.split('/')[:-1]), path.split('/')[-1][0], path.split('/')[-1][1],
                            path.split('/')[-1][2], path.split('/')[-1]) + '.jpg'
    if get_image_backend() == 'accimage':
        return accimage_loader(fullpath)
    else:
        return pil_loader(fullpath)


def imresize(img, imsize):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]


def collate_tuples_of_three(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]], [batch[0][2]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))], [batch[i][2] for i in range(len(batch))]


def collate_tuplesAsTensors(batch):
    """
    :param batch: list of batchSize tuples with :
            - tuple of 2+nbNeg images
            - target, Tensor size (2+NbNeg)
    :return:
            - images, Tensor shape (batchSize, 2+nbNeg, nbChannels, imageSize, imageSize)
            - target, Tensor shape (batchSize, 2+nbNeg)
    """
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]

    _, nbChannels, imageSize, _ = batch[0][0][0].shape
    images = torch.zeros(len(batch), len(batch[0][0]), 1, nbChannels, imageSize, imageSize)
    targets = torch.zeros(len(batch), len(batch[0][0]))

    for i in range(len(batch)):
        images[i] = torch.stack(batch[i][0])
        targets[i] = batch[i][1]

    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]


def collate_tuplesOfTensors(batch):
    """
    :param batch: list of tuple with :
            - images, list of 2+nbNeg Tensors shape (nbChannels, imageSize, imageSize)
            - target (2+nbNeg)
    :return:
            - images, Tensor shape (batchSize, 2+nbNeg, nbChannels, imageSize, imageSize)
            - target, Tensor shape (batchSize, 2+nbNeg)
    """
    # if len(batch) == 1:
    #     return torch.stack(batch[0][0]).squeeze_(), batch[0][1]

    _, nbChannels, imageSize, _ = batch[0][0][0].shape
    images = torch.zeros(len(batch), len(batch[0][0]), 1, nbChannels, imageSize, imageSize, requires_grad=False)
    targets = torch.zeros(len(batch), len(batch[0][0]), requires_grad=False)

    for i in range(len(batch)):
        images[i] = torch.stack(batch[i][0])
        targets[i] = batch[i][1]

    return torch.squeeze(images, dim=2), targets


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images[:min(max_dataset_size, len(images))]


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
