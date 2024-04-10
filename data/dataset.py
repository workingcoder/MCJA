"""MCJA/data/dataset.py
   It defines dataset classes for handling image data in the context of cross-modality person re-identification task.
"""

import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset


class SYSUDataset(Dataset):
    """
    A dataset class tailored for the SYSU-MM01 dataset, designed to support the loading and preprocessing of data for
    training, querying, and gallery modes in cross-modality (visible and infrared) person re-identification tasks. The
    class handles the specifics of dataset directory structure, selecting appropriate subsets of images based on the
    mode (train, gallery, query) and performing specified transformations on the visible and infrared images separately.
    This class ensures that images are correctly matched with their labels, camera IDs, and other relevant information,
    facilitating their use in training and evaluation of models for person re-identification.

    The constructor of this class takes the root directory of the SYSU-MM01 dataset, the mode of operation (training,
    gallery, or query), and optional transformations for RGB (visible) and IR (infrared) images. If `memory_loaded` is
    set to True, all images are loaded into memory at initialization for faster access during training or evaluation.
    This class is compatible with PyTorch's DataLoader, making it easy to batch and shuffle the dataset as needed.

    Args:
    - root (str): The root directory where the SYSU-MM01 dataset is stored.
    - mode (str): The mode of dataset usage, which can be 'train', 'gallery', or 'query'.
    - transform_rgb (callable, optional): A function that takes in an RGB image and returns a transformed version.
    - transform_ir (callable, optional): A function that takes in an IR image and returns a transformed version.
    - memory_loaded (bool): If True, all images will be loaded into memory at initialization.

    Attributes:
    - img_paths (list): A list of paths to images that belong to the selected mode and IDs.
    - cam_ids (list): Camera IDs corresponding to each image in `img_paths`.
    - num_ids (int): The number of unique identities present in the selected mode.
    - ids (list): A list of identity labels corresponding to each image.
    - img_data (list, optional): If `memory_loaded` is True, this list contains preloaded images from `img_paths`.

    Methods:
    - __len__(): Returns the total number of images in the img_paths.
    - __getitem__(item): Retrieves the image and its metadata at the specified index,
      applying the appropriate transformations based on the camera ID (modality labels).
    """

    def __init__(self, root, mode='train', transform_rgb=None, transform_ir=None, memory_loaded=False):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        if mode == 'train':
            train_ids = open(os.path.join(root, 'exp', 'train_id.txt')).readline()
            val_ids = open(os.path.join(root, 'exp', 'val_id.txt')).readline()

            train_ids = train_ids.strip('\n').split(',')
            val_ids = val_ids.strip('\n').split(',')
            selected_ids = train_ids + val_ids
        else:
            test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
            selected_ids = test_ids.strip('\n').split(',')

        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)

        img_paths = glob(os.path.join(root, '**/*.jpg'), recursive=True)
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]

        if mode == 'gallery':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]
        elif mode == 'query':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]
        self.num_ids = num_ids
        self.transform_rgb = transform_rgb
        self.transform_ir = transform_ir

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

        self.memory_loaded = memory_loaded
        if memory_loaded:
            self.img_data = [Image.open(path) for path in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        if self.memory_loaded:
            img = self.img_data[item]
        else:
            img = Image.open(path)

        label = torch.as_tensor(self.ids[item], dtype=torch.long)
        cam = torch.as_tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.as_tensor(item, dtype=torch.long)

        if cam == 3 or cam == 6:
            if self.transform_ir is not None:
                img = self.transform_ir(img)
        else:
            if self.transform_rgb is not None:
                img = self.transform_rgb(img)

        return img, label, cam, path, item


class RegDBDataset(Dataset):
    """
    A dataset class specifically designed for the RegDB dataset, facilitating the loading and preprocessing of images
    for the cross-modality (visible and infrared) person re-identification task. It supports different operational modes
    including training, gallery, and query, applying distinct preprocessing routines to RGB (visible) and IR (infrared)
    images as specified. This class handles the unique structure of the RegDB dataset, including its division into
    separate visible and thermal image sets, and it prepares the dataset for use in a PyTorch DataLoader, ensuring that
    images are appropriately matched with their labels and camera IDs.

    The constructor of this class takes several parameters including dataset's root directory, the mode of operation,
    optional transformations for both RGB and IR images, and a flag indicating whether images should be loaded into
    memory at initialization. This facilitates faster access during model training and evaluation, especially useful
    when working with large datasets or in environments where I/O speed is a bottleneck.

    Args:
    - root (str): The root directory where the RegDB dataset is stored.
    - mode (str): The mode of dataset usage, which can be 'train', 'gallery', or 'query'.
    - transform_rgb (callable, optional): A function/transform that applies to RGB images.
    - transform_ir (callable, optional): A function/transform that applies to IR images.
    - memory_loaded (bool): If set to True, all images are loaded into memory upfront for faster access.

    Attributes:
    - img_paths (list): A list of paths to images that belong to the selected mode and IDs.
    - cam_ids (list): Camera IDs derived from the image paths, (with visible cameras marked as 2 and thermal as 3).
    - num_ids (int): The number of unique identities present in the selected mode.
    - ids (list): A list of identity labels corresponding to each image.
    - img_data (list, optional): If `memory_loaded` is True, this list contains preloaded images from `img_paths`.

    Methods:
    - __len__(): Returns the total number of images in the img_paths.
    - __getitem__(item): Retrieves the image and its metadata at the specified index,
      applying the appropriate transformations based on the camera ID (modality labels).
    """

    def __init__(self, root, mode='train', transform_rgb=None, transform_ir=None, memory_loaded=False):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        num = '1'
        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_visible_' + num + '.txt', 'r'))
            index_IR = loadIdx(open(root + '/idx/train_thermal_' + num + '.txt', 'r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_visible_' + num + '.txt', 'r'))
            index_IR = loadIdx(open(root + '/idx/test_thermal_' + num + '.txt', 'r'))

        if mode == 'gallery':
            img_paths = [root + '/' + path for path, _ in index_RGB]
        elif mode == 'query':
            img_paths = [root + '/' + path for path, _ in index_IR]
        else:
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]

        selected_ids = [int(path.split('/')[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'Thermal') + 2 for path in img_paths]
        # Note: In SYSU-MM01 dataset, the visible cams are 1 2 4 5, and thermal cams are 3 6.
        # To simplify the code, visible cam is 2 and thermal cam is 3 in RegDB dataset.
        self.num_ids = num_ids
        self.transform_rgb = transform_rgb
        self.transform_ir = transform_ir

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

        self.memory_loaded = memory_loaded
        if memory_loaded:
            self.img_data = [Image.open(path) for path in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        if self.memory_loaded:
            img = self.img_data[item]
        else:
            path = self.img_paths[item]
            img = Image.open(path)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        if cam == 3 or cam == 6:
            if self.transform_ir is not None:
                img = self.transform_ir(img)
        else:
            if self.transform_rgb is not None:
                img = self.transform_rgb(img)

        return img, label, cam, path, item
