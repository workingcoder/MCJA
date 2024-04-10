"""MCJA/data/__init__.py
   It orchestrates the data handling process, focusing on data transforming and loader for the training and testing.
"""

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from data.dataset import SYSUDataset
from data.dataset import RegDBDataset

from data.sampler import CrossModalityIdentitySampler
from data.sampler import CrossModalityRandomSampler
from data.sampler import IdentityCrossModalitySampler
from data.sampler import NormTripletSampler

from data.transform import WeightedGrayscale
from data.transform import ChannelCutMix
from data.transform import SpectrumJitter
from data.transform import ChannelAugmentation
from data.transform import NoTransform


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle batches of data. This function is designed to process a batch of
    data by separating and recombining the elements of each data point in the batch, except for a specified element
    (e.g., image paths). The recombination is done in such a way that it preserves the integrity of multi-modal data
    or other structured data necessary for model training or evaluation. The function operates by zipping the batch
    (which combines elements from each data point across the batch), then selectively stacking the elements to form a
    new batch tensor. It specifically skips over a index (in this case, the image paths) and reinserts this non-tensor
    data back into its original position in the batch. This approach ensures compatibility with models expecting data
    in a specific format while accommodating for elements like paths that should not be converted into tensors.

    Args:
    - batch (list): A list of tuples, where each tuple represents a data point and contains elements,
      including images, labels, camera IDs, image paths, and image IDs (img, label, cam_id, img_path, img_id).

    Returns:
    - list: A list of tensors and other data types recombined from the input batch, with tensor elements
      stacked along a new dimension and non-tensor elements (e.g., paths) preserved in their original form.
    """

    samples = list(zip(*batch))

    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 3]
    data.insert(3, samples[3])
    return data


def get_train_loader(dataset, root, sample_method, batch_size, p_size, k_size, image_size,
                     random_flip=False, random_crop=False, random_erase=False, color_jitter=False, padding=0,
                     vimc_wg=False, vimc_cc=False, vimc_sj=False, num_workers=4):
    """
    Constructs and returns a DataLoader for training with specific datasets (SYSU or RegDB), incorporating a variety
    of data augmentation techniques and sampling strategies tailored for mixed-modality (visible and infrared) computer
    vision tasks. This function allows for extensive customization of the data preprocessing pipeline, including options
    for random flipping, cropping, erasing, color jitter, and innovative visible-infrared modality coordination (VIMC).
    The sampling strategy for forming batches can be selected from among several options, including norm triplet, cross
    modality random, cross modality identity, and identity cross modality samplers, to suit different training needs
    and objectives. This function plays a critical role in preparing the data for efficient and effective training by
    dynamically adjusting to the specified dataset, sample method, data augmentation preferences, etc.

    Args:
    - dataset (str): Name of the dataset to use ('sysu' or 'regdb').
    - root (str): Root directory where the dataset is stored.
    - sample_method (str): Method used for sampling data points to form batches.
    - batch_size (int): Number of data points in each batch.
    - p_size (int): Number of identities per batch (used in certain sampling methods).
    - k_size (int): Number of instances per identity (used in certain sampling methods).
    - image_size (tuple): The size to which the images are resized.
    - random_flip (bool): Whether to randomly flip images horizontally.
    - random_crop (bool): Whether to randomly crop images.
    - random_erase (bool): Whether to randomly erase parts of images.
    - color_jitter (bool): Whether to apply random color jittering.
    - padding (int): Padding size used for random cropping.
    - vimc_wg (bool): Whether to apply weighted grayscale conversion.
    - vimc_cc (bool): Whether to apply channel cutmix augmentation.
    - vimc_sj (bool): Whether to apply spectrum jitter.
    - num_workers (int): Number of worker threads to use for loading data.

    Returns:
    - DataLoader: A DataLoader object ready for training, with batches formed
      according to the specified sample method and data augmentation settings.
    """

    # Data Transform - RGB ---------------------------------------------------------------------------------------------
    t = [T.Resize(image_size)]

    t.append(T.RandomChoice([
        T.RandomApply([T.ColorJitter(hue=0.20)], p=0.5) if color_jitter else NoTransform(),

        ###### Visible-Infrared Modality Coordinator (VIMC) ######
        WeightedGrayscale(p=0.5) if vimc_wg else NoTransform(),
        ChannelCutMix(p=0.5) if vimc_cc else NoTransform(),
        SpectrumJitter(factor=(0.00, 1.00), p=0.5) if vimc_sj else NoTransform(),
    ]))

    if random_flip:
        t.append(T.RandomHorizontalFlip())

    if random_crop:
        t.append(T.RandomCrop(image_size, padding=padding, fill=127))

    t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if random_erase:
        t.append(T.RandomErasing(value=0, scale=(0.02, 0.30)))

    transform_rgb = T.Compose(t)

    # Data Transform - IR ----------------------------------------------------------------------------------------------
    t = [T.Resize(image_size)]

    if random_flip:
        t.append(T.RandomHorizontalFlip())

    if random_crop:
        t.append(T.RandomCrop(image_size, padding=padding, fill=127))

    t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if random_erase:
        t.append(T.RandomErasing(value=0, scale=(0.02, 0.30)))

    transform_ir = T.Compose(t)

    # Dataset ----------------------------------------------------------------------------------------------------------
    if dataset == 'sysu':
        train_dataset = SYSUDataset(root, mode='train', transform_rgb=transform_rgb, transform_ir=transform_ir)
    elif dataset == 'regdb':
        train_dataset = RegDBDataset(root, mode='train', transform_rgb=transform_rgb, transform_ir=transform_ir)
    else:
        raise NotImplementedError(f'Dataset - {dataset} is not supported')

    # DataSampler ------------------------------------------------------------------------------------------------------
    assert sample_method in ['none', 'norm_triplet',
                             'cross_modality_random',
                             'cross_modality_identity',
                             'identity_cross_modality']
    shuffle = False
    if sample_method == 'none':
        sampler = None
        shuffle = True
    elif sample_method == 'norm_triplet':
        batch_size = p_size * k_size
        sampler = NormTripletSampler(train_dataset, p_size * k_size, k_size)
    elif sample_method == 'cross_modality_random':
        sampler = CrossModalityRandomSampler(train_dataset, batch_size)
    elif sample_method == 'cross_modality_identity':
        batch_size = p_size * k_size
        sampler = CrossModalityIdentitySampler(train_dataset, p_size, k_size)
    elif sample_method == 'identity_cross_modality':
        batch_size = p_size * k_size
        sampler = IdentityCrossModalitySampler(train_dataset, p_size * k_size, k_size)
    ## Note:
    ## When sample_method is in [none, cross_modity_random],
    ## batch_size is adopted, and p_size and k_size are invalid.
    ## When sample_method is in [norm_triplet, cross_modity_identity, identity_cross_modity],
    ## p_size and k_size are adopted, and batch_size is invalid.

    # DataLoader -------------------------------------------------------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler,
                              shuffle=shuffle, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    return train_loader


def get_test_loader(dataset, root, batch_size, image_size, num_workers=4, mode=None):
    """
    Creates and returns DataLoader objects for the gallery and query datasets, intended for use in the testing phase of
    mixed-modality (visible and infrared) computer vision tasks. This function configures data preprocessing pipelines
    with transformations tailored for evaluation, including resizing, channel augmentation based on a specified mode,
    and normalization. It supports various modes for channel augmentation, enabling flexibility in how images are
    processed and potentially enhancing model robustness during evaluation. The function is designed to work with
    specific datasets (SYSU or RegDB), preparing both gallery and query sets for efficient and effective testing.

    Args:
    - dataset (str): The name of the dataset to be used ('sysu' or 'regdb').
    - root (str): The root directory where the dataset is stored.
    - batch_size (int): The number of data points in each batch.
    - image_size (tuple): The size to which the images are resized.
    - num_workers (int): The number of worker threads to use for data loading.
    - mode (str, optional): The mode of channel augmentation to apply to the RGB data.
      Options include None, 'avg', 'r', 'g', 'b', 'rand', 'wg', 'cc', 'sj', with each providing a different manner.

    Returns:
    - tuple: A tuple containing two DataLoader objects, one for the gallery dataset and one for the query dataset,
      both configured for testing with the specified transformations and settings.
    """

    assert mode in [None, 'avg', 'r', 'g', 'b', 'rand', 'wg', 'cc', 'sj']

    # Data Transform - RGB ---------------------------------------------------------------------------------------------
    transform_rgb = T.Compose([
        T.Resize(image_size),
        ChannelAugmentation(mode=mode),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Data Transform - IR ----------------------------------------------------------------------------------------------
    transform_ir = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset ----------------------------------------------------------------------------------------------------------
    if dataset == 'sysu':
        gallery_dataset = SYSUDataset(root, mode='gallery', transform_rgb=transform_rgb, transform_ir=transform_ir)
        query_dataset = SYSUDataset(root, mode='query', transform_rgb=transform_rgb, transform_ir=transform_ir)
    elif dataset == 'regdb':
        gallery_dataset = RegDBDataset(root, mode='gallery', transform_rgb=transform_rgb, transform_ir=transform_ir)
        query_dataset = RegDBDataset(root, mode='query', transform_rgb=transform_rgb, transform_ir=transform_ir)
    else:
        raise NotImplementedError(f'Dataset - {dataset} is not supported')

    # DataLoader -------------------------------------------------------------------------------------------------------
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn,
                                num_workers=num_workers)

    return gallery_loader, query_loader
