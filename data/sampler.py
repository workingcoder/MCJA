"""MCJA/data/sampler.py
   It defines several Sampler classes, designed to facilitate cross-modality (e.g.,RGB and IR) person re-identification.
"""

import copy
import numpy as np

from collections import defaultdict
from torch.utils.data import Sampler


class NormTripletSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    It does not distinguish modalities.

    Args:
    - dataset (Dataset): Instance of dataset class.
    - num_instances (int): Number of instances per identity in a batch.
    - batch_size (int): Number of examples in a batch.
    """

    def __init__(self, dataset, batch_size, num_instances):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(self.dataset.ids):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class CrossModalityRandomSampler(Sampler):
    """
    The first half of a batch are randomly selected RGB images,
    and the second half are randomly selected IR images.

    Args:
    - dataset (Dataset): Instance of dataset class.
    - batch_size (int): Total number of images in a batch.
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.rgb_list = []
        self.ir_list = []
        for i, cam in enumerate(dataset.cam_ids):
            if cam in [3, 6]:
                self.ir_list.append(i)
            else:
                self.rgb_list.append(i)

    def __len__(self):
        return max(len(self.rgb_list), len(self.ir_list)) * 2

    def __iter__(self):
        sample_list = []
        rgb_list = np.random.permutation(self.rgb_list).tolist()
        ir_list = np.random.permutation(self.ir_list).tolist()

        rgb_size = len(self.rgb_list)
        ir_size = len(self.ir_list)
        if rgb_size >= ir_size:
            diff = rgb_size - ir_size
            reps = diff // ir_size
            pad_size = diff % ir_size
            for _ in range(reps):
                ir_list.extend(np.random.permutation(self.ir_list).tolist())
            ir_list.extend(np.random.choice(self.ir_list, pad_size, replace=False).tolist())
        else:
            diff = ir_size - rgb_size
            reps = diff // ir_size
            pad_size = diff % ir_size
            for _ in range(reps):
                rgb_list.extend(np.random.permutation(self.rgb_list).tolist())
            rgb_list.extend(np.random.choice(self.rgb_list, pad_size, replace=False).tolist())

        assert len(rgb_list) == len(ir_list)

        half_bs = self.batch_size // 2
        for start in range(0, len(rgb_list), half_bs):
            sample_list.extend(rgb_list[start:start + half_bs])
            sample_list.extend(ir_list[start:start + half_bs])

        return iter(sample_list)


class CrossModalityIdentitySampler(Sampler):
    """
    The first half of a batch are randomly selected k_size/2 RGB images for each randomly selected p_size people,
    and the second half are randomly selected k_size/2 IR images for each the same p_size people.
    Batch - [id1_rgb, id1_rgb, ..., id2_rgb, id2_rgb, ..., id1_ir, id1_ir, ..., id2_ir, id2_ir, ...]

    Args:
    - dataset (Dataset): Instance of dataset class.
    - p_size (int): Number of identities per batch.
    - k_size (int): Number of instances per identity.
    """

    def __init__(self, dataset, p_size, k_size):
        self.dataset = dataset
        self.p_size = p_size
        self.k_size = k_size // 2
        self.batch_size = p_size * k_size * 2

        self.id2idx_rgb = defaultdict(list)
        self.id2idx_ir = defaultdict(list)
        for i, identity in enumerate(dataset.ids):
            if dataset.cam_ids[i] in [3, 6]:
                self.id2idx_ir[identity].append(i)
            else:
                self.id2idx_rgb[identity].append(i)

        self.num_base_samples = self.dataset.num_ids * self.k_size * 2

        self.num_repeats = len(dataset.ids) // self.num_base_samples
        self.num_samples = self.num_base_samples * self.num_repeats

        # num_ir, num_rgb = 0, 0
        # for c_id in dataset.cam_ids:
        #     if c_id in [3, 6]:
        #         num_ir += 1
        #     else:
        #         num_rgb += 1
        # self.num_repeats = (num_ir * 2) // self.num_base_samples
        # self.num_samples = self.num_base_samples * self.num_repeats

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        sample_list = []

        for r in range(self.num_repeats):
            id_perm = np.random.permutation(self.dataset.num_ids)
            for start in range(0, self.dataset.num_ids, self.p_size):
                selected_ids = id_perm[start:start + self.p_size]

                sample = []
                for identity in selected_ids:
                    replace = len(self.id2idx_rgb[identity]) < self.k_size
                    s = np.random.choice(self.id2idx_rgb[identity], size=self.k_size, replace=replace)
                    sample.extend(s)

                sample_list.extend(sample)

                sample.clear()
                for identity in selected_ids:
                    replace = len(self.id2idx_ir[identity]) < self.k_size
                    s = np.random.choice(self.id2idx_ir[identity], size=self.k_size, replace=replace)
                    sample.extend(s)

                sample_list.extend(sample)

        return iter(sample_list)


class IdentityCrossModalitySampler(Sampler):
    """
    It is equivalent to CrossModalityIdentitySampler, but the arrangement is different.
    Batch - [id1_ir, id1_rgb, id1_ir, id1_rgb, ..., id2_ir, id2_rgb, id2_ir, id2_rgb, ...]

    Args:
    - dataset (Dataset): Instance of dataset class.
    - batch_size (int): Number of examples in a batch.
    - num_instances (int): Number of instances per identity in a batch.
    """

    def __init__(self, dataset, batch_size, num_instances):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_R = defaultdict(list)
        self.index_dic_I = defaultdict(list)
        for i, identity in enumerate(dataset.ids):
            if dataset.cam_ids[i] in [3, 6]:
                self.index_dic_I[identity].append(i)
            else:
                self.index_dic_R[identity].append(i)
        self.pids = list(self.index_dic_I.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic_I[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __len__(self):
        return self.length

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            idxs_R = copy.deepcopy(self.index_dic_R[pid])
            if len(idxs_I) < self.num_instances // 2 and len(idxs_R) < self.num_instances // 2:
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_R = np.random.choice(idxs_R, size=self.num_instances // 2, replace=True)
            if len(idxs_I) > len(idxs_R):
                idxs_I = np.random.choice(idxs_I, size=len(idxs_R), replace=False)
            if len(idxs_R) > len(idxs_I):
                idxs_R = np.random.choice(idxs_R, size=len(idxs_I), replace=False)
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_R)
            batch_idxs = []
            for idx_I, idx_R in zip(idxs_I, idxs_R):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_R)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)
