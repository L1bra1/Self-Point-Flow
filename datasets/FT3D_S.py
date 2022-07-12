"""
References:
HPLFlowNet: https://github.com/laoreja/HPLFlowNet
"""

import sys, os
import os.path as osp
import numpy as np
import pptk
import torch.utils.data as data

__all__ = ['FT3D_S']


class FT3D_S(data.Dataset):
    """
    Generate the FlyingThing3D training dataset following HPLFlowNet

    Parameters
    ----------
    train (bool) : If True, creates dataset from training set, otherwise creates from test set.
    num_points (int) : Number of points in point clouds.
    data_root (str) : Path to dataset root directory.
    """
    def __init__(self,
                 train,
                 num_points,
                 data_root, DEPTH_THRESHOLD = 35.0):
        self.root = osp.join(data_root, 'FlyingThings3D_subset_processed_35m')
        self.train = train
        self.num_points = num_points
        self.samples = self.make_dataset()
        self.DEPTH_THRESHOLD = DEPTH_THRESHOLD


        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.train:
            pc1_transformed, pc2_transformed, sf_transformed, valid, pc1_norm, pc2_norm = self.pc_loader(self.samples[index])
            sf_transformed = np.zeros_like(sf_transformed)

        else:
            pc1_transformed, pc2_transformed, sf_transformed, valid, _, _ = self.pc_loader(self.samples[index])
            pc1_norm = np.zeros_like(sf_transformed)
            pc2_norm = np.zeros_like(sf_transformed)


        return pc1_transformed.astype(np.float32), pc2_transformed.astype(np.float32), \
               pc1_norm.astype(np.float32), pc2_norm.astype(np.float32),\
               sf_transformed.astype(np.float32), self.samples[index], valid


    def make_dataset(self):
        root = osp.realpath(osp.expanduser(self.root))
        root = osp.join(root, 'train_s_norm') if self.train else osp.join(root, 'val')

        all_paths = os.walk(root)
        useful_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])

        try:
            if self.train:
                assert (len(useful_paths) == 19640)
            else:
                assert (len(useful_paths) == 3824)
        except AssertionError:
            print(useful_paths)
            print('len(useful_paths) assert error', len(useful_paths))
            sys.exit(1)

        return useful_paths

    def pc_loader(self, fn):
        try:
            if self.train:
                pc1 = np.load(os.path.join(fn, 'pc1.npy'))
                pc2 = np.load(os.path.join(fn, 'pc2.npy'))
                sf = np.load(os.path.join(fn, 'sf.npy'))
                norm1 = np.load(os.path.join(fn, 'norm1.npy')).astype(np.float32)
                norm2 = np.load(os.path.join(fn, 'norm2.npy')).astype(np.float32)

                # No correspondence between the two sampled point clouds
                sample_num = np.min([pc1.shape[0], self.num_points])

                sampled_indices1 = np.random.choice(np.arange(pc1.shape[0]), size=sample_num, replace=False, p=None)
                sampled_indices2 = np.random.choice(np.arange(pc2.shape[0]), size=sample_num, replace=False, p=None)

                pc1 = pc1[sampled_indices1]
                norm1 = norm1[sampled_indices1]
                sf = sf[sampled_indices1]

                pc2 = pc2[sampled_indices2]
                norm2 = norm2[sampled_indices2]

            else:
                pc1 = np.load(os.path.join(fn, 'pc1.npy'))
                pc2 = np.load(os.path.join(fn, 'pc2.npy'))

                # multiply -1 only for subset datasets
                pc1[..., -1] *= -1
                pc2[..., -1] *= -1
                pc1[..., 0] *= -1
                pc2[..., 0] *= -1

                sf = pc2[:, :3] - pc1[:, :3]

                near_mask = np.logical_and(pc1[:, 2] < self.DEPTH_THRESHOLD, pc2[:, 2] < self.DEPTH_THRESHOLD)
                indices = np.where(near_mask)[0]

                sample_num = np.min([len(indices), 1 * self.num_points])

                sampled_indices1 = np.random.choice(indices, size=sample_num, replace=False, p=None)
                sampled_indices2 = np.random.choice(indices, size=sample_num, replace=False, p=None)

                pc1 = pc1[sampled_indices1]
                sf = sf[sampled_indices1]
                pc2 = pc2[sampled_indices2]
                norm1 = np.copy(pc1)
                norm2 = np.copy(pc2)

            if sample_num < self.num_points:
                subsample_indices = np.concatenate((np.arange(sample_num),
                                                    np.random.choice(sample_num, self.num_points - sample_num, replace=True)),axis=-1)
                pc1 = pc1[subsample_indices, :]
                pc2 = pc2[subsample_indices, :]
                sf = sf[subsample_indices, :]
                norm1 = norm1[subsample_indices, :]
                norm2 = norm2[subsample_indices, :]

            valid = 1

        except:
            pc1 = np.zeros([8192, 3])
            pc2 = np.zeros([8192, 3])
            sf = np.zeros([8192, 3])
            norm1 = np.zeros([8192, 3])
            norm2 = np.zeros([8192, 3])
            valid = 0


        return pc1, pc2, sf, valid, norm1, norm2
