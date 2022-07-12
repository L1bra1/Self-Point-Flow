import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import numpy as np
import os.path as osp
import pptk

from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True, help="path to the FT3D_s training data")
parser.add_argument('--save_root', type=str, required=True, help="save path")
parser.add_argument('--num_points', type=int, required=False, default=32768, help='number of points in each training scene')
parser.add_argument('--DEPTH_THRESHOLD', required=False, default=35.0)

args = parser.parse_args()

data_root = args.data_root
save_root = args.save_root
num_points = args.num_points
DEPTH_THRESHOLD = args.DEPTH_THRESHOLD

os.makedirs(save_root, exist_ok=True)

root = osp.realpath(osp.expanduser(data_root))
all_paths = os.walk(root)
useful_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])


def process_one_file(index):
    # print(index)
    fn = useful_paths[index]
    pos1 = np.load(os.path.join(fn, 'pc1.npy'))
    pos2 = np.load(os.path.join(fn, 'pc2.npy'))

    # multiply -1 only for subset datasets
    pos1[..., -1] *= -1
    pos2[..., -1] *= -1
    pos1[..., 0] *= -1
    pos2[..., 0] *= -1

    sf = pos2[:, :3] - pos1[:, :3]

    near_mask = np.logical_and(pos1[:, 2] < DEPTH_THRESHOLD, pos2[:, 2] < DEPTH_THRESHOLD)
    indices = np.where(near_mask)[0]

    # No correspondence between the two sampled point clouds
    sample_num = np.min([len(indices), num_points])

    sampled_indices1 = np.random.choice(indices, size=sample_num, replace=False, p=None)
    sampled_indices2 = np.random.choice(indices, size=sample_num, replace=False, p=None)

    pos1 = pos1[sampled_indices1]
    sf = sf[sampled_indices1]
    pos2 = pos2[sampled_indices2]

    # Compute surface normal
    norm1 = pptk.estimate_normals(pos1, k=32, r=0.4, verbose=False, num_procs=16)
    norm2 = pptk.estimate_normals(pos2, k=32, r=0.4, verbose=False, num_procs=16)

    data_name = fn.split('/')[-1]
    os.makedirs(osp.join(save_root, data_name), exist_ok=True)

    np.save(osp.join(save_root, data_name, 'pc1.npy'), pos1)
    np.save(osp.join(save_root, data_name, 'pc2.npy'), pos2)
    np.save(osp.join(save_root, data_name, 'sf.npy'), sf)

    norm1 = norm1.astype(np.float16)
    np.save(osp.join(save_root, data_name, 'norm1.npy'), norm1)

    norm2 = norm2.astype(np.float16)
    np.save(osp.join(save_root, data_name, 'norm2.npy'), norm2)


if __name__ == '__main__':
    list = np.arange(len(useful_paths))
    pool = Pool(4)
    pool.map(process_one_file, list)
    pool.close()
    pool.join()