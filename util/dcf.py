import os
import numpy as np
import SharedArray as SA

import torch
from torch.utils.data import Dataset

from util.voxelize import voxelize
from util.data_util import sa_create, collate_fn
from util.data_util import data_prepare_dcf as data_prepare
import glob


class DCF(Dataset):
    def __init__(self,  split='train', data_root='trainval', voxel_size=0.04, sigma=0.02, voxel_max=None, shuffle_index=False, coord_move=True):
        super().__init__()

        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.sigma = sigma
        self.shuffle_index = shuffle_index
        self.coord_move = coord_move
        if split == "train":
            # self.root = os.path.join(self.data_root, 'train')
            train_flg = 'train'
        else:
            # self.root = os.path.join(self.data_root, 'test')
            train_flg = 'test'
        self.data_path = self.load_path_dic(self.data_root, train_flg)

        print("voxel_size: ", voxel_size)
        print("Totally {} samples in {} set.".format(len(self.data_path), split))

    def load_path_dic(self, root_dir, train_flg):
        '''
        load dictionary type data
        root_dir, [root1, root2, root3,...]
        '''
        total_path = []
        for root in root_dir:
            path_ = [f for f in glob.glob(os.path.join(root, train_flg, '*')) if 'cube' in f]
            total_path += path_
        return total_path  # [f for f in glob.glob(os.path.join(root_dir, '*')) if 'cube' in f]


    def load_item(self, d_path):
        cubes = np.load(d_path, allow_pickle=True)
        sample_list, label_list, offset_list, id_list, param_list = [], [], [], [], []
        for i, cube in enumerate(cubes):
            samples = np.vstack((cube.get('f_samples'), cube.get('e_samples')))
            labels = np.concatenate((cube.get('f_labels'), cube.get('e_labels')))
            offsets = np.vstack((cube.get('f_offsets'), cube.get('e_offsets')))


            ids = np.ones(samples.shape[0])*i
            param = np.hstack((cube.get('centroid'), cube.get('lengths')))

            sample_list.append(samples)
            label_list.append(labels)
            offset_list.append(offsets)
            id_list.append(ids)
            param_list.append(param)

        data = np.concatenate(sample_list)
        labels = np.concatenate(label_list)
        offsets = np.concatenate(offset_list)
        params = np.asarray(param_list)
        feat = np.ones(data.shape)

        return data, labels, offsets, feat, params

    def __getitem__(self, idx):
        sid = idx % len(self.data_path)
        coord, label, offset, feat, box_labels = self.load_item(self.data_path[sid])

        # add random noise
        coord = coord + np.random.normal(scale=self.sigma, size=coord.shape)

        # random translation
        delta_trans = np.random.normal(scale=0.1, size=[1, 3])
        coord = coord + delta_trans

        coord, feat, label, offset = data_prepare(coord, feat, label, offset, self.split, self.voxel_size, self.voxel_max, self.shuffle_index, self.coord_move)
        return coord, feat, label, offset

    def __len__(self):
        # return len(self.data_idx) * self.loop
        return len(self.data_path)

