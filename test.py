import os
import glob
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections
import trimesh

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize
import torch_points_kernels as tp
import torch.nn.functional as F

from util.train_utils import *

random.seed(123)
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Classification / Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointweb.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointweb.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    from model.stratified_transformer import Stratified
    cuda = torch.device('cuda:%d' % args.train_gpu[0])
    torch.cuda.set_device(cuda)

    args.patch_size = args.grid_size * args.patch_size
    args.window_size = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
    args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
    args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

    model = Stratified(args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size, \
        args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
        rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz, num_classes=args.classes, \
        ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

    model = model.cuda()

    logger.info(model)
    model_path = os.path.join(args.weight, args.name, args.model_path)
    if os.path.isfile(model_path):
        logger.info("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=cuda)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        # todo what is the code below used for?
        # new_state_dict = collections.OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]
        #     new_state_dict[name.replace("item", "stem")] = v
        # model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))

    test(model)


def data_prepare():
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(args.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    elif args.data_name == 'scannetv2':
        data_list = sorted(os.listdir(args.data_root_val))
        data_list = [item[:-4] for item in data_list if '.pth' in item]
        # data_list = sorted(glob.glob(os.path.join(args.data_root_val, "*.pth")))
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    print("Totally {} samples in val set.".format(len(data_list)))
    return data_list

def data_prepare_custom():
    return glob.glob(os.path.join(args.eval_data_path, '*'))


def data_load_custom(data_path):
    samples, f, colors = load_obj_mesh(data_path)
    name = data_path.split('/')[-1].split('.')[0]

    # remove outlier especially for scannet filter data
    dbscan = DBSCAN(eps=0.1, min_samples=5).fit(samples)
    samples = [samples[dbscan.labels_ == idx] for idx in range(dbscan.labels_.max() + 1) if
               len(samples[dbscan.labels_ == idx]) > 400]

    samples = np.concatenate(samples)
    colors = np.ones(samples.shape)

    print(data_path)
    # todo preprocess - alignment with principal axis, translation with zero center, scale [-3,3]
    if 'cube' not in data_path:
        obb = trimesh.Trimesh(samples).bounding_box_oriented
        delta = -1.5 - np.min(samples, axis=0)[2]
        centroid = obb.centroid.copy()
        centroid[2] = -delta
        samples = samples - centroid

        egien_v = obb.principal_inertia_vectors.copy()
        if 'scannet' in data_path:
            samples = np.matmul(egien_v, samples.T)
            samples = samples.T

    save_obj(os.path.join('tmp', data_path.split('/')[-1].split('.')[0] + '.obj'), samples)
    coord, feat = samples, colors

    idx_data = []

    coord_min = np.min(coord, 0)
    coord -= coord_min
    idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
    for i in range(count.max()):
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
        idx_part = idx_sort[idx_select]
        idx_data.append(idx_part)

    return coord, feat, idx_data, name


def data_load(data_name):
    if args.data_name == 's3dis':
        data_path = os.path.join(args.data_root, data_name + '.npy')
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
    elif args.data_name == 'scannetv2':
        data_path = os.path.join(args.data_root_val, data_name + '.pth')
        data = torch.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[0], data[1], data[2]
        # print("type(coord): {}".format(type(coord)))

    idx_data = []
    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        idx_data.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_data


def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    if args.data_name == 's3dis':
        feat = feat / 255.
    return coord, feat


def test(model):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    args.batch_size_test = 5 
    # args.voxel_max = None
    with torch.no_grad():
        model.eval()

        check_makedirs(os.path.join(args.result_path, args.name, args.eval_folder))
        data_list = data_prepare_custom()
        for idx, item in enumerate(data_list):
            end = time.time()

            coord, feat, idx_data, data_name = data_load_custom(item)
            pred = torch.zeros((len(coord), args.classes)).cuda()
            pred_shift = torch.zeros((len(coord), 3)).cuda()

            idx_size = len(idx_data)
            idx_list, coord_list, feat_list, offset_list  = [], [], [], []
            for i in range(idx_size):
                logger.info('{}/{}: {}/{}/{}, {}'.format(idx + 1, len(data_list), i + 1, idx_size, idx_data[0].shape[0], item))
                idx_part = idx_data[i]
                coord_part, feat_part = coord[idx_part], feat[idx_part]
                if args.voxel_max and coord_part.shape[0] > args.voxel_max:
                    coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
                    while idx_uni.size != idx_part.shape[0]:
                        init_idx = np.argmin(coord_p)
                        dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                        idx_crop = np.argsort(dist)[:args.voxel_max]
                        coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
                        dist = dist[idx_crop]
                        delta = np.square(1 - dist / np.max(dist))
                        coord_p[idx_crop] += delta
                        coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
                        idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(feat_sub), offset_list.append(idx_sub.size)
                        idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
                        # cnt += 1; logger.info('cnt={}, idx_sub/idx={}/{}'.format(cnt, idx_uni.size, idx_part.shape[0]))
                else:
                    coord_part, feat_part = input_normalize(coord_part, feat_part)
                    idx_list.append(idx_part), coord_list.append(coord_part), feat_list.append(feat_part), offset_list.append(idx_part.size)

            batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
                idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i]
                idx_part = np.concatenate(idx_part)
                coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
                feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
                offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
                with torch.no_grad():

                    offset_ = offset_part.clone()
                    offset_[1:] = offset_[1:] - offset_[:-1]
                    batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long().cuda(non_blocking=True)

                    sigma = 1.0
                    radius = 2.5 * args.grid_size * sigma
                    neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord_part, coord_part, mode="partial_dense", batch_x=batch, batch_y=batch)[0]
                    neighbor_idx = neighbor_idx.cuda(non_blocking=True)

                    if args.concat_xyz:
                        feat_part = torch.cat([feat_part, coord_part], 1)

                    pred_part, shift_part = model(feat_part, coord_part, offset_part, batch, neighbor_idx)
                    pred_part = F.softmax(pred_part, -1) # Add softmax

                torch.cuda.empty_cache()
                pred[idx_part, :] += pred_part
                pred_shift[idx_part, :] += shift_part

            # pred = pred / (pred.sum(-1)[:, None]+1e-8)
            pred = pred.max(1)[1].data.cpu().numpy()
            shift_coord = coord + pred_shift.cpu().numpy()  # todo

            # save classification and regression results
            cls_res_path = os.path.join(args.result_path, args.name, args.eval_folder, data_name + '_res.obj')  # todo fix path
            save_obj_color_coding(cls_res_path, coord, pred)
            offset_res_path = os.path.join(args.result_path, args.name, args.eval_folder, data_name + '_res_offset.obj')  # todo fix path
            save_obj_color_coding(offset_res_path, shift_coord, pred)

            # instantiation
            inst_res_path = os.path.join(args.result_path, args.name, args.eval_folder)
            # mask = np.linalg.norm(pred_offset, axis=1) < 1
            # pts, pred_offset, pred = coord[mask], pred_offset[mask], pred[mask]
            instances = instantiation_eval(inst_res_path, data_name, coord, pred_shift.cpu().numpy(), pred)
            if len(instances) < 2:
                return

            # save instances before merging
            for k, inst in enumerate(instances):
                o3d_pts = o3d.geometry.PointCloud()
                o3d_pts.points = o3d.utility.Vector3dVector(inst)
                new_pts = np.asarray(
                    o3d_pts.points)  # np.asarray(o3d_pts.remove_radius_outlier(nb_points=3, radius=0.05)[0].points)

                obb = trimesh.points.PointCloud(new_pts).bounding_box
                save_obj_color_coding(os.path.join(inst_res_path, data_name + '_%d_instance.obj' % k), inst,
                                      np.ones(len(inst)) * (k % 20))
                trimesh.exchange.export.export_mesh(obb,
                                                    os.path.join(inst_res_path, data_name + '_%d_obb.obj' % k))

            # save instances after merging
            inst_list = list(instances)
            cnt, end_cnt = 0, len(instances)
            while (cnt < end_cnt):
                cur_inst = inst_list.pop(0)
                merge_list, remain_list = [], []
                merge_list.append(cur_inst)
                while (len(inst_list) != 0):
                    targ_inst = inst_list.pop(0)
                    # compute iou
                    cur_box = trimesh.points.PointCloud(cur_inst).bounding_box
                    targ_box = trimesh.points.PointCloud(targ_inst).bounding_box
                    cur_box_param = np.concatenate((cur_box.centroid, cur_box.extents))
                    targ_box_param = np.concatenate((targ_box.centroid, targ_box.extents))
                    is_overlap1, is_overlap2 = compute_partial_iou(cur_box_param, targ_box_param)

                    # check placed seamlessly by counting number of points adjacent cur_inst?
                    pc_thre = 0.2
                    num_neighbor = np.sum(np.min(distance.cdist(cur_inst, targ_inst), axis=0) < pc_thre)
                    is_seamless = num_neighbor > 10
                    # print(num_neighbor)
                    if (is_overlap1 or is_overlap2) and is_seamless:
                        # merging
                        merge_list.append(targ_inst)
                        # print('merge!')
                    else:
                        remain_list.append(targ_inst)

                # add to last
                new_inst = np.concatenate(merge_list)
                remain_list.append(new_inst)
                inst_list = remain_list

                cnt += 1

            # save new obb
            for k, inst in enumerate(inst_list):
                o3d_pts = o3d.geometry.PointCloud()
                o3d_pts.points = o3d.utility.Vector3dVector(inst)
                new_pts = np.asarray(o3d_pts.points)
                # np.asarray(o3d_pts.remove_radius_outlier(nb_points=3, radius=0.05)[0].points)

                obb = trimesh.points.PointCloud(new_pts).bounding_box
                save_obj_color_coding(os.path.join(inst_res_path, data_name + '_m_%d_instance.obj' % k), inst,
                                      np.ones(len(inst)) * (k % 20))
                trimesh.exchange.export.export_mesh(obb,
                                                    os.path.join(inst_res_path,
                                                                 data_name + '_m_%d_obb.obj' % k))

            batch_time.update(time.time() - end)

if __name__ == '__main__':
    main()
