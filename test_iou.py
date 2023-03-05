"""
evaluation code based on AP metric
only for data with bounding box gt
"""
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

# for evaluation
from util.evaluation import DetectionMAP as Evaluate_metric


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

def data_prepare_scannet_val():
    SCAN_NAMES = [os.path.join(args.eval_data_path, line.rstrip() +'_vert.npy') for line in open('meta_data/scannetv2_val.txt')]
    return SCAN_NAMES

# todo for AP evaluation
def data_load_scannet(data_path):

    '''
    scannet data for mAP evaluation
    CLASS_LABELS_SEMANTIC = ('*wall', '*floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                    '*door', '*window', 'bookshelf', 'picture', 'counter', 'desk',
                    'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                    'bathtub', 'otherfurniture', 'ceiling')

    CLASS_LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39, 22)
    '''
    root_dir = args.eval_data_path
    name = data_path.split('/')[-1][:-9]
    if args.eval_type=='scannet':
        filters = [1,2,8,9]
    elif args.eval_type=='s3dis':
        filters = [1,2,22]
    label = np.load(os.path.join(root_dir, name + '_sem_label.npy'))
    coord = np.load(os.path.join(root_dir, name+'_vert.npy'))[:,:3]  # axis-aligned todo need more n. of pts

    for f_idx in filters:
        coord = coord[label!=f_idx]
        label = label[label!=f_idx]

    coord_eval = coord  # original n. of coord for evaluation

    max_pts = 30000
    if len(coord) < max_pts:
        max_pts = len(coord)
    choices = np.random.choice(len(coord), max_pts, replace=False)
    coord, label = coord[choices], label[choices]

    # remove outlier especially for scannet filter data
    dbscan = DBSCAN(eps=0.1, min_samples=5).fit(coord)
    coord = [coord[dbscan.labels_ == idx] for idx in range(dbscan.labels_.max() + 1) if
               len(coord[dbscan.labels_ == idx]) > 50]
    coord = np.concatenate(coord)

    feat = np.ones(coord.shape)
    bboxes = np.load(os.path.join(root_dir, name+'_bbox.npy'))
    bbox_param, bbox_idx = bboxes[:, :-1], bboxes[:, -1:]
    # remove classes to be filtered
    bbox_param = bbox_param[(bbox_idx != filters).all(axis=1)]
    center, length = bbox_param[:, :3], bbox_param[:, 3:]
    box_gt = np.hstack((center - length / 2, center + length / 2))
    # viz box
    # save_obj('tmp/coord.obj', coord)
    # box_list = []
    # for i, param in enumerate(bbox_param):
    #     center, length = param[:3], param[3:]
    #     rigid_mat = np.eye(4)
    #     rigid_mat[:3, -1] = center
    #     box = trimesh.creation.box(extents=length, transform=rigid_mat)
    #     if box_list is None:
    #         box_list = box
    #     else:
    #         box_list = trimesh.util.concatenate(box_list, box)
    # trimesh.exchange.export.export_mesh(box_list, 'tmp/box_list.obj')

    idx_data, coord_min = [],0
    if args.coord_move:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord_eval -= coord_min
        # todo box 도 아예 옮겨버리기
        center -= coord_min
        box_gt = np.hstack((center - length / 2, center + length / 2))


    idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
    for i in range(count.max()):
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
        idx_part = idx_sort[idx_select]
        idx_data.append(idx_part)

    return coord, feat, box_gt, idx_data, name, coord_min, bbox_param, coord_eval

# should not to be used in this file
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
    mAP_CLASSIFICATION = Evaluate_metric(1, ignore_class=[0], overlap_threshold=0.25)
    model.eval()
    total_coverage = []
    # args.voxel_max = None
    with torch.no_grad():
        check_makedirs(os.path.join(args.result_path, args.name, args.eval_folder))
        data_list = data_prepare_scannet_val()  # data_prepare_custom()
        for idx, item in enumerate(data_list):
            print(item)
            # if idx>1000:
            #     break
            if args.is_mIOU and 'vert' not in item:
                continue
            end = time.time()

            coord, feat, gt_box, idx_data, data_name, coord_min, bbox_param, coord_eval = data_load_scannet(item)

            pred = torch.zeros((len(coord), args.classes)).cuda()
            pred_shift = torch.zeros((len(coord), 3)).cuda()
            torch.cuda.empty_cache()
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
                # with torch.no_grad():

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
            # move coord to original coord before instantiation --> why?
            # coord += coord_min
            pred = pred.max(1)[1].data.cpu().numpy()
            shift_coord = coord + pred_shift.cpu().numpy()  # todo

            # ---todo uncomment ----save classification and regression results
            cls_res_path = os.path.join(args.result_path, args.name, args.eval_folder, data_name + '_res.obj')  # todo fix path
            save_obj_color_coding(cls_res_path, coord, pred)
            offset_res_path = os.path.join(args.result_path, args.name, args.eval_folder, data_name + '_res_offset.obj')  # todo fix path
            save_obj_color_coding(offset_res_path, shift_coord, pred)
            # -----------------------------------------------

            # instantiation
            inst_res_path = os.path.join(args.result_path, args.name, args.eval_folder)
            # try:
            instances, polycubes, polycube_merge = instantiation_eval_v2(args, data_name, coord, pred_shift.cpu().numpy(), pred, is_rot=False)
            # except:
            #     print('erorr  occure in ', data_name)
            #     continue

            if len(instances) < 2:
                return

            # continue


            # chamx_list, chamy_list, new_supp = [], [], []
            threshold = 0.2
            # for poly in polycubes:
            #     samples = np.asarray(trimesh.sample.sample_surface_even(poly, 6000)[0])
            #     samples = np.asarray(samples)
            #     new_supp.append(samples)
            # new_supp = np.vstack(new_supp)
            new_supp = trimesh.sample.sample_surface_even(polycube_merge, 60000)[0]
            box_pcd = o3d.geometry.PointCloud()
            box_pcd.points = o3d.utility.Vector3dVector(new_supp)
            in_pcd = o3d.geometry.PointCloud()
            in_pcd.points = o3d.utility.Vector3dVector(coord_eval)
            cham_x = np.asarray(in_pcd.compute_point_cloud_distance(box_pcd))
            cham_y = np.asarray(box_pcd.compute_point_cloud_distance(in_pcd))

            pr_x = len(cham_x[cham_x <= threshold]) / len(cham_x)
            pr_y = len(cham_y[cham_y <= threshold]) / len(cham_y)
            pr = 0.5 * pr_x + 0.5 * pr_y
            total_coverage.append(pr)

            print('%s coverage' % data_name, pr)
            print('%s mean coverage '% data_name, np.mean(total_coverage))

            # continue
            inst_list = list(instances)

            # save instances before merging
            # for k, inst in enumerate(instances):
            #     o3d_pts = o3d.geometry.PointCloud()
            #     o3d_pts.points = o3d.utility.Vector3dVector(inst)
            #     new_pts = np.asarray(
            #         o3d_pts.points)  # np.asarray(o3d_pts.remove_radius_outlier(nb_points=3, radius=0.05)[0].points)

                # obb = trimesh.points.PointCloud(new_pts).bounding_box
                # save_obj_color_coding(os.path.join(inst_res_path, data_name + '_%d_instance.obj' % k), inst,
                #                       np.ones(len(inst)) * (k % 20))
                # trimesh.exchange.export.export_mesh(obb,
                #                                     os.path.join(inst_res_path, data_name + '_%d_obb.obj' % k))


            #-----------------------precision and recall------------------------------------------
            # save instances after merging
            # cnt, end_cnt = 0, len(instances)
            # while (cnt < end_cnt):
            #     cur_inst = inst_list.pop(0)
            #     merge_list, remain_list = [], []
            #     merge_list.append(cur_inst)
            #     while (len(inst_list) != 0):
            #         targ_inst = inst_list.pop(0)
            #         # compute iou
            #         cur_box = trimesh.points.PointCloud(cur_inst).bounding_box
            #         targ_box = trimesh.points.PointCloud(targ_inst).bounding_box
            #         cur_box_param = np.concatenate((cur_box.centroid, cur_box.extents))
            #         targ_box_param = np.concatenate((targ_box.centroid, targ_box.extents))
            #         is_overlap1, is_overlap2 = compute_partial_iou(cur_box_param, targ_box_param)
            #
            #         # check placed seamlessly by counting number of points adjacent cur_inst?
            #         pc_thre = 0.1
            #         num_neighbor = np.sum(np.min(distance.cdist(cur_inst, targ_inst), axis=0) < pc_thre)
            #         # print(num_neighbor)
            #         is_seamless = num_neighbor > 100
            #         # print(num_neighbor)
            #         if (is_overlap1 or is_overlap2) and is_seamless:
            #             # merging
            #             merge_list.append(targ_inst)
            #             # print('merge!')
            #         else:
            #             remain_list.append(targ_inst)
            #
            #     # add to last
            #     new_inst = np.concatenate(merge_list)
            #     remain_list.append(new_inst)
            #     inst_list = remain_list
            #     cnt += 1

            # save new obb
            pred_box = []
            pred_box_viz = None
            for k, inst in enumerate(inst_list):
                o3d_pts = o3d.geometry.PointCloud()
                o3d_pts.points = o3d.utility.Vector3dVector(inst)
                new_pts = np.asarray(o3d_pts.points)

                obb = trimesh.points.PointCloud(new_pts).bounding_box
                # save_obj_color_coding(os.path.join(inst_res_path, data_name + '_m_%d_instance.obj' % k), inst,
                #                       np.ones(len(inst)) * (k % 20))
                # trimesh.exchange.export.export_mesh(obb,
                #                                     os.path.join(inst_res_path,
                #                                                  data_name + '_m_%d_obb.obj' % k))
                if pred_box_viz is None:
                    pred_box_viz = obb
                else:
                    pred_box_viz = trimesh.util.concatenate(pred_box_viz, obb)
                pred_box.append(np.hstack((obb.centroid-obb.extents/2, obb.centroid+obb.extents/2)))

            pred_path = os.path.join(args.result_path, args.name, args.eval_folder, data_name + '_box.obj')
            trimesh.exchange.export.export_mesh(pred_box_viz, pred_path)
            pred_box = np.vstack(pred_box)

            # save gt box
            try:
                box_list = []
                for i, param in enumerate(bbox_param):
                    center, length = param[:3], param[3:]
                    rigid_mat = np.eye(4)
                    rigid_mat[:3, -1] = center
                    box = trimesh.creation.box(extents=length, transform=rigid_mat)
                    if box_list is None:
                        box_list = box
                    else:
                        box_list = trimesh.util.concatenate(box_list, box)
                gt_box_path = os.path.join(args.result_path, args.name, args.eval_folder, data_name + '_box_gt.obj')
                trimesh.exchange.export.export_mesh(box_list, gt_box_path)
            except:
                print('fail to save')
            # ------------------compute mAP (referred by 3D-SIS)------------------------------
            torch.cuda.empty_cache()
            # gt_box = box_gt  # [x1,y1,z1, x2, y2, z2]: shape [n_gt, 6]
            # gt_class = blobs['gt_box'][0][:, 6].numpy()
            # pred_box = clip_boxes(pred_box, net._scene_info[:3]).numpy()  # [x1,y1,z1,x2,y2,z2]: shape [n_pred, 6]

            # sort by confidence
            # sort_index = []
            # for conf_index in range(pred_conf.shape[0]):
            #     if pred_conf[conf_index] > cfg.CLASS_THRESH:
            #         sort_index.append(True)
            #     else:
            #         sort_index.append(False)
            # try:
            print('len of pred box', len(pred_box), 'len of gt box', len(gt_box))
            mAP_CLASSIFICATION.evaluate(
                pred_box, #[sort_index],
                # pred_class, #[sort_index],
                # pred_conf, #[sort_index],
                gt_box)

            # mAP_CLASSIFICATION.finalize_precision()
            # mAP_CLASSIFICATION.finalize_recall()
            # print('precision of box detection: {}'.format(mAP_CLASSIFICATION.mean_precision))
            # print('recall of box detection: {}'.format(mAP_CLASSIFICATION.mean_recall))
            try:
                TP = mAP_CLASSIFICATION.total_accumulators[0].TP
                FP = mAP_CLASSIFICATION.total_accumulators[0].FP
                FN = mAP_CLASSIFICATION.total_accumulators[0].FN
                print('accumulated precision: ', TP / (TP + FP))
                print('accumulated recall: ', TP / (TP + FN))
            except:
                print('fail to compute AP')

        # detected 된 전체 box/ gt 에서 값 추출
        TP = mAP_CLASSIFICATION.total_accumulators[0].TP
        FP = mAP_CLASSIFICATION.total_accumulators[0].FP
        FN = mAP_CLASSIFICATION.total_accumulators[0].FN
        print('final precision: ', TP / (TP + FP))
        print('final recall: ', TP / (TP + FN))

        mAP_CLASSIFICATION.finalize()
        mAP_CLASSIFICATION.finalize_precision()

        print('AP of box detection: {}'.format(mAP_CLASSIFICATION.mAP()))
        print('precision of box detection: {}'.format(mAP_CLASSIFICATION.mean_precision))
        print('recall of box detection: {}'.format(mAP_CLASSIFICATION.mean_recall))
        # -----------------------precision and recall------------------------------------------

if __name__ == '__main__':
    main()
