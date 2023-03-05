import os
import random

import torch
import numpy as np
import itertools
# import trimesh
import open3d as o3d
import trimesh

#import MinkowskiEngine as ME
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

from util.iostream import *

from scipy.spatial.transform import Rotation as R

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def normalize_color(color: torch.Tensor, is_color_in_range_0_255: bool = False) -> torch.Tensor:
    r"""
    Convert color in range [0, 1] to [-0.5, 0.5]. If the color is in range [0,
    255], use the argument `is_color_in_range_0_255=True`.
    `color` (torch.Tensor): Nx3 color feature matrix
    `is_color_in_range_0_255` (bool): If the color is in range [0, 255] not [0, 1], normalize the color to [0, 1].
    """
    if is_color_in_range_0_255:
        color /= 255
    color -= 0.5
    return color.float()


def calc_error(opt, net, cuda, test_data_loader, num_test):
    with torch.no_grad():
        l1loss = torch.nn.L1Loss(reduce=False)
        criterion = torch.nn.CrossEntropyLoss(reduce=False)
        error_arr, offset_arr, inout_arr, angle_arr, scale_arr = [], [], [], [], []
        for idx, test_data in enumerate(test_data_loader):
            if idx == num_test:
                break

            coords, feats, offsets, masks, labels = test_data["coords"], test_data["feats"], test_data["offsets"],test_data["masks"], test_data["labels"]

            sinput = ME.SparseTensor(feats.type(torch.float32), coords, device=cuda)

            out, out_offset, out_inout, out_angle, out_scale = net.forward(sinput)
            error = criterion(out.F, labels.to(cuda).long())
            error_offset = l1loss(out_offset.F, offsets.to(cuda).float())

            error_offset = error_offset[masks] #[masks==1]
            item_offset = error_offset.mean().item()
            offset_arr.append(item_offset)
            error_arr.append(error[masks].mean().item())
    return np.average(offset_arr), np.average(error_arr)


def viz_classification(opt, net, cuda, test_data_loader, num_test=1, epoch=0):
    with torch.no_grad():
        itor = iter(test_data_loader)
        for i in range(num_test):
            item = next(itor)
            coords, feats, labels, masks, offsets, samples, box_labels = \
                item["coords"], item["feats"], item["labels"], item["masks"], \
                item["offsets"], item["samples"], item["boxes"]
            sinput = ME.SparseTensor(feats.type(torch.float32), coords, device=cuda)

            if opt.layout or opt.type:
                out, out_offset, out_layout = net.forward(sinput)
            else:
                out = net.forward(sinput)

            # map to input
            logits = out.slice(sinput).F

            _, pred = logits.max(1)
            pred = pred.cpu().numpy()

            pred_color = np.array([CUBOID_COLOR_MAP[l] for l in pred])
            save_obj('%s/%s/%d_epoch_%d.obj' % (opt.results_path, opt.name, epoch, i), coords[coords[:,0] == 0, 1:], pred_color[coords[:,0] == 0])

            if opt.layout or opt.type:
                logits = out_layout.slice(sinput).F

                _, pred = logits.max(1)
                pred = pred.cpu().numpy()

                pred_color = np.array([CUBOID_COLOR_MAP[l] for l in pred])
                save_obj('%s/%s/%d_epoch_%d_layout.obj' % (opt.results_path, opt.name, epoch, i), coords[coords[:, 0] == 0, 1:],
                         pred_color[coords[:, 0] == 0])


def instantiation(opt, net, cuda, test_dataset, num_test=1, epoch=0):
    samples, pred_offset, pred_labels, pred_layout = get_offset(opt, net, cuda, test_dataset, num_test, epoch, visualize=False)
    # consider points only for objects
    samples, pred_offset, pred_labels = samples[pred_layout == 1], pred_offset[pred_layout == 1], pred_labels[pred_layout == 1]
    # instance segmentation based predicted offsets and labels
    samples_trans = samples + pred_offset
    cls_list, label_list = [], []
    inst_idx, indice_list = 0, []
    for i in range(max(pred_labels)+1):
        index_list = []
        pts_trans = samples_trans[pred_labels == i]
        pts_ori = samples[pred_labels == i]
        save_obj_color_coding('tmp/%d_pts.obj' % i, pts_trans, np.ones(len(pts_trans))*i)

        dbscan = DBSCAN(eps=0.05, min_samples=3).fit(pts_trans)
        instances = [pts_ori[dbscan.labels_ == j] for j in range(max(dbscan.labels_)+1)]
        # [save_obj_color_coding('tmp/%d_%d_instance.obj' % (i, k), item, np.ones(len(item))*k) for k, (item) in enumerate(instances)]

        cls_list.append(instances)
        for k in range(len(instances)):
            index_list.append(inst_idx)
            inst_idx += 1
        indice_list.append(index_list)

    # build face-edge-face connections based on spatial relation between face and edge
    f_cls_list, e_cls_list = cls_list[:6], cls_list[6:]
    pair_list = []
    # adjacent face indices along edge index
    lookup_face = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [2, 4], [3, 4], [1, 5], [2, 5], [3, 5], [4, 5]]
    for cls_idx, e_list in enumerate(e_cls_list):
        # f_idx1, f_idx2의 face instances에 대해 edge instance와 connection 만들기 based on euclidean distance
        f_idx1, f_idx2 = lookup_face[cls_idx][0], lookup_face[cls_idx][1]
        f_list1, f_list2 = f_cls_list[f_idx1], f_cls_list[f_idx2]
        f_inst_id1, f_inst_id2 = indice_list[f_idx1], indice_list[f_idx2]
        # edge를 기준으로 가까운 face search
        for cur_supp in e_list:
            dist1 = np.asarray([np.min(distance.cdist(cur_supp, f_supp)) for f_supp in f_list1])
            dist2 = np.asarray([np.min(distance.cdist(cur_supp, f_supp)) for f_supp in f_list2])
            min_dist1, min_dist_idx1 = np.min(dist1), np.argmin(dist1)
            min_dist2, min_dist_idx2 = np.min(dist2), np.argmin(dist2)
            # print(min_dist1, min_dist2)
            paired = []
            # if closest face, save its instance id
            if min_dist1 < 0.1:
                paired.append(f_inst_id1[min_dist_idx1])
            if min_dist2 < 0.1:
                paired.append(f_inst_id2[min_dist_idx2])
            if paired:
                pair_list.append(paired)

    # viz pairing results
    # item_flatten = list(itertools.chain(*cls_list[:6]))
    # for idx, paired in enumerate(pair_list):
    #     idx1, idx2 = paired[0], paired[0]
    #     if len(paired) == 2:
    #         idx2 = paired[1]
    #     supp = np.concatenate((item_flatten[idx1], item_flatten[idx2]))
    #     save_obj_color_coding('tmp/pair_%d.obj' % idx, supp, np.ones(len(supp)) * idx)
    #     if idx == 10:
    #         break

    # merging paris using set.intersection(set1, set2)
    terminate = False
    final_pair_list=[]
    while terminate is not True:
        is_intersect = False
        new_pair_list = []
        start = pair_list[0]
        pair_list = pair_list[1:]

        new_set = start
        # compare start item and others
        for pair in pair_list:
            intersect = set.intersection(set(start), set(pair))
            if intersect:
                new_set = new_set + pair
                is_intersect = True
            else:
                new_pair_list.append(pair)
        new_pair_list.append(list(set(new_set)))
        pair_list = new_pair_list

        #intersection이 없는 set -> no more need to compare
        if is_intersect is False:
            final_pair_list.append(pair_list[-1])
            pair_list.pop(-1)
            if len(pair_list) == 1:
                final_pair_list.append(pair_list[0])
                terminate = True

    box_supp_list = []
    item_flatten = list(itertools.chain(*cls_list[:6]))
    for i, pair_indices in enumerate(final_pair_list):
        supps = np.concatenate([item_flatten[idx] for idx in pair_indices])
        box_supp_list.append(supps)
        save_obj_color_coding('tmp/objsupp_%d.obj' % i, supps, np.ones(len(supps))*i)

    # grouping by just distance. not consider relations
    # start_insts = cls_list[0] # 0번 label이 기준
    # comp_insts = cls_list[1:]
    # box_list = []
    # for cur_supp in start_insts:
    #     # item: set of pts of each instance
    #     face_list = []
    #     for inst_list in comp_insts:
    #         if not inst_list:
    #             continue
    #         # check only smallest distance between all comparable sets of points (very unstable to noise data)
    #         dist_matrix = np.asarray([np.min(distance.cdist(comp_supp, cur_supp)) for comp_supp in inst_list])
    #         min_dist, min_dist_idx = np.min(dist_matrix), np.argmin(dist_matrix)
    #         if min_dist < 0.18:
    #             face_list.append(inst_list[min_dist_idx])
    #             inst_list.pop(min_dist_idx)
    #     if face_list:
    #         box_list.append(face_list)
    #
    # [save_obj_color_coding('tmp/%d_box_supp.obj' % k, np.concatenate(supp), np.ones(len(np.concatenate(supp)))*k)
    #  for k, (supp) in enumerate(box_list)]
    return None


def instantiation_train(samples, pred_offset, pred_labels):
    samples_trans = samples + pred_offset
    cls_list, label_list = [], []
    inst_idx, indice_list = 0, []
    # applying clustering algorithm for a set of points of each class
    for i in range(max(pred_labels) + 1):
        index_list = []
        pts_trans = samples_trans[pred_labels == i]
        pts_ori = samples[pred_labels == i]
        # save_obj_color_coding(os.path.join(path,'%s_%d_pts.obj' % (name, i)), pts_trans, np.ones(len(pts_trans))*i)

        dbscan = DBSCAN(eps=0.2, min_samples=5).fit(pts_trans)
        instances = [pts_ori[dbscan.labels_ == j] for j in range(max(dbscan.labels_) + 1)]
        # [save_obj_color_coding(os.path.join(path, '%s_%d_%d_instance.obj' % (name, i, k)), item, np.ones(len(item))*k) for k, (item) in enumerate(instances)]

        cls_list.append(instances)
        for k in range(len(instances)):
            index_list.append(inst_idx)
            inst_idx += 1
        indice_list.append(index_list)

    # build face-edge-face connections based on spatial relation between face and edge
    f_cls_list, e_cls_list = cls_list[:6], cls_list[6:]
    pair_list = []
    # adjacent face indices along edge index
    lookup_face = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [2, 4], [3, 4], [1, 5], [2, 5], [3, 5], [4, 5]]
    for cls_idx, e_list in enumerate(e_cls_list):
        # f_idx1, f_idx2의 face instances에 대해 edge instance와 connection 만들기 based on euclidean distance
        f_idx1, f_idx2 = lookup_face[cls_idx][0], lookup_face[cls_idx][1]
        f_list1, f_list2 = f_cls_list[f_idx1], f_cls_list[f_idx2]
        f_inst_id1, f_inst_id2 = indice_list[f_idx1], indice_list[f_idx2]
        # edge를 기준으로 가까운 face search
        for cur_supp in e_list:
            dist1 = np.asarray([np.min(distance.cdist(cur_supp, f_supp)) for f_supp in f_list1])
            dist2 = np.asarray([np.min(distance.cdist(cur_supp, f_supp)) for f_supp in f_list2])
            min_dist1, min_dist_idx1 = np.min(dist1), np.argmin(dist1)
            min_dist2, min_dist_idx2 = np.min(dist2), np.argmin(dist2)
            # print(min_dist1, min_dist2)
            paired = []
            # if closest face, save its instance id
            if min_dist1 < 0.1:
                paired.append(f_inst_id1[min_dist_idx1])
            if min_dist2 < 0.1:
                paired.append(f_inst_id2[min_dist_idx2])
            if paired:
                pair_list.append(paired)

    # merging paris using set.intersection(set1, set2)
    terminate = False
    final_pair_list = []
    while terminate is not True:
        is_intersect = False
        new_pair_list = []
        start = pair_list[0]
        pair_list = pair_list[1:]

        new_set = start
        # compare start item and others
        for pair in pair_list:
            intersect = set.intersection(set(start), set(pair))
            if intersect:
                new_set = new_set + pair
                is_intersect = True
            else:
                new_pair_list.append(pair)
        new_pair_list.append(list(set(new_set)))
        pair_list = new_pair_list

        # intersection이 없는 set -> no more need to compare
        if is_intersect is False:
            final_pair_list.append(pair_list[-1])
            pair_list.pop(-1)
            if len(pair_list) == 1:
                final_pair_list.append(pair_list[0])
                terminate = True

    box_supp_list = []
    item_flatten = list(itertools.chain(*cls_list[:6]))
    for i, pair_indices in enumerate(final_pair_list):
        supps = np.concatenate([item_flatten[idx] for idx in pair_indices])
        box_supp_list.append(supps)
        # save_obj_color_coding('tmp/train_object_supp_%d.obj' % i, supps, np.ones(len(supps)) * (i%20))
    # exit(0)
    return box_supp_list


def get_proposal_data(opt, batch_instances, batch_params):

    # instances -> b, n_instance, 3
    # box_labels -> b, n_instance, 6

    data_list, sample_list = [], []
    cnt =0
    primitives = None
    for (instances, params) in zip(batch_instances, batch_params):
        for (item, param) in zip(instances, params):
            centers = np.mean(item, axis=0)
            center_distance = np.min(np.linalg.norm(centers-params[:, :3], axis=1))
            gt_idx = np.argmin(np.linalg.norm(centers-params[:, :3], axis=1))
            if center_distance < 0.3:
                # print('propose +1')
                tmp_feats = np.ones(item.shape)
                coordsP,featsP = item, tmp_feats
                coordsP_center = coordsP - centers
                # replace center vector with residual vector
                final_gt = params[gt_idx]
                # centroid = final_gt[:3].copy()
                final_gt[:3] = final_gt[:3] - np.mean(item, axis=0)

                # quantization
                coordsP_center, featsP = ME.utils.sparse_quantize(
                    coordsP_center,
                    features=featsP,
                    quantization_size=opt.voxel_size
                )

                sample_list.append(coordsP)
                data_list.append({"coordinates": coordsP_center, #np.hstack((coordsP_center, coordsP)),
                                  "features": featsP,
                                  "labels": final_gt.reshape((1, -1))})

                # check paired gt results
                # save_obj('tmp/tmp%d.obj'% cnt, coordsP)
                # cnt += 1
                # m = np.eye(4)
                # m[:3, 3] = centers
                # cube = trimesh.creation.box(transform=m, extents=final_gt[3:])
                # if primitives is None:
                #     primitives = cube
                # else:
                #     primitives = trimesh.util.concatenate((primitives, cube))

        # trimesh.exchange.export.export_mesh(primitives, 'tmp/tmp_box.obj')
        # exit(0)


    # to sparse tensor and voxelization
    coords, feats, labels = ME.utils.sparse_collate(
        [d["coordinates"] for d in data_list],
        [d["features"] for d in data_list],
        [d["labels"] for d in data_list],
        dtype=torch.float32,
    )

    return coords, feats, labels, sample_list


# todo data loader 도 바꿔야함 box param return
def get_proposal_data_eval(opt, batch_instances, batch_params=None):
    data_list = []
    if batch_params is None:
        batch_params = batch_instances
    for (instances, params) in zip(batch_instances, batch_params):
        for (item, param) in zip(instances, params):
            tmp_feats = np.ones(item.shape)
            centers = np.mean(item, axis=0)
            coordsP,featsP = item, tmp_feats
            coordsP_center = coordsP - centers

            # quantization
            coordsP_center, featsP = ME.utils.sparse_quantize(
                coordsP_center,
                features=featsP,
                quantization_size=opt.voxel_size
            )

            data_list.append({"coordinates": coordsP_center,
                              "features": featsP,
                            })


    # to sparse tensor and voxelization
    coords, feats = ME.utils.sparse_collate(
        [d["coordinates"] for d in data_list],
        [d["features"] for d in data_list],
        dtype=torch.float32,
    )

    return coords, feats


def instance_eval_pseudo(samples, pred_offset, pred_labels, confidence):
    '''
    face segments for generating pseudo labels
    '''
    samples_trans = samples+pred_offset
    seg_list, seg_offset_list, seg_label_list, mask_list = [], [], [], []
    for i in range(max(pred_labels)+1):
        pts_trans = samples_trans[pred_labels == i]
        pts_ori = samples[pred_labels == i]
        pts_conf = confidence[pred_labels==i]  #todo

        eps, min_samples, thre = 0.15, 3, 10  # pts_trans
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_trans)

        # mask = np.ones(len(pts_ori))
        # mask[dbscan.labels_ == -1] = 0
        # mask_list.append(mask)
        for j in range(max(dbscan.labels_)+2):
            confi = pts_conf[dbscan.labels_ == j-1]
            segment = pts_ori[dbscan.labels_ == j-1]
            center = np.mean(segment, axis=0)
            offset = center-segment
            label = np.ones(len(segment))*i
            mask = np.ones(len(segment))
            if j == -1 or len(segment) < thre:
                mask = np.zeros(len(segment))
            # mask[confi < 0.8] = 0

            mask_list.append(mask)
            seg_list.append(segment)
            seg_offset_list.append(offset)
            seg_label_list.append(label)

    # check the segment's quality
    # [save_obj_color_coding(os.path.join('tmp', '%d_instance.obj' % k), item[mask.astype(bool)],
    #                        np.ones(len(item)) * np.random.choice(29, 1)) for k, (mask, item) in enumerate(zip(mask_list,seg_list)) if len(item[mask.astype(bool)])>10]
    return np.concatenate(seg_list), np.concatenate(seg_label_list), np.concatenate(seg_offset_list), np.concatenate(mask_list)


def instantiation_eval_face_only(path, name, samples, pred_offset, pred_labels):
    # instance segmentation based predicted offsets and labels
    samples_trans = samples + pred_offset
    # samples_trans[pred_labels>5] = samples[pred_labels>5]
    cls_list, label_list = [], []
    inst_idx, indice_list = 0, []
    for i in range(6):
        index_list = []
        pts_trans = samples_trans[pred_labels == i]
        pts_ori = samples[pred_labels == i]
        # save_obj_color_coding(os.path.join(path,'%s_%d_pts.obj' % (name, i)), pts_trans, np.ones(len(pts_trans))*i)

        eps, min_samples, inpts = 0.1, 5, pts_trans  # pts_trans
        thre = 50
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(inpts)

        segments = [pts_ori[dbscan.labels_ == j] for j in range(max(dbscan.labels_)+1) if len(pts_ori[dbscan.labels_ == j]) > thre]

        # [save_obj_color_coding(os.path.join(path, '%s_%d_%d_instance.obj' % (name, i, k)), item, np.ones(len(item))*np.random.choice(29, 1)) for k, (item) in enumerate(segments)]
        cls_list.append(segments)
        for k in range(len(segments)):
            index_list.append(inst_idx)
            inst_idx += 1
        indice_list.append(index_list)

    # exit(0)
    # todo save segments for fine-tuning

    # build face-edge-face connections based on spatial relation between face and edge
    f_cls_list = cls_list[:6]
    print('name', name, 'len of f', len(list(itertools.chain(*f_cls_list))))
    pair_list = []
    # adjacent face indices along edge index
    lookup_face = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [2, 4], [3, 4], [1, 5], [2, 5], [3, 5], [4, 5]]
    for cls_idx in lookup_face:
        # f_idx1, f_idx2의 face instances에 대해 edge instance와 connection 만들기 based on euclidean distance
        f_idx1, f_idx2 = cls_idx[0], cls_idx[1]
        f_list1, f_list2 = f_cls_list[f_idx1], f_cls_list[f_idx2]
        f_inst_id1, f_inst_id2 = indice_list[f_idx1], indice_list[f_idx2]
        if (len(f_list1)==0) or (len(f_list2)==0):
            continue

        # shortest distance between two segment
        for f_inst1, f_supp1 in zip(f_inst_id1, f_list1):
            tmp_n_supp=[]
            for f_inst2, f_supp2 in zip(f_inst_id2, f_list2):
                d1 = np.min(distance.cdist(f_supp2, f_supp1), axis=1)
                n_supp = np.sum(d1<0.1)
                tmp_n_supp.append(n_supp)
            if np.max(np.asarray(tmp_n_supp))<10:
                continue
            max_idx = np.argmax(np.asarray(tmp_n_supp))
            pair_list.append([f_inst1, f_inst_id2[max_idx]])
    print('number of paired list', len(pair_list))

    # todo all instance id - paired id 중 size 큰것 -> single face지만 object로 간주

    # exit(0)
    # tmp_list = list(itertools.chain(*cls_list))
    # for i, indices in enumerate(pair_list):
    #     pts = np.vstack((tmp_list[indices[0]], tmp_list[indices[1]]))
    #     clr = np.ones(len(pts)) * (i%28)
    #     save_obj_color_coding(os.path.join(path,'%s_%d_paired.obj' % (name, i )), pts, clr)
    # exit(0)

    for m in range(len(pair_list)):
        is_intersect = False
        new_pair_list = []
        start = pair_list[0]
        pair_list = pair_list[1:]

        new_set = start
        # compare start item and others
        for pair in pair_list:
            intersect = set.intersection(set(start), set(pair))
            if intersect:
                new_set = new_set + pair
                is_intersect = True
            else:
                new_pair_list.append(pair)
        new_pair_list.append(list(set(new_set)))
        pair_list = new_pair_list
    final_pair_list = pair_list
    #     if is_intersect is False:
    #         final_pair_list.append(pair_list[-1])
    #         pair_list.pop(-1)

        # terminate condition
        # if len(pair_list) == 1:
        #     final_pair_list.append(pair_list[0])
        #     terminate = True

    box_supp_list = []
    item_flatten = list(itertools.chain(*cls_list[:6]))
    for i, pair_indices in enumerate(final_pair_list):
        if len(pair_indices) ==1:
            continue
        supps = np.concatenate([item_flatten[idx] for idx in pair_indices])
        if len(supps)<20:
            continue
        box_supp_list.append(supps)

        # save instances for fine-tuning
        # save_obj_color_coding(os.path.join(path, '%s_objsupp_%d.obj' % (name, i)), supps, np.ones(len(supps))*(i%20))
    # exit(0)
    return box_supp_list

def generate_segment(coord, offset, label):
    # density based clustering for each class to generate segments
    cls_seg_list, cls_segId_list, cls_list, inst_idx = [], [], [], 0

    coord_shift = coord + offset
    for cls in range(max(label)+1):
        sub_coord_shift = coord_shift[label == cls]
        sub_coord = coord[label == cls]
        # print('cls', cls)
        if len(sub_coord) ==0:
            # warning. need to add dummy not to harm order of cls
            print('cls %d is not detected in point cloud'%cls)
            # todo temp
            sub_coord = np.zeros((20,3))
            sub_coord_shift = np.zeros((20, 3))
            # continue

        # save_obj_color_coding(os.path.join(path,'%s_%d_pts.obj' % (name, i)), pts_trans, np.ones(len(pts_trans))*i)
        if cls<6:
            eps, min_samples = 0.15, 3  # todo pts_ori vs pts_trans
            thre = 2
        else:
            eps, min_samples = 0.2, 3
            thre = 2
        if cls<6:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(sub_coord_shift)
        else:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(sub_coord)

        instances = [sub_coord[dbscan.labels_ == j] for j in range(max(dbscan.labels_)+1) if len(sub_coord[dbscan.labels_ == j]) > thre]

        # segment check
        # inst_label = [np.ones(len(sub_coord[dbscan.labels_ == j]))*(j%28) for j in range(max(dbscan.labels_)+1) if len(sub_coord[dbscan.labels_ == j]) > thre]
        # save_obj_color_coding('tmp/%d_cls_inst.obj'%cls, np.vstack(instances), np.concatenate(inst_label))

        cls_seg_list.append(instances)
        for k in range(len(instances)):
            cls_list.append(cls)

        index_list = []
        for k in range(len(instances)):
            index_list.append(inst_idx)
            inst_idx += 1
        cls_segId_list.append(index_list)

    return cls_seg_list, cls_segId_list, cls_list



def make_pairs(seg_list, segId_list):
    # adjacent face indices ordered by edge index
    f_lookup = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [2, 4], [3, 4], [1, 5], [2, 5], [3, 5], [4, 5]]
    normal_lookup = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
    pair_list, pair_cls_list = [], []

    f_seg_list, e_seg_list = seg_list[:6], seg_list[6:18]
    f_ind_list, e_ind_list = segId_list[:6], segId_list[6:18]
    # print('name', name, 'len of f', len(list(itertools.chain(*f_seg_list))), 'len of e',
    #       len(list(itertools.chain(*e_seg_list))))

    # for each edge class
    tmp_list = []
    tmp_cnt = 0
    for cls_idx, (eid_list, e_list) in enumerate(zip(e_ind_list, e_seg_list[:12])):
        f_idx1, f_idx2 = f_lookup[cls_idx][0], f_lookup[cls_idx][1]
        f_list1, f_list2 = f_seg_list[f_idx1], f_seg_list[f_idx2]
        f_inst_id1, f_inst_id2 = segId_list[f_idx1], segId_list[f_idx2]

        # tmp viz
        # save_obj('tmp%d.obj'%cls_idx,np.vstack(e_list))
        # save_obj('tmp%d_f1.obj' % cls_idx, np.vstack(f_list1))
        # save_obj('tmp%d_f2.obj' % cls_idx, np.vstack(f_list2))
        # continue

        # exception case due to wrong clustering
        if (len(f_list1)==0) or (len(f_list2)==0):
            continue

        # based on edge supp, get the closest face
        e_supp_rel, f_supp_rel1, f_supp_rel2 = None, None, None

        for (e_idx, e_supp) in zip(eid_list, e_list):
            paired, paired_cls = [], []
            max_r, pair1 = 0, None
            e_supp_rel = e_supp
            for k, fsup1 in enumerate(f_list1):
                # -------option 1. warning distance from edge samples --> may yield problem when edge is very noisy or spread
                dist1 = np.min(distance.cdist(e_supp, fsup1), axis=1)

                r = np.sum(dist1 < 0.15) / len(dist1)
                # print(r)
                if r > 0.4 and r > max_r:
                    pair1 = f_inst_id1[k]
                    max_r = r
                    f_supp_rel1 = fsup1
                    tmp_list.append(f_supp_rel1)

            max_r, pair2 = 0, None
            for k, fsup2 in enumerate(f_list2):
                #--------option1 --------------
                dist2 = np.min(distance.cdist(e_supp, fsup2), axis=1)
                r = np.sum(dist2 < 0.15) / len(dist2)
                # print(r)
                if r > 0.4 and r > max_r:
                    pair2 = f_inst_id2[k]
                    max_r = r
                    f_supp_rel2 = fsup2

                    tmp_list.append(f_supp_rel2)


            if pair1 is not None:
                paired.append(pair1)
                paired_cls.append(f_idx1)
            if pair2 is not None:
                paired.append(pair2)
                paired_cls.append(f_idx2)

            # if pair is created, check positional relation is correct
            # rule: swap + opposite sign
            if len(paired) == 2:
                v1_gt, v2_gt = -np.asarray(normal_lookup[f_idx2]), -np.asarray(normal_lookup[f_idx1])
                m_e, m_f1, m_f2 = np.mean(e_supp_rel, axis=0), np.mean(f_supp_rel1, axis=0), np.mean(f_supp_rel2, axis=0)
                v1, v2 = (m_f1-m_e)/np.linalg.norm(m_f1-m_e), (m_f2-m_e)/np.linalg.norm(m_f2-m_e)
                sim1, sim2 = np.dot(v1, v1_gt), np.dot(v2, v2_gt)
                # print('paired')
                # save_obj('tmp/%d.obj'%tmp_cnt, np.vstack((f_supp_rel1, f_supp_rel2)))
                # tmp_cnt += 1
                # print(sim1, sim2)

                if sim1<0 or sim2<0:
                    continue

            if paired:
                paired.append(e_idx) # add edge index to last
                pair_list.append(paired)

    # save paired data
    # save_obj('tmp.obj', np.vstack(tmp_list))
    return pair_list



'''
    aggregation code based on DFP
'''
def aggregate_pairs_v2(pair_list):
    # Generate data
    face_ids = set([])
    edge_ids = []
    edge_info = {}
    for pair in pair_list:
        edge_id = pair[-1]
        if len(pair) == 3:
            face_1_id = pair[0]
            face_2_id = pair[1]
            face_ids.add(face_1_id)
            face_ids.add(face_2_id)
        else:
            face_1_id = pair[0]
            face_2_id = None
            face_ids.add(face_1_id)
        edge_ids.append(edge_id)
        edge_info[edge_id] = (face_1_id, face_2_id)
    face_ids = list(face_ids)

    # Make adjacency list
    adj_list = {f: [] for f in face_ids}
    for e in edge_ids:
        f1 = edge_info[e][0]
        f2 = edge_info[e][1]
        if f1 == None or f2 == None: continue
        adj_list[f1].append(f2)
        adj_list[f2].append(f1)

    # DFS for finding connected component groups
    def explore(i, obj_num, obj_ids):
        obj_ids[i] = obj_num
        for j in adj_list[i]:
            if obj_ids[j] == -1:
                explore(j, obj_num, obj_ids)

    obj_ids = [-1 for i in range(max(face_ids + edge_ids) + 1)]
    obj_num = 0
    for f in face_ids:
        if obj_ids[f] == -1:
            explore(f, obj_num, obj_ids)
            obj_num += 1

    # Assign obj ids for edges
    for e in edge_ids:
        if edge_info[e][0] != None:
            obj_ids[e] = obj_ids[edge_info[e][0]]
        else:
            obj_ids[e] = obj_ids[edge_info[e][1]]

    return obj_ids



'''
polycube - cube oriented grid filling algo
'''
def instantiation_eval_v2(args, name, samples, pred_offset, pred_labels, is_rot=True):
    # 1. generate segments
    seg_list, segId_list, seg_cls_list = generate_segment(samples, pred_offset, pred_labels)

    # to save
    # seg_list = list(itertools.chain(*seg_list[:18]))
    # np.save('segment_list.npy', seg_list), np.save('pair_list.npy', pair_list)

    # 2. make a pair of face which share adjacent adge
    pair_list = make_pairs(seg_list, segId_list)
    print('number of paired list', len(pair_list))
    # 3. aggregate pairs to make instance

    # 3.2
    merged_pair_list = aggregate_pairs_v2(pair_list)
    merged_pair_list = np.asarray(merged_pair_list)

    # ** len(merged_pair_list) < len(final_seg_list)
    final_seg_list = np.asarray(list(itertools.chain(*seg_list[:18])))
    final_segcls_list = np.asarray(seg_cls_list[:len(merged_pair_list)])

    box_supp_list, polycube_list, polycube_viz_list = [], [], []
    data = np.vstack(
        [np.vstack(final_seg_list[np.where(merged_pair_list == i)[0]]) for i in range(max(merged_pair_list) + 1)])
    preds = np.concatenate(
        np.asarray([np.ones(len(np.vstack(final_seg_list[np.where(merged_pair_list == i)[0]]))) * (i % 20) for i in
                    range(max(merged_pair_list) + 1)]))
    seg_id = np.concatenate([np.ones(len(p)) * kk for kk, p in enumerate(final_seg_list)])

    # ------------save for viz iccv---------------
    # np.save('input_offset.npy', pred_offset+samples); np.save('input.npy', samples); np.save('input_cls.npy', pred_labels)
    # np.save('instance_pts.npy', data); np.save('instance_id.npy', preds)
    # np.save('segment_pts.npy', np.vstack(final_seg_list)); np.save('segment_id.npy', seg_id)
    # mask_pair = np.concatenate([random.choice(pair_list) for kk in range(3)])
    # non_mask = set(np.concatenate(pair_list)) - set(mask_pair)
    # non_pair = np.vstack([final_seg_list[p_id] for p_id in non_mask])
    # pair = np.vstack([final_seg_list[p_id] for p_id in mask_pair])
    # save_obj('tmp_pair.obj',pair); save_obj('tmp_nonpair.obj', non_pair)
    # save_obj_color_coding('tmp/%s_in.obj' % name, data, preds)
    #---------------------------------------------

    print('len of detected object', max(merged_pair_list) + 1)
    for i in range(max(merged_pair_list) + 1):
        obj_seg_cls = final_segcls_list[merged_pair_list == i]
        obj_seg = final_seg_list[np.where(merged_pair_list == i)[0]]
        supp = np.vstack(obj_seg)

        x1, x2, y1, y2, z = None, None, None,None,None
        if 0 in obj_seg_cls:
            tmp_subseg = obj_seg[obj_seg_cls == 0]
            x1 = np.vstack([x_tmp - np.mean(x_tmp, 0) for x_tmp in tmp_subseg])

        if 5 in obj_seg_cls:
            tmp_subseg = obj_seg[obj_seg_cls == 5]
            x2 = np.vstack([x_tmp - np.mean(x_tmp, 0) for x_tmp in tmp_subseg])

        if 1 in obj_seg_cls:
            tmp_subseg = obj_seg[obj_seg_cls == 1]
            y1 = np.vstack([y_tmp - np.mean(y_tmp, 0) for y_tmp in tmp_subseg])

        if 4 in obj_seg_cls:
            tmp_subseg = obj_seg[obj_seg_cls == 4]
            y2 = np.vstack([y_tmp - np.mean(y_tmp, 0) for y_tmp in tmp_subseg])

        # ori coord
        x_vec1, y_vec1 = np.asarray([-1, 0, 0]), np.asarray([0, -1, 0])
        x_vec2, y_vec2, z_vec = np.asarray([1,0,0]), np.asarray([0,1,0]), np.asarray([0,0,1])
        gt_rot = np.asarray([x_vec1, y_vec1, x_vec2, y_vec2, z_vec])
        # compute normal of each face segment
        if x1 is not None:
            x_eval, x_evec = np.linalg.eig(np.cov(x1.T))
            x_vec1 = x_evec[:,np.argmin(x_eval)]
            if x_vec1[0]>0:
                x_vec1 *= -1
            # x_rot = R.align_vectors(x_vec.reshape(1, -1), np.asarray([1, 0, 0]).reshape(1, -1))[0].as_matrix()
            # rot = np.matmul(x_rot, rot)
        if x2 is not None:
            x_eval, x_evec = np.linalg.eig(np.cov(x2.T))
            x_vec2 = x_evec[:, np.argmin(x_eval)]
            if x_vec2[0]<0:
                x_vec2 *= -1
        if y1 is not None:
            y_eval, y_evec = np.linalg.eig(np.cov(y1.T))
            y_vec1 = y_evec[:,np.argmin(y_eval)]
            if y_vec1[1] >0:
                y_vec1 *= -1
        if y2 is not None:
            y_eval, y_evec = np.linalg.eig(np.cov(y2.T))
            y_vec2 = y_evec[:, np.argmin(y_eval)]
            if y_vec2[1]<0:
                y_vec2 *= -1

        d = np.asarray([x_vec1, y_vec1,  x_vec2,y_vec2, z_vec])
        rot = R.align_vectors(gt_rot, d)[0].as_matrix()  # d to gt_rot
        rot_mat = R.from_matrix(rot)
        z_angle = rot_mat.as_euler('xyz', degrees=True)[2] #consider only along z-axis
        rot_angle = R.from_euler('z', z_angle, degrees=True)
        rot = rot_angle.as_matrix()

        trans_mean = np.mean(supp, axis=0)
        supp -= trans_mean
        if is_rot:
            supp = np.matmul(rot, supp.T).T

        # trans_min = np.min(supp, axis=0)
        # supp -= trans_min

        # save_obj('tmp/%s_sup_%d_aligned.obj'%(name,i), supp)
        # viz pred obj
        # obj_pred_cls = np.concatenate([np.ones(len(item)) * item_cls for kk, (item,item_cls) in enumerate(zip(obj_seg, obj_seg_cls))])
        # save_obj_color_coding('tmp/inst_f_%d.obj'%i, supp, obj_pred_cls)

        obb = trimesh.points.PointCloud(supp).bounding_box
        coord = [[], [], []]
        # compute center, min, max and normal of segment
        final_seg_center, final_seg_minmax, final_normal = [], [], []
        normal_lookup = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1], [0, 1, 0], [1, 0, 0]]

        obj_seg = obj_seg[obj_seg_cls < 6]
        obj_seg_cls = obj_seg_cls[obj_seg_cls < 6]

        # for kk, (seg, cls) in enumerate(zip(obj_seg, obj_seg_cls)):
        #     save_obj_color_coding('tmp/%s_%d_%d.obj'%(name,i,kk), seg, np.ones(len(seg))*cls)

        # todo rotation and translation - mean min?
        if is_rot:
            obj_seg = np.asarray([np.matmul(rot, (tmp_pts - trans_mean).T).T for tmp_pts in obj_seg])  #np.matmul(rot, (tmp_pts - trans_mean).T).T - trans_min
        else:
            obj_seg = np.asarray([(tmp_pts - trans_mean)  for tmp_pts in obj_seg])


        # for kk, (seg, cls) in enumerate(zip(obj_seg, obj_seg_cls)):
        #     save_obj_color_coding('tmp/tmprot%d.obj'%kk, seg, np.ones(len(seg))*cls)

        # add bounding box minmax to front and end of grid coordinate
        min_obb, max_obb = np.min(obb.vertices, axis=0), np.max(obb.vertices, axis=0)
        [coord[ii].append(max_obb[ii]) for ii in range(3)]
        [coord[ii].insert(0, min_obb[ii]) for ii in range(3)]

        # generate coordinate
        for (item, cls) in zip(obj_seg, obj_seg_cls):
            seg_normal = np.asarray(normal_lookup[cls])

            seg_mean = np.mean(item, axis=0)
            axis = np.where(seg_normal != 0)[0][0]
            minmax = np.concatenate((np.min(item, axis=0), np.max(item, axis=0)))
            final_seg_minmax.append(minmax)
            final_normal.append(seg_normal)

            # if there are redundant points after projecting to each axis, ignore this one
            v = seg_mean[axis]
            if ((abs(coord[axis] - v)) < 0.1).any():
                continue
            coord[axis].append(v)

        coord = [sorted(coord[i]) for i in range(3)]

        # viz grid
        final_coord = []
        cx, cy, cz = np.asarray(coord[0]), np.asarray(coord[1]), np.asarray(coord[2])
        for ii in range(len(cx)):
            for jj in range(len(cy)):
                for kk in range(len(cz)):
                    final_coord.append(np.asarray([cx[ii], cy[jj], cz[kk]]))

        # save_obj('tmp/%s_%d_grid.obj'%(name,i), np.vstack(final_coord))

        # assign boolean to grid
        bool_grid = np.zeros((len(coord[0]) - 1, len(coord[1]) - 1, len(coord[2]) - 1))

        for ii in range(bool_grid.shape[0]):
            for jj in range(bool_grid.shape[1]):
                for kk in range(bool_grid.shape[2]):
                    minxyz= coord[0][ii], coord[1][jj], coord[2][kk]
                    maxxyz = coord[0][ii+1], coord[1][jj+1], coord[2][kk+1]
                    minxyz, maxxyz = np.asarray(minxyz), np.asarray(maxxyz)

                    tmp_supp = []
                    # for each face of cell
                    for fi, normal in enumerate(zip(normal_lookup)):
                        # point to point distance
                        xyz1, xyz2 = np.zeros(3), np.zeros(3)
                        normal = np.asarray(normal).reshape(-1)
                        xyz1[normal==0], xyz2[normal==0] = minxyz[normal==0], maxxyz[normal==0]
                        eps = 0.1
                        if normal[normal!=0] < 0:
                            xyz1[normal!=0] = minxyz[normal!=0] -eps
                        else:
                            xyz1[normal != 0] = maxxyz[normal != 0]-eps
                        xyz2[normal!=0] = xyz1[normal!=0]+eps*2
                        # point of cls // ** warning: if object is occluded, some cls may be not included.
                        if len(obj_seg[obj_seg_cls == fi])==0:
                            continue
                        pts_seg = np.vstack(obj_seg[obj_seg_cls == fi])

                        # pts_seg -= trans
                        # pts_seg = np.matmul(rot, pts_seg.T).T

                        mask1, mask2 = (xyz1<pts_seg).all(axis=1), (pts_seg<xyz2).all(axis=1)
                        n_total = np.sum(mask1*mask2)
                        tmp_pts_seg = pts_seg[mask1*mask2]

                        # if len(tmp_pts_seg)!=0:
                        #     tmp_supp.append(tmp_pts_seg)
                        # print(n_total)
                        if n_total <30: # 범주내 포인트가 일정개수 이하면 없다고 판단
                            # print(n_total)
                            continue
                        bool_grid[ii,jj,kk]=1
                    # save_obj('tmp%d%d%d.obj'%(ii,jj,kk), np.vstack(tmp_supp))

        # exit(0)
        # generate cube based on boolean cube
        polycube, polycubeo3d = None, None
        tmp_polycube = []
        coord1, coord2, coord3 = coord[0], coord[1], coord[2]
        for ii in range(bool_grid.shape[0]):
            for jj in range(bool_grid.shape[1]):
                for kk in range(bool_grid.shape[2]):
                    if bool_grid[ii, jj, kk] == 1:
                        # print(ii, jj, kk)
                        gx, gy, gz = coord1[ii:ii + 2], coord2[jj:jj + 2], coord3[kk:kk + 2]
                        # print('gx gy gz', gx, gy, gz)
                        grid_pts = np.vstack([gx, gy, gz]).T

                        # -----remove if is_rot:
                        #     grid_pts += (trans_min + trans_mean) # np.matmul(np.linalg.inv(rot), grid_pts.T).T + trans_min + trans_mean
                        # else:-------------
                        grid_obb = trimesh.points.PointCloud(grid_pts).bounding_box

                        if polycube:
                            polycube = trimesh.util.concatenate(polycube, grid_obb)

                        else:
                            polycube = grid_obb

                        tmp_polycube.append(grid_obb)


        # is_fail = False
        # max_cnt, start_cnt = len(tmp_polycube)*5, 0
        # while len(tmp_polycube) != 0:
        #     item = tmp_polycube.pop()
        #     if final_poly is None:
        #         final_poly = item
        #     else:
        #         tmp_final_poly = final_poly.boolean_union(item)
        #         tmp_final_poly = o3d.t.geometry.TriangleMesh.to_legacy(tmp_final_poly)
        #         if len(tmp_final_poly.vertices) != 0:
        #             final_poly = final_poly.boolean_union(item)
        #         else:
        #             tmp_polycube.append(item)
        #     start_cnt+= 1
        #     if start_cnt >max_cnt:
        #         trimesh.exchange.export.export_mesh(polycube, 'tmp/%s_%d_fail.obj' % (name, i))
        #         print('fail to generate mesh')
        #         is_fail = True
        #         break


        if polycube is None:
            # print('none poly, need to rotate cube for axis-alignment')
            continue
        # union
        if len(tmp_polycube) == 1:
            final_poly = tmp_polycube[0]
        else:
            final_poly = None
            for kk in range(len(tmp_polycube)):
                ori_p = tmp_polycube[kk].copy()
                tmp_p = tmp_polycube[kk].copy()
                tmp_polycube.pop(kk)
                mask =[]
                for tmp_other in tmp_polycube:
                    # facet center list
                    p_center, o_center = [], []
                    for fc in tmp_p.facets:
                        p_center.append(tmp_p.vertices[np.unique(np.concatenate(tmp_p.faces[fc]))].mean(axis=0))

                    for pc in tmp_other.facets:
                        o_center.append(tmp_other.vertices[np.unique(np.concatenate(tmp_other.faces[pc]))].mean(axis=0))

                    p_center, o_center = np.asarray(p_center), np.asarray(o_center)
                    dist = distance.cdist(p_center, o_center)
                    if np.min(dist)<0.000001:
                        idx = np.where(np.min(dist, axis=1) < 0.000001)[0]
                        for f_idx in idx:
                            mask.append(tmp_p.facets[f_idx])
                tmp_polycube.insert(kk, ori_p)

                mask_total = set([i for i in range(len(tmp_p.faces))])
                if len(mask) != 0:
                    mask = set(np.concatenate(mask))
                    mask_final = list(mask_total-mask)
                else:
                    mask_final = list(mask_total)
                tmp_v = tmp_p.vertices.copy()
                tmp_f = tmp_p.faces.copy()
                tmp_f_n = tmp_p.face_normals.copy()
                new_mesh = trimesh.Trimesh(vertices=tmp_v, faces=tmp_f[mask_final], face_normals=tmp_f_n[mask_final])

                # trimesh.exchange.export.export_mesh(new_mesh, 'tmp_new%d.obj'%kk)
                if final_poly is None:
                    final_poly = new_mesh
                else:
                    final_poly = trimesh.util.concatenate(final_poly, new_mesh)
            final_poly.process()


        # transform to original coordinate
        transform = np.eye(4)
        # todo rotation
        if is_rot:
            transform[:3,:3] = np.linalg.inv(rot)

        transform[:3,3] = trans_mean  #+ trans_min
        final_poly.apply_transform(transform)

        polycube_list.append(final_poly)
        # trimesh.exchange.export.export_mesh(final_poly, 'tmp/%s_%d_out.obj'%(name, i))
        if polycube_viz_list:
            polycube_viz_list = trimesh.util.concatenate(polycube_viz_list, final_poly)
        else:
            polycube_viz_list = final_poly

        # global to local
        # todo rot
        if is_rot:
            supp = np.matmul(np.linalg.inv(rot), supp.T).T

        supp += trans_mean  #(trans_min + trans_mean)
        box_supp_list.append(supp)
    # viz
    # o3d.io.write_triangle_mesh('tmp/%s_out_o3d.obj' % name, total_poly)
    trimesh.exchange.export.export_mesh(polycube_viz_list, 'tmp/%s_out.obj' % name)
    inpath= os.path.join(args.result_path, args.name, args.eval_folder, name + '_in.obj')
    save_obj_color_coding(inpath, np.vstack(box_supp_list), np.concatenate(
        np.asarray([np.ones(len(box_supp_list[i])) * (i % 20) for i in range(len(box_supp_list))])))
    # exit(0)
    return box_supp_list, polycube_list, polycube_viz_list


'''
polycube - cube oriented grid filling algo
'''
def instantiation_eval_v2_backup(args, name, samples, pred_offset, pred_labels):
    # 1. generate segments
    seg_list, segId_list, seg_cls_list = generate_segment(samples, pred_offset, pred_labels)

    # to save
    # seg_list = list(itertools.chain(*seg_list[:18]))
    # np.save('segment_list.npy', seg_list), np.save('pair_list.npy', pair_list)

    # 2. make a pair of face which share adjacent adge
    pair_list = make_pairs(seg_list, segId_list)
    print('number of paired list', len(pair_list))
    # 3. aggregate pairs to make instance
    # merged_pair_list = aggregate_pairs(pair_list)

    # 3.2
    merged_pair_list = aggregate_pairs_v2(pair_list)
    merged_pair_list = np.asarray(merged_pair_list)

    # ** len(merged_pair_list) < len(final_seg_list)
    final_seg_list = np.asarray(list(itertools.chain(*seg_list[:18])))
    final_segcls_list = np.asarray(seg_cls_list[:len(merged_pair_list)])


    box_supp_list, polycube_list = [], []

    data = np.vstack(
        [np.vstack(final_seg_list[np.where(merged_pair_list == i)[0]]) for i in range(max(merged_pair_list) + 1)])
    preds = np.concatenate(
        np.asarray([np.ones(len(np.vstack(final_seg_list[np.where(merged_pair_list == i)[0]]))) * (i % 20) for i in
                    range(max(merged_pair_list) + 1)]))
    in_path = os.path.join(args.result_path, args.name, args.eval_folder, data_name + '_in.obj')
    save_obj_color_coding(in_path, data, preds)  #'tmp/%s_in.obj' % name

    print('len of detected object', max(merged_pair_list) + 1)
    for i in range(max(merged_pair_list) + 1):
        obj_seg_cls = final_segcls_list[merged_pair_list == i]
        obj_seg = final_seg_list[np.where(merged_pair_list == i)[0]]
        supp = np.vstack(obj_seg)
        supp_id = np.concatenate([np.ones(len(item))*kk for kk, item in enumerate(obj_seg)])
        # ----------------remove outlier
        # o3d_pts = o3d.geometry.PointCloud()
        # o3d_pts.points = o3d.utility.Vector3dVector(supp)
        # o3d_pts = o3d_pts.voxel_down_sample(voxel_size=0.04)
        # cl, ind = o3d_pts.remove_radius_outlier(nb_points=3, radius=0.08)
        # # new_supp = np.asarray(o3d_pts.remove_radius_outlier(nb_points=3, radius=0.05)[0].points)
        # new_supp, new_supp_id = supp[ind], supp_id[ind]
        # supp = [new_supp[new_supp_id == kk] for kk in range(int(max(new_supp_id)) + 1)]
        # # coodinate move todo rotation
        # supp = np.vstack(supp)
        #---------------------------------

        trans = np.min(supp, axis=0)
        supp -= trans

        obj_pred_cls = np.concatenate([np.ones(len(item)) * item_cls for kk, (item,item_cls) in enumerate(zip(obj_seg, obj_seg_cls))])
        # save_obj_color_coding('tmp/inst_f_%d.obj'%i, supp, obj_pred_cls)
        obb = trimesh.points.PointCloud(supp).bounding_box
        coord = [[], [], []]
        # compute center, min, max and normal of segment
        final_seg_center, final_seg_minmax, final_normal = [], [], []
        normal_lookup = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1], [0, 1, 0], [1, 0, 0]]

        obj_seg = obj_seg[obj_seg_cls < 6]
        obj_seg_cls = obj_seg_cls[obj_seg_cls < 6]
        # add bounding box minmax to front and end of grid coordinate
        min_obb, max_obb = np.min(obb.vertices, axis=0), np.max(obb.vertices, axis=0)
        [coord[ii].append(max_obb[ii]) for ii in range(3)]
        [coord[ii].insert(0, min_obb[ii]) for ii in range(3)]

        # generate coordinate
        for (item, cls) in zip(obj_seg, obj_seg_cls):
            seg_normal = np.asarray(normal_lookup[cls])

            # coordinate move todo
            item -= trans

            seg_mean = np.mean(item, axis=0)
            axis = np.where(seg_normal != 0)[0][0]
            minmax = np.concatenate((np.min(item, axis=0), np.max(item, axis=0)))
            final_seg_minmax.append(minmax)
            final_normal.append(seg_normal)

            # if there are redundant points after projecting to each axis, ignore this one
            v = seg_mean[axis]
            if ((abs(coord[axis] - v)) < 0.1).any():
                continue
            coord[axis].append(v)

        coord = [sorted(coord[i]) for i in range(3)]

        # viz grid
        final_coord = []
        cx, cy, cz = np.asarray(coord[0]), np.asarray(coord[1]), np.asarray(coord[2])
        for ii in range(len(cx)):
            for jj in range(len(cy)):
                for kk in range(len(cz)):
                    final_coord.append(np.asarray([cx[ii], cy[jj], cz[kk]]))

        # assign boolean to grid
        bool_grid = np.zeros((len(coord[0]) - 1, len(coord[1]) - 1, len(coord[2]) - 1))

        for ii in range(bool_grid.shape[0]):
            for jj in range(bool_grid.shape[1]):
                for kk in range(bool_grid.shape[2]):
                    minxyz= coord[0][ii], coord[1][jj], coord[2][kk]
                    maxxyz = coord[0][ii+1], coord[1][jj+1], coord[2][kk+1]
                    minxyz, maxxyz = np.asarray(minxyz), np.asarray(maxxyz)

                    for fi, normal in enumerate(zip(normal_lookup)):
                        # point to point distance
                        xyz1, xyz2 = np.zeros(3), np.zeros(3)
                        normal = np.asarray(normal).reshape(-1)
                        xyz1[normal==0], xyz2[normal==0] = minxyz[normal==0], maxxyz[normal==0]
                        eps = 0.1
                        if normal[normal!=0] < 0:
                            xyz1[normal!=0] = minxyz[normal!=0] -eps
                        else:
                            xyz1[normal != 0] = maxxyz[normal != 0]-eps
                        xyz2[normal!=0] = xyz1[normal!=0]+eps*2
                        # point of cls // ** warning: if object is occluded, some cls may be not included.
                        if len(obj_seg[obj_seg_cls == fi])==0:
                            continue
                        pts_seg = np.vstack(obj_seg[obj_seg_cls == fi])
                        mask1, mask2 = (xyz1<pts_seg).all(axis=1), (pts_seg<xyz2).all(axis=1)
                        n_total = np.sum(mask1*mask2)
                        print(n_total)
                        if n_total <20: # 범주내 포인트가 일정개수 이하면 없다고 판단
                            continue
                        bool_grid[ii,jj,kk]=1

        # generate cube based on boolean cube
        polycube = None
        coord1, coord2, coord3 = coord[0], coord[1], coord[2]
        for ii in range(bool_grid.shape[0]):
            for jj in range(bool_grid.shape[1]):
                for kk in range(bool_grid.shape[2]):
                    if bool_grid[ii, jj, kk] == 1:
                        print(ii, jj, kk)
                        gx, gy, gz = coord1[ii:ii + 2], coord2[jj:jj + 2], coord3[kk:kk + 2]
                        print('gx gy gz', gx, gy, gz)
                        grid_pts = np.vstack([gx, gy, gz]).T
                        # todo trans to original coordinate
                        grid_pts += trans
                        grid_obb = trimesh.points.PointCloud(grid_pts).bounding_box
                        if polycube:
                            polycube = trimesh.util.concatenate(polycube, grid_obb)
                        else:
                            polycube = grid_obb
        if polycube is None:
            print('none poly, need to rotate cube for axis-alignment')
            continue
        polycube.process()
        trimesh.exchange.export.export_mesh(polycube, 'tmp/%s_%d_out.obj'%(name, i))
        if polycube_list:
            polycube_list = trimesh.util.concatenate(polycube_list, polycube)
        else:
            polycube_list = polycube

        supp += trans
        box_supp_list.append(supp)
    # viz

    trimesh.exchange.export.export_mesh(polycube_list, 'tmp/%s_out.obj' % name)
    save_obj_color_coding('tmp/%s_in.obj' % name, np.vstack(box_supp_list), np.concatenate(
        np.asarray([np.ones(len(box_supp_list[i])) * (i % 20) for i in range(len(box_supp_list))])))
    exit(0)
    return box_supp_list


def instantiation_eval(path, name, samples, pred_offset, pred_labels):
    # instance segmentation based predicted offsets and labels
    samples_trans = samples  + pred_offset
    # samples_trans[pred_labels>5] = samples[pred_labels>5]
    cls_list, label_list = [], []
    inst_idx, indice_list = 0, []
    for i in range(max(pred_labels) + 1):
        index_list = []
        pts_trans = samples_trans[pred_labels == i]
        pts_ori = samples[pred_labels == i]
        # save_obj_color_coding(os.path.join(path,'%s_%d_pts.obj' % (name, i)), pts_trans, np.ones(len(pts_trans))*i)
        if i < 6:
            eps, min_samples, inpts = 0.1, 5, pts_trans  # todo pts_ori vs pts_trans
            thre = 0
        else:
            eps, min_samples, inpts = 0.1, 3, pts_trans
            thre = 0
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(inpts)

        instances = [pts_ori[dbscan.labels_ == j] for j in range(max(dbscan.labels_) + 1) if
                     len(pts_ori[dbscan.labels_ == j]) > thre]

        # NOTE; tried to remove outliers but differences in density between edge and face does not allow to set hyper parameters properly
        # instances = []
        # for j in range(max(dbscan.labels_)+1):
        #     if i == 6:
        #         print(' ')
        #     if len(pts_ori[dbscan.labels_ == j]) > 20:
        #         pts = pts_ori[dbscan.labels_ == j]
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(pts)
        #         pcd = pcd.voxel_down_sample(voxel_size=0.02)
        #         # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.01)
        #         cl, ind = pcd.remove_radius_outlier(nb_points=3, radius=0.1)
        #         if len(cl.points)<20:
        #             continue
        #         instances.append(np.asarray(cl.points))

        # [save_obj_color_coding(os.path.join(path, '%s_%d_%d_instance.obj' % (name, i, k)), item, np.ones(len(item))*np.random.choice(29, 1)) for k, (item) in enumerate(instances) if i<6 ]

        cls_list.append(instances)
        for k in range(len(instances)):
            index_list.append(inst_idx)
            inst_idx += 1
        indice_list.append(index_list)

    # exit(0)
    # build face-edge-face connections based on spatial relation between face and edge
    f_cls_list, e_cls_list = cls_list[:6], cls_list[6:18]
    f_ind_list, e_ind_list = indice_list[:6], indice_list[6:18]
    print('name', name, 'len of f', len(list(itertools.chain(*f_cls_list))), 'len of e',
          len(list(itertools.chain(*e_cls_list))))
    pair_list = []
    # adjacent face indices along edge index
    lookup_face = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [2, 4], [3, 4], [1, 5], [2, 5], [3, 5], [4, 5]]
    for cls_idx, (ei_list, e_list) in enumerate(zip(e_ind_list, e_cls_list[:12])):
        # f_idx1, f_idx2의 face instances에 대해 edge instance와 connection 만들기 based on euclidean distance
        f_idx1, f_idx2 = lookup_face[cls_idx][0], lookup_face[cls_idx][1]
        f_list1, f_list2 = f_cls_list[f_idx1], f_cls_list[f_idx2]
        f_inst_id1, f_inst_id2 = indice_list[f_idx1], indice_list[f_idx2]
        # save_obj('tmp_0edg.obj', np.vstack(e_list)), save_obj('tmp_0face.obj', np.vstack(f_list1)), save_obj('tmp_1face.obj', np.vstack(f_list2))
        if (len(f_list1) == 0) or (len(f_list2) == 0):
            continue
        # edge를 기준으로 가까운 face search
        for (e_idx, e_supp) in zip(ei_list, e_list):
            #     # -----option1 just using single minimum point------
            #     dist1 = np.asarray([np.min(distance.cdist(e_supp, f_supp1)) for f_supp1 in f_list1])
            #     dist2 = np.asarray([np.min(distance.cdist(e_supp, f_supp2)) for f_supp2 in f_list2])
            #     min_dist1, min_dist_idx1 = np.min(dist1), np.argmin(dist1)
            #     min_dist2, min_dist_idx2 = np.min(dist2), np.argmin(dist2)
            #     paired = []
            #     # if closest face, save its instance id
            #     if min_dist1 < 0.2:
            #         paired.append(f_inst_id1[min_dist_idx1])
            #         # paired.append(e_supp)
            #     if min_dist2 < 0.2:
            #         paired.append(f_inst_id2[min_dist_idx2])
            #         # paired.append(e_supp)
            # -----------------------------------------------

            # ---option2 distance between sets------
            # todo max num으로 뽑기
            paired = []
            max_r, pair1 = 0, None
            for k, fsup in enumerate(f_list1):
                dist1 = np.min(distance.cdist(e_supp, fsup), axis=1)
                r = np.sum(dist1 < 0.1) / len(dist1)  # 0.1
                # print(r)
                if r > 0.3 and r > max_r:  # 0.5
                    pair1 = f_inst_id1[k]
                    # paired.append(f_inst_id1[k])
                    max_r = r

            max_r, pair2 = 0, None
            for k, fsup in enumerate(f_list2):
                dist2 = np.min(distance.cdist(e_supp, fsup), axis=1)
                r = np.sum(dist2 < 0.1) / len(dist2)  # 0.1
                if r > 0.3 and r > max_r:  # 0.5
                    pair2 = f_inst_id2[k]
                    max_r = r
                    # paired.append(f_inst_id2[k])
                    # break
            # --------------------------------------

            if pair1 is not None:
                paired.append(pair1)
            if pair2 is not None:
                paired.append(pair2)
            if paired:
                paired.append(e_idx)  # add edge index
                pair_list.append(paired)

    # exit(0)
    print('number of paired list', len(pair_list))

    for m in range(len(pair_list)+100):
        new_pair_list = []
        start = pair_list[0]
        pair_list = pair_list[1:]

        new_set = start
        # compare start item with remained ones
        for pair in pair_list:
            intersect = set.intersection(set(start), set(pair))
            if intersect:
                new_set = new_set + pair
                # once_intersect = True
            else:
                new_pair_list.append(pair)
        new_pair_list.append(list(set(new_set)))
        pair_list = new_pair_list

    final_pair_list = pair_list

    box_supp_list = []
    # item_flatten = list(itertools.chain(*cls_list[:6]))
    item_flatten = list(itertools.chain(*cls_list[:18]))
    for i, pair_indices in enumerate(final_pair_list):
        # if len(pair_indices) ==1:
        #     continue
        supps = np.concatenate([item_flatten[idx] for idx in pair_indices])
        # outlier removal
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(supps)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        cl, ind = pcd.remove_radius_outlier(nb_points=3, radius=0.1)
        supps = np.asarray(cl.points)

        if len(supps) < 20:
            continue

        box_supp_list.append(supps)

    # save_obj_color_coding(os.path.join(path, '%s_objsupp_%d.obj' % (name, i)), supps, np.ones(len(supps))*(i%20))

    return box_supp_list


def get_offset(opt, net, cuda, test_dataset, num_test=1, epoch=0, visualize=True):
    with torch.no_grad():
        samples, label = test_dataset.return_data()
        # to save the reults in batch size
        # item = test_dataset[epoch%len(test_dataset)]
        # samples, label = item['samples'], item['labels']

        color = np.ones(samples.shape)
        samples, label, color = torch.Tensor(samples), torch.Tensor(label), torch.Tensor(color)
        for i in range(num_test):
            # Quantize the input
            coords, feats,  idx1, idx2 = ME.utils.sparse_quantize(
                samples,
                features=color,
                quantization_size=opt.voxel_size,
                return_index=True,
                return_inverse=True)
            # collate 추가 !!!!!!!!!!!!!!
            coords = torch.hstack((torch.zeros((len(coords),1)), coords))
            # print(coords.shape)
            sinput = ME.SparseTensor(feats.type(torch.float32), coords, device=cuda)

            out, out_offset, out_inout, out_angle, out_scale = net.forward(sinput)

            # map to input
            logits = out.slice(sinput).F

            _, pred = logits.max(1)
            pred = pred.cpu().numpy()
            pred_color = np.array([CUBOID_COLOR_MAP[l] for l in pred])

            samples = samples[idx1]
            save_obj('%s/%s/%d_epoch_%d.obj' % (opt.results_path, opt.name, epoch, i), samples, pred_color)

            if opt.offset:
                pred_offset = out_offset.slice(sinput).F
                pred_offset = pred_offset.cpu().numpy()
                # print(pred_offset.shape)
                # print(np.linalg.norm(pred_offset, axis=1))
                trans_coords = samples + pred_offset
                save_obj('%s/%s/%d_epoch_%d_offset.obj' % (opt.results_path, opt.name, epoch, i), trans_coords, pred_color[coords[:, 0] == 0])

            if opt.layout:
                logits_inout = out_inout.slice(sinput).F
                _, pred_inout = logits_inout.max(1)
                pred_inout = pred_inout.cpu().numpy()
                pred_color_inout = np.array([CUBOID_COLOR_MAP[l] for l in pred_inout])
                save_obj('%s/%s/%d_epoch_%d_inout.obj' % (opt.results_path, opt.name, epoch, i), samples, pred_color_inout)

            if opt.angle:
                logits_angle = out_angle.slice(sinput).F
                _, pred_angle = logits_angle.max(1)
                pred_angle = pred_angle.cpu().numpy()
                pred_color_angle = np.array([CUBOID_COLOR_MAP[l] for l in pred_angle])
                save_obj('%s/%s/%d_epoch_%d_angle.obj' % (opt.results_path, opt.name, epoch, i), samples, pred_color_angle)

                pred_scale = out_scale.slice(sinput).F
                pred_scale = pred_scale.cpu().numpy()
                save_offset('%s/%s/%d_epoch_%d_angle_offset.obj' % (opt.results_path, opt.name, epoch, i), samples, pred, pred_angle, pred_scale.squeeze())

        return samples, pred


def viz_offset_finetune(opt, net, cuda, test_dataset, num_test=1, epoch=0):
    with torch.no_grad():
        samples = test_dataset.return_data()
        color = np.ones(samples.shape)
        samples, color = torch.Tensor(samples), torch.Tensor(color)
        for i in range(num_test):
            # Quantize the input
            coords, feats,  idx1, idx2 = ME.utils.sparse_quantize(
                samples,
                features=color,
                quantization_size=opt.voxel_size,
                return_index=True,
                return_inverse=True)
            coords = torch.hstack((torch.zeros((len(coords),1)), coords))
            print(coords.shape)
            sinput = ME.SparseTensor(feats.type(torch.float32), coords, device=cuda)

            out, out_offset, out_inout, out_angle, out_scale = net.forward(sinput)

            # map to input
            logits = out.slice(sinput).F

            _, pred = logits.max(1)
            pred = pred.cpu().numpy()
            pred_color = np.array([CUBOID_COLOR_MAP[l] for l in pred])

            samples = samples[idx1]
            save_obj('%s/%s/%d_epoch_%d.obj' % (opt.results_path, opt.name, epoch, i), samples, pred_color)

            pred_offset = out_offset.slice(sinput).F
            pred_offset = pred_offset.cpu().numpy()
            print(pred_offset.shape)
            print(np.linalg.norm(pred_offset, axis=1))
            trans_coords = samples + pred_offset
            save_obj('%s/%s/%d_epoch_%d_offset.obj' % (opt.results_path, opt.name, epoch, i), trans_coords, pred_color[coords[:, 0] == 0])


def compute_partial_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
      Args:
          box_a, box_b: 6D of center and lengths
      Returns:
          iou
      """
    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return False, False

    intersection = (min_max - max_min).prod()
    union1 = box_a[3] * box_a[4] * box_a[5]
    union2 = box_b[3] * box_b[4] * box_b[5]

    thre=0.8
    return (intersection/union1)>thre, (intersection/union2)>thre

def minkowski_collate_fn(list_data):
    # collate provided by ME only considers data, feature and labels.
    coords, feats, labels = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["labels"] for d in list_data],
        dtype=torch.float32,
    )

    samples = [d["samples"] for d in list_data]
    box_labels = [d["box_labels"] for d in list_data]
    masks = [d["masks"] for d in list_data]
    angles = [d["angles"] for d in list_data]
    return {
        "coords": coords,
        "feats": feats,
        "labels": labels[:,0],
        "offsets": labels[:,-3:],
        "samples": samples,
        "boxes": box_labels,
        "masks": masks,
        "angles": torch.Tensor(np.concatenate(angles)),
        "scale": torch.linalg.norm(labels[:,-3:], axis=1),
    }


def minkowski_collate_fn_finetune(list_data):
    # collate provided by ME only considers data, feature and labels.
    coords, feats, labels = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["labels"] for d in list_data],
        dtype=torch.float32,
    )

    samples = [d["samples"] for d in list_data]
    masks = [d["masks"] for d in list_data]
    return {
        "coords": coords,
        "feats": feats,
        "offsets": labels[:,1:],
        "labels": labels[:,0],
        "samples": samples,
        "masks": masks,
    }
