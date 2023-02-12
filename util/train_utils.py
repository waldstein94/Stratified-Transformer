import os
import torch
import numpy as np
import itertools
# import trimesh
import open3d as o3d

#import MinkowskiEngine as ME
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

from util.iostream import *


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
        [save_obj_color_coding('tmp/%d_%d_instance.obj' % (i, k), item, np.ones(len(item))*k) for k, (item) in enumerate(instances)]

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



def instantiation_eval(path, name, samples, pred_offset, pred_labels):
    # instance segmentation based predicted offsets and labels
    samples_trans = samples + pred_offset
    # samples_trans[pred_labels>5] = samples[pred_labels>5]
    cls_list, label_list = [], []
    inst_idx, indice_list = 0, []
    for i in range(max(pred_labels)+1):
        index_list = []
        pts_trans = samples_trans[pred_labels == i]
        pts_ori = samples[pred_labels == i]
        # save_obj_color_coding(os.path.join(path,'%s_%d_pts.obj' % (name, i)), pts_trans, np.ones(len(pts_trans))*i)
        if i<6:
            eps, min_samples, inpts = 0.1, 5, pts_trans # todo pts_ori vs pts_trans
            thre = 5
        else:
            eps, min_samples, inpts = 0.1, 5, pts_trans
            thre = 5
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(inpts)

        instances = [pts_ori[dbscan.labels_ == j] for j in range(max(dbscan.labels_)+1) if len(pts_ori[dbscan.labels_ == j]) > thre]

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
    f_cls_list, e_cls_list = cls_list[:6], cls_list[6:]
    print('name', name, 'len of f', len(list(itertools.chain(*f_cls_list))), 'len of e', len(list(itertools.chain(*e_cls_list))))
    pair_list = []
    # adjacent face indices along edge index
    lookup_face = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [2, 4], [3, 4], [1, 5], [2, 5], [3, 5], [4, 5]]
    for cls_idx, e_list in enumerate(e_cls_list[:12]):
        # f_idx1, f_idx2의 face instances에 대해 edge instance와 connection 만들기 based on euclidean distance
        f_idx1, f_idx2 = lookup_face[cls_idx][0], lookup_face[cls_idx][1]
        f_list1, f_list2 = f_cls_list[f_idx1], f_cls_list[f_idx2]
        f_inst_id1, f_inst_id2 = indice_list[f_idx1], indice_list[f_idx2]
        if (len(f_list1)==0) or (len(f_list2)==0):
            continue
        # edge를 기준으로 가까운 face search
        for e_supp in e_list:
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
            #-----------------------------------------------

            # ---option2 distance between sets------
            paired = []
            for k, fsup in enumerate(f_list1):
                dist1 = np.min(distance.cdist(e_supp, fsup), axis=1)
                r = np.sum(dist1<0.1)/len(dist1)
                # print(r)
                if r > 0.5:
                    paired.append(f_inst_id1[k])
                    break
            for k, fsup in enumerate(f_list2):
                dist2 = np.min(distance.cdist(e_supp, fsup), axis=1)
                r = np.sum(dist2 < 0.1) / len(dist2)
                if r > 0.5:
                    paired.append(f_inst_id2[k])
                    break
            # --------------------------------------

            if paired:
                pair_list.append(paired)

    print('number of paired list', len(pair_list))
    # tmp_list = list(itertools.chain(*cls_list))
    # for i, indices in enumerate(pair_list):
    #     if len(indices) == 2:
    #         pts = np.vstack((tmp_list[indices[0]], tmp_list[indices[1]]))
    #     else:
    #         pts = tmp_list[indices[0]]
    #     pts = np.vstack((pts, indices[1]))
    #     clr = np.ones(len(pts)) * (i%28)
    #     save_obj_color_coding(os.path.join(path,'%s_%d_paired.obj' % (name, i )), pts, clr)
    # exit(0)
    # merging pairs using set.intersection(set1, set2)
    terminate = False
    final_pair_list=[]
    # while terminate is not True:
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

    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(pts)
    #         pcd = pcd.voxel_down_sample(voxel_size=0.02)
    #         # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.01)
    #         cl, ind = pcd.remove_radius_outlier(nb_points=3, radius=0.1)
    #         if len(cl.points)<20:
    #             continue
    #         instances.append(np.asarray(cl.points))


    box_supp_list = []
    item_flatten = list(itertools.chain(*cls_list[:6]))
    for i, pair_indices in enumerate(final_pair_list):
        # if len(pair_indices) ==1:
        #     continue
        supps = np.concatenate([item_flatten[idx] for idx in pair_indices])
        # outlier removal
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(supps)
        pcd = pcd.voxel_down_sample(voxel_size=0.04)
        cl, ind = pcd.remove_radius_outlier(nb_points=3, radius=0.1)
        supps = np.asarray(cl.points)

        if len(supps)<60:
            continue

        #----todo: concave check based on occupancy ratio in projected space-----------
        # not that good
        # unit = 0.1
        # pts2d = (supps[:, :2] // unit).astype(int)
        # minx, miny, maxx, maxy = np.min(pts2d[:, 0]), np.min(pts2d[:, 1]), np.max(pts2d[:, 0]), np.max(pts2d[:, 1])
        # nomi, denomi = len(np.unique(pts2d, axis=0)), (maxx - minx) * (maxy - miny)
        # print('ratio ', nomi / denomi)
        #--------------------------------------------------------------------------------
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

    thre=0.3
    return intersection/union1>thre, intersection/union2>thre

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
