import os
import time
import random
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel

import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import dataset, config
from util.s3dis import S3DIS
from util.scannet_v2 import Scannetv2
from util.dcf import DCF
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port, poly_learning_rate, smooth_loss
from util.data_util import collate_fn_dcf, collate_fn_dcf_eval
from util import transform
from util.logger import get_logger
from util.iostream import *

from functools import partial
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup
import torch_points_kernels as tp



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_stratified_transformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    cuda = torch.device('cuda:%d' % args.train_gpu[0])
    torch.cuda.set_device(cuda)

#    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(os.path.join(args.save_path, args.name)):
        os.makedirs(os.path.join(args.save_path, args.name))
    if not os.path.exists(os.path.join(args.weight, args.name)):
        os.makedirs(os.path.join(args.weight, args.name))
    if not os.path.exists(os.path.join(args.result_path, args.name)):
        os.makedirs(os.path.join(args.result_path, args.name))


    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    # get model
    from model.stratified_transformer import Stratified

    args.patch_size = args.grid_size * args.patch_size
    args.window_size = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
    args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
    args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

    model = Stratified(args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size, \
        args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
        rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz, num_classes=args.classes, \
        ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer,\
                       activation=args.activation)

    
    # set loss func 
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    if args.loss =='l1':
        offsetloss = torch.nn.L1Loss().cuda()
    else:
        offsetloss = torch.nn.MSELoss().cuda()

    # set optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "blocks" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "blocks" in n and p.requires_grad],
                "lr": args.base_lr * transformer_lr_scale,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger(os.path.join(args.save_path, args.name))
        writer = SummaryWriter(os.path.join(args.save_path, args.name))
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    else:
        model = model.cuda()  #  torch.nn.DataParallel(model.cuda())

    if args.weight and args.train_continue:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(os.path.join(args.weight, args.name)))
            checkpoint = torch.load(os.path.join(args.weight, args.name))
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_state_dict = checkpoint['scheduler']
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_split = args.get("train_split", "train")
    if main_process():
        logger.info("scannet. train_split: {}".format(train_split))
        train_data = DCF(split=train_split, data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, shuffle_index=True, coord_move=args.coord_move)

    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, \
        pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=partial(collate_fn_dcf, max_batch_points=args.max_batch_points, logger=logger if main_process() else None))

    val_transform = None
    val_data = DCF(split='val', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=800000, coord_move=args.coord_move)

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=True, num_workers=args.workers, \
            pin_memory=True, sampler=val_sampler, collate_fn=collate_fn_dcf_eval)
    val_viz_loader = torch.utils.data.DataLoader(val_data, batch_size=args.viz_size_val, shuffle=False,
                                             num_workers=args.workers, \
                                             pin_memory=True, sampler=val_sampler, collate_fn=collate_fn_dcf_eval)
    
    # set scheduler
    if args.scheduler == "MultiStepWithWarmup":
        assert args.scheduler_update == 'step'
        if main_process():
            logger.info("scheduler: MultiStepWithWarmup. scheduler_update: {}".format(args.scheduler_update))
        iter_per_epoch = len(train_loader)
        milestones = [int(args.epochs*0.6) * iter_per_epoch, int(args.epochs*0.8) * iter_per_epoch]
        scheduler = MultiStepWithWarmup(optimizer, milestones=milestones, gamma=0.1, warmup=args.warmup, \
            warmup_iters=args.warmup_iters, warmup_ratio=args.warmup_ratio)
    elif args.scheduler == 'MultiStep':
        assert args.scheduler_update == 'epoch'
        milestones = [int(x) for x in args.milestones.split(",")] if hasattr(args, "milestones") else [int(args.epochs*0.6), int(args.epochs*0.8)]
        gamma = args.gamma if hasattr(args, 'gamma') else 0.1
        if main_process():
            logger.info("scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(args.scheduler_update, milestones, gamma))
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif args.scheduler == 'Poly':
        if main_process():
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))
            
        loss_train, loss_offset = train(train_loader, model, criterion, offsetloss, optimizer, epoch, scaler, scheduler, args.name)

        if args.scheduler_update == 'epoch':
            scheduler.step()
        epoch_log = epoch + 1
        
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('loss_offset', loss_offset, epoch_log)


        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            validate_qualitative(args, epoch, val_viz_loader, model, criterion, l1loss)
            loss_val, loss_offset  = validate(val_loader, model, criterion, l1loss)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('offset_val', loss_offset, epoch_log)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(os.path.join(args.weight, args.name)):
                os.makedirs(os.path.join(args.weight, args.name))
            filename = os.path.join(args.weight, args.name, 'epoch_%d.pth' % epoch_log)
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            # if is_best:
            #     shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

        # torch.cuda.empty_cache()

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def train(train_loader, model, criterion, l1loss, optimizer, epoch, scaler, scheduler, name='name'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    offset_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, feat, target, offset, shift) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        # if i>3:
        #     break
        data_time.update(time.time() - end)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii, o in enumerate(offset_)], 0).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]
    
        coord, feat, target, offset, shift = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True), shift.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]
        
        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            output, out_shift = model(feat, coord, offset, batch, neighbor_idx)
            assert output.shape[1] == args.classes
            if target.shape[-1] == 1:
                target = target[:, 0]  # for cls
            loss = criterion(output, target)

            # for predicted offset vectors
            loss_shift = l1loss(out_shift, shift)
            loss_total = loss + args.offset_weight * loss_shift

        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            optimizer.step()

        if args.scheduler_update == 'step':
            scheduler.step()

        n = coord.size(0)
        loss_meter.update(loss.item(), n)
        offset_meter.update(loss_shift.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info('Name: {}  '
                        'Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Loss_offset {offset_meter.val:.4f} '
                        'Lr: {lr} '.format(name, epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          offset_meter=offset_meter,
                                                          lr=lr))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('loss_offset_batch', offset_meter.val, current_iter)
        # torch.cuda.empty_cache()

    return loss_meter.avg, offset_meter.avg  #, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion, l1loss):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    offset_meter = AverageMeter()
    # intersection_meter = AverageMeter()
    # union_meter = AverageMeter()
    target_meter = AverageMeter()

    # torch.cuda.empty_cache()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset, shift) in enumerate(val_loader):
        # if i>2:
        #     break
        data_time.update(time.time() - end)
    
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]
    
        coord, feat, target, offset, shift = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True), shift.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]
        
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        with torch.no_grad():
            output, out_shift = model(feat, coord, offset, batch, neighbor_idx)
            loss = criterion(output, target)
            loss_shift = l1loss(out_shift, shift)

        # output = output.max(1)[1]
        n = coord.size(0)


        # intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        #
        # intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        # intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
 
        loss_meter.update(loss.item(), n)
        offset_meter.update(loss_shift.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Loss_offset {offset_meter.val:.4f} ({offset_meter.avg:.4f}) '.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          offset_meter=offset_meter))

    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: class_loss/offset_loss {:.4f}/{:.4f}.'.format(loss_meter.avg, offset_meter.avg))
        # for i in range(args.classes):
        #     logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        # logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, offset_meter.avg #, mIoU, mAcc, allAcc


def validate_qualitative(args, epoch, val_loader, model, criterion, l1loss):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Qualitative Evaluation >>>>>>>>>>>>>>>>')

    # torch.cuda.empty_cache()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset, shift) in enumerate(val_loader):
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = \
        tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[
            0]

        coord, feat, target, offset, shift = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(
            non_blocking=True), offset.cuda(non_blocking=True), shift.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        with torch.no_grad():
            output, out_shift = model(feat, coord, offset, batch, neighbor_idx)

        output = output.max(1)[1].cpu().numpy()
        trans_coord = (coord+out_shift).cpu().numpy()

        cls_res_path = os.path.join(args.result_path, args.name, '%d_epoch_%d.obj' % (epoch, i))
        save_obj_color_coding(cls_res_path, coord, output)
        offset_res_path = os.path.join(args.result_path, args.name, '%d_epoch_%d_offset.obj' % (epoch, i))
        save_obj_color_coding(offset_res_path, trans_coord, output)
        break


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
