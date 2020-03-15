import argparse
import os
import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from nuscenes.nuscenes import NuScenes
from data_utils.NuScenesDataLoader import NuscenesDataset,NuscenesDataset2
import datetime
import logging
from pathlib import Path
from utils.utils import plot_loss_curve_iter, plot_loss_curve_epoch, plot_mAP_curve
from utils.evaluation import DetectionMAP as Evaluate_metric
from utils.evaluation2 import DetectionDistMAP as Evaluate_metric2
from model.pointmaskrcnn3 import PointMaskRCNN, compute_vote_loss, compute_rpn_class_bphere_loss, compute_mrcnn_bsphere_loss, compute_mrcnn_class_loss

def parse_args():
    parser = argparse.ArgumentParser('PointNet2')
    parser.add_argument('--mark', required=True, type=str, help='mark of the experiment. It decides where to store samples and models')
    parser.add_argument('--batchsize', type=int, default=12, help='input batch size')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs for training')
    parser.add_argument('--pretrained_RPN', type=str, default=None,help='whether use pretrained pointnet2 model')
    parser.add_argument('--use_semantic_loss', action='store_true',help='whether use semantic loss')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
    parser.add_argument('--model_name', type=str, default='PointMaskRCNN', help='Name of model')

    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.multi_gpu is None else '0,1,2,3'
    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) +'/%sNuScenes-'%args.model_name+ 'lr-'+str(args.learning_rate)+args.mark)
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    print('Load data...')

    nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/raid/chengnan/NuScenes/', verbose=True)
    dataset = NuscenesDataset(nusc, npoints_lidar=16384, npoints_radar=1024, split='train_detect')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, collate_fn=dataset.collate_fn,
                                             shuffle=True, num_workers=int(args.workers), drop_last=True)
    test_dataset = NuscenesDataset(nusc, npoints_lidar=16384, npoints_radar=1024, split='val_small')
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, collate_fn=test_dataset.collate_fn,
                                                 shuffle=True, num_workers=int(args.workers))

    if args.model_name == 'PointMaskRCNN':
        model = PointMaskRCNN()

    if args.pretrained_RPN is not None:
        model.RPN.load_state_dict(torch.load(args.pretrained_RPN))
        print('load model %s'%args.pretrained_RPN)
        logger.info('load model %s'%args.pretrained_RPN)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    #pretrain = args.pretrained_Pointnet2
    init_epoch = 0

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    LEARNING_RATE_CLIP = 1e-5

    '''GPU selection and multi-GPU'''
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model.cuda()

    history = defaultdict(lambda: list())
    best_mAP_dete_05 = 0.0
    best_mAP_dete_05_epoch = 0
    best_mAP_dete_1 = 0.0
    best_mAP_dete_1_epoch = 0
    best_mAP_dete_2 = 0.0
    best_mAP_dete_2_epoch = 0
    best_mAP_dete_4 = 0.0
    best_mAP_dete_4_epoch = 0
    best_mAP_dete_avg = 0.0
    best_mAP_dete_avg_epoch = 0.0

    for epoch in range(init_epoch,args.epoch):
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'],LEARNING_RATE_CLIP)
        print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        train_loss_sum = 0.0
        train_mAP_RPN_25 = Evaluate_metric(1, overlap_threshold=0.25)
        train_mAP_RPN_50 = Evaluate_metric(1, overlap_threshold=0.50)
        train_mAP_DETECTION_05 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=0.5)
        train_mAP_DETECTION_1 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=1.0)
        train_mAP_DETECTION_2 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=2.0)
        train_mAP_DETECTION_4 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=4.0)
        for i, data in enumerate(dataloader):
            if (i+1)%200 == 1:
                train_it_mAP_RPN_25 = Evaluate_metric(1, overlap_threshold=0.25)
                train_it_mAP_RPN_50 = Evaluate_metric(1, overlap_threshold=0.50)
                train_it_mAP_DETECTION_05 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=0.5)
                train_it_mAP_DETECTION_1 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=1.0)
                train_it_mAP_DETECTION_2 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=2.0)
                train_it_mAP_DETECTION_4 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=4.0)
            
            lidar_points, _, _, bounding_spheres_list, vote_labels, vote_masks, _, _, _ = data           
            lidar_points, vote_labels, vote_masks = lidar_points.float(), vote_labels.float(), vote_masks.float()
            lidar_points = lidar_points.transpose(2, 1)
            lidar_points, vote_labels, vote_masks = lidar_points.cuda(), vote_labels.cuda(), vote_masks.cuda()
            bounding_spheres_list = [b.float().cuda() for b in bounding_spheres_list]
             
            optimizer.zero_grad()
            model = model.train()
            
            vote_delta, agg_xyz, rpn_class_logits, rpn_bsphere, mrcnn_class_logits_batch, mrcnn_bsphere_batch, roi_gt_class_ids_batch, roi_bsphere_target_batch = model(lidar_points[:,:3,:], lidar_points[:,3:,:], bounding_spheres_list, mode='train')
            vote_loss = compute_vote_loss(vote_delta, vote_labels, vote_masks)
            rpn_class_loss, rpn_bsphere_loss = compute_rpn_class_bphere_loss(agg_xyz, rpn_class_logits, rpn_bsphere, bounding_spheres_list)
            mrcnn_bsphere_loss = compute_mrcnn_bsphere_loss(roi_bsphere_target_batch, roi_gt_class_ids_batch, mrcnn_bsphere_batch)
            mrcnn_class_loss = compute_mrcnn_class_loss(roi_gt_class_ids_batch, mrcnn_class_logits_batch)
            loss = vote_loss + rpn_class_loss + rpn_bsphere_loss + mrcnn_bsphere_loss + mrcnn_class_loss
            history['loss'].append(loss.item())
            train_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            # training metric
            model = model.eval()
            with torch.no_grad():
                #rpn_proposals, rpn_proposals_scores, final_detections, mrcnn_mask_all_points_batch = model(points[:,:3,:],points[:,3:,:],anchors,None,None,None,mode='test')
                rpn_proposals, rpn_proposals_scores, final_detections = model(lidar_points[:,:3,:], lidar_points[:,3:,:], None, mode='test')
            rpn_proposals = [p.cpu().numpy() for p in rpn_proposals]
            rpn_proposals_scores = [s.cpu().numpy() for s in rpn_proposals_scores]
            final_detections = [d.cpu().numpy() for d in final_detections]
            gt_bsphere_list = [b.cpu().numpy() for b in bounding_spheres_list]
            for k in range(len(final_detections)):
                rpn_pred_bspheres = rpn_proposals[k]
                rpn_pred_bspheres_labels = np.zeros(rpn_pred_bspheres.shape[0])
                rpn_pred_scores = rpn_proposals_scores[k]
                gt_bsphere = gt_bsphere_list[k][:,0:4]
                gt_bsphere_label = np.zeros(gt_bsphere.shape[0])
                train_mAP_RPN_25.evaluate(rpn_pred_bspheres, rpn_pred_bspheres_labels, rpn_pred_scores, gt_bsphere, gt_bsphere_label)
                train_it_mAP_RPN_25.evaluate(rpn_pred_bspheres, rpn_pred_bspheres_labels, rpn_pred_scores, gt_bsphere, gt_bsphere_label)
                train_mAP_RPN_50.evaluate(rpn_pred_bspheres, rpn_pred_bspheres_labels, rpn_pred_scores, gt_bsphere, gt_bsphere_label)
                train_it_mAP_RPN_50.evaluate(rpn_pred_bspheres, rpn_pred_bspheres_labels, rpn_pred_scores, gt_bsphere, gt_bsphere_label)
                pred_sphere = final_detections[k][:,0:4] if final_detections[k].shape[0]>0 else final_detections[k]
                pred_class = final_detections[k][:,4] if final_detections[k].shape[0]>0 else final_detections[k]
                pred_conf = final_detections[k][:,5] if final_detections[k].shape[0]>0 else final_detections[k]
                gt_sphere = gt_bsphere_list[k][:,0:4]
                gt_class = gt_bsphere_list[k][:,-1]
                train_mAP_DETECTION_05.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                train_it_mAP_DETECTION_05.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                train_mAP_DETECTION_1.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                train_it_mAP_DETECTION_1.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                train_mAP_DETECTION_2.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                train_it_mAP_DETECTION_2.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                train_mAP_DETECTION_4.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                train_it_mAP_DETECTION_4.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
            
            if (i+1)%5 == 0:
                print('[Epoch %d/%d] [Iteration %d/%d] TRAIN total_loss/vote_loss/rpn_cls_loss/rpn_bsph_loss/mrcnn_cls_loss/mrcnn_bsph_loss: %f/%f/%f/%f/%f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), loss.item(), vote_loss.item(), rpn_class_loss.item(), rpn_bsphere_loss.item(), mrcnn_class_loss.item(), mrcnn_bsphere_loss.item()))
                logger.info('[Epoch %d/%d] [Iteration %d/%d] TRAIN total_loss/vote_loss/rpn_cls_loss/rpn_bsph_loss/mrcnn_cls_loss/mrcnn_bsph_loss: %f/%f/%f/%f/%f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), loss.item(), vote_loss.item(), rpn_class_loss.item(), rpn_bsphere_loss.item(), mrcnn_class_loss.item(), mrcnn_bsphere_loss.item()))
            if (i+1)%200 == 0:
                train_it_mAP_RPN_25.finalize()
                history['train_it_mAP_RPN_25'].append(train_it_mAP_RPN_25.mAP())
                train_it_mAP_RPN_50.finalize()
                history['train_it_mAP_RPN_50'].append(train_it_mAP_RPN_50.mAP())
                train_it_mAP_DETECTION_05.finalize()
                mAP_DETECTION_05 = train_it_mAP_DETECTION_05.mAP()
                history['train_it_mAP_DETECTION_05'].append(mAP_DETECTION_05)
                train_it_mAP_DETECTION_1.finalize()
                mAP_DETECTION_1 = train_it_mAP_DETECTION_1.mAP()
                history['train_it_mAP_DETECTION_1'].append(mAP_DETECTION_1)
                train_it_mAP_DETECTION_2.finalize()
                mAP_DETECTION_2 = train_it_mAP_DETECTION_2.mAP()
                history['train_it_mAP_DETECTION_2'].append(mAP_DETECTION_2)
                train_it_mAP_DETECTION_4.finalize()
                mAP_DETECTION_4 = train_it_mAP_DETECTION_4.mAP()
                history['train_it_mAP_DETECTION_4'].append(mAP_DETECTION_4)
                mAP_DETECTION_avg = (mAP_DETECTION_05 + mAP_DETECTION_1 + mAP_DETECTION_2 + mAP_DETECTION_4)/4.0
                history['train_it_mAP_DETECTION_avg'].append(mAP_DETECTION_avg)
                train_loss_avg = np.sum(history['loss'][-200:])/200.0
                history['train_it_loss'].append(train_loss_avg)
                print('[Epoch %d/%d] [Iteration %d/%d] TRAIN AP_RPN_25/AP_RPN_50/mAP_dete_0.5/mAP_dete_1.0/mAP_dete_2.0/mAP_dete_4.0/mAP_dete_avg/loss: %f/%f/%f/%f/%f/%f/%f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), train_it_mAP_RPN_25.mAP(), train_it_mAP_RPN_50.mAP(), mAP_DETECTION_05, mAP_DETECTION_1, mAP_DETECTION_2, mAP_DETECTION_4, mAP_DETECTION_avg, train_loss_avg))
                logger.info('[Epoch %d/%d] [Iteration %d/%d] TRAIN AP_RPN_25/AP_RPN_50/mAP_dete_0.5/mAP_dete_1.0/mAP_dete_2.0/mAP_dete_4.0/mAP_dete_avg/loss: %f/%f/%f/%f/%f/%f/%f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), train_it_mAP_RPN_25.mAP(), train_it_mAP_RPN_50.mAP(), mAP_DETECTION_05, mAP_DETECTION_1, mAP_DETECTION_2, mAP_DETECTION_4, mAP_DETECTION_avg, train_loss_avg))
        
                # Validation
                test_mAP_RPN_25 = Evaluate_metric(1, overlap_threshold=0.25)
                test_mAP_RPN_50 = Evaluate_metric(1, overlap_threshold=0.50)
                test_mAP_DETECTION_05 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=0.5)
                test_mAP_DETECTION_1 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=1.0)
                test_mAP_DETECTION_2 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=2.0)
                test_mAP_DETECTION_4 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=4.0)        
                for j, data in enumerate(testdataloader):
                    with torch.no_grad():
                        lidar_points, _, _, bounding_spheres_list, _, _, _, _, _ = data          
                        lidar_points = lidar_points.float()
                        lidar_points = lidar_points.transpose(2, 1)
                        lidar_points = lidar_points.cuda()
                        bounding_spheres_list = [b.float().cuda() for b in bounding_spheres_list]
                
                        model = model.eval()
                        rpn_proposals, rpn_proposals_scores, final_detections = model(lidar_points[:,:3,:], lidar_points[:,3:,:], None, mode='test')
                    rpn_proposals = [p.cpu().numpy() for p in rpn_proposals]
                    rpn_proposals_scores = [s.cpu().numpy() for s in rpn_proposals_scores]
                    final_detections = [d.cpu().numpy() for d in final_detections]
                    gt_bsphere_list = [b.cpu().numpy() for b in bounding_spheres_list]
                    for k in range(len(final_detections)):
                        rpn_pred_bspheres = rpn_proposals[k]
                        rpn_pred_bspheres_labels = np.zeros(rpn_pred_bspheres.shape[0])
                        rpn_pred_scores = rpn_proposals_scores[k]
                        gt_bsphere = gt_bsphere_list[k][:,0:4]
                        gt_bsphere_label = np.zeros(gt_bsphere.shape[0])
                        test_mAP_RPN_25.evaluate(rpn_pred_bspheres, rpn_pred_bspheres_labels, rpn_pred_scores, gt_bsphere, gt_bsphere_label)
                        test_mAP_RPN_50.evaluate(rpn_pred_bspheres, rpn_pred_bspheres_labels, rpn_pred_scores, gt_bsphere, gt_bsphere_label)
                        pred_sphere = final_detections[k][:,0:4] if final_detections[k].shape[0]>0 else final_detections[k]
                        pred_class = final_detections[k][:,4] if final_detections[k].shape[0]>0 else final_detections[k]
                        pred_conf = final_detections[k][:,5] if final_detections[k].shape[0]>0 else final_detections[k]
                        gt_sphere = gt_bsphere_list[k][:,0:4]
                        gt_class = gt_bsphere_list[k][:,-1]
                        test_mAP_DETECTION_05.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                        test_mAP_DETECTION_1.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                        test_mAP_DETECTION_2.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                        test_mAP_DETECTION_4.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
        
                test_mAP_RPN_25.finalize()
                history['test_it_mAP_RPN_25'].append(test_mAP_RPN_25.mAP())
                test_mAP_RPN_50.finalize()
                history['test_it_mAP_RPN_50'].append(test_mAP_RPN_50.mAP())
                test_mAP_DETECTION_05.finalize()
                mAP_DETECTION_05 = test_mAP_DETECTION_05.mAP()
                history['test_it_mAP_DETECTION_05'].append(mAP_DETECTION_05)
                test_mAP_DETECTION_1.finalize()
                mAP_DETECTION_1 = test_mAP_DETECTION_1.mAP()
                history['test_it_mAP_DETECTION_1'].append(mAP_DETECTION_1)
                test_mAP_DETECTION_2.finalize()
                mAP_DETECTION_2 = test_mAP_DETECTION_2.mAP()
                history['test_it_mAP_DETECTION_2'].append(mAP_DETECTION_2)
                test_mAP_DETECTION_4.finalize()
                mAP_DETECTION_4 = test_mAP_DETECTION_4.mAP()
                history['test_it_mAP_DETECTION_4'].append(mAP_DETECTION_4)
                mAP_DETECTION_avg = (mAP_DETECTION_05 + mAP_DETECTION_1 + mAP_DETECTION_2 + mAP_DETECTION_4)/4.0
                history['test_it_mAP_DETECTION_avg'].append(mAP_DETECTION_avg)

                print('[Epoch %d/%d] [Iteration %d/%d] TEST AP_RPN_25/AP_RPN_50/mAP_dete_0.5/mAP_dete_1.0/mAP_dete_2.0/mAP_dete_4.0/mAP_dete_avg: %f/%f/%f/%f/%f/%f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), test_mAP_RPN_25.mAP(), test_mAP_RPN_50.mAP(), mAP_DETECTION_05, mAP_DETECTION_1, mAP_DETECTION_2, mAP_DETECTION_4, mAP_DETECTION_avg))
                logger.info('[Epoch %d/%d] [Iteration %d/%d] TEST AP_RPN_25/AP_RPN_50/mAP_dete_0.5/mAP_dete_1.0/mAP_dete_2.0/mAP_dete_4.0/mAP_dete_avg: %f/%f/%f/%f/%f/%f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), test_mAP_RPN_25.mAP(), test_mAP_RPN_50.mAP(), mAP_DETECTION_05, mAP_DETECTION_1, mAP_DETECTION_2, mAP_DETECTION_4, mAP_DETECTION_avg))
                for class_ind in range(11):
                    if class_ind not in test_mAP_DETECTION_05.ignore_class:
                        logger.info('AP of class %d: dete_0.5/dete_1.0/dete_2.0/dete_4.0/dete_avg: %f/%f/%f/%f/%f' % (class_ind, test_mAP_DETECTION_05.AP(class_ind), test_mAP_DETECTION_1.AP(class_ind), test_mAP_DETECTION_2.AP(class_ind), test_mAP_DETECTION_4.AP(class_ind), (test_mAP_DETECTION_05.AP(class_ind) + test_mAP_DETECTION_1.AP(class_ind) + test_mAP_DETECTION_2.AP(class_ind) + test_mAP_DETECTION_4.AP(class_ind))/4.0))
                
                if test_mAP_DETECTION_05.mAP() > best_mAP_dete_05:
                    best_mAP_dete_05 = test_mAP_DETECTION_05.mAP()
                    best_mAP_dete_05_epoch = epoch+1
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_dete_mAP_0.5.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_dete_05))
                    logger.info('Save best detection 0.5 mAP model..')
                    print('Save best detection 0.5 mAP model..')
                if test_mAP_DETECTION_1.mAP() > best_mAP_dete_1:
                    best_mAP_dete_1 = test_mAP_DETECTION_1.mAP()
                    best_mAP_dete_1_epoch = epoch+1
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_dete_mAP_1.0.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_dete_1))
                    logger.info('Save best detection 1.0 mAP model..')
                    print('Save best detection 1.0 mAP model..')
                if test_mAP_DETECTION_2.mAP() > best_mAP_dete_2:
                    best_mAP_dete_2 = test_mAP_DETECTION_2.mAP()
                    best_mAP_dete_2_epoch = epoch+1
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_dete_mAP_2.0.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_dete_2))
                    logger.info('Save best detection 2.0 mAP model..')
                    print('Save best detection 2.0 mAP model..')
                if test_mAP_DETECTION_4.mAP() > best_mAP_dete_4:
                    best_mAP_dete_4 = test_mAP_DETECTION_4.mAP()
                    best_mAP_dete_4_epoch = epoch+1
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_dete_mAP_4.0.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_dete_4))
                    logger.info('Save best detection 4.0 mAP model..')
                    print('Save best detection 4.0 mAP model..')
                if ((test_mAP_DETECTION_05.mAP() + test_mAP_DETECTION_1.mAP()+ test_mAP_DETECTION_2.mAP() + test_mAP_DETECTION_4.mAP())/4.0) > best_mAP_dete_avg:
                    best_mAP_dete_avg = (test_mAP_DETECTION_05.mAP() + test_mAP_DETECTION_1.mAP()+ test_mAP_DETECTION_2.mAP() + test_mAP_DETECTION_4.mAP())/4.0
                    best_mAP_dete_avg_epoch = epoch+1
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_dete_mAP_avg.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_dete_avg))
                    logger.info('Save best detection avg mAP model..')
                    print('Save best detection avg mAP model..')                 
                
                
        
        train_mAP_RPN_25.finalize()
        history['train_mAP_RPN_25'].append(train_mAP_RPN_25.mAP())
        train_mAP_RPN_50.finalize()
        history['train_mAP_RPN_50'].append(train_mAP_RPN_50.mAP())
        train_mAP_DETECTION_05.finalize()
        mAP_DETECTION_05 = train_mAP_DETECTION_05.mAP()
        history['train_mAP_DETECTION_05'].append(mAP_DETECTION_05)
        train_mAP_DETECTION_1.finalize()
        mAP_DETECTION_1 = train_mAP_DETECTION_1.mAP()
        history['train_mAP_DETECTION_1'].append(mAP_DETECTION_1)
        train_mAP_DETECTION_2.finalize()
        mAP_DETECTION_2 = train_mAP_DETECTION_2.mAP()
        history['train_mAP_DETECTION_2'].append(mAP_DETECTION_2)
        train_mAP_DETECTION_4.finalize()
        mAP_DETECTION_4 = train_mAP_DETECTION_4.mAP()
        history['train_mAP_DETECTION_4'].append(mAP_DETECTION_4)
        mAP_DETECTION_avg = (mAP_DETECTION_05 + mAP_DETECTION_1 + mAP_DETECTION_2 + mAP_DETECTION_4)/4.0
        history['train_mAP_DETECTION_avg'].append(mAP_DETECTION_avg)
        train_loss_avg = train_loss_sum/(len(dataloader)*1.0)
        history['train_loss'].append(train_loss_avg)
        print('[Epoch %d/%d] TRAIN AP_RPN_25/AP_RPN_50/mAP_dete_0.5/mAP_dete_1.0/mAP_dete_2.0/mAP_dete_4.0/mAP_dete_avg/loss: %f/%f/%f/%f/%f/%f/%f/%f ' % (epoch+1, args.epoch, train_mAP_RPN_25.mAP(), train_mAP_RPN_50.mAP(), mAP_DETECTION_05, mAP_DETECTION_1, mAP_DETECTION_2, mAP_DETECTION_4, mAP_DETECTION_avg, train_loss_avg))
        logger.info('[Epoch %d/%d] TRAIN AP_RPN_25/AP_RPN_50/mAP_dete_0.5/mAP_dete_1.0/mAP_dete_2.0/mAP_dete_4.0/mAP_dete_avg/loss: %f/%f/%f/%f/%f/%f/%f/%f ' % (epoch+1, args.epoch, train_mAP_RPN_25.mAP(), train_mAP_RPN_50.mAP(), mAP_DETECTION_05, mAP_DETECTION_1, mAP_DETECTION_2, mAP_DETECTION_4, mAP_DETECTION_avg, train_loss_avg))
        
        # Validation
        test_mAP_RPN_25 = Evaluate_metric(1, overlap_threshold=0.25)
        test_mAP_RPN_50 = Evaluate_metric(1, overlap_threshold=0.50)
        test_mAP_DETECTION_05 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=0.5)
        test_mAP_DETECTION_1 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=1.0)
        test_mAP_DETECTION_2 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=2.0)
        test_mAP_DETECTION_4 = Evaluate_metric2(11, ignore_class=[0], dist_threshold=4.0)        
        for j, data in enumerate(testdataloader):
            with torch.no_grad():
                lidar_points, _, _, bounding_spheres_list, _, _, _, _, _ = data          
                lidar_points = lidar_points.float()
                lidar_points = lidar_points.transpose(2, 1)
                lidar_points = lidar_points.cuda()
                bounding_spheres_list = [b.float().cuda() for b in bounding_spheres_list]
                
                model = model.eval()
                rpn_proposals, rpn_proposals_scores, final_detections = model(lidar_points[:,:3,:], lidar_points[:,3:,:], None, mode='test')
            rpn_proposals = [p.cpu().numpy() for p in rpn_proposals]
            rpn_proposals_scores = [s.cpu().numpy() for s in rpn_proposals_scores]
            final_detections = [d.cpu().numpy() for d in final_detections]
            gt_bsphere_list = [b.cpu().numpy() for b in bounding_spheres_list]
            for k in range(len(final_detections)):
                rpn_pred_bspheres = rpn_proposals[k]
                rpn_pred_bspheres_labels = np.zeros(rpn_pred_bspheres.shape[0])
                rpn_pred_scores = rpn_proposals_scores[k]
                gt_bsphere = gt_bsphere_list[k][:,0:4]
                gt_bsphere_label = np.zeros(gt_bsphere.shape[0])
                test_mAP_RPN_25.evaluate(rpn_pred_bspheres, rpn_pred_bspheres_labels, rpn_pred_scores, gt_bsphere, gt_bsphere_label)
                test_mAP_RPN_50.evaluate(rpn_pred_bspheres, rpn_pred_bspheres_labels, rpn_pred_scores, gt_bsphere, gt_bsphere_label)
                pred_sphere = final_detections[k][:,0:4] if final_detections[k].shape[0]>0 else final_detections[k]
                pred_class = final_detections[k][:,4] if final_detections[k].shape[0]>0 else final_detections[k]
                pred_conf = final_detections[k][:,5] if final_detections[k].shape[0]>0 else final_detections[k]
                gt_sphere = gt_bsphere_list[k][:,0:4]
                gt_class = gt_bsphere_list[k][:,-1]
                test_mAP_DETECTION_05.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                test_mAP_DETECTION_1.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                test_mAP_DETECTION_2.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
                test_mAP_DETECTION_4.evaluate(pred_sphere, pred_class, pred_conf, gt_sphere, gt_class)
        
        test_mAP_RPN_25.finalize()
        history['test_mAP_RPN_25'].append(test_mAP_RPN_25.mAP())
        test_mAP_RPN_50.finalize()
        history['test_mAP_RPN_50'].append(test_mAP_RPN_50.mAP())
        test_mAP_DETECTION_05.finalize()
        mAP_DETECTION_05 = test_mAP_DETECTION_05.mAP()
        history['test_mAP_DETECTION_05'].append(mAP_DETECTION_05)
        test_mAP_DETECTION_1.finalize()
        mAP_DETECTION_1 = test_mAP_DETECTION_1.mAP()
        history['test_mAP_DETECTION_1'].append(mAP_DETECTION_1)
        test_mAP_DETECTION_2.finalize()
        mAP_DETECTION_2 = test_mAP_DETECTION_2.mAP()
        history['test_mAP_DETECTION_2'].append(mAP_DETECTION_2)
        test_mAP_DETECTION_4.finalize()
        mAP_DETECTION_4 = test_mAP_DETECTION_4.mAP()
        history['test_mAP_DETECTION_4'].append(mAP_DETECTION_4)
        mAP_DETECTION_avg = (mAP_DETECTION_05 + mAP_DETECTION_1 + mAP_DETECTION_2 + mAP_DETECTION_4)/4.0
        history['test_mAP_DETECTION_avg'].append(mAP_DETECTION_avg)

        print('[Epoch %d/%d] TEST AP_RPN_25/AP_RPN_50/mAP_dete_0.5/mAP_dete_1.0/mAP_dete_2.0/mAP_dete_4.0/mAP_dete_avg: %f/%f/%f/%f/%f/%f/%f ' % (epoch+1, args.epoch, test_mAP_RPN_25.mAP(), test_mAP_RPN_50.mAP(), mAP_DETECTION_05, mAP_DETECTION_1, mAP_DETECTION_2, mAP_DETECTION_4, mAP_DETECTION_avg))
        logger.info('[Epoch %d/%d] TEST AP_RPN_25/AP_RPN_50/mAP_dete_0.5/mAP_dete_1.0/mAP_dete_2.0/mAP_dete_4.0/mAP_dete_avg: %f/%f/%f/%f/%f/%f/%f ' % (epoch+1, args.epoch, test_mAP_RPN_25.mAP(), test_mAP_RPN_50.mAP(), mAP_DETECTION_05, mAP_DETECTION_1, mAP_DETECTION_2, mAP_DETECTION_4, mAP_DETECTION_avg))
        for class_ind in range(11):
            if class_ind not in test_mAP_DETECTION_05.ignore_class:
                logger.info('AP of class %d: dete_0.5/dete_1.0/dete_2.0/dete_4.0/dete_avg: %f/%f/%f/%f/%f' % (class_ind, test_mAP_DETECTION_05.AP(class_ind), test_mAP_DETECTION_1.AP(class_ind), test_mAP_DETECTION_2.AP(class_ind), test_mAP_DETECTION_4.AP(class_ind), (test_mAP_DETECTION_05.AP(class_ind) + test_mAP_DETECTION_1.AP(class_ind) + test_mAP_DETECTION_2.AP(class_ind) + test_mAP_DETECTION_4.AP(class_ind))/4.0))
        
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1))
            logger.info('Save model..')
            print('Save model..')
            plot_loss_curve_iter(history['loss'], str(log_dir))
            plot_loss_curve_epoch(history['train_it_loss'],str(log_dir))
            plot_mAP_curve(history['train_it_mAP_RPN_25'], history['test_it_mAP_RPN_25'], str(log_dir), 'RPN_25')
            plot_mAP_curve(history['train_it_mAP_RPN_50'], history['test_it_mAP_RPN_50'], str(log_dir), 'RPN_50')
            plot_mAP_curve(history['train_it_mAP_DETECTION_05'], history['test_it_mAP_DETECTION_05'], str(log_dir), 'detection_0.5')
            plot_mAP_curve(history['train_it_mAP_DETECTION_1'], history['test_it_mAP_DETECTION_1'], str(log_dir), 'detection_1.0')
            plot_mAP_curve(history['train_it_mAP_DETECTION_2'], history['test_it_mAP_DETECTION_2'], str(log_dir), 'detection_2.0')
            plot_mAP_curve(history['train_it_mAP_DETECTION_4'], history['test_it_mAP_DETECTION_4'], str(log_dir), 'detection_4.0')
        if test_mAP_DETECTION_05.mAP() > best_mAP_dete_05:
            best_mAP_dete_05 = test_mAP_DETECTION_05.mAP()
            best_mAP_dete_05_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_dete_mAP_0.5.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_dete_05))
            logger.info('Save best detection 0.5 mAP model..')
            print('Save best detection 0.5 mAP model..')
        if test_mAP_DETECTION_1.mAP() > best_mAP_dete_1:
            best_mAP_dete_1 = test_mAP_DETECTION_1.mAP()
            best_mAP_dete_1_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_dete_mAP_1.0.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_dete_1))
            logger.info('Save best detection 1.0 mAP model..')
            print('Save best detection 1.0 mAP model..')
        if test_mAP_DETECTION_2.mAP() > best_mAP_dete_2:
            best_mAP_dete_2 = test_mAP_DETECTION_2.mAP()
            best_mAP_dete_2_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_dete_mAP_2.0.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_dete_2))
            logger.info('Save best detection 2.0 mAP model..')
            print('Save best detection 2.0 mAP model..')
        if test_mAP_DETECTION_4.mAP() > best_mAP_dete_4:
            best_mAP_dete_4 = test_mAP_DETECTION_4.mAP()
            best_mAP_dete_4_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_dete_mAP_4.0.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_dete_4))
            logger.info('Save best detection 4.0 mAP model..')
            print('Save best detection 4.0 mAP model..')
        if ((test_mAP_DETECTION_05.mAP() + test_mAP_DETECTION_1.mAP()+ test_mAP_DETECTION_2.mAP() + test_mAP_DETECTION_4.mAP())/4.0) > best_mAP_dete_avg:
            best_mAP_dete_avg = (test_mAP_DETECTION_05.mAP() + test_mAP_DETECTION_1.mAP()+ test_mAP_DETECTION_2.mAP() + test_mAP_DETECTION_4.mAP())/4.0
            best_mAP_dete_avg_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_dete_mAP_avg.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_dete_avg))
            logger.info('Save best detection avg mAP model..')
            print('Save best detection avg mAP model..')            
        
    print('Best mAP of detection 0.5 is %f at epoch %d.' % (best_mAP_dete_05, best_mAP_dete_05_epoch))
    logger.info('Best mAP of detection 0.5 is %f at epoch %d.' % (best_mAP_dete_05, best_mAP_dete_05_epoch))
    print('Best mAP of detection 1.0 is %f at epoch %d.' % (best_mAP_dete_1, best_mAP_dete_1_epoch))
    logger.info('Best mAP of detection 1.0 is %f at epoch %d.' % (best_mAP_dete_1, best_mAP_dete_1_epoch))
    print('Best mAP of detection 2.0 is %f at epoch %d.' % (best_mAP_dete_2, best_mAP_dete_2_epoch))
    logger.info('Best mAP of detection 2.0 is %f at epoch %d.' % (best_mAP_dete_2, best_mAP_dete_2_epoch))
    print('Best mAP of detection 4.0 is %f at epoch %d.' % (best_mAP_dete_4, best_mAP_dete_4_epoch))
    logger.info('Best mAP of detection 4.0 is %f at epoch %d.' % (best_mAP_dete_4, best_mAP_dete_4_epoch))
    print('Best mAP of detection avg is %f at epoch %d.' % (best_mAP_dete_avg, best_mAP_dete_avg_epoch))
    logger.info('Best mAP of detection avg is %f at epoch %d.' % (best_mAP_dete_avg, best_mAP_dete_avg_epoch))    
    
    plot_loss_curve_iter(history['loss'], str(log_dir))
    plot_loss_curve_epoch(history['train_it_loss'],str(log_dir))
    plot_mAP_curve(history['train_it_mAP_RPN_25'], history['test_it_mAP_RPN_25'], str(log_dir), 'RPN_25')
    plot_mAP_curve(history['train_it_mAP_RPN_50'], history['test_it_mAP_RPN_50'], str(log_dir), 'RPN_50')
    plot_mAP_curve(history['train_it_mAP_DETECTION_05'], history['test_it_mAP_DETECTION_05'], str(log_dir), 'detection_0.5')
    plot_mAP_curve(history['train_it_mAP_DETECTION_1'], history['test_it_mAP_DETECTION_1'], str(log_dir), 'detection_1.0')
    plot_mAP_curve(history['train_it_mAP_DETECTION_2'], history['test_it_mAP_DETECTION_2'], str(log_dir), 'detection_2.0')
    plot_mAP_curve(history['train_it_mAP_DETECTION_4'], history['test_it_mAP_DETECTION_4'], str(log_dir), 'detection_4.0')
    plot_mAP_curve(history['train_it_mAP_DETECTION_avg'], history['test_it_mAP_DETECTION_avg'], str(log_dir), 'detection_avg')
    print('FINISH.')
    logger.info('FINISH')



if __name__ == '__main__':
    args = parse_args()
    main(args)
