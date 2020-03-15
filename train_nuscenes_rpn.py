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
from utils.utils import plot_loss_curve, plot_loss_curves, plot_mAP_curve
from utils.evaluation import DetectionMAP as Evaluate_metric
from model.pointmaskrcnn3 import RPN, compute_fg_bg_loss, compute_vote_loss, compute_rpn_class_bphere_loss, proposal_layer, iou_spheres

def parse_args():
    parser = argparse.ArgumentParser('PointNet2')
    parser.add_argument('--mark', required=True, type=str, help='mark of the experiment. It decides where to store samples and models')
    parser.add_argument('--batchsize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs for training')
    parser.add_argument('--pretrained_Pointnet2', type=str, default=None,help='whether use pretrained pointnet2 model')
    parser.add_argument('--use_semantic_loss', action='store_true',help='whether use semantic loss')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
    parser.add_argument('--model_name', type=str, default='RPN', help='Name of model')

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

    model = RPN()

    if args.pretrained_Pointnet2 is not None:
        model.pointnet2.load_state_dict(torch.load(args.pretrained_Pointnet2), strict=False)
        print('load model %s'%args.pretrained_Pointnet2)
        logger.info('load model %s'%args.pretrained_Pointnet2)
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
    best_mAP_RPN_100_25 = 0.0
    best_mAP_RPN_100_25_epoch = 0
    best_mAP_RPN_100_50 = 0.0
    best_mAP_RPN_100_50_epoch = 0
    best_mAP_RPN_05_25 = 0.0
    best_mAP_RPN_05_25_epoch = 0
    best_mAP_RPN_05_50 = 0.0
    best_mAP_RPN_05_50_epoch = 0

    for epoch in range(init_epoch,args.epoch):
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'],LEARNING_RATE_CLIP)
        print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        train_loss_sum = 0.0
        train_fg_bg_cls_loss_sum = 0.0
        train_vote_loss_sum = 0.0
        train_rpn_class_loss_sum = 0.0
        train_rpn_bsphere_loss_sum = 0.0
        train_mAP_RPN_100_25 = Evaluate_metric(1, overlap_threshold=0.25)
        train_mAP_RPN_100_50 = Evaluate_metric(1, overlap_threshold=0.50)
        train_mAP_RPN_05_25 = Evaluate_metric(1, overlap_threshold=0.25)
        train_mAP_RPN_05_50 = Evaluate_metric(1, overlap_threshold=0.50)
        invalid_count = 0
        for i, data in enumerate(dataloader):
            if (i+1)%200 == 1:
                train_it_mAP_RPN_100_25 = Evaluate_metric(1, overlap_threshold=0.25)
                train_it_mAP_RPN_100_50 = Evaluate_metric(1, overlap_threshold=0.50)
                train_it_mAP_RPN_05_25 = Evaluate_metric(1, overlap_threshold=0.25)
                train_it_mAP_RPN_05_50 = Evaluate_metric(1, overlap_threshold=0.50)
            lidar_points, _, _, bounding_spheres_list, vote_labels, vote_masks, _, _, _ = data
            lidar_points, vote_labels, vote_masks = lidar_points.float(), vote_labels.float(), vote_masks.float()
            lidar_points = lidar_points.transpose(2, 1)
            lidar_points, vote_labels, vote_masks = lidar_points.cuda(), vote_labels.cuda(), vote_masks.cuda()
            bounding_spheres_list = [b.float().cuda() for b in bounding_spheres_list]
            
            optimizer.zero_grad()
            model = model.train()
            
            _, fg_bg_prob, vote_delta, agg_xyz, rpn_class_logits, rpn_probs, rpn_bsphere = model(lidar_points[:,:3,:], lidar_points[:,3:,:])
            fg_bg_cls_loss = compute_fg_bg_loss(fg_bg_prob, vote_masks)
            vote_loss = compute_vote_loss(vote_delta, vote_labels, vote_masks)
            rpn_class_loss, rpn_bsphere_loss = compute_rpn_class_bphere_loss(agg_xyz, rpn_class_logits, rpn_bsphere, bounding_spheres_list)
            loss = fg_bg_cls_loss + vote_loss + rpn_class_loss + rpn_bsphere_loss
            history['loss'].append(loss.item())
            train_loss_sum += loss.item()
            train_fg_bg_cls_loss_sum += fg_bg_cls_loss.item()
            train_vote_loss_sum += vote_loss.item()
            train_rpn_class_loss_sum += rpn_class_loss.item()
            train_rpn_bsphere_loss_sum += rpn_bsphere_loss.item()
            loss.backward()
            optimizer.step()
            # training metrics
            proposals, proposals_scores = proposal_layer(rpn_probs, rpn_bsphere, proposal_count=1000, nms_threshold=0.7)
            for k in range(len(proposals)):
                #print(proposals[k].size()[0])
                pred_bspheres_100 = proposals[k][0:100,:].cpu().detach().numpy()
                pred_bspheres_label_100 = np.zeros(pred_bspheres_100.shape[0])
                pred_scores_100 = proposals_scores[k][0:100].cpu().detach().numpy()
                gt_bspheres = bounding_spheres_list[k].cpu().numpy()
                gt_bspheres_label = np.zeros(gt_bspheres.shape[0])
                train_mAP_RPN_100_25.evaluate(pred_bspheres_100, pred_bspheres_label_100, pred_scores_100, gt_bspheres, gt_bspheres_label)
                train_it_mAP_RPN_100_25.evaluate(pred_bspheres_100, pred_bspheres_label_100, pred_scores_100, gt_bspheres, gt_bspheres_label)
                train_mAP_RPN_100_50.evaluate(pred_bspheres_100, pred_bspheres_label_100, pred_scores_100, gt_bspheres, gt_bspheres_label)
                train_it_mAP_RPN_100_50.evaluate(pred_bspheres_100, pred_bspheres_label_100, pred_scores_100, gt_bspheres, gt_bspheres_label)
                idx = np.where(proposals_scores[k].cpu().detach().numpy()>=0.5)[0]
                if idx.shape[0]>0:
                    pred_bspheres_05 = proposals[k].cpu().detach().numpy()[idx]
                    pred_bspheres_label_05 = np.zeros(pred_bspheres_05.shape[0])
                    pred_scores_05 = proposals_scores[k].cpu().detach().numpy()[idx]
                else:
                    pred_bspheres_05 = proposals[k].cpu().detach().numpy()[0:2]
                    pred_bspheres_label_05 = np.zeros(pred_bspheres_05.shape[0])
                    pred_scores_05 = proposals_scores[k].cpu().detach().numpy()[0:2]
                train_mAP_RPN_05_25.evaluate(pred_bspheres_05, pred_bspheres_label_05, pred_scores_05, gt_bspheres, gt_bspheres_label)
                train_it_mAP_RPN_05_25.evaluate(pred_bspheres_05, pred_bspheres_label_05, pred_scores_05, gt_bspheres, gt_bspheres_label)
                train_mAP_RPN_05_50.evaluate(pred_bspheres_05, pred_bspheres_label_05, pred_scores_05, gt_bspheres, gt_bspheres_label)
                train_it_mAP_RPN_05_50.evaluate(pred_bspheres_05, pred_bspheres_label_05, pred_scores_05, gt_bspheres, gt_bspheres_label)
            if (i+1)%5 == 0:
                print('[Epoch %d/%d] [Iteration %d/%d] TRAIN loss: %f ' % (epoch+1, args.epoch, i+1, len(dataloader), loss.item()))
                logger.info('[Epoch %d/%d] [Iteration %d/%d] TRAIN loss: %f ' % (epoch+1, args.epoch, i+1, len(dataloader), loss.item()))
                
            # validation every 300 iterations
            if (i+1)%200 == 0:
                train_it_mAP_RPN_100_25.finalize()
                history['train_it_mAP_RPN_100_25'].append(train_it_mAP_RPN_100_25.mAP())
                train_it_mAP_RPN_100_50.finalize()
                history['train_it_mAP_RPN_100_50'].append(train_it_mAP_RPN_100_50.mAP())
                train_it_mAP_RPN_05_25.finalize()
                history['train_it_mAP_RPN_05_25'].append(train_it_mAP_RPN_05_25.mAP())
                train_it_mAP_RPN_05_50.finalize()
                history['train_it_mAP_RPN_05_50'].append(train_it_mAP_RPN_05_50.mAP())
                train_loss_avg = np.sum(history['loss'][-200:])/200.0
                history['train_it_loss'].append(train_loss_avg)
                print('[Epoch %d/%d] [Iteration %d/%d] TRAIN AP_RPN_100_25/AP_RPN_100_50/AP_RPN_05_25/AP_RPN_05_50/loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), train_it_mAP_RPN_100_25.mAP(), train_it_mAP_RPN_100_50.mAP(), train_it_mAP_RPN_05_25.mAP(), train_it_mAP_RPN_05_50.mAP(), train_loss_avg))
                logger.info('[Epoch %d/%d] [Iteration %d/%d] TRAIN AP_RPN_100_25/AP_RPN_100_50/AP_RPN_05_25/AP_RPN_05_50/loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), train_it_mAP_RPN_100_25.mAP(), train_it_mAP_RPN_100_50.mAP(), train_it_mAP_RPN_05_25.mAP(), train_it_mAP_RPN_05_50.mAP(), train_loss_avg))
                
                # Test loss and metrics
                test_losses = []
                test_mAP_RPN_100_25 = Evaluate_metric(1, overlap_threshold=0.25)
                test_mAP_RPN_100_50 = Evaluate_metric(1, overlap_threshold=0.50)
                test_mAP_RPN_05_25 = Evaluate_metric(1, overlap_threshold=0.25)
                test_mAP_RPN_05_50 = Evaluate_metric(1, overlap_threshold=0.50)
        
                for j, data in enumerate(testdataloader):
                    with torch.no_grad():
                        lidar_points, _, _, bounding_spheres_list, vote_labels, vote_masks, _, _, _ = data          
                        lidar_points, vote_labels, vote_masks = lidar_points.float(), vote_labels.float(), vote_masks.float()
                        lidar_points = lidar_points.transpose(2, 1)
                        lidar_points, vote_labels, vote_masks = lidar_points.cuda(), vote_labels.cuda(), vote_masks.cuda()
                        bounding_spheres_list = [b.float().cuda() for b in bounding_spheres_list]            
                        model = model.eval()
                        _, fg_bg_prob, vote_delta, agg_xyz, rpn_class_logits, rpn_probs, rpn_bsphere = model(lidar_points[:,:3,:], lidar_points[:,3:,:])
                        fg_bg_cls_loss = compute_fg_bg_loss(fg_bg_prob, vote_masks)
                        vote_loss = compute_vote_loss(vote_delta, vote_labels, vote_masks)
                        rpn_class_loss, rpn_bsphere_loss = compute_rpn_class_bphere_loss(agg_xyz, rpn_class_logits, rpn_bsphere, bounding_spheres_list)                
                        loss = fg_bg_cls_loss + vote_loss + rpn_class_loss + rpn_bsphere_loss
                        test_losses.append(loss.item())
                        # testing mIoU of proposals
                        proposals, proposals_scores = proposal_layer(rpn_probs, rpn_bsphere, proposal_count=1000, nms_threshold=0.7)
                    for k in range(len(proposals)):
                        #print(proposals[k].size()[0])
                        pred_bspheres_100 = proposals[k][0:100,:].cpu().detach().numpy()
                        pred_bspheres_label_100 = np.zeros(pred_bspheres_100.shape[0])
                        pred_scores_100 = proposals_scores[k][0:100].cpu().detach().numpy()
                        gt_bspheres = bounding_spheres_list[k].cpu().numpy()
                        gt_bspheres_label = np.zeros(gt_bspheres.shape[0])
                        test_mAP_RPN_100_25.evaluate(pred_bspheres_100, pred_bspheres_label_100, pred_scores_100, gt_bspheres, gt_bspheres_label)
                        test_mAP_RPN_100_50.evaluate(pred_bspheres_100, pred_bspheres_label_100, pred_scores_100, gt_bspheres, gt_bspheres_label)
                        idx = np.where(proposals_scores[k].cpu().detach().numpy()>=0.5)[0]
                        if idx.shape[0]>0:
                            pred_bspheres_05 = proposals[k].cpu().detach().numpy()[idx]
                            pred_bspheres_label_05 = np.zeros(pred_bspheres_05.shape[0])
                            pred_scores_05 = proposals_scores[k].cpu().detach().numpy()[idx]
                        else:
                            pred_bspheres_05 = proposals[k].cpu().detach().numpy()[0:2]
                            pred_bspheres_label_05 = np.zeros(pred_bspheres_05.shape[0])
                            pred_scores_05 = proposals_scores[k].cpu().detach().numpy()[0:2]
                        test_mAP_RPN_05_25.evaluate(pred_bspheres_05, pred_bspheres_label_05, pred_scores_05, gt_bspheres, gt_bspheres_label)
                        test_mAP_RPN_05_50.evaluate(pred_bspheres_05, pred_bspheres_label_05, pred_scores_05, gt_bspheres, gt_bspheres_label)
        
                test_mAP_RPN_100_25.finalize()
                history['test_it_mAP_RPN_100_25'].append(test_mAP_RPN_100_25.mAP())
                test_mAP_RPN_100_50.finalize()
                history['test_it_mAP_RPN_100_50'].append(test_mAP_RPN_100_50.mAP())
                test_mAP_RPN_05_25.finalize()
                history['test_it_mAP_RPN_05_25'].append(test_mAP_RPN_05_25.mAP())
                test_mAP_RPN_05_50.finalize()
                history['test_it_mAP_RPN_05_50'].append(test_mAP_RPN_05_50.mAP())         
                test_loss = np.mean(test_losses)
                history['test_it_loss'].append(test_loss)

                print('[Epoch %d/%d] [Iteration %d/%d] TEST AP_RPN_100_25/AP_RPN_100_50/AP_RPN_05_25/AP_RPN_05_50/loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), test_mAP_RPN_100_25.mAP(), test_mAP_RPN_100_50.mAP(), test_mAP_RPN_05_25.mAP(), test_mAP_RPN_05_50.mAP(), test_loss))
                logger.info('[Epoch %d/%d] [Iteration %d/%d] TEST AP_RPN_100_25/AP_RPN_100_50/AP_RPN_05_25/AP_RPN_05_50/loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), test_mAP_RPN_100_25.mAP(), test_mAP_RPN_100_50.mAP(), test_mAP_RPN_05_25.mAP(), test_mAP_RPN_05_50.mAP(), test_loss))

                if test_mAP_RPN_100_25.mAP() > best_mAP_RPN_100_25:
                    best_mAP_RPN_100_25 = test_mAP_RPN_100_25.mAP()
                    best_mAP_RPN_100_25_epoch = epoch+1
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_RPN_100_AP_25.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_RPN_100_25))
                    logger.info('Save best AP 25 RPN 100 model..')
                    print('Save best AP 25 RPN 100 model..')
                if test_mAP_RPN_100_50.mAP() > best_mAP_RPN_100_50:
                    best_mAP_RPN_100_50 = test_mAP_RPN_100_50.mAP()
                    best_mAP_RPN_100_50_epoch = epoch+1
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_RPN_100_AP_50.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_RPN_100_50))
                    logger.info('Save best AP 50 RPN 100 model..')
                    print('Save best AP 50 RPN 100 model..')
                if test_mAP_RPN_05_25.mAP() > best_mAP_RPN_05_25:
                    best_mAP_RPN_05_25 = test_mAP_RPN_05_25.mAP()
                    best_mAP_RPN_05_25_epoch = epoch+1
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_RPN_05_AP_25.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_RPN_05_25))
                    logger.info('Save best AP 25 RPN 0.5 model..')
                    print('Save best AP 25 RPN 0.5 model..')
                if test_mAP_RPN_05_50.mAP() > best_mAP_RPN_05_50:
                    best_mAP_RPN_05_50 = test_mAP_RPN_05_50.mAP()
                    best_mAP_RPN_05_50_epoch = epoch+1
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_RPN_05_AP_50.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_RPN_05_50))
                    logger.info('Save best AP 50 RPN 0.5 model..')
                    print('Save best AP 50 RPN 0.5 model..')                
        
        # validation after one epoch ends
        train_mAP_RPN_100_25.finalize()
        history['train_mAP_RPN_100_25'].append(train_mAP_RPN_100_25.mAP())
        train_mAP_RPN_100_50.finalize()
        history['train_mAP_RPN_100_50'].append(train_mAP_RPN_100_50.mAP())
        train_mAP_RPN_05_25.finalize()
        history['train_mAP_RPN_05_25'].append(train_mAP_RPN_05_25.mAP())
        train_mAP_RPN_05_50.finalize()
        history['train_mAP_RPN_05_50'].append(train_mAP_RPN_05_50.mAP())
        train_loss_avg = train_loss_sum/(len(dataloader)*1.0)
        train_fg_bg_cls_loss_avg = train_fg_bg_cls_loss_sum/(len(dataloader)*1.0)
        train_vote_loss_avg = train_vote_loss_sum/(len(dataloader)*1.0)
        train_rpn_class_loss_avg = train_rpn_class_loss_sum/(len(dataloader)*1.0)
        train_rpn_bsphere_loss_avg = train_rpn_bsphere_loss_sum/(len(dataloader)*1.0)        
        history['train_loss'].append(train_loss_avg)
        print('[Epoch %d/%d] TRAIN AP_RPN_100_25/AP_RPN_100_50/AP_RPN_05_25/AP_RPN_05_50/loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, train_mAP_RPN_100_25.mAP(), train_mAP_RPN_100_50.mAP(), train_mAP_RPN_05_25.mAP(), train_mAP_RPN_05_50.mAP(), train_loss_avg))
        logger.info('[Epoch %d/%d] TRAIN AP_RPN_100_25/AP_RPN_100_50/AP_RPN_05_25/AP_RPN_05_50/loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, train_mAP_RPN_100_25.mAP(), train_mAP_RPN_100_50.mAP(), train_mAP_RPN_05_25.mAP(), train_mAP_RPN_05_50.mAP(), train_loss_avg))
        print('[Epoch %d/%d] TRAIN total_loss/fg_bg_cls_loss/vote_loss/rpn_class_loss/rpn_bsphere_loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, train_loss_avg, train_fg_bg_cls_loss_avg, train_vote_loss_avg, train_rpn_class_loss_avg, train_rpn_bsphere_loss_avg))
        logger.info('[Epoch %d/%d] TRAIN total_loss/fg_bg_cls_loss/vote_loss/rpn_class_loss/rpn_bsphere_loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, train_loss_avg, train_fg_bg_cls_loss_avg, train_vote_loss_avg, train_rpn_class_loss_avg, train_rpn_bsphere_loss_avg))        
        # Test loss and metrics
        test_losses = []
        test_fg_bg_cls_loss = []
        test_vote_loss = []
        test_rpn_class_loss = []
        test_rpn_bsphere_loss = []
        test_mAP_RPN_100_25 = Evaluate_metric(1, overlap_threshold=0.25)
        test_mAP_RPN_100_50 = Evaluate_metric(1, overlap_threshold=0.50)
        test_mAP_RPN_05_25 = Evaluate_metric(1, overlap_threshold=0.25)
        test_mAP_RPN_05_50 = Evaluate_metric(1, overlap_threshold=0.50)
        
        for j, data in enumerate(testdataloader):
            with torch.no_grad():
                lidar_points, _, _, bounding_spheres_list, vote_labels, vote_masks, _, _, _ = data          
                lidar_points, vote_labels, vote_masks = lidar_points.float(), vote_labels.float(), vote_masks.float()
                lidar_points = lidar_points.transpose(2, 1)
                lidar_points, vote_labels, vote_masks = lidar_points.cuda(), vote_labels.cuda(), vote_masks.cuda()
                bounding_spheres_list = [b.float().cuda() for b in bounding_spheres_list]            
                model = model.eval()
                _, fg_bg_prob, vote_delta, agg_xyz, rpn_class_logits, rpn_probs, rpn_bsphere = model(lidar_points[:,:3,:], lidar_points[:,3:,:])
                fg_bg_cls_loss = compute_fg_bg_loss(fg_bg_prob, vote_masks)
                vote_loss = compute_vote_loss(vote_delta, vote_labels, vote_masks)
                rpn_class_loss, rpn_bsphere_loss = compute_rpn_class_bphere_loss(agg_xyz, rpn_class_logits, rpn_bsphere, bounding_spheres_list)                
                loss = fg_bg_cls_loss + vote_loss + rpn_class_loss + rpn_bsphere_loss
                test_losses.append(loss.item())
                test_fg_bg_cls_loss.append(fg_bg_cls_loss.item())
                test_vote_loss.append(vote_loss.item())
                test_rpn_class_loss.append(rpn_class_loss.item())
                test_rpn_bsphere_loss.append(rpn_bsphere_loss.item())
                # testing mIoU of proposals
                proposals, proposals_scores = proposal_layer(rpn_probs, rpn_bsphere, proposal_count=1000, nms_threshold=0.7)
            for k in range(len(proposals)):
                #print(proposals[k].size()[0])
                pred_bspheres_100 = proposals[k][0:100,:].cpu().detach().numpy()
                pred_bspheres_label_100 = np.zeros(pred_bspheres_100.shape[0])
                pred_scores_100 = proposals_scores[k][0:100].cpu().detach().numpy()
                gt_bspheres = bounding_spheres_list[k].cpu().numpy()
                gt_bspheres_label = np.zeros(gt_bspheres.shape[0])
                test_mAP_RPN_100_25.evaluate(pred_bspheres_100, pred_bspheres_label_100, pred_scores_100, gt_bspheres, gt_bspheres_label)
                test_mAP_RPN_100_50.evaluate(pred_bspheres_100, pred_bspheres_label_100, pred_scores_100, gt_bspheres, gt_bspheres_label)
                idx = np.where(proposals_scores[k].cpu().detach().numpy()>=0.5)[0]
                if idx.shape[0]>0:
                    pred_bspheres_05 = proposals[k].cpu().detach().numpy()[idx]
                    pred_bspheres_label_05 = np.zeros(pred_bspheres_05.shape[0])
                    pred_scores_05 = proposals_scores[k].cpu().detach().numpy()[idx]
                else:
                    pred_bspheres_05 = proposals[k].cpu().detach().numpy()[0:2]
                    pred_bspheres_label_05 = np.zeros(pred_bspheres_05.shape[0])
                    pred_scores_05 = proposals_scores[k].cpu().detach().numpy()[0:2]
                test_mAP_RPN_05_25.evaluate(pred_bspheres_05, pred_bspheres_label_05, pred_scores_05, gt_bspheres, gt_bspheres_label)
                test_mAP_RPN_05_50.evaluate(pred_bspheres_05, pred_bspheres_label_05, pred_scores_05, gt_bspheres, gt_bspheres_label)
        
        test_mAP_RPN_100_25.finalize()
        history['test_mAP_RPN_100_25'].append(test_mAP_RPN_100_25.mAP())
        test_mAP_RPN_100_50.finalize()
        history['test_mAP_RPN_100_50'].append(test_mAP_RPN_100_50.mAP())
        test_mAP_RPN_05_25.finalize()
        history['test_mAP_RPN_05_25'].append(test_mAP_RPN_05_25.mAP())
        test_mAP_RPN_05_50.finalize()
        history['test_mAP_RPN_05_50'].append(test_mAP_RPN_05_50.mAP())         
        test_loss = np.mean(test_losses)
        history['test_loss'].append(test_loss)
        test_fg_bg_cls_loss = np.mean(test_fg_bg_cls_loss)
        test_vote_loss = np.mean(test_vote_loss)
        test_rpn_class_loss = np.mean(test_rpn_class_loss)
        test_rpn_bsphere_loss = np.mean(test_rpn_bsphere_loss)

        print('[Epoch %d/%d] TEST AP_RPN_100_25/AP_RPN_100_50/AP_RPN_05_25/AP_RPN_05_50/loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, test_mAP_RPN_100_25.mAP(), test_mAP_RPN_100_50.mAP(), test_mAP_RPN_05_25.mAP(), test_mAP_RPN_05_50.mAP(), test_loss))
        logger.info('[Epoch %d/%d] TEST AP_RPN_100_25/AP_RPN_100_50/AP_RPN_05_25/AP_RPN_05_50/loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, test_mAP_RPN_100_25.mAP(), test_mAP_RPN_100_50.mAP(), test_mAP_RPN_05_25.mAP(), test_mAP_RPN_05_50.mAP(), test_loss))
        print('[Epoch %d/%d] TEST total_loss/fg_bg_cls_loss/vote_loss/rpn_class_loss/rpn_bsphere_loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, test_loss, test_fg_bg_cls_loss, test_vote_loss, test_rpn_class_loss, test_rpn_bsphere_loss))
        logger.info('[Epoch %d/%d] TEST total_loss/fg_bg_cls_loss/vote_loss/rpn_class_loss/rpn_bsphere_loss: %f/%f/%f/%f/%f ' % (epoch+1, args.epoch, test_loss, test_fg_bg_cls_loss, test_vote_loss, test_rpn_class_loss, test_rpn_bsphere_loss))          
            
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1))
            logger.info('Save model..')
            print('Save model..')
        if test_mAP_RPN_100_25.mAP() > best_mAP_RPN_100_25:
            best_mAP_RPN_100_25 = test_mAP_RPN_100_25.mAP()
            best_mAP_RPN_100_25_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_RPN_100_AP_25.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_RPN_100_25))
            logger.info('Save best AP 25 RPN 100 model..')
            print('Save best AP 25 RPN 100 model..')
        if test_mAP_RPN_100_50.mAP() > best_mAP_RPN_100_50:
            best_mAP_RPN_100_50 = test_mAP_RPN_100_50.mAP()
            best_mAP_RPN_100_50_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_RPN_100_AP_50.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_RPN_100_50))
            logger.info('Save best AP 50 RPN 100 model..')
            print('Save best AP 50 RPN 100 model..')
        if test_mAP_RPN_05_25.mAP() > best_mAP_RPN_05_25:
            best_mAP_RPN_05_25 = test_mAP_RPN_05_25.mAP()
            best_mAP_RPN_05_25_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_RPN_05_AP_25.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_RPN_05_25))
            logger.info('Save best AP 25 RPN 0.5 model..')
            print('Save best AP 25 RPN 0.5 model..')
        if test_mAP_RPN_05_50.mAP() > best_mAP_RPN_05_50:
            best_mAP_RPN_05_50 = test_mAP_RPN_05_50.mAP()
            best_mAP_RPN_05_50_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_best_RPN_05_AP_50.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mAP_RPN_05_50))
            logger.info('Save best AP 50 RPN 0.5 model..')
            print('Save best AP 50 RPN 0.5 model..')
        
    print('Best AP of RPN100 0.25 is %f at epoch %d.' % (best_mAP_RPN_100_25, best_mAP_RPN_100_25_epoch))
    logger.info('Best AP of RPN100 0.25 is %f at epoch %d.' % (best_mAP_RPN_100_25, best_mAP_RPN_100_25_epoch))
    print('Best AP of RPN100 0.50 is %f at epoch %d.' % (best_mAP_RPN_100_50, best_mAP_RPN_100_50_epoch))
    logger.info('Best AP of RPN100 0.50 is %f at epoch %d.' % (best_mAP_RPN_100_50, best_mAP_RPN_100_50_epoch))
    print('Best AP of RPN0.5 0.25 is %f at epoch %d.' % (best_mAP_RPN_05_25, best_mAP_RPN_05_25_epoch))
    logger.info('Best AP of RPN0.5 0.25 is %f at epoch %d.' % (best_mAP_RPN_05_25, best_mAP_RPN_05_25_epoch))
    print('Best AP of RPN0.5 0.50 is %f at epoch %d.' % (best_mAP_RPN_05_50, best_mAP_RPN_05_50_epoch))
    logger.info('Best AP of RPN0.5 0.50 is %f at epoch %d.' % (best_mAP_RPN_05_50, best_mAP_RPN_05_50_epoch))
    
    plot_loss_curve(history['loss'], str(log_dir))
    plot_loss_curves(history['train_it_loss'], history['test_it_loss'], str(log_dir))
    plot_mAP_curve(history['train_it_mAP_RPN_100_25'], history['test_it_mAP_RPN_100_25'], str(log_dir), '100_AP_25')
    plot_mAP_curve(history['train_it_mAP_RPN_100_50'], history['test_it_mAP_RPN_100_50'], str(log_dir), '100_AP_50')
    plot_mAP_curve(history['train_it_mAP_RPN_05_25'], history['test_it_mAP_RPN_05_25'], str(log_dir), '05_AP_25')
    plot_mAP_curve(history['train_it_mAP_RPN_05_50'], history['test_it_mAP_RPN_05_50'], str(log_dir), '05_AP_50')
    print('FINISH.')
    logger.info('FINISH')



if __name__ == '__main__':
    args = parse_args()
    main(args)
