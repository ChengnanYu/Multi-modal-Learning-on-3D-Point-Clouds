import argparse
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from data_utils.ScanNetDataLoader import ScannetDatasetRGBImg, ScannetDatasetWholeSceneRGBImg
import datetime
import logging
from pathlib import Path
from utils.utils import plot_loss_curve, plot_acc_curve
#from tqdm import tqdm
from utils.pc_util import point_cloud_label_to_surface_voxel_label_fast
from model.pointnet2multiview import PointNet2Multiview2
from utils.projection import ProjectionHelper

#seg_classes = class2label
#seg_label_to_cat = {}
#for i,cat in enumerate(seg_classes.keys()):
#    seg_label_to_cat[i] = cat

def parse_args():
    parser = argparse.ArgumentParser('PointNet2Multiview')
    parser.add_argument('--batchsize', type=int, default=24, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
    parser.add_argument('--model_name', type=str, default='pointnet2Multiview', help='Name of model')

    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.multi_gpu is None else '0,1,2,3'
    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) +'/%sScanNetSemSeg-'%args.model_name+ str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
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
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s_semseg.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    print('Load data...')

    dataset = ScannetDatasetRGBImg(root = './data', split='train', npoints=8192, num_images=3)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, collate_fn=dataset.collate_fn,
                                             shuffle=True, num_workers=int(args.workers))
    test_dataset = ScannetDatasetRGBImg(root = './data', split='test', npoints=8192, num_images=3)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, collate_fn=test_dataset.collate_fn,
                                                 shuffle=True, num_workers=int(args.workers))

    num_classes = 21
    model = PointNet2Multiview2(num_classes)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    #loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    pretrain = args.pretrain
    init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    LEARNING_RATE_CLIP = 1e-5

    '''GPU selection and multi-GPU'''
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model.cuda()
        
    intrinsic = [[37.01983, 0, 20, 0],[0, 38.52470, 15.5, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
    projection = ProjectionHelper(intrinsic, 0.1, 4.0, [41,32], 0.05)

    history = defaultdict(lambda: list())
    best_acc = 0
    best_acc_epoch = 0
    best_mIoU = 0
    best_mIoU_epoch = 0

    for epoch in range(init_epoch,args.epoch):
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'],LEARNING_RATE_CLIP)
        print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        invalid_count = 0
        for i, data in enumerate(dataloader):
            points, target, sample_weights, image, depth, pose = data
            batch_size = points.shape[0]
            num_points = points.shape[1]
            num_images = image[0].shape[0]
            points, target = points.float(), target.long()
            points = points.transpose(2, 1)
            points, target, sample_weights = points.cuda(), target.cuda(), sample_weights.cuda()
            depth = [d.cuda() for d in depth]
            pose = [p.cuda() for p in pose]
            # Compute projection mapping
            points_projection = torch.repeat_interleave(points.transpose(2, 1)[:,:,0:3], num_images, dim=0) # For each scene chunk, we have num_images images. We repeat each point cloud num_images times to compute the projection
            proj_mapping = [[projection.compute_projection(p, d, c, num_points) for p, d, c in zip(points_projection[k*num_images:(k+1)*num_images], depth[k], pose[k])] for k in range(batch_size)]
            jump_flag = False
            for k in range(batch_size):
                if None in proj_mapping[k]: #invalid sample
                    print('invalid sample')
                    invalid_count = invalid_count+1
                    jump_flag = True
                    break
            if jump_flag:
                continue
            proj_ind_3d = []
            proj_ind_2d = []
            for k in range(batch_size):
                proj_mapping0, proj_mapping1 = zip(*proj_mapping[k])
                proj_ind_3d.append(torch.stack(proj_mapping0))
                proj_ind_2d.append(torch.stack(proj_mapping1))
            
            optimizer.zero_grad()
            model = model.train()
            model.enet_fixed = model.enet_fixed.eval()
            model.enet_trainable = model.enet_trainable.eval()
            for param in model.enet_trainable.parameters():
                param.requires_grad = False
            
            pred = model(points[:,:3,:], image, proj_ind_3d, proj_ind_2d)
            #pred = model(points[:,:3,:], points[:,3:6,:], image, proj_ind_3d, proj_ind_2d)
            
            pred = pred.contiguous().view(-1, num_classes)
            target = target.view(pred.size(0))
            weights = sample_weights.view(pred.size(0))
            loss = loss_function(pred, target)
            loss = loss * weights
            loss = torch.mean(loss)
            history['loss'].append(loss.item())
            train_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            # Train acc
            pred_val = torch.argmax(pred, 1)
            correct = torch.sum(((pred_val == target)&(target>0)&(weights>0)).float())
            seen = torch.sum(((target>0)&(weights>0)).float())+1e-08
            train_acc = correct/seen if seen!=0 else correct
            train_acc_sum += train_acc.item()
            if (i+1)%5 == 0:
                print('[Epoch %d/%d] [Iteration %d/%d] TRAIN acc/loss: %f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), train_acc.item(), loss.item()))
                logger.info('[Epoch %d/%d] [Iteration %d/%d] TRAIN acc/loss: %f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), train_acc.item(), loss.item()))
        train_loss_avg = train_loss_sum/(len(dataloader)-invalid_count)
        train_acc_avg = train_acc_sum/(len(dataloader)-invalid_count)
        history['train_acc'].append(train_acc_avg)
        print('[Epoch %d/%d] TRAIN acc/loss: %f/%f ' % (epoch+1, args.epoch, train_acc_avg, train_loss_avg))
        logger.info('[Epoch %d/%d] TRAIN acc/loss: %f/%f ' % (epoch+1, args.epoch, train_acc_avg, train_loss_avg))
        
        #Test acc
        test_losses = []
        total_correct = 0
        total_seen = 0
        total_correct_class = [0 for _ in range(num_classes)]
        total_seen_class = [0 for _ in range(num_classes)]
        total_intersection_class = [0 for _ in range(num_classes)]
        total_union_class = [0 for _ in range(num_classes)]
        
        total_correct_vox = 0
        total_seen_vox = 0
        total_seen_class_vox = [0 for _ in range(num_classes)]
        total_correct_class_vox = [0 for _ in range(num_classes)]
        total_intersection_class_vox = [0 for _ in range(num_classes)]
        total_union_class_vox = [0 for _ in range(num_classes)]
        
        
        labelweights = np.zeros(num_classes)
        labelweights_vox = np.zeros(num_classes)
        
        for j, data in enumerate(testdataloader):
            with torch.no_grad():
                points, target, sample_weights, image, depth, pose = data
                batch_size = points.shape[0]
                num_points = points.shape[1]
                num_images = image[0].shape[0]
                points, target, sample_weights = points.float(), target.long(), sample_weights.float()
                points = points.transpose(2, 1)
                points, target, sample_weights = points.cuda(), target.cuda(), sample_weights.cuda()
                depth = [d.cuda() for d in depth]
                pose = [p.cuda() for p in pose]
                # Compute projection mapping
                points_projection = torch.repeat_interleave(points.transpose(2, 1)[:,:,0:3], num_images, dim=0) # For each scene chunk, we have num_images images. We repeat each point cloud num_images times to compute the projection
                proj_mapping = [[projection.compute_projection(p, d, c, num_points) for p, d, c in zip(points_projection[k*num_images:(k+1)*num_images], depth[k], pose[k])] for k in range(batch_size)]
                jump_flag = False
                for k in range(batch_size):
                    if None in proj_mapping[k]: #invalid sample
                        print('invalid sample')
                        jump_flag = True
                        break
                if jump_flag:
                    continue
                proj_ind_3d = []
                proj_ind_2d = []
                for k in range(batch_size):
                    proj_mapping0, proj_mapping1 = zip(*proj_mapping[k])
                    proj_ind_3d.append(torch.stack(proj_mapping0))
                    proj_ind_2d.append(torch.stack(proj_mapping1))
                model = model.eval()
                pred = model(points[:,:3,:], image, proj_ind_3d, proj_ind_2d)
                #pred = model(points[:,:3,:], points[:,3:6,:], image, proj_ind_3d, proj_ind_2d)
                pred_2d = pred.contiguous().view(-1, num_classes)
                target_1d = target.view(pred_2d.size(0))
                weights_1d = sample_weights.view(pred_2d.size(0))
                loss = loss_function(pred_2d, target_1d)
                loss = loss * weights_1d
                loss = torch.mean(loss)
                test_losses.append(loss.item())
            #first convert torch tensor to numpy array
            pred_np = pred.cpu().numpy() #[B,N,C]
            target_np = target.cpu().numpy() #[B,N]
            weights_np = sample_weights.cpu().numpy() #[B,N]
            points_np = points.transpose(2, 1).cpu().numpy() #[B,N,3]
            # point wise acc
            pred_val = np.argmax(pred_np, 2) #[B,N]
            correct = np.sum((pred_val == target_np) & (target_np>0) & (weights_np>0))
            total_correct += correct
            total_seen += np.sum((target_np>0) & (weights_np>0))
            
            tmp,_ = np.histogram(target_np,range(num_classes+1))
            labelweights += tmp
            
            # point wise acc and IoU per class
            for l in range(num_classes):
                total_seen_class[l] += np.sum((target_np==l) & (weights_np>0))
                total_correct_class[l] += np.sum((pred_val==l) & (target_np==l) & (weights_np>0))
                total_intersection_class[l] += np.sum((pred_val==l) & (target_np==l) & (weights_np>0))
                total_union_class[l] += np.sum(((pred_val==l) | (target_np==l)) & (weights_np>0))
            
            # voxel wise acc
            for b in range(target_np.shape[0]):
                _, uvlabel, _ = point_cloud_label_to_surface_voxel_label_fast(points_np[b,weights_np[b,:]>0,:], np.concatenate((np.expand_dims(target_np[b,weights_np[b,:]>0],1),np.expand_dims(pred_val[b,weights_np[b,:]>0],1)),axis=1), res=0.02)
                total_correct_vox += np.sum((uvlabel[:,0]==uvlabel[:,1])&(uvlabel[:,0]>0))
                total_seen_vox += np.sum(uvlabel[:,0]>0)
                tmp,_ = np.histogram(uvlabel[:,0],range(num_classes+1))
                labelweights_vox += tmp
                # voxel wise acc and IoU per class
                for l in range(num_classes):
                    total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
                    total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))
                    total_intersection_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))
                    total_union_class_vox[l] += np.sum((uvlabel[:,0]==l) | (uvlabel[:,1]==l))
                
        test_loss = np.mean(test_losses)
        test_point_acc = total_correct/float(total_seen)
        history['test_point_acc'].append(test_point_acc)
        test_voxel_acc = total_correct_vox/float(total_seen_vox)
        history['test_voxel_acc'].append(test_voxel_acc)
        test_avg_class_point_acc = np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))
        history['test_avg_class_point_acc'].append(test_avg_class_point_acc)
        test_avg_class_voxel_acc = np.mean(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6))
        history['test_avg_class_voxel_acc'].append(test_avg_class_voxel_acc)
        test_avg_class_point_IoU = np.mean(np.array(total_intersection_class[1:])/(np.array(total_union_class[1:],dtype=np.float)+1e-6))
        history['test_avg_class_point_IoU'].append(test_avg_class_point_IoU)
        test_avg_class_voxel_IoU = np.mean(np.array(total_intersection_class_vox[1:])/(np.array(total_union_class_vox[1:],dtype=np.float)+1e-6))
        history['test_avg_class_voxel_IoU'].append(test_avg_class_voxel_IoU)
        labelweights = labelweights[1:].astype(np.float32)/np.sum(labelweights[1:].astype(np.float32))
        labelweights_vox = labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
        #caliweights = np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
        #test_cali_voxel_acc = np.average(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6),weights=caliweights)
        #history['test_cali_voxel_acc'].append(test_cali_voxel_acc)
        #test_cali_point_acc = np.average(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6),weights=caliweights)
        #history['test_cali_point_acc'].append(test_cali_point_acc)

        print('[Epoch %d/%d] TEST acc/loss: %f/%f ' % (epoch+1, args.epoch, test_voxel_acc, test_loss))
        logger.info('[Epoch %d/%d] TEST acc/loss: %f/%f ' % (epoch+1, args.epoch, test_voxel_acc, test_loss))
        print('Whole scene point wise accuracy: %f' % (test_point_acc))
        logger.info('Whole scene point wise accuracy: %f' % (test_point_acc))
        print('Whole scene voxel wise accuracy: %f' % (test_voxel_acc))
        logger.info('Whole scene voxel wise accuracy: %f' % (test_voxel_acc))
        print('Whole scene class averaged point wise accuracy: %f' % (test_avg_class_point_acc))
        logger.info('Whole scene class averaged point wise accuracy: %f' % (test_avg_class_point_acc))
        print('Whole scene class averaged voxel wise accuracy: %f' % (test_avg_class_voxel_acc))
        logger.info('Whole scene class averaged voxel wise accuracy: %f' % (test_avg_class_voxel_acc))
        #print('Whole scene calibrated point wise accuracy: %f' % (test_cali_point_acc))
        #logger.info('Whole scene calibrated point wise accuracy: %f' % (test_cali_point_acc))
        #print('Whole scene calibrated voxel wise accuracy: %f' % (test_cali_voxel_acc))
        #logger.info('Whole scene calibrated voxel wise accuracy: %f' % (test_cali_voxel_acc))
        print('Whole scene class averaged point wise IoU: %f' % (test_avg_class_point_IoU))
        logger.info('Whole scene class averaged point wise IoU: %f' % (test_avg_class_point_IoU))
        print('Whole scene class averaged voxel wise IoU: %f' % (test_avg_class_voxel_IoU))
        logger.info('Whole scene class averaged voxel wise IoU: %f' % (test_avg_class_voxel_IoU))
        
        per_class_voxel_str = 'voxel based --------\n'
        for l in range(1,num_classes):
            per_class_voxel_str += 'class %d weight: %f, acc: %f, IoU: %f;\n' % (l,labelweights_vox[l-1],total_correct_class_vox[l]/float(total_seen_class_vox[l]), total_intersection_class_vox[l]/(float(total_union_class_vox[l])+1e-6))
        logger.info(per_class_voxel_str)
        
        per_class_point_str = 'point based --------\n'
        for l in range(1,num_classes):
            per_class_point_str += 'class %d weight: %f, acc: %f, IoU: %f;\n' % (l,labelweights[l-1],total_correct_class[l]/float(total_seen_class[l]), total_intersection_class[l]/(float(total_union_class[l])+1e-6))
        logger.info(per_class_point_str)
            
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), '%s/%s_%.3d.pth' % (checkpoints_dir,args.model_name, epoch+1))
            logger.info('Save model..')
            print('Save model..')
        if test_voxel_acc > best_acc:
            best_acc = test_voxel_acc
            best_acc_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4f_bestacc.pth' % (checkpoints_dir,args.model_name, epoch+1, best_acc))
            logger.info('Save best acc model..')
            print('Save best acc model..')
        if test_avg_class_voxel_IoU > best_mIoU:
            best_mIoU = test_avg_class_voxel_IoU
            best_mIoU_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4f_bestmIoU.pth' % (checkpoints_dir,args.model_name, epoch+1, best_mIoU))
            logger.info('Save best mIoU model..')
            print('Save best mIoU model..')
    print('Best voxel wise accuracy is %f at epoch %d.' % (best_acc, best_acc_epoch))
    logger.info('Best voxel wise accuracy is %f at epoch %d.' % (best_acc, best_acc_epoch))
    print('Best class averaged voxel wise IoU is %f at epoch %d.' % (best_mIoU, best_mIoU_epoch))
    logger.info('Best class averaged voxel wise IoU is %f at epoch %d.' % (best_mIoU, best_mIoU_epoch))
    plot_loss_curve(history['loss'], str(log_dir))
    plot_acc_curve(history['train_acc'], history['test_voxel_acc'], str(log_dir))
    plot_acc_curve(history['train_acc'], history['test_avg_class_voxel_IoU'], str(log_dir))
    print('FINISH.')
    logger.info('FINISH')



if __name__ == '__main__':
    args = parse_args()
    main(args)
