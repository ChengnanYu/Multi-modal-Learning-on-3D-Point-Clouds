import argparse
import os
import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from nuscenes.nuscenes import NuScenes
from data_utils.NuScenesDataLoader import NuscenesDatasetSemseg
import datetime
import logging
from pathlib import Path
from utils.utils import plot_loss_curve, plot_acc_curve
from model.pointnet2 import PointNet2SemSeg, PointNet2SemSeg2, PointNet2SemSeg3, PointNet2SemSeg4, PointNet2SemSeg5, PointNet2SemSeg6

def parse_args():
    parser = argparse.ArgumentParser('PointNet2')
    parser.add_argument('--batchsize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
    parser.add_argument('--model_name', type=str, default='pointnet2', help='Name of model')

    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.multi_gpu is None else '0,1,2,3'
    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) +'/%sNuScenesSemSeg-'%args.model_name+ str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
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

    nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/raid/chengnan/NuScenes/', verbose=True)
    dataset = NuscenesDatasetSemseg(nusc, npoints_lidar=16384, npoints_radar=1024, split='train_detect')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, collate_fn=dataset.collate_fn,
                                             shuffle=True, num_workers=int(args.workers))
    test_dataset = NuscenesDatasetSemseg(nusc, npoints_lidar=16384, npoints_radar=1024, split='val_small')
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, collate_fn=dataset.collate_fn,
                                                 shuffle=True, num_workers=int(args.workers))

    num_classes = 11
    model = PointNet2SemSeg5(num_classes)
    labelweight = torch.tensor([0.84423709, 3.562451, 8.4504371, 5.04570442, 1.69708204, 6.41300778, 6.44816675, 4.88638126, 5.20078234, 4.82712436, 3.74396562])
    loss_function = torch.nn.CrossEntropyLoss(labelweight.cuda())
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
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
        for i, data in enumerate(dataloader):
            lidar_points, _, _, _, target, _, _, _   = data
            lidar_points, target = lidar_points.float(), target.long()
            lidar_points = lidar_points.transpose(2, 1)
            lidar_points, target = lidar_points.cuda(), target.cuda()
            optimizer.zero_grad()
            model = model.train()
            pred = model(lidar_points[:,:3,:],lidar_points[:,3:,:])
            #pred = model(lidar_points[:,:3,:],None)            
            loss = loss_function(pred, target)
            history['loss'].append(loss.item())
            train_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            # Train acc
            pred_val = torch.argmax(pred, 1)
            correct = torch.sum(((pred_val == target)&(target>0)).float())
            seen = torch.sum((target>0).float())+1e-08
            train_acc = correct/seen if seen!=0 else correct
            history['train_acc_per_iter'].append(train_acc.item())
            train_acc_sum += train_acc.item()
            if (i+1)%5 == 0:
                print('[Epoch %d/%d] [Iteration %d/%d] TRAIN acc/loss: %f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), train_acc.item(), loss.item()))
                logger.info('[Epoch %d/%d] [Iteration %d/%d] TRAIN acc/loss: %f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), train_acc.item(), loss.item()))
            if (i+1)%200 == 0:
                
                train_loss_avg = np.sum(history['loss'][-200:])/200.0
                train_acc_avg = np.sum(history['train_acc_per_iter'][-200:])/200.0
                history['train_it_acc'].append(train_acc_avg)
                print('[Epoch %d/%d] [Iteration %d/%d] TRAIN acc/loss: %f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), train_acc_avg, train_loss_avg))
                logger.info('[Epoch %d/%d] [Iteration %d/%d] TRAIN acc/loss: %f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), train_acc_avg, train_loss_avg))
                
                #Test acc
                test_losses = []
                total_correct = 0
                total_seen = 0
                total_correct_class = [0 for _ in range(num_classes)]
                total_seen_class = [0 for _ in range(num_classes)]
                total_intersection_class = [0 for _ in range(num_classes)]
                total_union_class = [0 for _ in range(num_classes)]
        
                for j, data in enumerate(testdataloader):
                    lidar_points, _, _, _, target, _, _, _   = data
                    lidar_points, target = lidar_points.float(), target.long()
                    lidar_points = lidar_points.transpose(2, 1)
                    lidar_points, target = lidar_points.cuda(), target.cuda()
                    model = model.eval()
                    with torch.no_grad():
                        pred = model(lidar_points[:,:3,:],lidar_points[:,3:,:])
                        #pred = model(lidar_points[:,:3,:],None)
                    loss = loss_function(pred, target)
                    test_losses.append(loss.item())
                    #first convert torch tensor to numpy array
                    pred_np = pred.cpu().numpy() #[B,C,N]
                    target_np = target.cpu().numpy() #[B,N]
                    # point wise acc
                    pred_val = np.argmax(pred_np, 1) #[B,N]
                    correct = np.sum((pred_val == target_np) & (target_np>0))
                    total_correct += correct
                    total_seen += np.sum(target_np>0)
            
            
                    # point wise acc and IoU per class
                    for l in range(num_classes):
                        total_seen_class[l] += np.sum((target_np==l))
                        total_correct_class[l] += np.sum((pred_val==l) & (target_np==l))
                        total_intersection_class[l] += np.sum((pred_val==l) & (target_np==l))
                        total_union_class[l] += np.sum(((pred_val==l) | (target_np==l)))
                
                test_loss = np.mean(test_losses)
                test_point_acc = total_correct/float(total_seen)
                history['test_it_point_acc'].append(test_point_acc)
                test_avg_class_point_acc = np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))
                history['test_it_avg_class_point_acc'].append(test_avg_class_point_acc)
                test_avg_class_point_IoU = np.mean(np.array(total_intersection_class[1:])/(np.array(total_union_class[1:],dtype=np.float)+1e-6))
                history['test_it_avg_class_point_IoU'].append(test_avg_class_point_IoU)

                print('[Epoch %d/%d] [Iteration %d/%d] TEST acc/loss: %f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), test_point_acc, test_loss))
                logger.info('[Epoch %d/%d] [Iteration %d/%d] TEST acc/loss: %f/%f ' % (epoch+1, args.epoch, i+1, len(dataloader), test_point_acc, test_loss))
                print('Whole scene point wise accuracy: %f' % (test_point_acc))
                logger.info('Whole scene point wise accuracy: %f' % (test_point_acc))
                print('Whole scene class averaged point wise accuracy: %f' % (test_avg_class_point_acc))
                logger.info('Whole scene class averaged point wise accuracy: %f' % (test_avg_class_point_acc))
                print('Whole scene class averaged point wise IoU: %f' % (test_avg_class_point_IoU))
                logger.info('Whole scene class averaged point wise IoU: %f' % (test_avg_class_point_IoU))
        
                per_class_point_str = 'point based --------\n'
                for l in range(0,num_classes):
                    per_class_point_str += 'class %d weight: %f, acc: %f, IoU: %f;\n' % (l,labelweight[l],total_correct_class[l]/float(total_seen_class[l]), total_intersection_class[l]/(float(total_union_class[l])+1e-6))
                logger.info(per_class_point_str)                
                if test_point_acc > best_acc:
                    best_acc = test_point_acc
                    best_acc_epoch = epoch+1
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_bestacc.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_acc))
                    logger.info('Save best acc model..')
                    print('Save best acc model..')
                if test_avg_class_point_IoU > best_mIoU:
                    best_mIoU = test_avg_class_point_IoU
                    best_mIoU_epoch = epoch+1
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_bestmIoU.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mIoU))
                    logger.info('Save best mIoU model..')
                    print('Save best mIoU model..')                
                
                
        train_loss_avg = train_loss_sum/len(dataloader)
        train_acc_avg = train_acc_sum/len(dataloader)
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
        
        for j, data in enumerate(testdataloader):
            lidar_points, _, _, _, target, _, _, _   = data
            lidar_points, target = lidar_points.float(), target.long()
            lidar_points = lidar_points.transpose(2, 1)
            lidar_points, target = lidar_points.cuda(), target.cuda()
            model = model.eval()
            with torch.no_grad():
                pred = model(lidar_points[:,:3,:],lidar_points[:,3:,:])
                #pred = model(lidar_points[:,:3,:],None)
            loss = loss_function(pred, target)
            test_losses.append(loss.item())
            #first convert torch tensor to numpy array
            pred_np = pred.cpu().numpy() #[B,C,N]
            target_np = target.cpu().numpy() #[B,N]
            # point wise acc
            pred_val = np.argmax(pred_np, 1) #[B,N]
            correct = np.sum((pred_val == target_np) & (target_np>0))
            total_correct += correct
            total_seen += np.sum(target_np>0)
            
            
            # point wise acc and IoU per class
            for l in range(num_classes):
                total_seen_class[l] += np.sum((target_np==l))
                total_correct_class[l] += np.sum((pred_val==l) & (target_np==l))
                total_intersection_class[l] += np.sum((pred_val==l) & (target_np==l))
                total_union_class[l] += np.sum(((pred_val==l) | (target_np==l)))
                
        test_loss = np.mean(test_losses)
        test_point_acc = total_correct/float(total_seen)
        history['test_point_acc'].append(test_point_acc)
        test_avg_class_point_acc = np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))
        history['test_avg_class_point_acc'].append(test_avg_class_point_acc)
        test_avg_class_point_IoU = np.mean(np.array(total_intersection_class[1:])/(np.array(total_union_class[1:],dtype=np.float)+1e-6))
        history['test_avg_class_point_IoU'].append(test_avg_class_point_IoU)

        print('[Epoch %d/%d] TEST acc/loss: %f/%f ' % (epoch+1, args.epoch, test_point_acc, test_loss))
        logger.info('[Epoch %d/%d] TEST acc/loss: %f/%f ' % (epoch+1, args.epoch, test_point_acc, test_loss))
        print('Whole scene point wise accuracy: %f' % (test_point_acc))
        logger.info('Whole scene point wise accuracy: %f' % (test_point_acc))
        print('Whole scene class averaged point wise accuracy: %f' % (test_avg_class_point_acc))
        logger.info('Whole scene class averaged point wise accuracy: %f' % (test_avg_class_point_acc))
        print('Whole scene class averaged point wise IoU: %f' % (test_avg_class_point_IoU))
        logger.info('Whole scene class averaged point wise IoU: %f' % (test_avg_class_point_IoU))
        
        per_class_point_str = 'point based --------\n'
        for l in range(0,num_classes):
            per_class_point_str += 'class %d weight: %f, acc: %f, IoU: %f;\n' % (l,labelweight[l],total_correct_class[l]/float(total_seen_class[l]), total_intersection_class[l]/(float(total_union_class[l])+1e-6))
        logger.info(per_class_point_str)
            
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1))
            logger.info('Save model..')
            print('Save model..')
        if test_point_acc > best_acc:
            best_acc = test_point_acc
            best_acc_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_bestacc.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_acc))
            logger.info('Save best acc model..')
            print('Save best acc model..')
        if test_avg_class_point_IoU > best_mIoU:
            best_mIoU = test_avg_class_point_IoU
            best_mIoU_epoch = epoch+1
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4d_%.4f_bestmIoU.pth' % (checkpoints_dir,args.model_name, epoch+1, i+1, best_mIoU))
            logger.info('Save best mIoU model..')
            print('Save best mIoU model..')
    print('Best point wise accuracy is %f at epoch %d.' % (best_acc, best_acc_epoch))
    logger.info('Best point wise accuracy is %f at epoch %d.' % (best_acc, best_acc_epoch))
    print('Best class averaged point wise IoU is %f at epoch %d.' % (best_mIoU, best_mIoU_epoch))
    logger.info('Best class averaged point wise IoU is %f at epoch %d.' % (best_mIoU, best_mIoU_epoch))
    plot_loss_curve(history['loss'], str(log_dir))
    plot_acc_curve(history['train_it_acc'], history['test_it_point_acc'], str(log_dir))
    #plot_acc_curve(history['train_it_acc'], history['test_it_avg_class_point_IoU'], str(log_dir))
    print('FINISH.')
    logger.info('FINISH')



if __name__ == '__main__':
    args = parse_args()
    main(args)
