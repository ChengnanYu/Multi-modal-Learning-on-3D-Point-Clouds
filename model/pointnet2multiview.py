import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from model.pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
from model.enet import create_enet_for_3d
from utils.projection import Projection


class PointNet2Multiview(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2Multiview, self).__init__()
        self.enet_fixed, self.enet_trainable, self.enet_classifier = create_enet_for_3d(41, './scannetv2_enet.pth', 21)
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 128 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128+128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, image, projection_indices_3d, projection_indices_2d):
        batch_size = xyz.shape[0]
        num_points = xyz.shape[2]
        image_features = []
        num_images = image[0].shape[0]
        for i in range(batch_size):
            imageft = self.enet_fixed(image[i].cuda())
            imageft = self.enet_trainable(imageft)
            imageft = [Projection.apply(ft, ind3d, ind2d, num_points) for ft, ind3d, ind2d in zip(imageft, projection_indices_3d[i], projection_indices_2d[i])]
            imageft = torch.stack(imageft, dim=2) # (input_channels, num_points_sample, num_images)
            sz = imageft.shape
            imageft = imageft.view(sz[0], -1, num_images)
            imageft = F.max_pool1d(imageft, kernel_size=num_images) # (input_channels, num_points_sample, 1)
            imageft = imageft.view(sz[0], sz[1], 1)
            image_features.append(imageft)
        image_features = torch.cat(image_features,dim=2)
        image_features = image_features.permute(2, 0, 1) # shape: (batch_size, input_channels, num_points_sample)
        
        l1_xyz, l1_points = self.sa1(xyz, image_features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, image_features, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class PointNet2Multiview2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2Multiview2, self).__init__()
        self.enet_fixed, self.enet_trainable, self.enet_classifier = create_enet_for_3d(41, './scannetv2_enet.pth', 21)
        self.sa1_geo = PointNetSetAbstraction(1024, 0.1, 32, 0 + 3, [32, 32, 64], False)
        self.sa2_geo = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa1_feat = PointNetSetAbstraction(1024, 0.1, 32, 128 + 3, [32, 32, 64], False)
        self.sa2_feat = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 256 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(512, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, image, projection_indices_3d, projection_indices_2d):
        batch_size = xyz.shape[0]
        num_points = xyz.shape[2]
        image_features = []
        num_images = image[0].shape[0]
        for i in range(batch_size):
            imageft = self.enet_fixed(image[i].cuda())
            imageft = self.enet_trainable(imageft)
            imageft = [Projection.apply(ft, ind3d, ind2d, num_points) for ft, ind3d, ind2d in zip(imageft, projection_indices_3d[i], projection_indices_2d[i])]
            imageft = torch.stack(imageft, dim=2) # (input_channels, num_points_sample, num_images)
            sz = imageft.shape
            imageft = imageft.view(sz[0], -1, num_images)
            #imageft = F.max_pool1d(imageft, kernel_size=num_images) # (input_channels, num_points_sample, 1)
            for j in range(imageft.shape[2]):
                if j==0:
                    imageft_final = imageft[:,:,j]
                else:
                    mask = ((imageft_final==0).sum(0)==128).nonzero().squeeze(1)
                    imageft_final[:,mask] = imageft[:,mask,j]
            imageft = imageft_final.view(sz[0], sz[1], 1)
            image_features.append(imageft)
        image_features = torch.cat(image_features,dim=2)
        image_features = image_features.permute(2, 0, 1) # shape: (batch_size, input_channels, num_points_sample)
        
        l1_xyz, l1_points = self.sa1_geo(xyz, None)
        l2_xyz, l2_points = self.sa2_geo(l1_xyz, l1_points)
        l1_xyz_feat, l1_points_feat = self.sa1_feat(xyz, image_features)
        l2_xyz_feat, l2_points_feat = self.sa2_feat(l1_xyz_feat, l1_points_feat)
        l2_points = torch.cat((l2_points,l2_points_feat),dim=1)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x        
        
class PointNet2Multiview2_backup(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2Multiview2_backup, self).__init__()
        self.enet_fixed, self.enet_trainable, self.enet_classifier = create_enet_for_3d(41, './scannetv2_enet.pth', 21)
        self.sa1_geo = PointNetSetAbstraction(1024, 0.1, 32, 0 + 3, [32, 32, 64], False)
        self.sa2_geo = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa1_feat = PointNetSetAbstraction(1024, 0.1, 32, 128 + 3, [32, 32, 64], False)
        self.sa2_feat = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 256 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(512, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, image, projection_indices_3d, projection_indices_2d):
        batch_size = xyz.shape[0]
        num_points = xyz.shape[2]
        image_features = []
        num_images = image[0].shape[0]
        for i in range(batch_size):
            imageft = self.enet_fixed(image[i].cuda())
            imageft = self.enet_trainable(imageft)
            imageft = [Projection.apply(ft, ind3d, ind2d, num_points) for ft, ind3d, ind2d in zip(imageft, projection_indices_3d[i], projection_indices_2d[i])]
            imageft = torch.stack(imageft, dim=2) # (input_channels, num_points_sample, num_images)
            sz = imageft.shape
            imageft = imageft.view(sz[0], -1, num_images)
            imageft = F.max_pool1d(imageft, kernel_size=num_images) # (input_channels, num_points_sample, 1)
            imageft = imageft.view(sz[0], sz[1], 1)
            image_features.append(imageft)
        image_features = torch.cat(image_features,dim=2)
        image_features = image_features.permute(2, 0, 1) # shape: (batch_size, input_channels, num_points_sample)
        
        l1_xyz, l1_points = self.sa1_geo(xyz, None)
        l2_xyz, l2_points = self.sa2_geo(l1_xyz, l1_points)
        l1_xyz_feat, l1_points_feat = self.sa1_feat(xyz, image_features)
        l2_xyz_feat, l2_points_feat = self.sa2_feat(l1_xyz_feat, l1_points_feat)
        l2_points = torch.cat((l2_points,l2_points_feat),dim=1)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class PointNet2Multiview2Msg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2Multiview2Msg, self).__init__()
        self.enet_fixed, self.enet_trainable, self.enet_classifier = create_enet_for_3d(41, './scannetv2_enet.pth', 21)
        self.sa1_geo = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 0, [[16, 16, 32], [32, 32, 64]])
        self.sa2_geo = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 96, [[64, 64, 128], [64, 96, 128]])
        self.sa1_feat = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 128, [[16, 16, 32], [32, 32, 64]])
        self.sa2_feat = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 96, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 512, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 512, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(1536, [512, 512])
        self.fp3 = PointNetFeaturePropagation(1024, [512, 512])
        self.fp2 = PointNetFeaturePropagation(608, [256, 256])
        self.fp1 = PointNetFeaturePropagation(256, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, image, projection_indices_3d, projection_indices_2d):
        batch_size = xyz.shape[0]
        num_points = xyz.shape[2]
        image_features = []
        num_images = image[0].shape[0]
        for i in range(batch_size):
            imageft = self.enet_fixed(image[i].cuda())
            imageft = self.enet_trainable(imageft)
            imageft = [Projection.apply(ft, ind3d, ind2d, num_points) for ft, ind3d, ind2d in zip(imageft, projection_indices_3d[i], projection_indices_2d[i])]
            imageft = torch.stack(imageft, dim=2) # (input_channels, num_points_sample, num_images)
            sz = imageft.shape
            imageft = imageft.view(sz[0], -1, num_images)
            imageft = F.max_pool1d(imageft, kernel_size=num_images) # (input_channels, num_points_sample, 1)
            imageft = imageft.view(sz[0], sz[1], 1)
            image_features.append(imageft)
        image_features = torch.cat(image_features,dim=2)
        image_features = image_features.permute(2, 0, 1) # shape: (batch_size, input_channels, num_points_sample)
        
        l1_xyz, l1_points = self.sa1_geo(xyz, None)
        l2_xyz, l2_points = self.sa2_geo(l1_xyz, l1_points)
        l1_xyz_feat, l1_points_feat = self.sa1_feat(xyz, image_features)
        l2_xyz_feat, l2_points_feat = self.sa2_feat(l1_xyz_feat, l1_points_feat)
        l2_points = torch.cat((l2_points,l2_points_feat),dim=1)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x        
        
class PointNet2Multiview2Plus(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2Multiview2Plus, self).__init__()
        self.enet_fixed, self.enet_trainable, self.enet_classifier = create_enet_for_3d(41, './scannetv2_enet.pth', 21)
        self.sa1_geo = PointNetSetAbstraction(1024, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2_geo = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa1_feat = PointNetSetAbstraction(1024, 0.1, 32, 128 + 3, [32, 32, 64], False)
        self.sa2_feat = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 256 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(512, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(3+128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, points, image, projection_indices_3d, projection_indices_2d):
        batch_size = xyz.shape[0]
        num_points = xyz.shape[2]
        image_features = []
        num_images = image[0].shape[0]
        for i in range(batch_size):
            imageft = self.enet_fixed(image[i].cuda())
            imageft = self.enet_trainable(imageft)
            imageft = [Projection.apply(ft, ind3d, ind2d, num_points) for ft, ind3d, ind2d in zip(imageft, projection_indices_3d[i], projection_indices_2d[i])]
            imageft = torch.stack(imageft, dim=2) # (input_channels, num_points_sample, num_images)
            sz = imageft.shape
            imageft = imageft.view(sz[0], -1, num_images)
            imageft = F.max_pool1d(imageft, kernel_size=num_images) # (input_channels, num_points_sample, 1)
            imageft = imageft.view(sz[0], sz[1], 1)
            image_features.append(imageft)
        image_features = torch.cat(image_features,dim=2)
        image_features = image_features.permute(2, 0, 1) # shape: (batch_size, input_channels, num_points_sample)
        
        l1_xyz, l1_points = self.sa1_geo(xyz, points)
        l2_xyz, l2_points = self.sa2_geo(l1_xyz, l1_points)
        l1_xyz_feat, l1_points_feat = self.sa1_feat(xyz, image_features)
        l2_xyz_feat, l2_points_feat = self.sa2_feat(l1_xyz_feat, l1_points_feat)
        l2_points = torch.cat((l2_points,l2_points_feat),dim=1)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, points, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x
        
class PointNet2Multiview3(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2Multiview3, self).__init__()
        self.enet_fixed, self.enet_trainable, self.enet_classifier = create_enet_for_3d(41, './scannetv2_enet.pth', 21)
        self.sa1_geo = PointNetSetAbstraction(1024, 0.1, 32, 0 + 3, [32, 32, 64], False)
        self.sa2_geo = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa1_feat = PointNetSetAbstraction(1024, 0.1, 32, 128 + 3, [32, 32, 64], False)
        self.sa2_feat = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, image, projection_indices_3d, projection_indices_2d):
        batch_size = xyz.shape[0]
        num_points = xyz.shape[2]
        image_features = []
        num_images = image[0].shape[0]
        for i in range(batch_size):
            imageft = self.enet_fixed(image[i].cuda())
            imageft = self.enet_trainable(imageft)
            imageft = [Projection.apply(ft, ind3d, ind2d, num_points) for ft, ind3d, ind2d in zip(imageft, projection_indices_3d[i], projection_indices_2d[i])]
            imageft = torch.stack(imageft, dim=2) # (input_channels, num_points_sample, num_images)
            sz = imageft.shape
            imageft = imageft.view(sz[0], -1, num_images)
            imageft = F.max_pool1d(imageft, kernel_size=num_images) # (input_channels, num_points_sample, 1)
            imageft = imageft.view(sz[0], sz[1], 1)
            image_features.append(imageft)
        image_features = torch.cat(image_features,dim=2)
        image_features = image_features.permute(2, 0, 1) # shape: (batch_size, input_channels, num_points_sample)
        
        l1_xyz, l1_points = self.sa1_geo(xyz, None)
        l2_xyz, l2_points = self.sa2_geo(l1_xyz, l1_points)
        l1_xyz_feat, l1_points_feat = self.sa1_feat(xyz, image_features)
        l2_xyz_feat, l2_points_feat = self.sa2_feat(l1_xyz_feat, l1_points_feat)
        l2_xyz = torch.cat((l2_xyz,l2_xyz_feat),dim=2)
        l2_points = torch.cat((l2_points,l2_points_feat),dim=2)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x        
        
class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2SemSeg, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128+3, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz,points):
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, points, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    label = torch.randn(8,16)
    model = PointNet2PartSeg_msg_one_hot(num_classes=50)
    output= model(input,input,label)
    print(output.size())

