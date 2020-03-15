import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from . import pointnet2_utils
from model.pointnet_util import PointNetSetAbstraction, PointNetFeaturePropagation, sample_and_group

class PointNet2(nn.Module):
    def __init__(self):
        super(PointNet2, self).__init__()
        self.sa1 = PointNetSetAbstraction(4096, 1.0, 32, 2 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(1024, 2.0, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(256, 4.0, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(64, 8.0, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

    def forward(self, xyz, points):
        
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        return l0_points

class PointNet2_backup(nn.Module):
    def __init__(self):
        super(PointNet2_backup, self).__init__()
        self.sa1 = PointNetSetAbstraction(4096, 0.5, 32, 2 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(1024, 2.0, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(256, 8.0, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(64, 32.0, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

    def forward(self, xyz, points):
        
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        return l0_points

class PointNet2_backup2(nn.Module):
    def __init__(self):
        super(PointNet2_backup2, self).__init__()
        self.sa1 = PointNetSetAbstraction(4096, 0.5, 32, 2 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(1024, 1.0, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(256, 2.0, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(64, 4.0, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

    def forward(self, xyz, points):
        
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        return l0_points

class RPN(nn.Module):
    """
    Builds the model of Region Proposal Network.

    Returns:
        class_logits: [batch, npoints, 11] (before softmax)
        class_probs: [batch, npoints, 11]
        bsphere: [batch, npoints, 4]
    """

    def __init__(self):
        super(RPN, self).__init__()
        self.pointnet2 = PointNet2()
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 2, 1)        
        
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 3+128, 1)
        
        self.vote_aggregation = PointNetSetAbstraction(128, 4.0, 32, 128+3, [128,128,128],False)
        self.conv5 = nn.Conv1d(128,128,1)
        self.bn5 = nn.BatchNorm1d(128)
        #self.conv6 = nn.Conv1d(128,128,1)
        #self.bn6 = nn.BatchNorm1d(128)
        self.conv_class = nn.Conv1d(128, 11, 1)
        self.conv_bsphere = nn.Conv1d(128, 4, 1)
        self.conv_whl = nn.Conv1d(128,3,1)
        self.conv_yaw = nn.Conv1d(128,1,1)
        self.conv_velocity = nn.Conv1d(128,2,1)

    def forward(self, xyz, points):
        # extract point features using pointnet++
        features = self.pointnet2(xyz, points)
        
        x = F.relu(self.bn1(self.conv1(features)))
        fg_bg_logits = self.conv2(x)
        fg_bg_prob = F.softmax(fg_bg_logits, dim=1) # (batch_size, 2, 16384)
        mask = (fg_bg_prob[:,[1],:]>0.7).float()
        
        x = F.relu(self.bn3(self.conv3(features)))
        delta = self.conv4(x) # (batch_size, (3+128), 16384)
        
        vote_xyz = (xyz + delta[:,0:3,:])*mask
        vote_features = (features + delta[:,3:,:])*mask
        
        agg_xyz, agg_feature = self.vote_aggregation(vote_xyz, vote_features)
        x = F.relu(self.bn5(self.conv5(agg_feature)))
        #x = F.relu(self.bn6(self.conv6(x)))
        
        # class Score. [batch, 2, npoints=128].
        class_logits = self.conv_class(x)
        # Reshape to [batch, npoints=128, 11]
        class_logits = class_logits.permute(0,2,1)
        # Softmax on last dimension
        class_probs = F.softmax(class_logits, dim=2)        

        # bounding sphere. [batch, 4, npoints=128]
        bsphere = self.conv_bsphere(x)        
        bsphere[:,0:3,:] = bsphere[:,0:3,:]+agg_xyz
        # Reshape to [batch, npoints=128, 4]
        bsphere = bsphere.permute(0,2,1)
        
        whl = self.conv_whl(x)
        # Reshape to [batch, npoints=128, 3]
        whl = whl.permute(0,2,1)
        whl = whl + 2*bsphere[:,:,[3]]
        
        yaw = self.conv_yaw(x)
        # Reshape to [batch, npoints=128, 1]
        yaw = yaw.permute(0,2,1)
        
        velocity = self.conv_velocity(x)
        # Reshape to [batch, npoints=128, 2]
        velocity = velocity.permute(0,2,1)

        return features, fg_bg_logits, delta[:,0:3,:].permute(0,2,1), agg_xyz.permute(0,2,1), class_logits, class_probs, bsphere, whl, yaw, velocity 

def compute_fg_bg_loss(fg_bg_logits, vote_mask):
    """
    fg_bg_prob: [batch, npoints]
    vote_mask: [batch, npoints]
    """
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.2,2.0]).cuda())
    fg_bg_loss = criterion(fg_bg_logits, vote_mask.long())
    return fg_bg_loss
        
def compute_vote_loss(vote_delta, vote_label, vote_mask):
    """
    vote_delta: [batch, npoints, 3]
    vote_label: [batch, npoints, 3]
    vote_mask: [batch, npoints]
    """
    criterion = nn.SmoothL1Loss(reduction='none')
    vote_loss = criterion(vote_delta, vote_label)
    vote_loss = torch.sum(vote_loss * vote_mask.unsqueeze(2))/(torch.sum(vote_mask)*3+1e-6)
    return vote_loss
    
        
def compute_detection_loss(agg_xyz, class_logits, bsphere, whl, yaw, velocity, gt_bspheres, gt_bboxes):
    """
    agg_xyz: [batch, nproposal, 3]
    class_logits: [batch, nproposal, 11]
    bsphere: [batch, nproposal, 4]
    whl: [batch, nproposal, 3]
    yaw: [batch, nproposal, 1]
    velocity: [batch, nproposal, 2]
    gt_bspheres: list, length is batch size, each element size [bsphere_count, 8]
    gt_bboxes: list, length is batch size, each element size [bsphere_count, 10]
    """
    objectness_mask = torch.zeros([agg_xyz.size()[0], agg_xyz.size()[1]], device=agg_xyz.device)
    class_label = torch.zeros([agg_xyz.size()[0], agg_xyz.size()[1]], device=agg_xyz.device).long()
    bspheres_label = torch.zeros([agg_xyz.size()[0], agg_xyz.size()[1], 4], device=agg_xyz.device)
    whl_label = torch.zeros([agg_xyz.size()[0], agg_xyz.size()[1], 3], device=agg_xyz.device)
    yaw_label = torch.zeros([agg_xyz.size()[0], agg_xyz.size()[1], 1], device=agg_xyz.device)
    velocity_label = torch.zeros([agg_xyz.size()[0], agg_xyz.size()[1], 2], device=agg_xyz.device)
    for i in range(len(gt_bspheres)):
        points = agg_xyz[i,:,:].unsqueeze(1).expand(-1,gt_bspheres[i].size()[0],-1) #shape [nproposal, bsphere_count, 3]
        sphere_centers = gt_bspheres[i][:,0:3].unsqueeze(0).expand(agg_xyz.size()[1],-1,-1) #shape [nproposal, bsphere_count, 3]
        dist = torch.norm(points - sphere_centers,p=2,dim=2) #shape [nproposal, bsphere_count]        
        min_dist, min_idx = torch.min(dist, dim=1) # [nproposal]
        objectness_mask[i, min_dist<1.5] = 1
        class_label[i,:] = gt_bspheres[i][min_idx,-1]
        class_label[i,objectness_mask[i,:]==0] = 0
        bspheres_label[i,:,:] = gt_bspheres[i][min_idx,0:4]
        whl_label[i,:,:] = gt_bboxes[i][min_idx,3:6]
        yaw_label[i,:,:] = gt_bspheres[i][min_idx,4:5]
        velocity_label[i,:,:] = gt_bspheres[i][min_idx,5:7]
        
    criterion1 = nn.CrossEntropyLoss(weight=torch.Tensor([0.84423709, 3.562451, 8.4504371, 5.04570442, 1.69708204, 6.41300778, 6.44816675, 4.88638126, 5.20078234, 4.82712436, 3.74396562]).cuda())
    class_loss = criterion1(class_logits.permute(0,2,1), class_label.long())
    criterion2 = nn.SmoothL1Loss(reduction='none')
    #bsphere_loss = criterion2(bsphere, bspheres_label)
    #bsphere_loss = torch.sum(bsphere_loss * objectness_mask.float().unsqueeze(2))/(torch.sum(objectness_mask.float())*4+1e-6)
    center_loss = criterion2(bsphere[:,:,0:3], bspheres_label[:,:,0:3])
    center_loss = torch.sum(center_loss * objectness_mask.float().unsqueeze(2))/(torch.sum(objectness_mask.float())*3+1e-6)
    r_loss = criterion2(bsphere[:,:,3:4], bspheres_label[:,:,3:4])
    r_loss = torch.sum(r_loss * objectness_mask.float().unsqueeze(2))/(torch.sum(objectness_mask.float())+1e-6)
    whl_loss = criterion2(whl, whl_label)
    whl_loss = torch.sum(whl_loss * objectness_mask.float().unsqueeze(2))/(torch.sum(objectness_mask.float())*3+1e-6)
    yaw_loss = criterion2(yaw, yaw_label)
    yaw_loss = torch.sum(yaw_loss * objectness_mask.float().unsqueeze(2))/(torch.sum(objectness_mask.float())+1e-6)
    velocity_loss = criterion2(velocity, velocity_label)
    velocity_loss = torch.sum(velocity_loss * objectness_mask.float().unsqueeze(2))/(torch.sum(objectness_mask.float())*2+1e-6)
    return class_loss, center_loss, r_loss, whl_loss, yaw_loss, velocity_loss   

def iou_spheres(spheres_a, spheres_b, no_grad=False):
    """
    Computes IoU overlaps between two sets of spheres.
    spheres_a: [M, 4]
    spheres_b: [N, 4].
    return: iou table of shape [M, N]
    """
    if no_grad:
        with torch.no_grad():
            spheres1 = spheres_a.unsqueeze(1).expand(-1,spheres_b.size()[0],-1) #shape: [M,N,4]
            spheres2 = spheres_b.unsqueeze(0).expand(spheres_a.size()[0],-1,-1) #shape: [M,N,4]
            dist = torch.norm(spheres1[:,:,0:3] - spheres2[:,:,0:3],p=2,dim=2)
            r_a, r_b = spheres1[:,:,3], spheres2[:,:,3]
            iou = torch.zeros([spheres_a.size()[0], spheres_b.size()[0]], device=spheres1.device)
            # one sphere fully inside the other (includes coincident)
            # take volume of smaller sphere as intersection
            # take volume of larger sphere as union
            # iou is (min(r_a, r_b)/max(r_a, r_b))**3
            idx = torch.nonzero(dist <= abs(r_a - r_b))
            if idx.size()[0]>0:
                min_r = torch.min(r_a[idx[:,0],idx[:,1]], r_b[idx[:,0],idx[:,1]])
                max_r = torch.max(r_a[idx[:,0],idx[:,1]], r_b[idx[:,0],idx[:,1]])
                iou[idx[:,0],idx[:,1]] = (1.0*min_r/max_r)**3
    
            # spheres partially overlap, calculate intersection as per
            # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
            idx = torch.nonzero((dist > abs(r_a - r_b))&(dist < r_a + r_b))
            if idx.size()[0]>0:
                intersection = (r_a[idx[:,0],idx[:,1]] + r_b[idx[:,0],idx[:,1]] - dist[idx[:,0],idx[:,1]])**2
                intersection *= (dist[idx[:,0],idx[:,1]]**2 + 2*dist[idx[:,0],idx[:,1]]*(r_a[idx[:,0],idx[:,1]] + r_b[idx[:,0],idx[:,1]]) - 3*(r_a[idx[:,0],idx[:,1]] - r_b[idx[:,0],idx[:,1]])**2)
                intersection *= np.pi / (12*dist[idx[:,0],idx[:,1]])
                union = 4/3. * np.pi * (r_a[idx[:,0],idx[:,1]]**3 + r_b[idx[:,0],idx[:,1]]**3) - intersection
                iou[idx[:,0],idx[:,1]] = intersection / union
    else:
        spheres1 = spheres_a.unsqueeze(1).expand(-1,spheres_b.size()[0],-1) #shape: [M,N,4]
        spheres2 = spheres_b.unsqueeze(0).expand(spheres_a.size()[0],-1,-1) #shape: [M,N,4]
        dist = torch.norm(spheres1[:,:,0:3] - spheres2[:,:,0:3],p=2,dim=2)
        r_a, r_b = spheres1[:,:,3], spheres2[:,:,3]
        iou = torch.zeros([spheres_a.size()[0], spheres_b.size()[0]], device=spheres1.device)
        # one sphere fully inside the other (includes coincident)
        # take volume of smaller sphere as intersection
        # take volume of larger sphere as union
        # iou is (min(r_a, r_b)/max(r_a, r_b))**3
        idx = torch.nonzero(dist <= abs(r_a - r_b))
        if idx.size()[0]>0:
            min_r = torch.min(r_a[idx[:,0],idx[:,1]], r_b[idx[:,0],idx[:,1]])
            max_r = torch.max(r_a[idx[:,0],idx[:,1]], r_b[idx[:,0],idx[:,1]])
            iou[idx[:,0],idx[:,1]] = (1.0*min_r/max_r)**3
    
        # spheres partially overlap, calculate intersection as per
        # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        idx = torch.nonzero((dist > abs(r_a - r_b))&(dist < r_a + r_b))
        if idx.size()[0]>0:
            intersection = (r_a[idx[:,0],idx[:,1]] + r_b[idx[:,0],idx[:,1]] - dist[idx[:,0],idx[:,1]])**2
            intersection *= (dist[idx[:,0],idx[:,1]]**2 + 2*dist[idx[:,0],idx[:,1]]*(r_a[idx[:,0],idx[:,1]] + r_b[idx[:,0],idx[:,1]]) - 3*(r_a[idx[:,0],idx[:,1]] - r_b[idx[:,0],idx[:,1]])**2)
            intersection *= np.pi / (12*dist[idx[:,0],idx[:,1]])
            union = 4/3. * np.pi * (r_a[idx[:,0],idx[:,1]]**3 + r_b[idx[:,0],idx[:,1]]**3) - intersection
            iou[idx[:,0],idx[:,1]] = intersection / union        
    
    return iou
    
def nms(bspheres, scores, threshold=0.7):
    """
    Apply non-maximum suppression on bounding spheres
    Parameters:
        bspheres: [N,4]
        scores: [N]
    """
    _, order = scores.sort(0, descending=True)
    iou_table = iou_spheres(bspheres, bspheres,no_grad=True)
    keep = []
    # loop when order is not empty
    while order.numel() > 0:
        # when order has only one element, save this last element
        if order.numel() == 1:
            i = order
            keep.append(i)
            break
        # otherwise, save the element with the highest score
        else:
            i = order[0]
            keep.append(i)
        # from iou_table take out IoUs between sphere[i] and other remaining spheres
        iou = iou_table[i, order[1:]]  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # idx: [N-1,], order: [N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # add one to match the index difference
    return torch.tensor(keep, device=bspheres.device).long()
    
def proposal_layer(class_probs, bsphere, whl, yaw, velocity, nms_threshold=0.3):
    """
    Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    sphere refinment deltas to anchors.
    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bsphere: [batch, anchors, (dx, dy, dz, log(dr))]
        anchors: [batch, anchors, (x, y, z, r)]
    Returns:
        proposals: list of tensor [proposal_count, (x, y, z, r)], length is batchsize
    """
    _, class_id_preds = torch.max(class_probs,dim=2) # [batchsize, nproposal=128]
    class_id_preds = class_id_preds.unsqueeze(2) # [batchsize, nproposal=128, 1]
    class_scores = torch.gather(class_probs, dim=2, index=class_id_preds) # [batchsize, nproposal=128, 1]
    rois = torch.cat((bsphere, whl, yaw, velocity),dim=2) # [batchsize, nproposal=128, 10] [x,y,z,r,w,h,l,yaw,vx,vy]
    final_detections = []
    for i in range(class_probs.shape[0]):
        keep_bool = (class_id_preds[i,:,0]>0)&(class_scores[i,:,0]>=0.2) #[nproposal=128]
        if torch.sum(keep_bool)>0:
            keep = torch.nonzero(keep_bool)[:,0]
            # Apply per-class NMS
            pre_nms_class_ids = class_id_preds[i,keep,0]
            pre_nms_scores = class_scores[i,keep,0]
            pre_nms_rois = rois[i,keep,:]
            class_keep_list = []
            for j, class_id in enumerate(torch.unique(pre_nms_class_ids)):
                # Pick detections of this class
                ixs = torch.nonzero(pre_nms_class_ids == class_id)[:,0]
                # Apply nms
                nms_keep = nms(pre_nms_rois[ixs,0:4],pre_nms_scores[ixs],threshold=nms_threshold)
                class_keep = keep[ixs[nms_keep]]
                class_keep_list.append(class_keep)
            keep = torch.cat(class_keep_list)
            # detection shape: [ndetetcion, 12], where 12 corresponds to [x,y,z,r,class_id,score,w,h,l,yaw,vx,vy]
            detection = torch.cat((rois[i,keep,0:4], class_id_preds[i,keep,:].float(), class_scores[i,keep,:], rois[i,keep,4:]),dim=1)
            final_detections.append(detection)
        else:
            final_detections.append(torch.tensor([],device=rois.device))
    return final_detections
    
class PointMaskRCNN(nn.Module):
    """
    Builds the model of whole Network.

    Returns:
        rpn_logits: [batch, npoints * anchors per point, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, npoints * anchors per point, 2] Anchor classifier probabilities.
        rpn_bsphere: [batch, npoints * anchors per point, (dx, dy, dz, log(dr))] Deltas to be applied to anchors.
    """

    def __init__(self):
        super(PointMaskRCNN, self).__init__()
        self.RPN = RPN()
        self.detectionHead = MrcnnDetectionHead(num_class=11)

    def forward(self, xyz, points, gt_bounding_spheres_list, mode):
        if mode == 'train':
            # generate RPN outputs using RPN network
            features, vote_delta, agg_xyz, rpn_class_logits, rpn_probs, rpn_bsphere = self.RPN(xyz, points)
        
            # generate proposals using proposal layer and RPN outputs
            rpn_proposals, _ = proposal_layer(rpn_probs, rpn_bsphere, proposal_count=1000, nms_threshold=0.7)
            
            # include gt bounding spheres into proposals
            for i in range(len(rpn_proposals)):
                rpn_proposals[i] = torch.cat((rpn_proposals[i], gt_bounding_spheres_list[i][:,0:4]))

            # take out points and their features in proposals
            proposals, roi_point_features_batch = roi_points_layer(rpn_proposals, xyz, features)
            
            # subsamples proposals and build target for subsampled proposals
            roi_point_features_batch, roi_bsphere_proposals_batch, roi_gt_class_ids_batch, roi_bsphere_target_batch = mrcnn_target_layer(proposals, roi_point_features_batch, gt_bounding_spheres_list)
            
            # run detection head and mask head
            mrcnn_class_logits_batch, mrcnn_class_probs_batch, mrcnn_bsphere_batch = self.detectionHead(roi_point_features_batch)
            
            #refine detections and generate final detections
            #final_detections = refine_detections(roi_bsphere_proposals_batch, mrcnn_class_probs_batch, mrcnn_bsphere_batch)
            
            return vote_delta, agg_xyz, rpn_class_logits, rpn_bsphere, mrcnn_class_logits_batch, mrcnn_bsphere_batch, roi_gt_class_ids_batch, roi_bsphere_target_batch
        
        if mode == 'test':
            # generate RPN outputs using RPN network
            features, _, _, rpn_class_logits, rpn_probs, rpn_bsphere = self.RPN(xyz, points)
            
            # generate proposals using proposal layer and RPN outputs
            proposals, proposals_scores = proposal_layer(rpn_probs, rpn_bsphere, proposal_count=1000, nms_threshold=0.7)
            rpn_proposals = proposals[:]
            rpn_proposals_scores = proposals_scores[:]
            
            # take out points and their features in proposals
            proposals, roi_point_features_batch = roi_points_layer(proposals, xyz, features)
            
            # run detection head
            mrcnn_class_logits_batch, mrcnn_class_probs_batch, mrcnn_bsphere_batch = self.detectionHead(roi_point_features_batch)
            
            #refine detections and generate final detections
            final_detections = refine_detections(proposals, mrcnn_class_probs_batch, mrcnn_bsphere_batch)
            
            return rpn_proposals, rpn_proposals_scores, final_detections
    
    
if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    label = torch.randn(8,16)
    model = PointNet2PartSeg_msg_one_hot(num_classes=50)
    output= model(input,input,label)
    print(output.size())

