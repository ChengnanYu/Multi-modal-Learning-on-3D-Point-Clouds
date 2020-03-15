import pickle
import os
import sys
import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import utils.pc_util as pc_util
import utils.scene_util as scene_util
import random
import math
from scipy import misc
from PIL import Image
import torchvision.transforms as transforms
from utils.projection import ProjectionHelper
from model.pointnet_util import pc_normalize

intrinsic = [[37.01983, 0, 20, 0],[0, 38.52470, 15.5, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
projection = ProjectionHelper(intrinsic, 0.1, 4.0, [41,32], 0.05)
        

class ScannetDatasetRGBImg(Dataset):
    def __init__(self, root, npoints=8192, split='train', num_images=5):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.num_images = num_images
        data_list = os.path.join(self.root, 'scannetv2_%s.txt'%(split))
        datalist = open(data_list, 'r')
        self.scenes = [x.strip() for x in datalist.readlines()]
        self.data_filename = os.path.join(self.root, 'scannetv2_%s.pickle'%(split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='bytes')
            self.semantic_labels_list = pickle.load(fp, encoding='bytes')
        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(21)
        
    def __getitem__(self, index):
        scene = self.scenes[index]
        # get points and labels
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        semantic_seg = semantic_seg[:,0]
        coordmax = np.max(point_set[:,0:3],axis=0)
        coordmin = np.min(point_set[:,0:3],axis=0)
        smpmin = np.maximum(coordmax-[1.5,1.5,3.0], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin,[1.5,1.5,3.0])
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],0:3]
            curmin = curcenter-[0.75,0.75,1.5]
            curmax = curcenter+[0.75,0.75,1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set[:,0:3]>=(curmin-0.2))*(point_set[:,0:3]<=(curmax+0.2)),axis=1)==3
            cur_point_set = point_set[curchoice,:]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg)==0:
                continue
            mask = np.sum((cur_point_set[:,0:3]>=(curmin-0.01))*(cur_point_set[:,0:3]<=(curmax+0.01)),axis=1)==3
            vidx = np.ceil((cur_point_set[mask,0:3]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
            vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
            isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        point_set = cur_point_set[choice,:]
        #point_set[:,3:6] = point_set[:,3:6]/255.0 - 0.5
        point_set[:,3:6] = pc_normalize(point_set[:,0:3])
        # TODO: point set augmentation
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask
        # get images
        base_image_path = os.path.join(self.root, 'frames_square', scene)
        pose_files = os.listdir(os.path.join(base_image_path, 'pose'))
        #find best 5 images 
        frame_ids = []
        poseDict = {}
        for poseFile in pose_files:
            pose = self.load_pose(os.path.join(base_image_path, 'pose', poseFile))
            with torch.no_grad():
                corners = projection.compute_frustum_corners(torch.from_numpy(pose))[:, :3, 0] # Corners of Frustum
                normals = projection.compute_frustum_normals(corners) # Normals of frustum
                num_valid_points = projection.points_in_frustum_cpu(corners.double(), normals.double(), torch.DoubleTensor(point_set[:,0:3])) # Checks for each point if it lies on the correct side of the normals of the frustum
                poseDict[poseFile[:-4]] = num_valid_points
        for i in range(self.num_images):# find maxima
            maximum = max(poseDict, key=poseDict.get)
            num_valid_points = poseDict[maximum]
            if (i==0)|(num_valid_points.numpy()>100):
                frame_ids.append(int(maximum))
                del poseDict[maximum]
            else:
                frame_ids.append(frame_ids[0])
        depths = []
        images = []
        poses = []
        for i in frame_ids:
            depth_file = os.path.join(base_image_path, 'depth', str(i) + '.png')
            image_file = os.path.join(base_image_path, 'color', str(i) + '.jpg')
            pose_file = os.path.join(base_image_path, 'pose', str(i) + '.txt')
            poses.append(self.load_pose(pose_file)) #[4,4]
            depths.append(self.load_depth(depth_file, [41,32])) #[32,41]
            im_pre = self.load_image(image_file, [328,256]) #[3,256,328]
            images.append(im_pre)
        image = np.stack(images,0) #[num_images,3,256,328]
        depth = np.array(depths) #[num_images,32,41]
        pose = np.array(poses)   #[num_images, 4, 4]
        return point_set, semantic_seg, sample_weight, image, depth, pose
        
    def __len__(self):
        return len(self.scene_points_list)
        
    def load_pose(self, filename):
        pose = np.zeros((4, 4))
        lines = open(filename).read().splitlines()
        assert len(lines) == 4
        lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
        return np.asarray(lines).astype(np.float32)
        
    def resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)
        return image
        
    def load_depth(self, file, image_dims):
        depth_image = misc.imread(file)
        # preprocess
        depth_image = self.resize_crop_image(depth_image, image_dims)
        depth_image = depth_image.astype(np.float32) / 1000.0
        return depth_image
        
    def load_image(self, file, image_dims):
        image = misc.imread(file)
        # preprocess
        image = self.resize_crop_image(image, image_dims)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=[0.496342, 0.466664, 0.440796], std=[0.277856, 0.28623, 0.291129])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise
        return image
        
    def collate_fn(self, batch):
        '''
        return:
        a tensor of point clouds
        a tensor of semantic labels
        a tensor of sample weights
        '''
        point_set_list = []
        semantic_seg_list = []
        sample_weight_list = []
        image_list = []
        depth_list = []
        pose_list = []
        for b in batch:
            point_set_list.append(torch.from_numpy(b[0]))
            semantic_seg_list.append(torch.from_numpy(b[1]))
            sample_weight_list.append(torch.from_numpy(b[2]))
            image_list.append(torch.from_numpy(b[3]))
            depth_list.append(torch.from_numpy(b[4]))
            pose_list.append(torch.from_numpy(b[5]))
        point_sets = torch.stack(point_set_list, dim=0) #shape: [batchsize, npoints, 6]
        semantic_segs = torch.stack(semantic_seg_list, dim=0) #shape: [batchsize, npoints]
        sample_weights = torch.stack(sample_weight_list, dim=0) #shape: [batchsize, npoints]
        #depth = torch.cat(depth_list,dim=0) #shape: [batchsize*num_images,32,41]
        #pose = torch.cat(pose_list,dim=0) #shape: [batchsize*num_images,4,4]
        # return tensor, tensor, tensor, list of N tensors, list of N tensors, list of N tensors
        return point_sets, semantic_segs, sample_weights, image_list, depth_list, pose_list

class ScannetDatasetWholeSceneRGBImg(Dataset):
    def __init__(self, root, npoints=8192, split='train', num_images=5):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.num_images = num_images
        data_list = os.path.join(self.root, 'scannetv2_%s.txt'%(split))
        datalist = open(data_list, 'r')
        self.scenes = [x.strip() for x in datalist.readlines()]
        self.data_filename = os.path.join(self.root, 'scannetv2_%s.pickle'%(split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='bytes')
            self.semantic_labels_list = pickle.load(fp, encoding='bytes')
        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(21)
        #for each scene, get points of every subvolumn, stack them up
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()        
        self.pc_scene = list()
        for index in range(len(self.scene_points_list)):
            scene = self.scenes[index]
            point_set_ini = self.scene_points_list[index]
            semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
            semantic_seg_ini = semantic_seg_ini[:,0]
            coordmax = np.max(point_set_ini[:,0:3],axis=0)
            coordmin = np.min(point_set_ini[:,0:3],axis=0)
            nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
            nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
            isvalid = False
            for i in range(nsubvolume_x):
                for j in range(nsubvolume_y):
                    curmin = coordmin+[i*1.5,j*1.5,0]
                    curmax = coordmin+[(i+1)*1.5,(j+1)*1.5,coordmax[2]-coordmin[2]]
                    curchoice = np.sum((point_set_ini[:,0:3]>=(curmin-0.2))*(point_set_ini[:,0:3]<=(curmax+0.2)),axis=1)==3
                    cur_point_set = point_set_ini[curchoice,:]
                    cur_semantic_seg = semantic_seg_ini[curchoice]
                    if len(cur_semantic_seg)==0:
                        continue
                    mask = np.sum((cur_point_set[:,0:3]>=(curmin-0.001))*(cur_point_set[:,0:3]<=(curmax+0.001)),axis=1)==3
                    choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                    point_set = cur_point_set[choice,:] # Nx6
                    point_set[:,3:6] = point_set[:,3:6]/255.0 - 0.5
                    semantic_seg = cur_semantic_seg[choice] # N
                    mask = mask[choice]
                    if sum(mask)/float(len(mask))<0.01:
                        continue
                    sample_weight = self.labelweights[semantic_seg]
                    sample_weight *= mask # N
                    point_sets.append(np.expand_dims(point_set,0)) # 1xNx6
                    semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                    sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN                    
                    self.pc_scene.append(scene)
        self.point_sets = np.concatenate(tuple(point_sets),axis=0)
        self.semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        self.sample_weights = np.concatenate(tuple(sample_weights),axis=0)
        #print('whole size is %d' % self.point_sets.shape[0])
        
    def __getitem__(self, index):
        scene = self.pc_scene[index]
        # get images
        base_image_path = os.path.join(self.root, 'frames_square', scene)
        pose_files = os.listdir(os.path.join(base_image_path, 'pose'))
        #find best 5 images 
        frame_ids = []
        poseDict = {}
        for poseFile in pose_files:
            pose = self.load_pose(os.path.join(base_image_path, 'pose', poseFile))
            with torch.no_grad():
                corners = projection.compute_frustum_corners(torch.from_numpy(pose))[:, :3, 0] # Corners of Frustum
                normals = projection.compute_frustum_normals(corners) # Normals of frustum
                num_valid_points = projection.points_in_frustum_cpu(corners.double(), normals.double(), torch.DoubleTensor(self.point_sets[index,:,0:3])) # Checks for each point if it lies on the correct side of the normals of the frustum
                poseDict[poseFile[:-4]] = num_valid_points
        for i in range(self.num_images):# find maxima
            maximum = max(poseDict, key=poseDict.get)
            num_valid_points = poseDict[maximum]
            if (i==0)|(num_valid_points.numpy()>100):
                frame_ids.append(int(maximum))
                del poseDict[maximum]
            else:
                frame_ids.append(frame_ids[0])
        depths = []
        images = []
        poses = []
        for i in frame_ids:
            depth_file = os.path.join(base_image_path, 'depth', str(i) + '.png')
            image_file = os.path.join(base_image_path, 'color', str(i) + '.jpg')
            pose_file = os.path.join(base_image_path, 'pose', str(i) + '.txt')
            poses.append(self.load_pose(pose_file)) #[4,4]
            depths.append(self.load_depth(depth_file, [41,32])) #[32,41]
            im_pre = self.load_image(image_file, [328,256]) #[3,256,328]
            images.append(im_pre)
        image = np.stack(images,0) #[num_images,3,256,328]
        depth = np.array(depths) #[num_images,32,41]
        pose = np.array(poses)   #[num_images, 4, 4]        
        return self.point_sets[index], self.semantic_segs[index], self.sample_weights[index], image, depth, pose
        
    def __len__(self):
        return self.point_sets.shape[0]
        
    def load_pose(self, filename):
        pose = np.zeros((4, 4))
        lines = open(filename).read().splitlines()
        assert len(lines) == 4
        lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
        return np.asarray(lines).astype(np.float32)
        
    def resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)
        return image
        
    def load_depth(self, file, image_dims):
        depth_image = misc.imread(file)
        # preprocess
        depth_image = self.resize_crop_image(depth_image, image_dims)
        depth_image = depth_image.astype(np.float32) / 1000.0
        return depth_image
        
    def load_image(self, file, image_dims):
        image = misc.imread(file)
        # preprocess
        image = self.resize_crop_image(image, image_dims)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=[0.496342, 0.466664, 0.440796], std=[0.277856, 0.28623, 0.291129])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise
        return image
        
    def collate_fn(self, batch):
        '''
        return:
        a tensor of point clouds
        a tensor of semantic labels
        a tensor of sample weights
        '''
        point_set_list = []
        semantic_seg_list = []
        sample_weight_list = []
        image_list = []
        depth_list = []
        pose_list = []
        for b in batch:
            point_set_list.append(torch.from_numpy(b[0]))
            semantic_seg_list.append(torch.from_numpy(b[1]))
            sample_weight_list.append(torch.from_numpy(b[2]))
            image_list.append(torch.from_numpy(b[3]))
            depth_list.append(torch.from_numpy(b[4]))
            pose_list.append(torch.from_numpy(b[5]))
        point_sets = torch.stack(point_set_list, dim=0) #shape: [batchsize, npoints, 6]
        semantic_segs = torch.stack(semantic_seg_list, dim=0) #shape: [batchsize, npoints]
        sample_weights = torch.stack(sample_weight_list, dim=0) #shape: [batchsize, npoints]
        #depth = torch.cat(depth_list,dim=0) #shape: [batchsize*num_images,32,41]
        #pose = torch.cat(pose_list,dim=0) #shape: [batchsize*num_images,4,4]
        # return tensor, tensor, tensor, list of N tensors
        return point_sets, semantic_segs, sample_weights, image_list, depth_list, pose_list

class ScannetDataset(Dataset):
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannetv2_%s.pickle'%(split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='bytes')
            self.semantic_labels_list = pickle.load(fp, encoding='bytes')
        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(21)
    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        semantic_seg = semantic_seg[:,0]
        coordmax = np.max(point_set[:,0:3],axis=0)
        coordmin = np.min(point_set[:,0:3],axis=0)
        smpmin = np.maximum(coordmax-[1.5,1.5,3.0], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin,[1.5,1.5,3.0])
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],0:3]
            curmin = curcenter-[0.75,0.75,1.5]
            curmax = curcenter+[0.75,0.75,1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set[:,0:3]>=(curmin-0.2))*(point_set[:,0:3]<=(curmax+0.2)),axis=1)==3
            cur_point_set = point_set[curchoice,:]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg)==0:
                continue
            mask = np.sum((cur_point_set[:,0:3]>=(curmin-0.01))*(cur_point_set[:,0:3]<=(curmax+0.01)),axis=1)==3
            vidx = np.ceil((cur_point_set[mask,0:3]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
            vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
            isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        point_set = cur_point_set[choice,:]
        point_set[:,3:6] = point_set[:,3:6]/255.0 - 0.5
        # TODO: point set augmentation
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask
        return point_set, semantic_seg, sample_weight
    def __len__(self):
        return len(self.scene_points_list)
        
class ScannetDatasetWholeScene(Dataset):
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannetv2_%s.pickle'%(split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='bytes')
            self.semantic_labels_list = pickle.load(fp, encoding='bytes')
        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(21)
        #for each scene, get points of every subvolumn, stack them up
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()        
        for index in range(len(self.scene_points_list)):
            point_set_ini = self.scene_points_list[index]
            semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
            semantic_seg_ini = semantic_seg_ini[:,0]
            coordmax = np.max(point_set_ini[:,0:3],axis=0)
            coordmin = np.min(point_set_ini[:,0:3],axis=0)
            nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
            nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
            isvalid = False
            for i in range(nsubvolume_x):
                for j in range(nsubvolume_y):
                    curmin = coordmin+[i*1.5,j*1.5,0]
                    curmax = coordmin+[(i+1)*1.5,(j+1)*1.5,coordmax[2]-coordmin[2]]
                    curchoice = np.sum((point_set_ini[:,0:3]>=(curmin-0.2))*(point_set_ini[:,0:3]<=(curmax+0.2)),axis=1)==3
                    cur_point_set = point_set_ini[curchoice,:]
                    cur_semantic_seg = semantic_seg_ini[curchoice]
                    if len(cur_semantic_seg)==0:
                        continue
                    mask = np.sum((cur_point_set[:,0:3]>=(curmin-0.001))*(cur_point_set[:,0:3]<=(curmax+0.001)),axis=1)==3
                    choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                    point_set = cur_point_set[choice,:] # Nx6
                    point_set[:,3:6] = point_set[:,3:6]/255.0 - 0.5
                    semantic_seg = cur_semantic_seg[choice] # N
                    mask = mask[choice]
                    if sum(mask)/float(len(mask))<0.01:
                        continue
                    sample_weight = self.labelweights[semantic_seg]
                    sample_weight *= mask # N
                    point_sets.append(np.expand_dims(point_set,0)) # 1xNx6
                    semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                    sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
        self.point_sets = np.concatenate(tuple(point_sets),axis=0)
        self.semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        self.sample_weights = np.concatenate(tuple(sample_weights),axis=0)
        #print('whole size is %d' % self.point_sets.shape[0])
    def __getitem__(self, index):
        return self.point_sets[index], self.semantic_segs[index], self.sample_weights[index]
    def __len__(self):
        return self.point_sets.shape[0]

class ScannetDatasetVirtualScan(Dataset):
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannetv2_%s.pickle'%(split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='bytes')
            self.semantic_labels_list = pickle.load(fp, encoding='bytes')
        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(21)
    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        sample_weight_ini = self.labelweights[semantic_seg_ini]
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in range(8):
            smpidx = scene_util.virtual_scan(point_set_ini,mode=i)
            if len(smpidx)<300:
                continue
            point_set = point_set_ini[smpidx,:]
            semantic_seg = semantic_seg_ini[smpidx]
            sample_weight = sample_weight_ini[smpidx]
            choice = np.random.choice(len(semantic_seg), self.npoints, replace=True)
            point_set = point_set[choice,:] # Nx3
            semantic_seg = semantic_seg[choice] # N
            sample_weight = sample_weight[choice] # N
            point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
            semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
            sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)
        return point_sets, semantic_segs, sample_weights
    def __len__(self):
        return len(self.scene_points_list)

if __name__=='__main__':
    #d = ScannetDatasetWholeScene(root = './data', split='test', npoints=8192)
    dataset = ScannetDatasetRGBImg(root = './data', split='test', npoints=8192)
    #dataset = ScannetDataset(root = './data', split='test', npoints=8192)
    #dataset = ScannetDatasetVirtualScan(root = './data', split='test', npoints=8192)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn,
                                             shuffle=False, num_workers=4)
    labelweights_vox = np.zeros(21)
    for ii, data in enumerate(dataloader):
    #for ii in range(len(d)):
        print(ii)
        ps,seg,smpw,image,depth,pose = data
        print(ps.shape)
        print(seg.shape)
        print(smpw.shape)
        for b in range(len(image)):
            print(image[b].shape)
            print(depth[b].shape)
            print(pose[b].shape)
        #for b in range(ps.shape[0]):
        #    _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(ps[b,smpw[b,:]>0,:], seg[b,smpw[b,:]>0], res=0.02)
        #    tmp,_ = np.histogram(uvlabel,range(22))
        #    labelweights_vox += tmp
    #print(labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32)))
    exit()


