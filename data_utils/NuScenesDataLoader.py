import os
import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
import math
from scipy.spatial import Delaunay
from scipy import misc
from PIL import Image
import torchvision.transforms as transforms
from utils.projection import ProjectionHelper


class NuscenesDataset(Dataset):
    def __init__(self, nusc, npoints_lidar=16384, npoints_radar=1024, split='train_detect'):
        self.npoints_lidar = npoints_lidar
        self.npoints_radar = npoints_radar
        self.split = split
        self.nusc = nusc
        split_dict = create_splits_scenes()
        self.NameMapping = {
            'movable_object.barrier': 'barrier',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }
        self.LabelMapping = {
            'barrier': 1,
            'bicycle': 2,
            'bus': 3,
            'car': 4,
            'construction_vehicle': 5,
            'motorcycle': 6,
            'pedestrian': 7,
            'traffic_cone': 8,
            'trailer': 9,
            'truck':10
        }
        self.MaxDistMapping = {
            1: 30.0,
            2: 40.0,
            3: 50.0,
            4: 50.0,
            5: 50.0,
            6: 40.0,
            7: 40.0,
            8: 30.0,
            9: 50.0,
            10: 50.0
        }        
        self.DefaultAttribute = {
            "car": "vehicle.parked",
            "pedestrian": "pedestrian.moving",
            "trailer": "vehicle.parked",
            "truck": "vehicle.parked",
            "bus": "vehicle.parked",
            "motorcycle": "cycle.without_rider",
            "construction_vehicle": "vehicle.parked",
            "bicycle": "cycle.without_rider",
            "barrier": "",
            "traffic_cone": ""
        }
        self.scene = []
        for s in self.nusc.scene:
            if s['name'] in split_dict[split]:
                self.scene.append(s)
        self.sample = []
        for s in self.scene:
            sample = self.nusc.get('sample', s['first_sample_token'])
            self.sample.append(sample)
            while not sample['next'] == "":
                sample = self.nusc.get('sample', sample['next'])
                self.sample.append(sample)
        print(len(self.scene))
        print(len(self.sample))
    
    def __getitem__(self, index):
        sample = self.sample[index]
        #if self.split in ['train_detect', 'train_track', 'train']:
        #    # augmentation
        #    aug_angle = np.random.uniform(-np.pi/36.0, np.pi/36.0)
        #    aug_quat = Quaternion(axis=[0, 0, 1], angle=aug_angle)
        #    aug_rot_mat = aug_quat.rotation_matrix
        #    aug_trans_x = np.random.uniform(-0.5, 0.5)
        #    aug_trans_y = np.random.uniform(-0.5, 0.5)
        #    aug_translation = np.array([aug_trans_x, aug_trans_y, 0.0])
        #    aug_trans_mat = aug_quat.transformation_matrix
        #    aug_trans_mat[0:3,3] = aug_translation
        # get box list, filter objects and transform to np array
        _, BoxList, _ = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        gt_BoxList = []
        gt_SphereList = []
        gt_BoxCornerList = []
        for box in BoxList:
            if box.name in self.NameMapping:
                box.label = self.LabelMapping[self.NameMapping[box.name]]
                if np.linalg.norm(box.center[:2]) <= self.MaxDistMapping[box.label]:
                    #if self.split in ['train_detect', 'train_track', 'train']:
                    #    #augmentation
                    #    box.rotate(aug_quat)
                    #    box.translate(aug_translation)
                    center = box.center
                    size = box.wlh
                    yaw = np.array([box.orientation.yaw_pitch_roll[0]])
                    velocity = box.velocity[0:2]
                    if np.sum(np.isnan(velocity))>0:
                        velocity = np.array([0.0, 0.0])
                    label = np.array([box.label])
                    # box encode: 0,1,2: center, 3,4,5: size(whl), 6: yaw(rad), 7,8: vx,vy, 9:label
                    new_box = np.concatenate((center, size, yaw, velocity, label))
                    gt_BoxList.append(new_box)
                    r = np.array([np.linalg.norm(size)/2.0])
                    # sphere encode: 0,1,2: center, 3: size(r), 4: yaw(rad), 5,6: vx,vy, 7:label
                    new_sphere = np.concatenate((center, r, yaw, velocity, label))
                    gt_SphereList.append(new_sphere)
                    box_corners = box.corners().T
                    gt_BoxCornerList.append(box_corners)                    
                #else:
                    #print('GT box out of detection range')
                    #print(np.linalg.norm(box.center[:2]))
                    #print(self.MaxDistMapping[box.label])
        bounding_boxes = np.array(gt_BoxList)
        bounding_spheres = np.array(gt_SphereList)
        bounding_boxes_corners = np.array(gt_BoxCornerList)
        if bounding_boxes.shape[0]==0:
            #print('No object in this sample')
            return self.__getitem__(index-1)
        
        # get lidar point cloud
        lidar_sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', 5)
        lidar_points = np.concatenate((pc.points,times),0).T #size [N, 5], 5 corresponds to [x,y,z,intensity,time]
        depth = np.linalg.norm(lidar_points[:,0:3], axis=1)       
        lidar_points = lidar_points[depth<52.0,:]
        if self.npoints_lidar < lidar_points.shape[0]:
            choice = np.random.choice(lidar_points.shape[0], self.npoints_lidar, replace=False)
        else:
            choice = np.arange(0, lidar_points.shape[0], dtype=np.int32)
            if self.npoints_lidar > lidar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_lidar - lidar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        lidar_points = lidar_points[choice,:] #size [npoints_lidar, 5]
        lidar_points[:,3] = lidar_points[:,3]/255.0 # scale intensity from 0-255 to 0-1
        #if self.split in ['train_detect', 'train_track', 'train']:
        #    #augmentation
        #    lidar_points[:,0:3] = np.dot(aug_rot_mat, lidar_points[:,0:3].T).T + aug_translation
        
        
        '''
        # get radar point cloud
        radar_points_list = []
        radar_sensor = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT_RIGHT']
        for sensor in radar_sensor:
            radar_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            radar_cs_record = self.nusc.get('calibrated_sensor', radar_sd_record['calibrated_sensor_token'])
            pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, sensor, 'LIDAR_TOP', 6)
            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the point cloud.
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(lidar_cs_record['rotation']).rotation_matrix.T, velocities)            
            velocities[2, :] = np.zeros(pc.points.shape[1])
            radar_points = np.concatenate((pc.points[0:3,:],velocities[0:2,:],times),0).T #size [N, 6], 6 corresponds to [x,y,z,vx,vy,time]
            radar_points_list.append(radar_points)
        radar_points = np.concatenate(radar_points_list, 0)
        depth = np.linalg.norm(radar_points[:,0:3], axis=1)
        radar_points = radar_points[depth<52.0,:]
        if self.npoints_radar < radar_points.shape[0]:
            choice = np.random.choice(radar_points.shape[0], self.npoints_radar, replace=False)
        else:
            choice = np.arange(0, radar_points.shape[0], dtype=np.int32)
            if self.npoints_radar > radar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_radar - radar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        radar_points = radar_points[choice,:] #size [npoints_radar, 5]
        
        # get images, camera intrinsic, camera pose to lidar coordinate system
        images = []
        poses = []
        intrinsics = []
        camera_sensor = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        for sensor in camera_sensor:
            cam_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            cam_cs_record = self.nusc.get('calibrated_sensor', cam_sd_record['calibrated_sensor_token'])
            img_path = os.path.join(self.nusc.dataroot, cam_sd_record['filename'])
            img = self.load_image(img_path, [640,360]) #[3,360,640], feature map will be [128,45,80]
            images.append(img)
            
            cam_intrinsic = np.eye(4)
            cam_intrinsic[0:3,0:3] = np.array(cam_cs_record['camera_intrinsic'])
            # adjust intrinsic for the size of feature map
            cam_intrinsic[0,0] *= float(80)/float(1600)
            cam_intrinsic[1,1] *= float(45)/float(900)
            cam_intrinsic[0,2] *= float(80-1)/float(1600-1)
            cam_intrinsic[1,2] *= float(45-1)/float(900-1)
            intrinsics.append(cam_intrinsic)
            
            # compute camera pose
            cam_to_ego = np.eye(4)
            cam_to_ego[0:3,0:3] = Quaternion(cam_cs_record['rotation']).rotation_matrix
            cam_to_ego[0:3,3] = np.array(cam_cs_record['translation'])            
            poserecord = self.nusc.get('ego_pose', cam_sd_record['ego_pose_token'])
            ego_to_global = np.eye(4)
            ego_to_global[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix
            ego_to_global[0:3,3] = np.array(poserecord['translation'])            
            poserecord = self.nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
            global_to_ego = np.eye(4)
            global_to_ego[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix.T
            global_to_ego[0:3,3] = global_to_ego[0:3,0:3].dot(-np.array(poserecord['translation']))            
            ego_to_lidar = np.eye(4)
            ego_to_lidar[0:3,0:3] = Quaternion(lidar_cs_record['rotation']).rotation_matrix.T
            ego_to_lidar[0:3,3] = ego_to_lidar[0:3,0:3].dot(-np.array(lidar_cs_record['translation']))            
            cam_to_lidar = ego_to_lidar.dot(global_to_ego).dot(ego_to_global).dot(cam_to_ego)
            poses.append(cam_to_lidar)
        image = np.stack(images,0) #[num_images=6,3,360,640]
        intrinsic = np.array(intrinsics) #[num_images=6,4,4]
        pose = np.array(poses).astype(np.float32) #[num_images=6,4,4]
        '''
        radar_points = np.array([])
        image = np.array([])
        intrinsic = np.array([])
        pose = np.array([])
        
        # generate anchors, shape: [npoints*anchors per point, 4]

        # 10 anchors kmeans
        #anchor_radius = np.array([0.59519359, 1.04024921, 1.51894038, 2.33694095, 2.58570698, 2.82699181, 3.34568567, 4.45521787, 6.1408409, 8.17982825])
        # 12 anchors kmeans
        #anchor_radius = np.array([0.47506766, 0.73028391, 1.06012893, 1.53382215, 2.3336855, 2.57559099, 2.79544339, 3.2186541, 3.96466492, 4.88161637, 6.35588674, 8.30185038])
        # 14 anchors kmeans
        #anchor_radius = np.array([0.47202531, 0.70919491, 0.98827069, 1.19303585, 1.61575141, 2.16229479, 2.44651622, 2.63216558, 2.84913011, 3.27342676, 4.0436326, 4.98453866, 6.42535928, 8.33202872])
        
        # only use lidar points
        point_set = lidar_points[:,0:3]
        # use both lidar and radar points
        #point_set = np.concatenate((lidar_points[:,0:3],radar_points[:,0:3]),0)
        
        '''
        points = np.expand_dims(point_set, axis=1)
        points = points.repeat(repeats=bounding_spheres.shape[0], axis=1) #shape [npoints, bsphere_count, 3]
        sphere_centers = np.expand_dims(bounding_spheres[:,0:3], axis=0)
        sphere_centers = sphere_centers.repeat(repeats=point_set.shape[0], axis=0) #shape [npoints, bsphere_count, 3]
        # compute distance between every point and every object center
        dist_table = np.linalg.norm(points-sphere_centers, axis=2) #shape [npoints, bsphere_count]
        # points that out of all bounding spheres do not count to the loss
        vote_mask = (np.sum(dist_table <= bounding_spheres[:,3],axis=1)>0)*1.0
        if np.sum(vote_mask)==0:
            #print('No point in any object bounding box!')
            return self.__getitem__(index-1)
        # mask out "background" points and their distance
        dist_table[dist_table > bounding_spheres[:,3]] = 100000.0
        # find the sphere center that is nearist to the points
        idx = np.argmin(dist_table,axis=1)
        vote_label = bounding_spheres[idx,0:3] - point_set
        vote_label[vote_mask==0,:] = 0.0
        '''
        vote_mask = np.zeros(point_set.shape[0])
        vote_label = np.zeros((point_set.shape[0],3))
        #sem_label = np.zeros(point_set.shape[0])
        for i in range(bounding_boxes_corners.shape[0]):
            inds_in_box = in_hull(point_set, bounding_boxes_corners[i,:,:])
            vote_mask[inds_in_box] = 1.0
            vote_label[inds_in_box,:] = bounding_spheres[i, 0:3] - point_set[inds_in_box,:]
            #sem_label[inds_in_box] = bounding_spheres[i,-1]
        if np.sum(vote_mask)==0:
            #print('No point in any object bounding box!')
            return self.__getitem__(index-1)        
               
        return lidar_points, radar_points, bounding_boxes, bounding_spheres, vote_label, vote_mask, image, intrinsic, pose
    
    def __len__(self):
        return len(self.sample)
        
    def resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)
        return image
        
    def load_image(self, file, image_dims):
        image = misc.imread(file)
        # preprocess
        image = self.resize_crop_image(image, image_dims)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=[0.290101, 0.328081, 0.286964], std=[0.182954, 0.186566, 0.184475])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise
        return image
    
    def collate_fn(self, batch):
        '''
        Since each point set may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        param batch: an iterable of N sets from __getitem__()
        return:
        a tensor of point clouds
        a tensor of semantic labels
        a tensor of instance labels
        a list of varying-size tensors of bounding spheres
        a tensor of sample weights
        '''
        lidar_points_list = []
        radar_points_list = []
        bounding_boxes_list = []
        bounding_spheres_list = []
        vote_label_list = []
        vote_mask_list = []
        image_list = []
        intrinsic_list = []
        pose_list = []
        for b in batch:
            lidar_points_list.append(torch.from_numpy(b[0]))
            radar_points_list.append(torch.from_numpy(b[1]))
            bounding_boxes_list.append(torch.from_numpy(b[2]))
            bounding_spheres_list.append(torch.from_numpy(b[3]))
            vote_label_list.append(torch.from_numpy(b[4]))
            vote_mask_list.append(torch.from_numpy(b[5]))
            image_list.append(torch.from_numpy(b[6]))
            intrinsic_list.append(torch.from_numpy(b[7]))
            pose_list.append(torch.from_numpy(b[8]))
        lidar_points = torch.stack(lidar_points_list, dim=0) #shape: [batchsize, npoints, 5]
        radar_points = torch.stack(radar_points_list, dim=0) #shape: [batchsize, npoints, 6]
        vote_labels = torch.stack(vote_label_list, dim=0) #shape: [batchsize, npoints, 3]
        vote_masks = torch.stack(vote_mask_list, dim=0) #shape: [batchsize, npoints]
        # return tensor, tensor, tensor, list of N tensors, tensor
        return lidar_points, radar_points, bounding_boxes_list, bounding_spheres_list, vote_labels, vote_masks, image_list, intrinsic_list, pose_list      

class NuscenesDatasetSemseg(Dataset):
    def __init__(self, nusc, npoints_lidar=16384, npoints_radar=2048, split='train_detect'):
        self.npoints_lidar = npoints_lidar
        self.npoints_radar = npoints_radar
        self.split = split
        self.nusc = nusc
        split_dict = create_splits_scenes()
        self.NameMapping = {
            'movable_object.barrier': 'barrier',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }
        self.LabelMapping = {
            'barrier': 1,
            'bicycle': 2,
            'bus': 3,
            'car': 4,
            'construction_vehicle': 5,
            'motorcycle': 6,
            'pedestrian': 7,
            'traffic_cone': 8,
            'trailer': 9,
            'truck':10
        }
        self.MaxDistMapping = {
            1: 30.0,
            2: 40.0,
            3: 50.0,
            4: 50.0,
            5: 50.0,
            6: 40.0,
            7: 40.0,
            8: 30.0,
            9: 50.0,
            10: 50.0
        }        
        self.DefaultAttribute = {
            "car": "vehicle.parked",
            "pedestrian": "pedestrian.moving",
            "trailer": "vehicle.parked",
            "truck": "vehicle.parked",
            "bus": "vehicle.parked",
            "motorcycle": "cycle.without_rider",
            "construction_vehicle": "vehicle.parked",
            "bicycle": "cycle.without_rider",
            "barrier": "",
            "traffic_cone": ""
        }
        self.scene = []
        for s in self.nusc.scene:
            if s['name'] in split_dict[split]:
                self.scene.append(s)
        self.sample = []
        for s in self.scene:
            sample = self.nusc.get('sample', s['first_sample_token'])
            self.sample.append(sample)
            while not sample['next'] == "":
                sample = self.nusc.get('sample', sample['next'])
                self.sample.append(sample)
        print(len(self.scene))
        print(len(self.sample))
    
    def __getitem__(self, index):
        sample = self.sample[index]
        # get box list, filter objects and transform to np array
        _, BoxList, _ = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        gt_BoxList = []
        gt_SphereList = []
        gt_BoxCornerList = []
        for box in BoxList:
            if box.name in self.NameMapping:
                box.label = self.LabelMapping[self.NameMapping[box.name]]
                center = box.center
                if np.linalg.norm(box.center[:2]) <= self.MaxDistMapping[box.label]:
                    size = box.wlh
                    yaw = np.array([box.orientation.yaw_pitch_roll[0]])
                    velocity = box.velocity[0:2]
                    label = np.array([box.label])
                    # box encode: 0,1,2: center, 3,4,5: size(whl), 6: yaw(rad), 7,8: vx,vy, 9:label
                    new_box = np.concatenate((center, size, yaw, velocity, label))
                    gt_BoxList.append(new_box)
                    r = np.array([np.linalg.norm(size)/2.0])
                    # sphere encode: 0,1,2: center, 3: size(r), 4: yaw(rad), 5,6: vx,vy, 7:label
                    new_sphere = np.concatenate((center, r, yaw, velocity, label))
                    gt_SphereList.append(new_sphere)
                    box_corners = box.corners().T
                    gt_BoxCornerList.append(box_corners)
                #else:
                    #print('GT box out of detection range')
                    #print(np.linalg.norm(box.center[:2]))
                    #print(self.MaxDistMapping[box.label])
        bounding_boxes = np.array(gt_BoxList)
        bounding_spheres = np.array(gt_SphereList)
        bounding_boxes_corners = np.array(gt_BoxCornerList)
        if bounding_boxes.shape[0]==0:
            #print('No object in this sample')
            return self.__getitem__(index-1)
        
        # get lidar point cloud
        lidar_sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', 5)
        lidar_points = np.concatenate((pc.points,times),0).T #size [N, 5], 5 corresponds to [x,y,z,intensity,time]
        depth = np.linalg.norm(lidar_points[:,0:3], axis=1)       
        lidar_points = lidar_points[depth<52.0,:]
        if self.npoints_lidar < lidar_points.shape[0]:
            choice = np.random.choice(lidar_points.shape[0], self.npoints_lidar, replace=False)
        else:
            choice = np.arange(0, lidar_points.shape[0], dtype=np.int32)
            if self.npoints_lidar > lidar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_lidar - lidar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        lidar_points = lidar_points[choice,:] #size [npoints_lidar, 5]
        lidar_points[:,3] = lidar_points[:,3]/255.0 # scale intensity from 0-255 to 0-1
        
        '''
        # get radar point cloud
        radar_points_list = []
        radar_sensor = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT_RIGHT']
        for sensor in radar_sensor:
            radar_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            radar_cs_record = self.nusc.get('calibrated_sensor', radar_sd_record['calibrated_sensor_token'])
            pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, sensor, 'LIDAR_TOP', 6)
            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the point cloud.
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(lidar_cs_record['rotation']).rotation_matrix.T, velocities)            
            velocities[2, :] = np.zeros(pc.points.shape[1])
            radar_points = np.concatenate((pc.points[0:3,:],velocities[0:2,:],times),0).T #size [N, 6], 6 corresponds to [x,y,z,vx,vy,time]
            radar_points_list.append(radar_points)
        radar_points = np.concatenate(radar_points_list, 0)
        if self.npoints_radar < radar_points.shape[0]:
            choice = np.random.choice(radar_points.shape[0], self.npoints_radar, replace=False)
        else:
            choice = np.arange(0, radar_points.shape[0], dtype=np.int32)
            if self.npoints_radar > radar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_radar - radar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        radar_points = radar_points[choice,:] #size [npoints_radar, 5]
        
        # get images, camera intrinsic, camera pose to lidar coordinate system
        images = []
        poses = []
        intrinsics = []
        camera_sensor = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        for sensor in camera_sensor:
            cam_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            cam_cs_record = self.nusc.get('calibrated_sensor', cam_sd_record['calibrated_sensor_token'])
            img_path = os.path.join(self.nusc.dataroot, cam_sd_record['filename'])
            img = self.load_image(img_path, [640,360]) #[3,360,640], feature map will be [128,45,80]
            images.append(img)
            
            cam_intrinsic = np.eye(4)
            cam_intrinsic[0:3,0:3] = np.array(cam_cs_record['camera_intrinsic'])
            # adjust intrinsic for the size of feature map
            cam_intrinsic[0,0] *= float(80)/float(1600)
            cam_intrinsic[1,1] *= float(45)/float(900)
            cam_intrinsic[0,2] *= float(80-1)/float(1600-1)
            cam_intrinsic[1,2] *= float(45-1)/float(900-1)
            intrinsics.append(cam_intrinsic)
            
            # compute camera pose
            cam_to_ego = np.eye(4)
            cam_to_ego[0:3,0:3] = Quaternion(cam_cs_record['rotation']).rotation_matrix
            cam_to_ego[0:3,3] = np.array(cam_cs_record['translation'])            
            poserecord = self.nusc.get('ego_pose', cam_sd_record['ego_pose_token'])
            ego_to_global = np.eye(4)
            ego_to_global[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix
            ego_to_global[0:3,3] = np.array(poserecord['translation'])            
            poserecord = self.nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
            global_to_ego = np.eye(4)
            global_to_ego[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix.T
            global_to_ego[0:3,3] = global_to_ego[0:3,0:3].dot(-np.array(poserecord['translation']))            
            ego_to_lidar = np.eye(4)
            ego_to_lidar[0:3,0:3] = Quaternion(lidar_cs_record['rotation']).rotation_matrix.T
            ego_to_lidar[0:3,3] = ego_to_lidar[0:3,0:3].dot(-np.array(lidar_cs_record['translation']))            
            cam_to_lidar = ego_to_lidar.dot(global_to_ego).dot(ego_to_global).dot(cam_to_ego)
            poses.append(cam_to_lidar)
        image = np.stack(images,0) #[num_images=6,3,360,640]
        intrinsic = np.array(intrinsics) #[num_images=6,4,4]
        pose = np.array(poses).astype(np.float32) #[num_images=6,4,4]
        '''
        radar_points = np.array([])
        image = np.array([])
        intrinsic = np.array([])
        pose = np.array([])
        
        # generate anchors, shape: [npoints*anchors per point, 4]

        # 10 anchors kmeans
        #anchor_radius = np.array([0.59519359, 1.04024921, 1.51894038, 2.33694095, 2.58570698, 2.82699181, 3.34568567, 4.45521787, 6.1408409, 8.17982825])
        # 12 anchors kmeans
        #anchor_radius = np.array([0.47506766, 0.73028391, 1.06012893, 1.53382215, 2.3336855, 2.57559099, 2.79544339, 3.2186541, 3.96466492, 4.88161637, 6.35588674, 8.30185038])
        # 14 anchors kmeans
        #anchor_radius = np.array([0.47202531, 0.70919491, 0.98827069, 1.19303585, 1.61575141, 2.16229479, 2.44651622, 2.63216558, 2.84913011, 3.27342676, 4.0436326, 4.98453866, 6.42535928, 8.33202872])
        
        # only use lidar points
        point_set = lidar_points[:,0:3]
        # use both lidar and radar points
        #point_set = np.concatenate((lidar_points[:,0:3],radar_points[:,0:3]),0)
        
        '''
        points = np.expand_dims(point_set, axis=1)
        points = points.repeat(repeats=bounding_spheres.shape[0], axis=1) #shape [npoints, bsphere_count, 3]
        sphere_centers = np.expand_dims(bounding_spheres[:,0:3], axis=0)
        sphere_centers = sphere_centers.repeat(repeats=point_set.shape[0], axis=0) #shape [npoints, bsphere_count, 3]
        # compute distance between every point and every object center
        dist_table = np.linalg.norm(points-sphere_centers, axis=2) #shape [npoints, bsphere_count]
        # points that out of all bounding spheres do not count to the loss
        vote_mask = (np.sum(dist_table <= bounding_spheres[:,3],axis=1)>0)*1.0
        if np.sum(vote_mask)==0:
            #print('No point in any object bounding box!')
            return self.__getitem__(index-1)
        # mask out "background" points and their distance
        dist_table[dist_table > bounding_spheres[:,3]] = 100000.0
        # find the sphere center that is nearist to the points
        idx = np.argmin(dist_table,axis=1)
        sem_label = bounding_spheres[idx,-1]
        sem_label[vote_mask==0] = 0.0
        '''
        
        vote_mask = np.zeros(point_set.shape[0])
        vote_label = np.zeros((point_set.shape[0],3))
        sem_label = np.zeros(point_set.shape[0])
        for i in range(bounding_boxes_corners.shape[0]):
            inds_in_box = in_hull(point_set, bounding_boxes_corners[i,:,:])
            vote_mask[inds_in_box] = 1.0
            vote_label[inds_in_box,:] = bounding_spheres[i, 0:3] - point_set[inds_in_box,:]
            sem_label[inds_in_box] = bounding_spheres[i,-1]
        if np.sum(vote_mask)==0:
            #print('No point in any object bounding box!')
            return self.__getitem__(index-1)
               
        return lidar_points, radar_points, bounding_boxes, bounding_spheres, sem_label, image, intrinsic, pose
    
    def __len__(self):
        return len(self.sample)
        
    def resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)
        return image
        
    def load_image(self, file, image_dims):
        image = misc.imread(file)
        # preprocess
        image = self.resize_crop_image(image, image_dims)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=[0.290101, 0.328081, 0.286964], std=[0.182954, 0.186566, 0.184475])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise
        return image
    
    def collate_fn(self, batch):
        '''
        Since each point set may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        param batch: an iterable of N sets from __getitem__()
        return:
        a tensor of point clouds
        a tensor of semantic labels
        a tensor of instance labels
        a list of varying-size tensors of bounding spheres
        a tensor of sample weights
        '''
        lidar_points_list = []
        radar_points_list = []
        bounding_boxes_list = []
        bounding_spheres_list = []
        sem_label_list = []
        image_list = []
        intrinsic_list = []
        pose_list = []
        for b in batch:
            lidar_points_list.append(torch.from_numpy(b[0]))
            radar_points_list.append(torch.from_numpy(b[1]))
            bounding_boxes_list.append(torch.from_numpy(b[2]))
            bounding_spheres_list.append(torch.from_numpy(b[3]))
            sem_label_list.append(torch.from_numpy(b[4]))
            image_list.append(torch.from_numpy(b[5]))
            intrinsic_list.append(torch.from_numpy(b[6]))
            pose_list.append(torch.from_numpy(b[7]))
        lidar_points = torch.stack(lidar_points_list, dim=0) #shape: [batchsize, npoints, 5]
        radar_points = torch.stack(radar_points_list, dim=0) #shape: [batchsize, npoints, 6]
        sem_labels = torch.stack(sem_label_list, dim=0) #shape: [batchsize, npoints]
        # return tensor, tensor, tensor, list of N tensors, tensor
        return lidar_points, radar_points, bounding_boxes_list, bounding_spheres_list, sem_labels, image_list, intrinsic_list, pose_list        

class NuscenesDatasetSemsegLR(Dataset):
    def __init__(self, nusc, npoints_lidar=16384, npoints_radar=1024, split='train_detect'):
        self.npoints_lidar = npoints_lidar
        self.npoints_radar = npoints_radar
        self.split = split
        self.nusc = nusc
        split_dict = create_splits_scenes()
        self.NameMapping = {
            'movable_object.barrier': 'barrier',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }
        self.LabelMapping = {
            'barrier': 1,
            'bicycle': 2,
            'bus': 3,
            'car': 4,
            'construction_vehicle': 5,
            'motorcycle': 6,
            'pedestrian': 7,
            'traffic_cone': 8,
            'trailer': 9,
            'truck':10
        }
        self.MaxDistMapping = {
            1: 30.0,
            2: 40.0,
            3: 50.0,
            4: 50.0,
            5: 50.0,
            6: 40.0,
            7: 40.0,
            8: 30.0,
            9: 50.0,
            10: 50.0
        }        
        self.DefaultAttribute = {
            "car": "vehicle.parked",
            "pedestrian": "pedestrian.moving",
            "trailer": "vehicle.parked",
            "truck": "vehicle.parked",
            "bus": "vehicle.parked",
            "motorcycle": "cycle.without_rider",
            "construction_vehicle": "vehicle.parked",
            "bicycle": "cycle.without_rider",
            "barrier": "",
            "traffic_cone": ""
        }
        self.scene = []
        for s in self.nusc.scene:
            if s['name'] in split_dict[split]:
                self.scene.append(s)
        self.sample = []
        for s in self.scene:
            sample = self.nusc.get('sample', s['first_sample_token'])
            self.sample.append(sample)
            while not sample['next'] == "":
                sample = self.nusc.get('sample', sample['next'])
                self.sample.append(sample)
        print(len(self.scene))
        print(len(self.sample))
    
    def __getitem__(self, index):
        sample = self.sample[index]
        # get box list, filter objects and transform to np array
        _, BoxList, _ = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        gt_BoxList = []
        gt_SphereList = []
        gt_BoxCornerList = []
        for box in BoxList:
            if box.name in self.NameMapping:
                box.label = self.LabelMapping[self.NameMapping[box.name]]
                center = box.center
                if np.linalg.norm(box.center[:2]) <= self.MaxDistMapping[box.label]:
                    size = box.wlh
                    yaw = np.array([box.orientation.yaw_pitch_roll[0]])
                    velocity = box.velocity[0:2]
                    label = np.array([box.label])
                    # box encode: 0,1,2: center, 3,4,5: size(whl), 6: yaw(rad), 7,8: vx,vy, 9:label
                    new_box = np.concatenate((center, size, yaw, velocity, label))
                    gt_BoxList.append(new_box)
                    r = np.array([np.linalg.norm(size)/2.0])
                    # sphere encode: 0,1,2: center, 3: size(r), 4: yaw(rad), 5,6: vx,vy, 7:label
                    new_sphere = np.concatenate((center, r, yaw, velocity, label))
                    gt_SphereList.append(new_sphere)
                    box_corners = box.corners().T
                    gt_BoxCornerList.append(box_corners)                    
                #else:
                    #print('GT box out of detection range')
                    #print(np.linalg.norm(box.center[:2]))
                    #print(self.MaxDistMapping[box.label])
        bounding_boxes = np.array(gt_BoxList)
        bounding_spheres = np.array(gt_SphereList)
        bounding_boxes_corners = np.array(gt_BoxCornerList)
        if bounding_boxes.shape[0]==0:
            #print('No object in this sample')
            return self.__getitem__(index-1)
        
        # get lidar point cloud
        lidar_sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', 5)
        lidar_points = np.concatenate((pc.points,times),0).T #size [N, 5], 5 corresponds to [x,y,z,intensity,time]
        depth = np.linalg.norm(lidar_points[:,0:3], axis=1)       
        lidar_points = lidar_points[depth<52.0,:]
        if self.npoints_lidar < lidar_points.shape[0]:
            choice = np.random.choice(lidar_points.shape[0], self.npoints_lidar, replace=False)
        else:
            choice = np.arange(0, lidar_points.shape[0], dtype=np.int32)
            if self.npoints_lidar > lidar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_lidar - lidar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        lidar_points = lidar_points[choice,:] #size [npoints_lidar, 5]
        lidar_points[:,3] = lidar_points[:,3]/255.0 # scale intensity from 0-255 to 0-1
        
        
        # get radar point cloud
        radar_points_list = []
        radar_sensor = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT_RIGHT']
        for sensor in radar_sensor:
            radar_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            radar_cs_record = self.nusc.get('calibrated_sensor', radar_sd_record['calibrated_sensor_token'])
            pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, sensor, 'LIDAR_TOP', 6)
            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the point cloud.
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(lidar_cs_record['rotation']).rotation_matrix.T, velocities)            
            velocities[2, :] = np.zeros(pc.points.shape[1])
            radar_points = np.concatenate((pc.points[0:3,:],velocities[0:2,:],times),0).T #size [N, 6], 6 corresponds to [x,y,z,vx,vy,time]
            radar_points_list.append(radar_points)
        radar_points = np.concatenate(radar_points_list, 0)
        depth = np.linalg.norm(radar_points[:,0:3], axis=1)
        radar_points = radar_points[depth<52.0,:]
        if self.npoints_radar < radar_points.shape[0]:
            choice = np.random.choice(radar_points.shape[0], self.npoints_radar, replace=False)
        else:
            choice = np.arange(0, radar_points.shape[0], dtype=np.int32)
            if self.npoints_radar > radar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_radar - radar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        radar_points = radar_points[choice,:] #size [npoints_radar, 6]
        
        '''
        # get images, camera intrinsic, camera pose to lidar coordinate system
        images = []
        poses = []
        intrinsics = []
        camera_sensor = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        for sensor in camera_sensor:
            cam_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            cam_cs_record = self.nusc.get('calibrated_sensor', cam_sd_record['calibrated_sensor_token'])
            img_path = os.path.join(self.nusc.dataroot, cam_sd_record['filename'])
            img = self.load_image(img_path, [640,360]) #[3,360,640], feature map will be [128,45,80]
            images.append(img)
            
            cam_intrinsic = np.eye(4)
            cam_intrinsic[0:3,0:3] = np.array(cam_cs_record['camera_intrinsic'])
            # adjust intrinsic for the size of feature map
            cam_intrinsic[0,0] *= float(80)/float(1600)
            cam_intrinsic[1,1] *= float(45)/float(900)
            cam_intrinsic[0,2] *= float(80-1)/float(1600-1)
            cam_intrinsic[1,2] *= float(45-1)/float(900-1)
            intrinsics.append(cam_intrinsic)
            
            # compute camera pose
            cam_to_ego = np.eye(4)
            cam_to_ego[0:3,0:3] = Quaternion(cam_cs_record['rotation']).rotation_matrix
            cam_to_ego[0:3,3] = np.array(cam_cs_record['translation'])            
            poserecord = self.nusc.get('ego_pose', cam_sd_record['ego_pose_token'])
            ego_to_global = np.eye(4)
            ego_to_global[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix
            ego_to_global[0:3,3] = np.array(poserecord['translation'])            
            poserecord = self.nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
            global_to_ego = np.eye(4)
            global_to_ego[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix.T
            global_to_ego[0:3,3] = global_to_ego[0:3,0:3].dot(-np.array(poserecord['translation']))            
            ego_to_lidar = np.eye(4)
            ego_to_lidar[0:3,0:3] = Quaternion(lidar_cs_record['rotation']).rotation_matrix.T
            ego_to_lidar[0:3,3] = ego_to_lidar[0:3,0:3].dot(-np.array(lidar_cs_record['translation']))            
            cam_to_lidar = ego_to_lidar.dot(global_to_ego).dot(ego_to_global).dot(cam_to_ego)
            poses.append(cam_to_lidar)
        image = np.stack(images,0) #[num_images=6,3,360,640]
        intrinsic = np.array(intrinsics) #[num_images=6,4,4]
        pose = np.array(poses).astype(np.float32) #[num_images=6,4,4]
        '''
        #radar_points = np.array([])
        image = np.array([])
        intrinsic = np.array([])
        pose = np.array([])
        
        # generate anchors, shape: [npoints*anchors per point, 4]

        # 10 anchors kmeans
        #anchor_radius = np.array([0.59519359, 1.04024921, 1.51894038, 2.33694095, 2.58570698, 2.82699181, 3.34568567, 4.45521787, 6.1408409, 8.17982825])
        # 12 anchors kmeans
        #anchor_radius = np.array([0.47506766, 0.73028391, 1.06012893, 1.53382215, 2.3336855, 2.57559099, 2.79544339, 3.2186541, 3.96466492, 4.88161637, 6.35588674, 8.30185038])
        # 14 anchors kmeans
        #anchor_radius = np.array([0.47202531, 0.70919491, 0.98827069, 1.19303585, 1.61575141, 2.16229479, 2.44651622, 2.63216558, 2.84913011, 3.27342676, 4.0436326, 4.98453866, 6.42535928, 8.33202872])
        
        # only use lidar points
        point_set = lidar_points[:,0:3]
        # use both lidar and radar points
        #point_set = np.concatenate((lidar_points[:,0:3],radar_points[:,0:3]),0)
        
        '''
        points = np.expand_dims(point_set, axis=1)
        points = points.repeat(repeats=bounding_spheres.shape[0], axis=1) #shape [npoints, bsphere_count, 3]
        sphere_centers = np.expand_dims(bounding_spheres[:,0:3], axis=0)
        sphere_centers = sphere_centers.repeat(repeats=point_set.shape[0], axis=0) #shape [npoints, bsphere_count, 3]
        # compute distance between every point and every object center
        dist_table = np.linalg.norm(points-sphere_centers, axis=2) #shape [npoints, bsphere_count]
        # points that out of all bounding spheres do not count to the loss
        vote_mask = (np.sum(dist_table <= bounding_spheres[:,3],axis=1)>0)*1.0
        if np.sum(vote_mask)==0:
            #print('No point in any object bounding box!')
            return self.__getitem__(index-1)
        # mask out "background" points and their distance
        dist_table[dist_table > bounding_spheres[:,3]] = 100000.0
        # find the sphere center that is nearist to the points
        idx = np.argmin(dist_table,axis=1)
        sem_label = bounding_spheres[idx,-1]
        sem_label[vote_mask==0] = 0.0
        '''
        
        vote_mask = np.zeros(point_set.shape[0])
        vote_label = np.zeros((point_set.shape[0],3))
        sem_label = np.zeros(point_set.shape[0])
        for i in range(bounding_boxes_corners.shape[0]):
            inds_in_box = in_hull(point_set, bounding_boxes_corners[i,:,:])
            vote_mask[inds_in_box] = 1.0
            vote_label[inds_in_box,:] = bounding_spheres[i, 0:3] - point_set[inds_in_box,:]
            sem_label[inds_in_box] = bounding_spheres[i,-1]
        if np.sum(vote_mask)==0:
            #print('No point in any object bounding box!')
            return self.__getitem__(index-1)        
               
        return lidar_points, radar_points, bounding_boxes, bounding_spheres, sem_label, image, intrinsic, pose
    
    def __len__(self):
        return len(self.sample)
        
    def resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)
        return image
        
    def load_image(self, file, image_dims):
        image = misc.imread(file)
        # preprocess
        image = self.resize_crop_image(image, image_dims)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=[0.290101, 0.328081, 0.286964], std=[0.182954, 0.186566, 0.184475])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise
        return image
    
    def collate_fn(self, batch):
        '''
        Since each point set may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        param batch: an iterable of N sets from __getitem__()
        return:
        a tensor of point clouds
        a tensor of semantic labels
        a tensor of instance labels
        a list of varying-size tensors of bounding spheres
        a tensor of sample weights
        '''
        lidar_points_list = []
        radar_points_list = []
        bounding_boxes_list = []
        bounding_spheres_list = []
        sem_label_list = []
        image_list = []
        intrinsic_list = []
        pose_list = []
        for b in batch:
            lidar_points_list.append(torch.from_numpy(b[0]))
            radar_points_list.append(torch.from_numpy(b[1]))
            bounding_boxes_list.append(torch.from_numpy(b[2]))
            bounding_spheres_list.append(torch.from_numpy(b[3]))
            sem_label_list.append(torch.from_numpy(b[4]))
            image_list.append(torch.from_numpy(b[5]))
            intrinsic_list.append(torch.from_numpy(b[6]))
            pose_list.append(torch.from_numpy(b[7]))
        lidar_points = torch.stack(lidar_points_list, dim=0) #shape: [batchsize, npoints, 5]
        radar_points = torch.stack(radar_points_list, dim=0) #shape: [batchsize, npoints, 6]
        sem_labels = torch.stack(sem_label_list, dim=0) #shape: [batchsize, npoints]
        # return tensor, tensor, tensor, list of N tensors, tensor
        return lidar_points, radar_points, bounding_boxes_list, bounding_spheres_list, sem_labels, image_list, intrinsic_list, pose_list        

class NuscenesDatasetSemsegLRGB(Dataset):
    def __init__(self, nusc, npoints_lidar=16384, npoints_radar=1024, split='train_detect'):
        self.npoints_lidar = npoints_lidar
        self.npoints_radar = npoints_radar
        self.split = split
        self.nusc = nusc
        split_dict = create_splits_scenes()
        self.NameMapping = {
            'movable_object.barrier': 'barrier',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }
        self.LabelMapping = {
            'barrier': 1,
            'bicycle': 2,
            'bus': 3,
            'car': 4,
            'construction_vehicle': 5,
            'motorcycle': 6,
            'pedestrian': 7,
            'traffic_cone': 8,
            'trailer': 9,
            'truck':10
        }
        self.MaxDistMapping = {
            1: 30.0,
            2: 40.0,
            3: 50.0,
            4: 50.0,
            5: 50.0,
            6: 40.0,
            7: 40.0,
            8: 30.0,
            9: 50.0,
            10: 50.0
        }        
        self.DefaultAttribute = {
            "car": "vehicle.parked",
            "pedestrian": "pedestrian.moving",
            "trailer": "vehicle.parked",
            "truck": "vehicle.parked",
            "bus": "vehicle.parked",
            "motorcycle": "cycle.without_rider",
            "construction_vehicle": "vehicle.parked",
            "bicycle": "cycle.without_rider",
            "barrier": "",
            "traffic_cone": ""
        }
        self.scene = []
        for s in self.nusc.scene:
            if s['name'] in split_dict[split]:
                self.scene.append(s)
        self.sample = []
        for s in self.scene:
            sample = self.nusc.get('sample', s['first_sample_token'])
            self.sample.append(sample)
            while not sample['next'] == "":
                sample = self.nusc.get('sample', sample['next'])
                self.sample.append(sample)
        print(len(self.scene))
        print(len(self.sample))
    
    def __getitem__(self, index):
        sample = self.sample[index]
        # get box list, filter objects and transform to np array
        _, BoxList, _ = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        gt_BoxList = []
        gt_SphereList = []
        gt_BoxCornerList = []
        for box in BoxList:
            if box.name in self.NameMapping:
                box.label = self.LabelMapping[self.NameMapping[box.name]]
                center = box.center
                if np.linalg.norm(box.center[:2]) <= self.MaxDistMapping[box.label]:
                    size = box.wlh
                    yaw = np.array([box.orientation.yaw_pitch_roll[0]])
                    velocity = box.velocity[0:2]
                    label = np.array([box.label])
                    # box encode: 0,1,2: center, 3,4,5: size(whl), 6: yaw(rad), 7,8: vx,vy, 9:label
                    new_box = np.concatenate((center, size, yaw, velocity, label))
                    gt_BoxList.append(new_box)
                    r = np.array([np.linalg.norm(size)/2.0])
                    # sphere encode: 0,1,2: center, 3: size(r), 4: yaw(rad), 5,6: vx,vy, 7:label
                    new_sphere = np.concatenate((center, r, yaw, velocity, label))
                    gt_SphereList.append(new_sphere)
                    box_corners = box.corners().T
                    gt_BoxCornerList.append(box_corners)                    
                #else:
                    #print('GT box out of detection range')
                    #print(np.linalg.norm(box.center[:2]))
                    #print(self.MaxDistMapping[box.label])
        bounding_boxes = np.array(gt_BoxList)
        bounding_spheres = np.array(gt_SphereList)
        bounding_boxes_corners = np.array(gt_BoxCornerList)
        if bounding_boxes.shape[0]==0:
            #print('No object in this sample')
            return self.__getitem__(index-1)
        
        # get lidar point cloud
        lidar_sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', 5)
        lidar_points = np.concatenate((pc.points,times),0).T #size [N, 5], 5 corresponds to [x,y,z,intensity,time]
        depth = np.linalg.norm(lidar_points[:,0:3], axis=1)       
        lidar_points = lidar_points[depth<52.0,:]
        if self.npoints_lidar < lidar_points.shape[0]:
            choice = np.random.choice(lidar_points.shape[0], self.npoints_lidar, replace=False)
        else:
            choice = np.arange(0, lidar_points.shape[0], dtype=np.int32)
            if self.npoints_lidar > lidar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_lidar - lidar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        lidar_points = lidar_points[choice,:] #size [npoints_lidar, 5]
        lidar_points[:,3] = lidar_points[:,3]/255.0 # scale intensity from 0-255 to 0-1
        
        '''
        # get radar point cloud
        radar_points_list = []
        radar_sensor = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT_RIGHT']
        for sensor in radar_sensor:
            radar_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            radar_cs_record = self.nusc.get('calibrated_sensor', radar_sd_record['calibrated_sensor_token'])
            pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, sensor, 'LIDAR_TOP', 6)
            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the point cloud.
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(lidar_cs_record['rotation']).rotation_matrix.T, velocities)            
            velocities[2, :] = np.zeros(pc.points.shape[1])
            radar_points = np.concatenate((pc.points[0:3,:],velocities[0:2,:],times),0).T #size [N, 6], 6 corresponds to [x,y,z,vx,vy,time]
            radar_points_list.append(radar_points)
        radar_points = np.concatenate(radar_points_list, 0)
        depth = np.linalg.norm(radar_points[:,0:3], axis=1)
        radar_points = radar_points[depth<52.0,:]
        if self.npoints_radar < radar_points.shape[0]:
            choice = np.random.choice(radar_points.shape[0], self.npoints_radar, replace=False)
        else:
            choice = np.arange(0, radar_points.shape[0], dtype=np.int32)
            if self.npoints_radar > radar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_radar - radar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        radar_points = radar_points[choice,:] #size [npoints_radar, 6]
        '''
        
        # get images, camera intrinsic, camera pose to lidar coordinate system
        images = []
        #poses = []
        #intrinsics = []
        proj_mapping = []
        camera_sensor = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        for sensor in camera_sensor:
            cam_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            cam_cs_record = self.nusc.get('calibrated_sensor', cam_sd_record['calibrated_sensor_token'])
            img_path = os.path.join(self.nusc.dataroot, cam_sd_record['filename'])
            img = self.load_image(img_path, [640,360]) #[3,360,640], feature map will be [128,45,80]
            images.append(img)
            
            cam_intrinsic = np.eye(4)
            cam_intrinsic[0:3,0:3] = np.array(cam_cs_record['camera_intrinsic'])
            # adjust intrinsic for the size of feature map
            cam_intrinsic[0,0] *= float(80)/float(1600)
            cam_intrinsic[1,1] *= float(45)/float(900)
            cam_intrinsic[0,2] *= float(80-1)/float(1600-1)
            cam_intrinsic[1,2] *= float(45-1)/float(900-1)
            #intrinsics.append(cam_intrinsic)
            projection = ProjectionHelper(torch.from_numpy(cam_intrinsic), 0.5, 60.0, [80,45], 0.05)
            
            # compute camera pose
            cam_to_ego = np.eye(4)
            cam_to_ego[0:3,0:3] = Quaternion(cam_cs_record['rotation']).rotation_matrix
            cam_to_ego[0:3,3] = np.array(cam_cs_record['translation'])            
            poserecord = self.nusc.get('ego_pose', cam_sd_record['ego_pose_token'])
            ego_to_global = np.eye(4)
            ego_to_global[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix
            ego_to_global[0:3,3] = np.array(poserecord['translation'])            
            poserecord = self.nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
            global_to_ego = np.eye(4)
            global_to_ego[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix.T
            global_to_ego[0:3,3] = global_to_ego[0:3,0:3].dot(-np.array(poserecord['translation']))            
            ego_to_lidar = np.eye(4)
            ego_to_lidar[0:3,0:3] = Quaternion(lidar_cs_record['rotation']).rotation_matrix.T
            ego_to_lidar[0:3,3] = ego_to_lidar[0:3,0:3].dot(-np.array(lidar_cs_record['translation']))            
            cam_to_lidar = ego_to_lidar.dot(global_to_ego).dot(ego_to_global).dot(cam_to_ego)
            #poses.append(cam_to_lidar)
            proj_mapping.append(projection.compute_projection(torch.from_numpy(lidar_points[:,0:3]), torch.from_numpy(cam_to_lidar), self.npoints_lidar))
            
        if None in proj_mapping:
            return self.__getitem__(index-1)
        image = np.stack(images,0) #[num_images=6,3,360,640]
        #intrinsic = np.array(intrinsics) #[num_images=6,4,4]
        #pose = np.array(poses).astype(np.float32) #[num_images=6,4,4]
        proj_mapping0, proj_mapping1 = zip(*proj_mapping)
        proj_ind_3d = torch.stack(proj_mapping0)
        proj_ind_2d = torch.stack(proj_mapping1)
        
        radar_points = np.array([])
        #image = np.array([])
        #intrinsic = np.array([])
        #pose = np.array([])
        
        # generate anchors, shape: [npoints*anchors per point, 4]

        # 10 anchors kmeans
        #anchor_radius = np.array([0.59519359, 1.04024921, 1.51894038, 2.33694095, 2.58570698, 2.82699181, 3.34568567, 4.45521787, 6.1408409, 8.17982825])
        # 12 anchors kmeans
        #anchor_radius = np.array([0.47506766, 0.73028391, 1.06012893, 1.53382215, 2.3336855, 2.57559099, 2.79544339, 3.2186541, 3.96466492, 4.88161637, 6.35588674, 8.30185038])
        # 14 anchors kmeans
        #anchor_radius = np.array([0.47202531, 0.70919491, 0.98827069, 1.19303585, 1.61575141, 2.16229479, 2.44651622, 2.63216558, 2.84913011, 3.27342676, 4.0436326, 4.98453866, 6.42535928, 8.33202872])
        
        # only use lidar points
        point_set = lidar_points[:,0:3]
        # use both lidar and radar points
        #point_set = np.concatenate((lidar_points[:,0:3],radar_points[:,0:3]),0)
        
        '''
        points = np.expand_dims(point_set, axis=1)
        points = points.repeat(repeats=bounding_spheres.shape[0], axis=1) #shape [npoints, bsphere_count, 3]
        sphere_centers = np.expand_dims(bounding_spheres[:,0:3], axis=0)
        sphere_centers = sphere_centers.repeat(repeats=point_set.shape[0], axis=0) #shape [npoints, bsphere_count, 3]
        # compute distance between every point and every object center
        dist_table = np.linalg.norm(points-sphere_centers, axis=2) #shape [npoints, bsphere_count]
        # points that out of all bounding spheres do not count to the loss
        vote_mask = (np.sum(dist_table <= bounding_spheres[:,3],axis=1)>0)*1.0
        if np.sum(vote_mask)==0:
            #print('No point in any object bounding box!')
            return self.__getitem__(index-1)
        # mask out "background" points and their distance
        dist_table[dist_table > bounding_spheres[:,3]] = 100000.0
        # find the sphere center that is nearist to the points
        idx = np.argmin(dist_table,axis=1)
        sem_label = bounding_spheres[idx,-1]
        sem_label[vote_mask==0] = 0.0
        '''
        vote_mask = np.zeros(point_set.shape[0])
        vote_label = np.zeros((point_set.shape[0],3))
        sem_label = np.zeros(point_set.shape[0])
        for i in range(bounding_boxes_corners.shape[0]):
            inds_in_box = in_hull(point_set, bounding_boxes_corners[i,:,:])
            vote_mask[inds_in_box] = 1.0
            vote_label[inds_in_box,:] = bounding_spheres[i, 0:3] - point_set[inds_in_box,:]
            sem_label[inds_in_box] = bounding_spheres[i,-1]
        if np.sum(vote_mask)==0:
            #print('No point in any object bounding box!')
            return self.__getitem__(index-1)        
               
        return lidar_points, radar_points, bounding_boxes, bounding_spheres, sem_label, image, proj_ind_3d, proj_ind_2d
    
    def __len__(self):
        return len(self.sample)
        
    def resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)
        return image
        
    def load_image(self, file, image_dims):
        image = misc.imread(file)
        # preprocess
        image = self.resize_crop_image(image, image_dims)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise
        return image
    
    def collate_fn(self, batch):
        '''
        Since each point set may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        param batch: an iterable of N sets from __getitem__()
        return:
        a tensor of point clouds
        a tensor of semantic labels
        a tensor of instance labels
        a list of varying-size tensors of bounding spheres
        a tensor of sample weights
        '''
        lidar_points_list = []
        radar_points_list = []
        bounding_boxes_list = []
        bounding_spheres_list = []
        sem_label_list = []
        image_list = []
        proj_ind_3d_list = []
        proj_ind_2d_list = []
        for b in batch:
            lidar_points_list.append(torch.from_numpy(b[0]))
            radar_points_list.append(torch.from_numpy(b[1]))
            bounding_boxes_list.append(torch.from_numpy(b[2]))
            bounding_spheres_list.append(torch.from_numpy(b[3]))
            sem_label_list.append(torch.from_numpy(b[4]))
            image_list.append(torch.from_numpy(b[5]))
            proj_ind_3d_list.append(b[6])
            proj_ind_2d_list.append(b[7])
        lidar_points = torch.stack(lidar_points_list, dim=0) #shape: [batchsize, npoints, 5]
        radar_points = torch.stack(radar_points_list, dim=0) #shape: [batchsize, npoints, 6]
        sem_labels = torch.stack(sem_label_list, dim=0) #shape: [batchsize, npoints]
        # return tensor, tensor, tensor, list of N tensors, tensor
        return lidar_points, radar_points, bounding_boxes_list, bounding_spheres_list, sem_labels, image_list, proj_ind_3d_list, proj_ind_2d_list

class NuscenesDatasetSemsegLRRGB(Dataset):
    def __init__(self, nusc, npoints_lidar=16384, npoints_radar=1024, split='train_detect'):
        self.npoints_lidar = npoints_lidar
        self.npoints_radar = npoints_radar
        self.split = split
        self.nusc = nusc
        split_dict = create_splits_scenes()
        self.NameMapping = {
            'movable_object.barrier': 'barrier',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }
        self.LabelMapping = {
            'barrier': 1,
            'bicycle': 2,
            'bus': 3,
            'car': 4,
            'construction_vehicle': 5,
            'motorcycle': 6,
            'pedestrian': 7,
            'traffic_cone': 8,
            'trailer': 9,
            'truck':10
        }
        self.MaxDistMapping = {
            1: 30.0,
            2: 40.0,
            3: 50.0,
            4: 50.0,
            5: 50.0,
            6: 40.0,
            7: 40.0,
            8: 30.0,
            9: 50.0,
            10: 50.0
        }        
        self.DefaultAttribute = {
            "car": "vehicle.parked",
            "pedestrian": "pedestrian.moving",
            "trailer": "vehicle.parked",
            "truck": "vehicle.parked",
            "bus": "vehicle.parked",
            "motorcycle": "cycle.without_rider",
            "construction_vehicle": "vehicle.parked",
            "bicycle": "cycle.without_rider",
            "barrier": "",
            "traffic_cone": ""
        }
        self.scene = []
        for s in self.nusc.scene:
            if s['name'] in split_dict[split]:
                self.scene.append(s)
        self.sample = []
        for s in self.scene:
            sample = self.nusc.get('sample', s['first_sample_token'])
            self.sample.append(sample)
            while not sample['next'] == "":
                sample = self.nusc.get('sample', sample['next'])
                self.sample.append(sample)
        print(len(self.scene))
        print(len(self.sample))
    
    def __getitem__(self, index):
        sample = self.sample[index]
        # get box list, filter objects and transform to np array
        _, BoxList, _ = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        gt_BoxList = []
        gt_SphereList = []
        gt_BoxCornerList = []
        for box in BoxList:
            if box.name in self.NameMapping:
                box.label = self.LabelMapping[self.NameMapping[box.name]]
                center = box.center
                if np.linalg.norm(box.center[:2]) <= self.MaxDistMapping[box.label]:
                    size = box.wlh
                    yaw = np.array([box.orientation.yaw_pitch_roll[0]])
                    velocity = box.velocity[0:2]
                    label = np.array([box.label])
                    # box encode: 0,1,2: center, 3,4,5: size(whl), 6: yaw(rad), 7,8: vx,vy, 9:label
                    new_box = np.concatenate((center, size, yaw, velocity, label))
                    gt_BoxList.append(new_box)
                    r = np.array([np.linalg.norm(size)/2.0])
                    # sphere encode: 0,1,2: center, 3: size(r), 4: yaw(rad), 5,6: vx,vy, 7:label
                    new_sphere = np.concatenate((center, r, yaw, velocity, label))
                    gt_SphereList.append(new_sphere)
                    box_corners = box.corners().T
                    gt_BoxCornerList.append(box_corners)                    
                #else:
                    #print('GT box out of detection range')
                    #print(np.linalg.norm(box.center[:2]))
                    #print(self.MaxDistMapping[box.label])
        bounding_boxes = np.array(gt_BoxList)
        bounding_spheres = np.array(gt_SphereList)
        bounding_boxes_corners = np.array(gt_BoxCornerList)
        if bounding_boxes.shape[0]==0:
            #print('No object in this sample')
            return self.__getitem__(index-1)
        
        # get lidar point cloud
        lidar_sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', 5)
        lidar_points = np.concatenate((pc.points,times),0).T #size [N, 5], 5 corresponds to [x,y,z,intensity,time]
        depth = np.linalg.norm(lidar_points[:,0:3], axis=1)       
        lidar_points = lidar_points[depth<52.0,:]
        if self.npoints_lidar < lidar_points.shape[0]:
            choice = np.random.choice(lidar_points.shape[0], self.npoints_lidar, replace=False)
        else:
            choice = np.arange(0, lidar_points.shape[0], dtype=np.int32)
            if self.npoints_lidar > lidar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_lidar - lidar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        lidar_points = lidar_points[choice,:] #size [npoints_lidar, 5]
        lidar_points[:,3] = lidar_points[:,3]/255.0 # scale intensity from 0-255 to 0-1
        

        # get radar point cloud
        radar_points_list = []
        radar_sensor = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT_RIGHT']
        for sensor in radar_sensor:
            radar_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            radar_cs_record = self.nusc.get('calibrated_sensor', radar_sd_record['calibrated_sensor_token'])
            pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, sensor, 'LIDAR_TOP', 6)
            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the point cloud.
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(lidar_cs_record['rotation']).rotation_matrix.T, velocities)            
            velocities[2, :] = np.zeros(pc.points.shape[1])
            radar_points = np.concatenate((pc.points[0:3,:],velocities[0:2,:],times),0).T #size [N, 6], 6 corresponds to [x,y,z,vx,vy,time]
            radar_points_list.append(radar_points)
        radar_points = np.concatenate(radar_points_list, 0)
        depth = np.linalg.norm(radar_points[:,0:3], axis=1)
        radar_points = radar_points[depth<52.0,:]
        if self.npoints_radar < radar_points.shape[0]:
            choice = np.random.choice(radar_points.shape[0], self.npoints_radar, replace=False)
        else:
            choice = np.arange(0, radar_points.shape[0], dtype=np.int32)
            if self.npoints_radar > radar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_radar - radar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        radar_points = radar_points[choice,:] #size [npoints_radar, 6]
        
        # get images, camera intrinsic, camera pose to lidar coordinate system
        images = []
        #poses = []
        #intrinsics = []
        proj_mapping = []
        camera_sensor = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        for sensor in camera_sensor:
            cam_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            cam_cs_record = self.nusc.get('calibrated_sensor', cam_sd_record['calibrated_sensor_token'])
            img_path = os.path.join(self.nusc.dataroot, cam_sd_record['filename'])
            img = self.load_image(img_path, [640,360]) #[3,360,640], feature map will be [128,45,80]
            images.append(img)
            
            cam_intrinsic = np.eye(4)
            cam_intrinsic[0:3,0:3] = np.array(cam_cs_record['camera_intrinsic'])
            # adjust intrinsic for the size of feature map
            cam_intrinsic[0,0] *= float(80)/float(1600)
            cam_intrinsic[1,1] *= float(45)/float(900)
            cam_intrinsic[0,2] *= float(80-1)/float(1600-1)
            cam_intrinsic[1,2] *= float(45-1)/float(900-1)
            #intrinsics.append(cam_intrinsic)
            projection = ProjectionHelper(torch.from_numpy(cam_intrinsic), 0.5, 60.0, [80,45], 0.05)
            
            # compute camera pose
            cam_to_ego = np.eye(4)
            cam_to_ego[0:3,0:3] = Quaternion(cam_cs_record['rotation']).rotation_matrix
            cam_to_ego[0:3,3] = np.array(cam_cs_record['translation'])            
            poserecord = self.nusc.get('ego_pose', cam_sd_record['ego_pose_token'])
            ego_to_global = np.eye(4)
            ego_to_global[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix
            ego_to_global[0:3,3] = np.array(poserecord['translation'])            
            poserecord = self.nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
            global_to_ego = np.eye(4)
            global_to_ego[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix.T
            global_to_ego[0:3,3] = global_to_ego[0:3,0:3].dot(-np.array(poserecord['translation']))            
            ego_to_lidar = np.eye(4)
            ego_to_lidar[0:3,0:3] = Quaternion(lidar_cs_record['rotation']).rotation_matrix.T
            ego_to_lidar[0:3,3] = ego_to_lidar[0:3,0:3].dot(-np.array(lidar_cs_record['translation']))            
            cam_to_lidar = ego_to_lidar.dot(global_to_ego).dot(ego_to_global).dot(cam_to_ego)
            #poses.append(cam_to_lidar)
            proj_mapping.append(projection.compute_projection(torch.from_numpy(np.concatenate((lidar_points[:,0:3],radar_points[:,0:3]),0)), torch.from_numpy(cam_to_lidar), self.npoints_lidar+self.npoints_radar))
            
        if None in proj_mapping:
            return self.__getitem__(index-1)
        image = np.stack(images,0) #[num_images=6,3,360,640]
        #intrinsic = np.array(intrinsics) #[num_images=6,4,4]
        #pose = np.array(poses).astype(np.float32) #[num_images=6,4,4]
        proj_mapping0, proj_mapping1 = zip(*proj_mapping)
        proj_ind_3d = torch.stack(proj_mapping0)
        proj_ind_2d = torch.stack(proj_mapping1)
        
        #radar_points = np.array([])
        #image = np.array([])
        #intrinsic = np.array([])
        #pose = np.array([])
        
        # generate anchors, shape: [npoints*anchors per point, 4]

        # 10 anchors kmeans
        #anchor_radius = np.array([0.59519359, 1.04024921, 1.51894038, 2.33694095, 2.58570698, 2.82699181, 3.34568567, 4.45521787, 6.1408409, 8.17982825])
        # 12 anchors kmeans
        #anchor_radius = np.array([0.47506766, 0.73028391, 1.06012893, 1.53382215, 2.3336855, 2.57559099, 2.79544339, 3.2186541, 3.96466492, 4.88161637, 6.35588674, 8.30185038])
        # 14 anchors kmeans
        #anchor_radius = np.array([0.47202531, 0.70919491, 0.98827069, 1.19303585, 1.61575141, 2.16229479, 2.44651622, 2.63216558, 2.84913011, 3.27342676, 4.0436326, 4.98453866, 6.42535928, 8.33202872])
        
        # only use lidar points
        point_set = lidar_points[:,0:3]
        # use both lidar and radar points
        #point_set = np.concatenate((lidar_points[:,0:3],radar_points[:,0:3]),0)
        
        '''
        points = np.expand_dims(point_set, axis=1)
        points = points.repeat(repeats=bounding_spheres.shape[0], axis=1) #shape [npoints, bsphere_count, 3]
        sphere_centers = np.expand_dims(bounding_spheres[:,0:3], axis=0)
        sphere_centers = sphere_centers.repeat(repeats=point_set.shape[0], axis=0) #shape [npoints, bsphere_count, 3]
        # compute distance between every point and every object center
        dist_table = np.linalg.norm(points-sphere_centers, axis=2) #shape [npoints, bsphere_count]
        # points that out of all bounding spheres do not count to the loss
        vote_mask = (np.sum(dist_table <= bounding_spheres[:,3],axis=1)>0)*1.0
        if np.sum(vote_mask)==0:
            #print('No point in any object bounding box!')
            return self.__getitem__(index-1)
        # mask out "background" points and their distance
        dist_table[dist_table > bounding_spheres[:,3]] = 100000.0
        # find the sphere center that is nearist to the points
        idx = np.argmin(dist_table,axis=1)
        sem_label = bounding_spheres[idx,-1]
        sem_label[vote_mask==0] = 0.0
        '''
        vote_mask = np.zeros(point_set.shape[0])
        vote_label = np.zeros((point_set.shape[0],3))
        sem_label = np.zeros(point_set.shape[0])
        for i in range(bounding_boxes_corners.shape[0]):
            inds_in_box = in_hull(point_set, bounding_boxes_corners[i,:,:])
            vote_mask[inds_in_box] = 1.0
            vote_label[inds_in_box,:] = bounding_spheres[i, 0:3] - point_set[inds_in_box,:]
            sem_label[inds_in_box] = bounding_spheres[i,-1]
        if np.sum(vote_mask)==0:
            #print('No point in any object bounding box!')
            return self.__getitem__(index-1)        
               
        return lidar_points, radar_points, bounding_boxes, bounding_spheres, sem_label, image, proj_ind_3d, proj_ind_2d
    
    def __len__(self):
        return len(self.sample)
        
    def resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)
        return image
        
    def load_image(self, file, image_dims):
        image = misc.imread(file)
        # preprocess
        image = self.resize_crop_image(image, image_dims)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise
        return image
    
    def collate_fn(self, batch):
        '''
        Since each point set may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        param batch: an iterable of N sets from __getitem__()
        return:
        a tensor of point clouds
        a tensor of semantic labels
        a tensor of instance labels
        a list of varying-size tensors of bounding spheres
        a tensor of sample weights
        '''
        lidar_points_list = []
        radar_points_list = []
        bounding_boxes_list = []
        bounding_spheres_list = []
        sem_label_list = []
        image_list = []
        proj_ind_3d_list = []
        proj_ind_2d_list = []
        for b in batch:
            lidar_points_list.append(torch.from_numpy(b[0]))
            radar_points_list.append(torch.from_numpy(b[1]))
            bounding_boxes_list.append(torch.from_numpy(b[2]))
            bounding_spheres_list.append(torch.from_numpy(b[3]))
            sem_label_list.append(torch.from_numpy(b[4]))
            image_list.append(torch.from_numpy(b[5]))
            proj_ind_3d_list.append(b[6])
            proj_ind_2d_list.append(b[7])
        lidar_points = torch.stack(lidar_points_list, dim=0) #shape: [batchsize, npoints, 5]
        radar_points = torch.stack(radar_points_list, dim=0) #shape: [batchsize, npoints, 6]
        sem_labels = torch.stack(sem_label_list, dim=0) #shape: [batchsize, npoints]
        # return tensor, tensor, tensor, list of N tensors, tensor
        return lidar_points, radar_points, bounding_boxes_list, bounding_spheres_list, sem_labels, image_list, proj_ind_3d_list, proj_ind_2d_list
        
class NuscenesDatasetSemsegOverfitting(Dataset):
    def __init__(self, nusc, npoints_lidar=16384, npoints_radar=2048, split='train_detect'):
        self.npoints_lidar = npoints_lidar
        self.npoints_radar = npoints_radar
        self.split = split
        self.nusc = nusc
        split_dict = create_splits_scenes()
        self.NameMapping = {
            'movable_object.barrier': 'barrier',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }
        self.LabelMapping = {
            'barrier': 1,
            'bicycle': 2,
            'bus': 3,
            'car': 4,
            'construction_vehicle': 5,
            'motorcycle': 6,
            'pedestrian': 7,
            'traffic_cone': 8,
            'trailer': 9,
            'truck':10
        }
        self.MaxDistMapping = {
            1: 30.0,
            2: 40.0,
            3: 50.0,
            4: 50.0,
            5: 50.0,
            6: 40.0,
            7: 40.0,
            8: 30.0,
            9: 50.0,
            10: 50.0
        }        
        self.DefaultAttribute = {
            "car": "vehicle.parked",
            "pedestrian": "pedestrian.moving",
            "trailer": "vehicle.parked",
            "truck": "vehicle.parked",
            "bus": "vehicle.parked",
            "motorcycle": "cycle.without_rider",
            "construction_vehicle": "vehicle.parked",
            "bicycle": "cycle.without_rider",
            "barrier": "",
            "traffic_cone": ""
        }
        self.scene = []
        for s in self.nusc.scene:
            if s['name'] in split_dict[split]:
                self.scene.append(s)
        self.sample = []
        for s in self.scene:
            sample = self.nusc.get('sample', s['first_sample_token'])
            self.sample.append(sample)
            while not sample['next'] == "":
                sample = self.nusc.get('sample', sample['next'])
                self.sample.append(sample)
        print(len(self.scene))
        print(len(self.sample))
    
    def __getitem__(self, index):
        index = index%1
        sample = self.sample[index]
        # get box list, filter objects and transform to np array
        _, BoxList, _ = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        gt_BoxList = []
        gt_SphereList = []
        for box in BoxList:
            if box.name in self.NameMapping:
                box.label = self.LabelMapping[self.NameMapping[box.name]]
                center = box.center
                if np.linalg.norm(box.center[:2]) <= self.MaxDistMapping[box.label]:
                    size = box.wlh
                    yaw = np.array([box.orientation.yaw_pitch_roll[0]])
                    velocity = box.velocity[0:2]
                    label = np.array([box.label])
                    # box encode: 0,1,2: center, 3,4,5: size(whl), 6: yaw(rad), 7,8: vx,vy, 9:label
                    new_box = np.concatenate((center, size, yaw, velocity, label))
                    gt_BoxList.append(new_box)
                    r = np.array([np.linalg.norm(size)/2.0])
                    # sphere encode: 0,1,2: center, 3: size(r), 4: yaw(rad), 5,6: vx,vy, 7:label
                    new_sphere = np.concatenate((center, r, yaw, velocity, label))
                    gt_SphereList.append(new_sphere)
                #else:
                    #print('GT box out of detection range')
                    #print(np.linalg.norm(box.center[:2]))
                    #print(self.MaxDistMapping[box.label])
        bounding_boxes = np.array(gt_BoxList)
        bounding_spheres = np.array(gt_SphereList)
        if bounding_boxes.shape[0]==0:
            #print('No object in this sample')
            return self.__getitem__(index-1)
        
        # get lidar point cloud
        lidar_sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', 5)
        lidar_points = np.concatenate((pc.points,times),0).T #size [N, 5], 5 corresponds to [x,y,z,intensity,time]
        depth = np.linalg.norm(lidar_points[:,0:3], axis=1)       
        lidar_points = lidar_points[depth<52.0,:]
        if self.npoints_lidar < lidar_points.shape[0]:
            choice = np.random.choice(lidar_points.shape[0], self.npoints_lidar, replace=False)
        else:
            choice = np.arange(0, lidar_points.shape[0], dtype=np.int32)
            if self.npoints_lidar > lidar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_lidar - lidar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        lidar_points = lidar_points[choice,:] #size [npoints_lidar, 5]
        lidar_points[:,3] = lidar_points[:,3]/255.0 # scale intensity from 0-255 to 0-1
        
        '''
        # get radar point cloud
        radar_points_list = []
        radar_sensor = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT_RIGHT']
        for sensor in radar_sensor:
            radar_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            radar_cs_record = self.nusc.get('calibrated_sensor', radar_sd_record['calibrated_sensor_token'])
            pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, sensor, 'LIDAR_TOP', 6)
            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the point cloud.
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(lidar_cs_record['rotation']).rotation_matrix.T, velocities)            
            velocities[2, :] = np.zeros(pc.points.shape[1])
            radar_points = np.concatenate((pc.points[0:3,:],velocities[0:2,:],times),0).T #size [N, 6], 6 corresponds to [x,y,z,vx,vy,time]
            radar_points_list.append(radar_points)
        radar_points = np.concatenate(radar_points_list, 0)
        if self.npoints_radar < radar_points.shape[0]:
            choice = np.random.choice(radar_points.shape[0], self.npoints_radar, replace=False)
        else:
            choice = np.arange(0, radar_points.shape[0], dtype=np.int32)
            if self.npoints_radar > radar_points.shape[0]:
                extra_choice = np.random.choice(choice, self.npoints_radar - radar_points.shape[0], replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
        radar_points = radar_points[choice,:] #size [npoints_radar, 5]
        
        # get images, camera intrinsic, camera pose to lidar coordinate system
        images = []
        poses = []
        intrinsics = []
        camera_sensor = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        for sensor in camera_sensor:
            cam_sd_record = self.nusc.get('sample_data', sample['data'][sensor])
            cam_cs_record = self.nusc.get('calibrated_sensor', cam_sd_record['calibrated_sensor_token'])
            img_path = os.path.join(self.nusc.dataroot, cam_sd_record['filename'])
            img = self.load_image(img_path, [640,360]) #[3,360,640], feature map will be [128,45,80]
            images.append(img)
            
            cam_intrinsic = np.eye(4)
            cam_intrinsic[0:3,0:3] = np.array(cam_cs_record['camera_intrinsic'])
            # adjust intrinsic for the size of feature map
            cam_intrinsic[0,0] *= float(80)/float(1600)
            cam_intrinsic[1,1] *= float(45)/float(900)
            cam_intrinsic[0,2] *= float(80-1)/float(1600-1)
            cam_intrinsic[1,2] *= float(45-1)/float(900-1)
            intrinsics.append(cam_intrinsic)
            
            # compute camera pose
            cam_to_ego = np.eye(4)
            cam_to_ego[0:3,0:3] = Quaternion(cam_cs_record['rotation']).rotation_matrix
            cam_to_ego[0:3,3] = np.array(cam_cs_record['translation'])            
            poserecord = self.nusc.get('ego_pose', cam_sd_record['ego_pose_token'])
            ego_to_global = np.eye(4)
            ego_to_global[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix
            ego_to_global[0:3,3] = np.array(poserecord['translation'])            
            poserecord = self.nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
            global_to_ego = np.eye(4)
            global_to_ego[0:3,0:3] = Quaternion(poserecord['rotation']).rotation_matrix.T
            global_to_ego[0:3,3] = global_to_ego[0:3,0:3].dot(-np.array(poserecord['translation']))            
            ego_to_lidar = np.eye(4)
            ego_to_lidar[0:3,0:3] = Quaternion(lidar_cs_record['rotation']).rotation_matrix.T
            ego_to_lidar[0:3,3] = ego_to_lidar[0:3,0:3].dot(-np.array(lidar_cs_record['translation']))            
            cam_to_lidar = ego_to_lidar.dot(global_to_ego).dot(ego_to_global).dot(cam_to_ego)
            poses.append(cam_to_lidar)
        image = np.stack(images,0) #[num_images=6,3,360,640]
        intrinsic = np.array(intrinsics) #[num_images=6,4,4]
        pose = np.array(poses).astype(np.float32) #[num_images=6,4,4]
        '''
        radar_points = np.array([])
        image = np.array([])
        intrinsic = np.array([])
        pose = np.array([])
        
        # generate anchors, shape: [npoints*anchors per point, 4]

        # 10 anchors kmeans
        #anchor_radius = np.array([0.59519359, 1.04024921, 1.51894038, 2.33694095, 2.58570698, 2.82699181, 3.34568567, 4.45521787, 6.1408409, 8.17982825])
        # 12 anchors kmeans
        #anchor_radius = np.array([0.47506766, 0.73028391, 1.06012893, 1.53382215, 2.3336855, 2.57559099, 2.79544339, 3.2186541, 3.96466492, 4.88161637, 6.35588674, 8.30185038])
        # 14 anchors kmeans
        #anchor_radius = np.array([0.47202531, 0.70919491, 0.98827069, 1.19303585, 1.61575141, 2.16229479, 2.44651622, 2.63216558, 2.84913011, 3.27342676, 4.0436326, 4.98453866, 6.42535928, 8.33202872])
        
        # only use lidar points
        point_set = lidar_points[:,0:3]
        # use both lidar and radar points
        #point_set = np.concatenate((lidar_points[:,0:3],radar_points[:,0:3]),0)
        
        points = np.expand_dims(point_set, axis=1)
        points = points.repeat(repeats=bounding_spheres.shape[0], axis=1) #shape [npoints, bsphere_count, 3]
        sphere_centers = np.expand_dims(bounding_spheres[:,0:3], axis=0)
        sphere_centers = sphere_centers.repeat(repeats=point_set.shape[0], axis=0) #shape [npoints, bsphere_count, 3]
        # compute distance between every point and every object center
        dist_table = np.linalg.norm(points-sphere_centers, axis=2) #shape [npoints, bsphere_count]
        # points that out of all bounding spheres do not count to the loss
        vote_mask = (np.sum(dist_table <= bounding_spheres[:,3],axis=1)>0)*1.0
        if np.sum(vote_mask)==0:
            #print('No point in any object bounding box!')
            return self.__getitem__(index-1)
        # mask out "background" points and their distance
        dist_table[dist_table > bounding_spheres[:,3]] = 100000.0
        # find the sphere center that is nearist to the points
        idx = np.argmin(dist_table,axis=1)
        sem_label = bounding_spheres[idx,-1]
        sem_label[vote_mask==0] = 0.0
               
        return lidar_points, radar_points, bounding_boxes, bounding_spheres, sem_label, image, intrinsic, pose
    
    def __len__(self):
        return 1
        
    def resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)
        return image
        
    def load_image(self, file, image_dims):
        image = misc.imread(file)
        # preprocess
        image = self.resize_crop_image(image, image_dims)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=[0.290101, 0.328081, 0.286964], std=[0.182954, 0.186566, 0.184475])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise
        return image
    
    def collate_fn(self, batch):
        '''
        Since each point set may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        param batch: an iterable of N sets from __getitem__()
        return:
        a tensor of point clouds
        a tensor of semantic labels
        a tensor of instance labels
        a list of varying-size tensors of bounding spheres
        a tensor of sample weights
        '''
        lidar_points_list = []
        radar_points_list = []
        bounding_boxes_list = []
        bounding_spheres_list = []
        sem_label_list = []
        image_list = []
        intrinsic_list = []
        pose_list = []
        for b in batch:
            lidar_points_list.append(torch.from_numpy(b[0]))
            radar_points_list.append(torch.from_numpy(b[1]))
            bounding_boxes_list.append(torch.from_numpy(b[2]))
            bounding_spheres_list.append(torch.from_numpy(b[3]))
            sem_label_list.append(torch.from_numpy(b[4]))
            image_list.append(torch.from_numpy(b[5]))
            intrinsic_list.append(torch.from_numpy(b[6]))
            pose_list.append(torch.from_numpy(b[7]))
        lidar_points = torch.stack(lidar_points_list, dim=0) #shape: [batchsize, npoints, 5]
        radar_points = torch.stack(radar_points_list, dim=0) #shape: [batchsize, npoints, 6]
        sem_labels = torch.stack(sem_label_list, dim=0) #shape: [batchsize, npoints]
        # return tensor, tensor, tensor, list of N tensors, tensor
        return lidar_points, radar_points, bounding_boxes_list, bounding_spheres_list, sem_labels, image_list, intrinsic_list, pose_list  
        
def in_hull(p, hull):
    #from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def iou_spheres(spheres_a, spheres_b):
    """
    Computes IoU overlaps between two sets of spheres.
    spheres_a: [M, 4] 
    spheres_b: [N, 4].
    return: iou table of shape: [M, N]
    """    
    spheres1 = np.expand_dims(spheres_a,axis=1) #shape: [M,1,4]
    spheres1 = spheres1.repeat(repeats=spheres_b.shape[0], axis=1) #shape: [M,N,4]
    spheres2 = np.expand_dims(spheres_b,axis=0) #shape: [1,N,4]
    spheres2 = spheres2.repeat(repeats=spheres_a.shape[0], axis=0) #shape: [M,N,4]
    dist = np.linalg.norm(spheres1[:,:,0:3] - spheres2[:,:,0:3],axis=2)
    r_a, r_b = spheres1[:,:,3], spheres2[:,:,3]
    iou = np.zeros([spheres_a.shape[0], spheres_b.shape[0]])
    # one sphere fully inside the other (includes coincident)
    # take volume of smaller sphere as intersection
    # take volume of larger sphere as union
    # iou is (min(r_a, r_b)/max(r_a, r_b))**3
    idx = np.where(dist <= abs(r_a - r_b))
    min_r = np.minimum(r_a[idx], r_b[idx])
    max_r = np.maximum(r_a[idx], r_b[idx])
    iou[idx] = (1.0*min_r/max_r)**3
    
    # spheres partially overlap, calculate intersection as per
    # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    idx = np.where((dist > abs(r_a - r_b))&(dist < r_a + r_b))
    intersection = (r_a[idx] + r_b[idx] - dist[idx])**2
    intersection *= (dist[idx]**2 + 2*dist[idx]*(r_a[idx] + r_b[idx]) - 3*(r_a[idx] - r_b[idx])**2)
    intersection *= np.pi / (12*dist[idx])
    union = 4/3. * np.pi * (r_a[idx]**3 + r_b[idx]**3) - intersection
    iou[idx] = intersection / union
    
    return iou 

if __name__=='__main__':
    dataset = NuscenesDataset(root = '/mnt/raid/chengnan/NuScenes/', npoints_lidar=16384, npoints_radar=2048, split='train_detect')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn,
                                             shuffle=True, num_workers=1)
    for ii, data in enumerate(dataloader):
        print(ii)
        lidar_points, radar_points, bounding_boxes, bounding_spheres, vote_labels, vote_masks, image, intrinsic, pose = data
        print(lidar_points.shape)
        print(radar_points.shape)
        print(vote_labels.shape)
        print(vote_masks.shape)
        print(torch.sum(vote_masks, dim=1))
        '''
        for b in range(len(bounding_boxes)):
            print(bounding_boxes[b].shape)
            print(bounding_spheres[b].shape)
            print(image[b].shape)
            print(intrinsic[b].shape)
            print(pose[b].shape)
        '''
        
    exit()


