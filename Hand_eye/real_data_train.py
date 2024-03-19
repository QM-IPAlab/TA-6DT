import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
import numpy.ma as ma
import copy
import math
import scipy.misc
import scipy.io as scio
import cv2
import _pickle as cPickle
from numpy.linalg import inv

import sys
sys.path.append("..")
from libs.transformations import euler_matrix

class Dataset(data.Dataset):
    def __init__(self, mode, root, add_noise, num_pt, num_cates, count, cate_id):
        self.root = root
        self.add_noise = add_noise
        self.mode = mode
        self.num_pt = num_pt
        self.num_cates = num_cates

        self.mesh = []
        input_file = open('../dataset/sphere.xyz', 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            self.mesh.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        self.mesh = np.array(self.mesh) * 0.6

        self.xmap = np.array([[j for i in range(1280)] for j in range(720)])
        self.ymap = np.array([[i for i in range(1280)] for j in range(720)])

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.trancolor = transforms.ColorJitter(0.8, 0.5, 0.5, 0.05)
        self.length = count

    def divide_scale(self, scale, pts):
        pts[:, 0] = pts[:, 0] / scale[0]
        pts[:, 1] = pts[:, 1] / scale[1]
        pts[:, 2] = pts[:, 2] / scale[2]

        return pts

    def get_anchor_box(self, ori_bbox):
        bbox = ori_bbox
        limit = np.array(search_fit(bbox))
        num_per_axis = 5
        gap_max = num_per_axis - 1

        small_range = [1, 3]

        gap_x = (limit[1] - limit[0]) / float(gap_max)
        gap_y = (limit[3] - limit[2]) / float(gap_max)
        gap_z = (limit[5] - limit[4]) / float(gap_max)

        ans = []
        scale = [max(limit[1], -limit[0]), max(limit[3], -limit[2]), max(limit[5], -limit[4])]

        for i in range(0, num_per_axis):
            for j in range(0, num_per_axis):
                for k in range(0, num_per_axis):
                    ans.append([limit[0] + i * gap_x, limit[2] + j * gap_y, limit[4] + k * gap_z])

        ans = np.array(ans)
        scale = np.array(scale)

        ans = self.divide_scale(scale, ans)

        return ans, scale

    def change_to_scale(self, scale, cloud_fr, cloud_to):
        cloud_fr = self.divide_scale(scale, cloud_fr)
        cloud_to = self.divide_scale(scale, cloud_to)

        return cloud_fr, cloud_to

    def enlarge_bbox(self, target):

        limit = np.array(search_fit(target))
        longest = max(limit[1] - limit[0], limit[3] - limit[2], limit[5] - limit[4])
        longest = longest * 1.3

        scale1 = longest / (limit[1] - limit[0])
        scale2 = longest / (limit[3] - limit[2])
        scale3 = longest / (limit[5] - limit[4])

        target[:, 0] *= scale1
        target[:, 1] *= scale2
        target[:, 2] *= scale3

        return target

    def load_depth(self, depth_path):
        depth = cv2.imread(depth_path, -1)

        if len(depth.shape) == 3:
            depth16 = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'

        return depth16

    def get_pose(self, pose_path, index):
        has_pose = []
        pose = {}
        with open(pose_path, 'rb') as f:
            pose_data = cPickle.load(f, encoding="bytes")

        ans = pose_data[index]

        ans_r = ans[:3, :3]
        ans_t = ans[:3, 3].flatten()

        return ans_r, ans_t


    def get_frame_camera_coord(self, data_location, index, target):

        img_location = data_location + '/raw/{}.png'.format(str(index).zfill(5))
        img = Image.open(img_location)
        img_org = np.array(img)
        img_org = np.transpose(img_org[:, :, :3], (2, 0, 1))

        depth_location = data_location + '/depth/{}.png'.format(str(index).zfill(5))
        depth = np.array(self.load_depth(depth_location))
        depth_org = depth

        pose_location = data_location + "/poses.pickle"
        target_r, target_t = self.get_pose(pose_location, index)

        with open('/import/smartcameras-002/long/CS207_real_dataset/long_dataset/cam_intrinsics.pkl', 'rb') as f:
            cam_intr = cPickle.load(f)

        cam_cx = cam_intr[0][2]
        cam_cy = cam_intr[1][2]
        cam_fx = cam_intr[0][0]
        cam_fy = cam_intr[1][1]
        cam_scale = 1.0

        ## mug
        points_3d_noextend = np.array([[0.0,    0.0,   0.0],
                                       [ 0.07,  0.05,  0.0],
                                       [-0.04,  0.05,  0.0],
                                       [ 0.07, -0.05,  0.0],
                                       [-0.04, -0.05,  0.0],
                                       [ 0.07,  0.05, -0.1],
                                       [-0.04,  0.05, -0.1],
                                       [ 0.07, -0.05, -0.1],
                                       [-0.04, -0.05, -0.1]])
        offset = np.tile([0.197, 0.118, 0.0], [9,1])
        target = points_3d_noextend + offset

        target = self.enlarge_bbox(copy.deepcopy(target))


        target_tmp = np.dot(target, target_r.T) + target_t
        target_tmp[:, 0] *= -1.0
        target_tmp[:, 1] *= -1.0
        rmin, rmax, cmin, cmax = get_2dbbox(target_tmp, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale)
        limit = search_fit(target)

        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        img = img / 255.0


        mask_location = data_location + '/mask/{}.png'.format(str(index).zfill(5))
        mask_target = (cv2.imread(mask_location))[rmin:rmax, cmin:cmax]
        mask_target = mask_target[:, :, 0]
        mask_target = 255 - mask_target

        mask_org = cv2.imread(mask_location)
        mask_org = mask_org[:, :, 0]
        mask_org = 255 - mask_org

        depth = depth[rmin:rmax, cmin:cmax]

        depth_mask = depth * mask_target
        choose = (depth_mask.flatten() != 0.0).nonzero()[0]

        if len(choose) == 0:
            return 0
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)
        cloud = cloud / 1000.0

        target = target / 1000.0

        return img, choose, cloud, target_r, target_t, target, mask_target, depth, img_org, depth_org, mask_org




    def get_frame(self, data_location, index, target):

        img_location = data_location + '/raw/{}.png'.format(str(index).zfill(5))
        img = Image.open(img_location)
        img_org = np.array(img)
        img_org = np.transpose(img_org[:, :, :3], (2, 0, 1))

        depth_location = data_location + '/depth/{}.png'.format(str(index).zfill(5))
        depth = np.array(self.load_depth(depth_location))
        pose_location = data_location + "/poses.pickle"
        target_r, target_t = self.get_pose(pose_location, index)
        with open('/import/smartcameras-002/long/CS207_real_dataset/long_dataset/cam_intrinsics.pkl', 'rb') as f:
        # with open('/import/smartcameras-002/long/CS207_real_dataset/05062023/cam_intrinsics_v2.pkl', 'rb') as f:
            cam_intr = cPickle.load(f)
        cam_cx = cam_intr[0][2]
        cam_cy = cam_intr[1][2]
        cam_fx = cam_intr[0][0]
        cam_fy = cam_intr[1][1]
        cam_scale = 1.0
        # cracker_box
        points_3d_noextend = np.array([[0.0, 0.0, 0.0],
                                       [0.0675, 0.0275, 0.0],
                                       [-0.0675, 0.0275, 0.0],
                                       [0.0675, -0.0275, 0.0],
                                       [-0.0675, -0.0275, 0.0],
                                       [0.0675, 0.0275, -0.16],
                                       [-0.0675, 0.0275, -0.16],
                                       [0.0675, -0.0275, -0.16],
                                       [-0.0675, -0.0275, -0.16]])

        offset = np.tile([0.197, 0.118, 0.0], [9,1])
        target = points_3d_noextend + offset

        target = target * 1000.0
        target = self.enlarge_bbox(copy.deepcopy(target))

        delta = math.pi / 10.0
        noise_trans = 0.05
        r = euler_matrix(random.uniform(-delta, delta), random.uniform(-delta, delta), random.uniform(-delta, delta))[
            :3, :3]
        t = np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)]) * 1000.0

        target_tmp = target - (np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)]) * 3000.0)
        target_tmp = np.dot(target_tmp, target_r.T) + target_t
        target_tmp[:, 0] *= -1.0
        target_tmp[:, 1] *= -1.0

        bbox_2d_path = data_location + '/mask/{}.npy'.format(str(index).zfill(5))
        rmin, rmax, cmin, cmax = get_2dbbox_v2(bbox_2d_path)

        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        img = img / 255.0


        mask_location = data_location + '/mask/{}.png'.format(str(index).zfill(5))
        mask_target = (cv2.imread(mask_location))[rmin:rmax, cmin:cmax]
        mask_target = mask_target[:, :, 0]

        choose = (mask_target.flatten() != False).nonzero()[0]
        if len(choose) == 0:
            return 0

        depth = depth[rmin:rmax, cmin:cmax]

        depth_mask = depth * mask_target
        choose = (depth_mask.flatten() != 0.0).nonzero()[0]
        xmap = np.array([[j for i in range(1280)] for j in range(720)])
        ymap = np.array([[i for i in range(1280)] for j in range(720)])

        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)

        pc_xz_norm_np = np.linalg.norm(cloud[:, [0, 2]], axis=1)

        pc_radius_mask_np = pc_xz_norm_np <= np.partition(pc_xz_norm_np, -1000)[::-1][1000 - 1]
        cloud = cloud[pc_radius_mask_np, :]
        choose = choose[pc_radius_mask_np]

        if cloud.shape[0] >= 500:
            choice_idx = np.random.choice(cloud.shape[0], 500, replace=False)
            cloud = cloud[choice_idx, :]
            choose = choose[choice_idx]

        if cloud.shape[0] < 500:
            padding_size = 500 - cloud.shape[0]
            padding = np.zeros((padding_size, 3))
            cloud = np.concatenate((cloud, padding))
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        t = t / 1000.0
        cloud = cloud / 1000.0
        target = target / 1000.0
        target_t = target_t / 1000.0



        return img, choose, cloud, target_r, target_t, target, mask_target, depth


    def re_scale(self, target_fr, target_to):
        ans_scale = target_fr / target_to
        ans_target = target_fr
        ans_scale = ans_scale[0][0]

        return ans_target, ans_scale

    def compute_pose_change(self, rotation1, rotation2, translation1, translation2):
        delta_rotation = np.dot(rotation2, rotation1.T)
        delta_translation = translation2 - translation1

        return delta_rotation, delta_translation

    def __getitem__(self, index):
        syn_or_real = (random.randint(1, 10) > 15)
        if self.mode == 'val':
            syn_or_real = False

        if syn_or_real:
            while 1:
                try:
                    choose_obj = random.sample(self.obj_name_list[self.cate_id], 1)[0]
                    choose_frame = random.sample(self.obj_list[self.cate_id][choose_obj], 2)

                    img_fr, choose_fr, cloud_fr, r_fr, t_fr, target_fr, mesh_pts_fr, mesh_bbox_fr, mask_target = self.get_frame(
                        choose_frame[0], choose_obj, syn_or_real)
                    if np.max(abs(target_fr)) > 1.0:
                        continue
                    img_to, choose_to, cloud_to, r_to, t_to, target_to, _, _, _, = self.get_frame(choose_frame[1],
                                                                                                  choose_obj,
                                                                                                  syn_or_real)
                    if np.max(abs(target_to)) > 1.0:
                        continue

                    target, scale_factor = self.re_scale(target_fr, target_to)
                    target_mesh_fr, scale_factor_mesh_fr = self.re_scale(target_fr, mesh_bbox_fr)

                    cloud_to = cloud_to * scale_factor
                    mesh = mesh_pts_fr * scale_factor_mesh_fr
                    t_to = t_to * scale_factor
                    break
                except:
                    continue

        else:
            while 1:
                try:
                    obj = 'car_model'
                    path = '/import/smartcameras-002/long/CS207_real_dataset/05062023/{}'.format(obj)
                    index_1 = 141
                    index_2 = random.randint(141, 961)


                    img_fr, choose_fr, cloud_fr, r_fr, t_fr, target, mask_fr, depth_fr = self.get_frame(path, index_1, obj)
                    img_to, choose_to, cloud_to, r_to, t_to, target, mask_to, depth_to = self.get_frame(path, index_2, obj)

                    if np.max(abs(target)) > 1.0:
                        continue
                    break
                except:
                    continue

        if False:
            p_img = np.transpose(img_fr, (1, 2, 0))
            scipy.misc.imsave('temp/{0}_img_fr.png'.format(index), p_img)

            p_img = np.transpose(img_to, (1, 2, 0))
            scipy.misc.imsave('temp/{0}_img_to.png'.format(index), p_img)

            scipy.misc.imsave('temp/{0}_mask_fr.png'.format(index), mask_target.astype(np.int64))

            fw = open('temp/{0}_cld_fr.xyz'.format(index), 'w')
            for it in cloud_fr:
                fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_cld_to.xyz'.format(index), 'w')
            for it in cloud_to:
                fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()


        anchor_box, scale_buyong = self.get_anchor_box(target)



        anchor_box, scale = self.get_anchor_box(target)

        diagonal_len_fr = 0.12
        diagonal_len_to = 0.12

        T_fr = t_fr[:, np.newaxis]
        T_to = t_to[:, np.newaxis]
        RT_1 = np.hstack((r_fr, T_fr))
        RT_2 = np.hstack((r_to, T_to))
        homo = np.array([0, 0, 0, 1])
        RT_1_homo = np.vstack((RT_1, homo))
        RT_2_homo = np.vstack((RT_2, homo))
        delta_RT = np.dot(RT_1_homo, inv(RT_2_homo))



        if False:
            fw = open('temp/{0}_aft_cld_fr.xyz'.format(index), 'w')
            for it in cloud_fr:
                fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_aft_cld_to.xyz'.format(index), 'w')
            for it in cloud_to:
                fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_cld_mesh.xyz'.format(index), 'w')
            for it in mesh:
                fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_target.xyz'.format(index), 'w')
            for it in target:
                fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_anchor.xyz'.format(index), 'w')
            for it in anchor_box:
                fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_small_anchor.xyz'.format(index), 'w')
            for it in small_anchor_box:
                fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_pose_fr.xyz'.format(index), 'w')
            for it in r_fr:
                fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            it = t_fr
            fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.write('{0}\n'.format(choose_frame[0]))
            fw.close()

            fw = open('temp/{0}_pose_to.xyz'.format(index), 'w')
            for it in r_to:
                fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            it = t_to
            fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.write('{0}\n'.format(choose_frame[1]))
            fw.close()


        return self.norm(torch.from_numpy(img_fr.astype(np.float32))), \
            torch.LongTensor(choose_fr.astype(np.int32)), \
            torch.from_numpy(cloud_fr.astype(np.float32)), \
            self.norm(torch.from_numpy(depth_fr.astype(np.float32))), \
            self.norm(torch.from_numpy(mask_fr.astype(np.float32))), \
            self.norm(torch.from_numpy(img_to.astype(np.float32))), \
            torch.LongTensor(choose_to.astype(np.int32)), \
            torch.from_numpy(cloud_to.astype(np.float32)), \
            self.norm(torch.from_numpy(depth_to.astype(np.float32)))




    def __len__(self):
        return self.length


border_list = [-1, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 720
img_length = 1280


def get_2dbbox(cloud, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale):
    rmin = 10000
    rmax = -10000
    cmin = 10000
    cmax = -10000
    for tg in cloud:
        p1 = int(tg[0] * cam_fx / tg[2] + cam_cx)
        p0 = int(tg[1] * cam_fy / tg[2] + cam_cy)
        if p0 < rmin:
            rmin = p0
        if p0 > rmax:
            rmax = p0
        if p1 < cmin:
            cmin = p1
        if p1 > cmax:
            cmax = p1
    rmax += 1
    cmax += 1
    if rmin < 0:
        rmin = 0
    if cmin < 0:
        cmin = 0
    if rmax >= 720:
        rmax = 719
    if cmax >= 1280:
        cmax = 1279

    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt

    if ((rmax - rmin) in border_list) and ((cmax - cmin) in border_list):
        return rmin, rmax, cmin, cmax
    else:
        return 0


def search_fit(points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])

    return [min_x, max_x, min_y, max_y, min_z, max_z]


def get_2dbbox_v2(path):
    points = np.load(path)
    x_coordinates = [point[0] for point in points]
    y_coordinates = [point[1] for point in points]
    min_x = min(x_coordinates)
    max_x = max(x_coordinates)
    min_y = min(y_coordinates)
    max_y = max(y_coordinates)

    return min_y, max_y, min_x, max_x
