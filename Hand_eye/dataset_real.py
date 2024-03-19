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


class Dataset():
    def __init__(self, mode, root, add_noise, num_pts, count):
        self.root = root
        self.add_noise = add_noise
        self.mode = mode
        self.num_pts = num_pts

        self.real_obj_list = {}

        self.xmap = np.array([[j for i in range(1280)] for j in range(720)])
        self.ymap = np.array([[i for i in range(1280)] for j in range(720)])

        self.color = np.array(
            [[255, 69, 0], [124, 252, 0], [0, 238, 238], [238, 238, 0], [155, 48, 255], [0, 0, 238], [255, 131, 250],
             [189, 183, 107], [165, 42, 42], [0, 234, 0]])

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.trancolor = transforms.ColorJitter(0.9, 0.5, 0.5, 0.05)
        self.length = count

        self.choose_obj = ''
        self.index = 0
        self.video_id = ''

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

    def change_to_scale(self, scale, cloud_fr):
        cloud_fr = self.divide_scale(scale, cloud_fr)

        return cloud_fr

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


    def get_frame(self, choose_frame, index, current_r, current_t):

        img_location = choose_frame + '/raw/{}.png'.format(str(index).zfill(5))
        img = Image.open(img_location)

        depth_location = choose_frame + '/depth/{}.png'.format(str(index).zfill(5))
        depth = np.array(self.load_depth(depth_location))

        pose_location = choose_frame + "/poses.pickle"
        target_r, target_t = self.get_pose(pose_location, index)

        with open('/import/smartcameras-002/long/CS207_real_dataset/long_dataset/cam_intrinsics.pkl', 'rb') as f:
            cam_intr = cPickle.load(f)

        cam_cx = cam_intr[0][2]
        cam_cy = cam_intr[1][2]
        cam_fx = cam_intr[0][0]
        cam_fy = cam_intr[1][1]
        cam_scale = 1.0
        ### mug
        # points_3d_noextend = np.array([[0.0,    0.0,   0.0],
        #                                [ 0.07,  0.05,  0.0],
        #                                [-0.04,  0.05,  0.0],
        #                                [ 0.07, -0.05,  0.0],
        #                                [-0.04, -0.05,  0.0],
        #                                [ 0.07,  0.05, -0.1],
        #                                [-0.04,  0.05, -0.1],
        #                                [ 0.07, -0.05, -0.1],
        #                                [-0.04, -0.05, -0.1]])
        # points_3d_noextend = np.array([[0.0,    0.0,   0.0],
        #                                [ 0.07,  0.05,  0.1],
        #                                [-0.07,  0.05,  0.1],
        #                                [ 0.07, -0.05,  0.1],
        #                                [-0.07, -0.05,  0.1],
        #                                [ 0.07,  0.05, -0.1],
        #                                [-0.07,  0.05, -0.1],
        #                                [ 0.07, -0.05, -0.1],
        #                                [-0.07, -0.05, -0.1]])

        #### laptop
        points_3d_noextend = np.array([
                                        [0.09, 0.04, 0.0],
                                        [-0.09, 0.04, 0.0],
                                        [0.09, -0.04, 0.0],
                                        [-0.09, -0.04, 0.0],
                                        [0.09, 0.04, -0.06],
                                        [-0.09, 0.04, -0.06],
                                        [0.09, -0.04, -0.06],
                                        [-0.09, -0.04, -0.06]])

        offset = np.tile([0.197, 0.118, 0.0], [8,1])
        target = points_3d_noextend + offset
        target = target * 1000.0
        target = self.enlarge_bbox(copy.deepcopy(target))

        target_tmp = np.dot(target, target_r.T) + target_t
        target_tmp[:, 0] *= -1.0
        target_tmp[:, 1] *= -1.0


        #### get 2D bounding box
        bbox_2d_path = choose_frame + '/mask/{}.npy'.format(str(index).zfill(5))
        rmin, rmax, cmin, cmax = get_2dbbox_v2(bbox_2d_path)

        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        img = img / 255.0
        print("img ", img_location, img.shape)

        img_crop = np.transpose(img, (1, 2, 0))
        print("img_crop", img_crop.shape)
        import scipy.misc
        import scipy.io as scio
        scipy.misc.imsave('./crop.png', img_crop)


        mask_location = choose_frame + '/mask/{}.png'.format(str(index).zfill(5))
        mask_target = (cv2.imread(mask_location))[rmin:rmax, cmin:cmax]
        mask_target = mask_target[:, :, 0]
        # mask_target = 255 - mask_target

        depth = depth[rmin:rmax, cmin:cmax]

        depth_mask = depth * mask_target
        choose = (depth_mask.flatten() != 0.0).nonzero()[0]

        # xmap = np.array([[j for i in range(640)] for j in range(480)])
        # ymap = np.array([[i for i in range(640)] for j in range(480)])

        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)

        pc_xz_norm_np = np.linalg.norm(cloud[:, [0, 2]], axis=1)

        # pc_radius_mask_np = pc_xz_norm_np <= 1200
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


        cloud = cloud / 1000.0

        return img, choose, cloud, target




    # def get_frame_camera_coord(self, choose_frame, index, current_r, current_t):
    #
    #     img_location = choose_frame + '/raw/{}.png'.format(str(index).zfill(5))
    #     img = Image.open(img_location)
    #     depth_location = choose_frame + '/depth/{}.png'.format(str(index).zfill(5))
    #     depth = np.array(self.load_depth(depth_location))
    #
    #     pose_location = choose_frame + "/poses.pickle"
    #     target_r, target_t = self.get_pose(pose_location, index)
    #
    #     with open('/import/smartcameras-002/long/CS207_real_dataset/long_dataset/cam_intrinsics.pkl', 'rb') as f:
    #         cam_intr = cPickle.load(f)
    #
    #     cam_cx = cam_intr[0][2]
    #     cam_cy = cam_intr[1][2]
    #     cam_fx = cam_intr[0][0]
    #     cam_fy = cam_intr[1][1]
    #     cam_scale = 1.0
    #
    #     # target = []
    #     # input_file = open(choose_frame + '/model_scales_v2.txt', 'r')
    #     # for i in range(8):
    #     #     input_line = input_file.readline()
    #     #     if input_line[-1:] == '\n':
    #     #         input_line = input_line[:-1]
    #     #     input_line = input_line.split(' ')
    #     #     target.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    #     # input_file.close()
    #     # target = np.array(target)
    #
    #     ### mug
    #     # points_3d_noextend = np.array([[0.0,    0.0,   0.0],
    #     #                                [ 0.07,  0.05,  0.0],
    #     #                                [-0.04,  0.05,  0.0],
    #     #                                [ 0.07, -0.05,  0.0],
    #     #                                [-0.04, -0.05,  0.0],
    #     #                                [ 0.07,  0.05, -0.1],
    #     #                                [-0.04,  0.05, -0.1],
    #     #                                [ 0.07, -0.05, -0.1],
    #     #                                [-0.04, -0.05, -0.1]])
    #     # points_3d_noextend = np.array([[0.0,    0.0,   0.0],
    #     #                                [ 0.07,  0.05,  0.1],
    #     #                                [-0.07,  0.05,  0.1],
    #     #                                [ 0.07, -0.05,  0.1],
    #     #                                [-0.07, -0.05,  0.1],
    #     #                                [ 0.07,  0.05, -0.1],
    #     #                                [-0.07,  0.05, -0.1],
    #     #                                [ 0.07, -0.05, -0.1],
    #     #                                [-0.07, -0.05, -0.1]])
    #
    #     #### laptop
    #     points_3d_noextend = np.array([[0.0,    0.0,   0.0],
    #                                    [ 0.17,  0.13,  0.0],
    #                                    [-0.15,  0.13,  0.0],
    #                                    [ 0.17, -0.13,  0.0],
    #                                    [-0.15, -0.13,  0.0],
    #                                    [ 0.17,  0.13, -0.2],
    #                                    [-0.15,  0.13, -0.2],
    #                                    [ 0.17, -0.13, -0.2],
    #                                    [-0.15, -0.13, -0.2]])
    #
    #     # # BOX
    #     # points_3d_noextend = np.array([[0.0,    0.0,   0.0],
    #     #                                [ 0.07,  0.03,  0.16],
    #     #                                [-0.07,  0.03,  0.16],
    #     #                                [ 0.07, -0.03,  0.16],
    #     #                                [-0.07, -0.03,  0.16],
    #     #                                [ 0.07,  0.03, -0.16],
    #     #                                [-0.07,  0.03, -0.16],
    #     #                                [ 0.07, -0.03, -0.16],
    #     #                                [-0.07, -0.03, -0.16]])
    #
    #     offset = np.tile([0.197, 0.118, 0.0], [9,1])
    #     target = points_3d_noextend + offset
    #
    #
    #     target = self.enlarge_bbox(copy.deepcopy(target))
    #
    #     target_tmp = np.dot(target, target_r.T) + target_t
    #     target_tmp[:, 0] *= -1.0
    #     target_tmp[:, 1] *= -1.0
    #     rmin, rmax, cmin, cmax = get_2dbbox(target_tmp, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale)
    #
    #     limit = search_fit(target)
    #
    #     img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
    #     img = img / 255.0
    #
    #
    #     mask_location = choose_frame + '/mask/{}.png'.format(str(index).zfill(5))
    #     mask_target = (cv2.imread(mask_location))[rmin:rmax, cmin:cmax]
    #     mask_target = mask_target[:, :, 0]
    #     mask_target = 255 - mask_target
    #     # print("mask_target ", mask_target, mask_target)
    #     #         #
    #     #         # choose = (mask_target.flatten() != False).nonzero()[0]
    #     #         # print("chooose   xxx ", len(choose))
    #     #         # if len(choose) == 0:
    #     #         #     return 0
    #
    #     depth = depth[rmin:rmax, cmin:cmax]
    #
    #     depth_mask = depth * mask_target
    #     choose = (depth_mask.flatten() != 0.0).nonzero()[0]
    #
    #     if len(choose) == 0:
    #         return 0
    #     if len(choose) > self.num_pts:
    #         c_mask = np.zeros(len(choose), dtype=int)
    #         c_mask[:self.num_pts] = 1
    #         np.random.shuffle(c_mask)
    #         choose = choose[c_mask.nonzero()]
    #     else:
    #         choose = np.pad(choose, (0, self.num_pts - len(choose)), 'wrap')
    #
    #
    #     depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
    #     xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    #     ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    #     pt2 = depth_masked / cam_scale
    #     pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    #     pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    #     cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)
    #     cloud = cloud / 1000.0
    #
    #     target = target / 1000.0
    #
    #     return img, choose, cloud, target


    def re_scale(self, target_fr, target_to):
        ans_scale = target_fr / target_to
        ans_target = target_fr

        ans_scale = ans_scale[0][0]

        return ans_target, ans_scale

    def getone(self, frame_location, index, current_r, current_t):

        data_location = frame_location

        img_fr, choose_fr, cloud_fr, target = self.get_frame(data_location, index, current_r, current_t)

        anchor_box, scale = self.get_anchor_box(target)

        return self.norm(torch.from_numpy(img_fr.astype(np.float32))).unsqueeze(0), \
            torch.LongTensor(choose_fr.astype(np.int32)).unsqueeze(0), \
            torch.from_numpy(cloud_fr.astype(np.float32)).unsqueeze(0), \
            data_location

    def getfirst(self, frame_location, index):

        pose_location = frame_location + "/poses.pickle"
        current_r, current_t = self.get_pose(pose_location, index)

        return current_r, current_t

    def build_frame(self, min_x, max_x, min_y, max_y, min_z, max_z):
        bbox = []
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, min_y, min_z])
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, min_y, max_z])
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, max_y, min_z])
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, max_y, max_z])

        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([min_x, i, min_z])
        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([min_x, i, max_z])
        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([max_x, i, min_z])
        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([max_x, i, max_z])

        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([min_x, min_y, i])
        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([min_x, max_y, i])
        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([max_x, min_y, i])
        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([max_x, max_y, i])
        bbox = np.array(bbox)

        return bbox

    def projection(self, path, choose_frame, inx, current_r, current_t, add_on):

        img_location = choose_frame + '/raw/{}.png'.format(str(inx).zfill(5))
        img = np.array(Image.open(img_location))

        with open('/import/smartcameras-002/long/CS207_real_dataset/long_dataset/cam_intrinsics.pkl', 'rb') as f:
            cam_intr = cPickle.load(f)

        cam_cx = cam_intr[0][2]
        cam_cy = cam_intr[1][2]
        cam_fx = cam_intr[0][0]
        cam_fy = cam_intr[1][1]

        cam_scale = 1.0

        target_r = current_r
        target_t = current_t

        target = []
        input_file = open(choose_frame + '/model_scale_v2.txt', 'r')
        for i in range(8):
            input_line = input_file.readline()
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            target.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        target = np.array(target)
        target = target * 1000.0

        limit = search_fit(target)
        bbox = self.build_frame(limit[0], limit[1], limit[2], limit[3], limit[4], limit[5])
        anchor_box, scale = self.get_anchor_box(bbox)
        anchor_box = anchor_box * scale
        bbox = np.dot(bbox, target_r.T) + target_t
        bbox[:, 0] *= -1.0
        bbox[:, 1] *= -1.0

        anchor_box = np.dot(anchor_box, target_r.T) + target_t
        anchor_box[:, 0] *= -1.0
        anchor_box[:, 1] *= -1.0

        target = self.enlarge_bbox(copy.deepcopy(target))

        # target = Kp.detach().cpu().numpy()[0] * 1000.0
        # kkk = np.dot(target, target_r.T) + target_t

        if not add_on:
            fw = open('{0}_{1}_pose_False.txt'.format(path, self.index), 'w')
        else:
            fw = open('{0}_{1}_pose.txt'.format(path, self.index), 'w')

        for it in target_r:
            fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        it = target_t
        fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.write('{0}\n'.format(score))
        fw.close()

        # kkk[:, 0] *= -1.0
        # kkk[:, 1] *= -1.0
        #
        # for tg in kkk:
        #     y = int(tg[0] * cam_fx / tg[2] + cam_cx)
        #     x = int(tg[1] * cam_fy / tg[2] + cam_cy)
        #
        #     if x - 3 < 0 or x + 3 > 479 or y - 3 < 0 or y + 3 > 639:
        #         continue
        #
        #     for xxx in range(x - 3, x + 4):
        #         for yyy in range(y - 3, y + 4):
        #             img[xxx][yyy] = self.color[0]

        for tg in bbox:
            y = int(tg[0] * cam_fx / tg[2] + cam_cx)
            x = int(tg[1] * cam_fy / tg[2] + cam_cy)

            if x - 3 < 0 or x + 3 > 479 or y - 3 < 0 or y + 3 > 639:
                continue

            for xxx in range(x - 2, x + 3):
                for yyy in range(y - 2, y + 3):
                    img[xxx][yyy] = self.color[1]

        # tg = anchor_box[www]
        #
        # y = int(tg[0] * cam_fx / tg[2] + cam_cx)
        # x = int(tg[1] * cam_fy / tg[2] + cam_cy)
        #
        # if x - 5 >= 0 and x + 5 <= 479 and y - 5 >= 0 and y + 5 <= 639:
        #     for xxx in range(x - 4, x + 5):
        #         for yyy in range(y - 4, y + 5):
        #             img[xxx][yyy] = self.color[2]

        print(path, img.shape)
        scipy.misc.imsave(path, img)

        if add_on:
            self.index += 1

    def __len__(self):
        return self.length


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
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