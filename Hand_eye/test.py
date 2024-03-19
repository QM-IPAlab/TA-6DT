import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from dataset_real import Dataset
import cv2
import pickle
from PIL import Image
import scipy.misc
import scipy.io as scio
import math

import sys
sys.path.append("..")
from libs.network import KeyNet
from libs.loss import Loss
from libs.transformations import euler_matrix

import copy


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '/import/smartcameras-002/long/CS207_real_dataset/cracker_box', help='dataset root dir')
parser.add_argument('--eval_id', type=int, default = 1, help='the evaluation id')
parser.add_argument('--ite', type=int, default=10, help='first frame fix iteration')
parser.add_argument('--num_kp', type=int, default = 8, help='num of kp')
parser.add_argument('--num_points', type=int, default = 500, help='num of input points')
parser.add_argument('--num_cates', type=int, default = 6, help='number of categories')
parser.add_argument('--outf', type=str, default = 'models/', help='load model dir')
parser.add_argument('--checkpoint', type=str, default = '', help='trained model location')
opt = parser.parse_args()





model = KeyNet(num_points = opt.num_points, num_key = opt.num_kp, num_cates = opt.num_cates)
model.cuda()
model.eval()


if opt.checkpoint != "":
    model.load_state_dict(torch.load(opt.checkpoint))
    print("Yes, pre-trained model is loaded")

pconf = torch.ones(opt.num_kp) / opt.num_kp
pconf = Variable(pconf).cuda()


test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, 1000)
criterion = Loss(opt.num_kp, opt.num_cates)


def project_3d_to_2d(keypoints_3d, intrinsic_matrix, rotation_matrix, translation_vector):
    extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(3, 1)))

    # Project 3D keypoints to 2D image plane
    keypoints_2d, _ = cv2.projectPoints(keypoints_3d, extrinsic_matrix, intrinsic_matrix, distCoeffs=None)

    return keypoints_2d.reshape(-1, 2)




frame_location = '{0}/raw/{1}.png'.format(opt.dataset_root, str(1).zfill(5))

img_fr, choose_fr, cloud_fr = test_dataset.getone(frame_location)
img_fr, choose_fr, cloud_fr = Variable(img_fr).cuda(), \
                            Variable(choose_fr).cuda(), \
                            Variable(cloud_fr).cuda()



for j in range(1, 2000):
    path = '{}/raw/{}.png'.format(opt.dataset_root, str(j).zfill(5))

    if os.path.exists(path):
        img_to, choose_to, cloud_to = test_dataset.getone(path)
        img_to, choose_to, cloud_to = Variable(img_fr).cuda(), \
                                    Variable(choose_fr).cuda(), \
                                    Variable(cloud_fr).cuda()


        Kp_fr, Kp_to = model.eval_one_anchor_forward(img_fr, choose_fr, cloud_fr, img_to, choose_to, cloud_to)



        save_pointcloud = cloud_to.detach().cpu().numpy()
        save_keypoint = Kp_to.detach().cpu().numpy()
        save_keypoint = np.squeeze(save_keypoint)
        save_keypoint[:, 0] *= -1.0
        save_keypoint[:, 1] *= -1.0


        _, pred_r, pred_t, _ = criterion.ev_one_anchor(Kp_fr, Kp_to)

        pred_t = pred_t / 1000.0





        left_x = (13.5) / 200.0
        right_x = (13.5) / 200.0
        width = (5.) / 200.0
        height = (16.) / 100.0

        points_3d_noextend = np.array([[0.0, 0.0, 0.0],
                                       [left_x, width, 0.0],
                                       [-right_x, width, 0.0],
                                       [left_x, -width, 0.0],
                                       [-right_x, -width, 0.0],
                                       [left_x, width, height],
                                       [-right_x, width, height],
                                       [left_x, -width, height],
                                       [-right_x, -width, height]])

        offset = np.tile([0.197, 0.118, 0.0], [9,1])
        points_3d_noextend = points_3d_noextend + offset



        with open(opt.dataset_root + '/cam_intrinsics.pkl', 'rb') as f:
            cam3intrinsics = pickle.load(f)

        points2d, _ = cv2. projectPoints(points_3d_noextend, pred_r, pred_t, cam3intrinsics, np.array([0., 0., 0., 0., 0.]))

        keypoints_2d_projected, _ = cv2.projectPoints(save_keypoint, np.zeros((3, 1)), np.zeros((3, 1)), cam3intrinsics, None)

        img_location = frame_location + '/raw/{}.png'.format(str(j).zfill(5))
        img = np.array(Image.open(img_location))



        cam3_image_temp = img
        links = [[1, 3], [3, 4], [4, 2], [2, 1],
                 [5, 7], [7, 8], [8, 6], [6, 5],
                 [1, 5], [3, 7], [4, 8], [2, 6]]

        for l in links:
            cam3_image_temp = cv2.line(cam3_image_temp, (int(points2d[l[0]][0][0]), int(points2d[l[0]][0][1])),
                                       (int(points2d[l[1]][0][0]), int(points2d[l[1]][0][1])), (0, 255, 0), 4)

        save_path = '/eval_results/{0}.png'.format(str(j).zfill(5))
        print(save_path)

        scipy.misc.imsave(save_path, cam3_image_temp)
        print("next frame")








