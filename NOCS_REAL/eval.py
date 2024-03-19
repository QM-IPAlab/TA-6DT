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
from dataset.eval_nocs_dataset import Dataset
from libs.network import KeyNet
from libs.loss import Loss
import copy

import cv2

choose_cate_list = [5]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--eval_id', type=int, default = 1, help='the evaluation id')
parser.add_argument('--ite', type=int, default=10, help='first frame fix iteration')
parser.add_argument('--num_kp', type=int, default = 8, help='num of kp')
parser.add_argument('--num_points', type=int, default = 500, help='num of input points')
parser.add_argument('--num_cates', type=int, default = 6, help='number of categories')
parser.add_argument('--checkpoint', type=str, default = '', help='load model dir')
opt = parser.parse_args()


if not os.path.exists('eval_results'):
    os.makedirs('eval_results')

if not os.path.exists('eval_results/TEST_{0}'.format(opt.eval_id)):
    os.makedirs('eval_results/TEST_{0}'.format(opt.eval_id))
    for item in choose_cate_list:
        os.makedirs('eval_results/TEST_{0}/temp_{1}'.format(opt.eval_id, item))

model = KeyNet(num_points = opt.num_points, num_key = opt.num_kp, num_cates = opt.num_cates)
model.cuda()
model.eval()

if opt.checkpoint != "":
    model.load_state_dict(torch.load(opt.checkpoint))
    print("Yes, pre-trained model is loaded")


pconf = torch.ones(opt.num_kp) / opt.num_kp
pconf = Variable(pconf).cuda()

test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, choose_cate_list[0], 1000)
criterion = Loss(opt.num_kp, opt.num_cates)



choose_obj = 'mug_daniel_norm'
choose_video = 'scene_1'
frame_location = '{0}/real_val/{1}'.format(opt.dataset_root, choose_video)





def find_next_frame(location, index):
    index = index + 1
    path = location + '/{}_color.png'.format(str(index).zfill(4))

    while not os.path.exists(path):
        index = index + 1
        path = frame_location + '/{}_color.png'.format(str(index).zfill(4))
        if os.path.exists(path):
            break
    return index

### first frame ###
i = 0
current_r, current_t = test_dataset.getfirst(frame_location, choose_obj, i)
current_r_1, current_t_1 = current_r, current_t
img_fr, choose_fr, cloud_fr, anchor, scale, choose_frame_1 = test_dataset.getone(frame_location, choose_obj, i, current_r, current_t)
img_fr, choose_fr, cloud_fr, anchor, scale = Variable(img_fr).cuda(), \
                                        Variable(choose_fr).cuda(), \
                                        Variable(cloud_fr).cuda(), \
                                        Variable(anchor).cuda(), \
                                        Variable(scale).cuda()


for j in range(1, 1000):
    path = frame_location + '/{}_color.png'.format(str(i).zfill(4))
    if os.path.exists(path):

        img_to, choose_to, cloud_to, anchor, scale, choose_frame_2 = test_dataset.getone(frame_location, choose_obj, j, current_r, current_t)
        img_to, choose_to, cloud_to, anchor, scale = Variable(img_to).cuda(), \
                                                    Variable(choose_to).cuda(), \
                                                    Variable(cloud_to).cuda(), \
                                                    Variable(anchor).cuda(), \
                                                    Variable(scale).cuda()

        Kp_fr, Kp_to = model.eval_one_anchor_forward(img_fr, choose_fr, cloud_fr, img_to, choose_to, cloud_to)

        save_pointcloud = cloud_to.detach().cpu().numpy()
        save_keypoint = Kp_to.detach().cpu().numpy()


        Kp_real, new_r, new_t, kp_dis = criterion.ev_one_anchor(Kp_fr, Kp_to)
        new_t = new_t/1000.0

        best_r = new_r
        best_t = new_t

        current_t = current_t + np.dot(best_t, current_r.T)
        current_r = np.dot(current_r, best_r)


        save_path = '/Visualization/{0}_{1}_{2}_bbox.png'.format(choose_obj, choose_video, j)
        img_1 = test_dataset.projection(save_path, choose_frame_2, choose_obj, Kp_to, current_r, current_t, True)




