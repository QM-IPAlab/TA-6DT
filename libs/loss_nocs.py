from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import math
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from libs.knn.__init__ import KNearestNeighbor
import torch.distributions as tdist
import copy


class Loss(_Loss):
    def __init__(self, num_key, num_cate):
        super(Loss, self).__init__(True)
        self.num_key = num_key
        self.num_cate = num_cate

        self.oneone = Variable(torch.ones(1)).cuda()

        self.normal = tdist.Normal(torch.tensor([0.0]), torch.tensor([0.0005]))

        self.pconf = torch.ones(num_key) / num_key
        self.pconf = Variable(self.pconf).cuda()

        self.sym_axis = Variable(torch.from_numpy(np.array([0, 1, 0]).astype(np.float32))).cuda().view(1, 3, 1)
        self.threezero = Variable(torch.from_numpy(np.array([0, 0, 0]).astype(np.float32))).cuda()

        self.zeros = torch.FloatTensor([0.0 for j in range(num_key - 1) for i in range(num_key)]).cuda()

        self.select1 = torch.tensor([i for j in range(num_key - 1) for i in range(num_key)]).cuda()
        self.select2 = torch.tensor([(i % num_key) for j in range(1, num_key) for i in range(j, j + num_key)]).cuda()

        self.knn = KNearestNeighbor(1)

    def estimate_rotation(self, pt0, pt1, sym_or_not):
        pconf2 = self.pconf.view(1, self.num_key, 1)
        cent0 = torch.sum(pt0 * pconf2, dim=1).repeat(1, self.num_key, 1).contiguous()
        cent1 = torch.sum(pt1 * pconf2, dim=1).repeat(1, self.num_key, 1).contiguous()

        diag_mat = torch.diag(self.pconf).unsqueeze(0)
        x = (pt0 - cent0).transpose(2, 1).contiguous()
        y = pt1 - cent1

        pred_t = cent1 - cent0

        cov = torch.bmm(torch.bmm(x, diag_mat), y).contiguous().squeeze(0)

        u, _, v = torch.svd(cov)

        u = u.transpose(1, 0).contiguous()
        d = torch.det(torch.mm(v, u)).contiguous().view(1, 1, 1).contiguous()
        u = u.transpose(1, 0).contiguous().unsqueeze(0)

        ud = torch.cat((u[:, :, :-1], u[:, :, -1:] * d), dim=2)
        v = v.transpose(1, 0).contiguous().unsqueeze(0)

        pred_r = torch.bmm(ud, v).transpose(2, 1).contiguous()

        if sym_or_not:
            pred_r = torch.bmm(pred_r, self.sym_axis).contiguous().view(-1).contiguous()

        return pred_r

    def estimate_pose(self, pt0, pt1):
        pconf2 = self.pconf.view(1, self.num_key, 1)
        cent0 = torch.sum(pt0 * pconf2, dim=1).repeat(1, self.num_key, 1).contiguous()
        cent1 = torch.sum(pt1 * pconf2, dim=1).repeat(1, self.num_key, 1).contiguous()

        diag_mat = torch.diag(self.pconf).unsqueeze(0)
        x = (pt0 - cent0).transpose(2, 1).contiguous()
        y = pt1 - cent1

        pred_t = cent1 - cent0

        cov = torch.bmm(torch.bmm(x, diag_mat), y).contiguous().squeeze(0)

        u, _, v = torch.svd(cov)

        u = u.transpose(1, 0).contiguous()
        d = torch.det(torch.mm(v, u)).contiguous().view(1, 1, 1).contiguous()
        u = u.transpose(1, 0).contiguous().unsqueeze(0)

        ud = torch.cat((u[:, :, :-1], u[:, :, -1:] * d), dim=2)
        v = v.transpose(1, 0).contiguous().unsqueeze(0)

        pred_r = torch.bmm(ud, v).transpose(2, 1).contiguous()
        return pred_r, pred_t[:, 0, :].view(1, 3)

    def recover_6d_pose(self, keypoints1, keypoints2):
        """
        Recovers the 6D pose between two frames given matched 3D keypoints.

        Arguments:
        keypoints1 -- tensor of shape (num_keypoints, 3) representing 3D keypoints in frame 1
        keypoints2 -- tensor of shape (num_keypoints, 3) representing 3D keypoints in frame 2

        Returns:
        translation -- tensor of shape (3,) representing the translation vector from frame 1 to frame 2
        rotation -- tensor of shape (3, 3) representing the rotation matrix from frame 1 to frame 2
        """
        keypoints1 = keypoints1.squeeze(0)
        keypoints2 = keypoints2.squeeze(0)
        # compute the centroids of the two sets of keypoints
        centroid1 = keypoints1.mean(dim=0)
        centroid2 = keypoints2.mean(dim=0)

        # subtract the centroids from the keypoints to center them
        centered1 = keypoints1 - centroid1
        centered2 = keypoints2 - centroid2

        # compute the 3x3 covariance matrix
        covariance = torch.matmul(centered2.transpose(0, 1), centered1)

        # perform SVD on the covariance matrix to get the rotation matrix
        u, s, v = torch.svd(covariance)
        rotation = torch.matmul(v, u.transpose(0, 1))

        # compute the translation vector
        translation = centroid2 - torch.matmul(rotation, centroid1.unsqueeze(-1)).squeeze()

        return translation, rotation

    def change_to_ver(self, Kp):
        pconf2 = self.pconf.view(1, self.num_key, 1)
        cent0 = torch.sum(Kp * pconf2, dim=1).view(-1).contiguous()

        num_kp = self.num_key
        ver_Kp_1 = Kp[:, :, 1].view(1, num_kp, 1).contiguous()

        kk_1 = Kp[:, :, 0].view(1, num_kp, 1).contiguous()
        kk_2 = Kp[:, :, 2].view(1, num_kp, 1).contiguous()
        rad = torch.cat((kk_1, kk_2), dim=2).contiguous()
        ver_Kp_2 = torch.norm(rad, dim=2).view(1, num_kp, 1).contiguous()

        tmp_aim_0 = torch.cat((Kp[:, 1:, :], Kp[:, 0:1, :]), dim=1).contiguous()
        aim_0_x = tmp_aim_0[:, :, 0].view(-1).contiguous()
        aim_0_y = tmp_aim_0[:, :, 2].view(-1).contiguous()

        aim_1_x = Kp[:, :, 0].view(-1).contiguous()
        aim_1_y = Kp[:, :, 2].view(-1).contiguous()

        angle = torch.atan2(aim_1_y, aim_1_x) - torch.atan2(aim_0_y, aim_0_x)
        angle[angle < 0] += 2 * math.pi
        ver_Kp_3 = angle.view(1, num_kp, 1).contiguous() * 0.01

        ver_Kp = torch.cat((ver_Kp_1, ver_Kp_2, ver_Kp_3), dim=2).contiguous()

        return ver_Kp, cent0


    def forward(self, Kp_fr, Kp_to, RT, Cloud_fr, Cloud_to):


        num_kp = self.num_key

        sub_tensors = torch.split(RT, [3, 1], dim=2)
        GT_R = sub_tensors[0]
        GT_T = sub_tensors[1].squeeze(-1)



        ############# Different View Loss
        num_points = Kp_to.shape[1]
        ones_column = torch.ones(1, num_points, 1, dtype=torch.float32, device='cuda')
        Kp_to_trans = torch.cat([Kp_to, ones_column], dim=2)

        Kp_to_trans = torch.bmm(RT, Kp_to_trans.transpose(1, 2))
        Kp_to_trans = Kp_to_trans.transpose(1, 2)

        Kp_dis = torch.mean(torch.norm((Kp_to_trans - Kp_fr), dim=2), dim=1)
        cent_fr = torch.mean(Kp_to_trans, dim=1).view(-1).contiguous()
        cent_to = torch.mean(Kp_fr, dim=1).view(-1).contiguous()
        Kp_cent_dis = (torch.norm(cent_fr - self.threezero) + torch.norm(cent_to - self.threezero)) / 2.0

        ############# Pose Error Loss
        pred_r, pred_t = self.estimate_pose(Kp_fr, Kp_to)


        mse = torch.mean((pred_r - GT_R)**2)
        loss_rot = mse

        mse_t = torch.mean((pred_t - GT_T)**2)
        loss_tra = mse_t



        ############# Close To Surface Loss
        kp_expanded = Kp_fr.expand(Cloud_fr.shape[1], -1, -1)  # shape: 500*8*3
        pc_expanded = Cloud_fr.expand(Kp_fr.shape[1], -1, -1).transpose(0, 1)  # shape: 500*8*3
        distances = torch.norm(kp_expanded - pc_expanded, dim=2)  # shape: 500*8
        min_distances = torch.min(distances, dim=0)[0]  # shape: 8
        loss_fr = torch.mean(min_distances)

        kp_expanded = Kp_to.expand(Cloud_to.shape[1], -1, -1)  # shape: 500*8*3
        pc_expanded = Cloud_to.expand(Kp_to.shape[1], -1, -1).transpose(0, 1)  # shape: 500*8*3
        distances = torch.norm(kp_expanded - pc_expanded, dim=2)  # shape: 500*8
        min_distances = torch.min(distances, dim=0)[0]  # shape: 8
        loss_to = torch.mean(min_distances)

        loss_surf = (loss_fr + loss_to)/2.0



        threshold = 0.04
        batch_size, num_keypoints, _ = Kp_fr.shape
        dists_fr = torch.norm(Kp_fr.unsqueeze(1) - Kp_fr.unsqueeze(2), dim=-1)
        mask_fr = 1 - torch.eye(num_keypoints, device=Kp_fr.device)
        dists_fr = dists_fr * mask_fr
        loss_sep_fr = torch.sum(torch.relu(threshold - dists_fr))
        loss_sep_fr /= (batch_size * num_keypoints * (num_keypoints - 1) / 2)

        dists_to = torch.norm(Kp_to.unsqueeze(1) - Kp_to.unsqueeze(2), dim=-1)
        mask_to = 1 - torch.eye(num_keypoints, device=Kp_fr.device)
        dists_to = dists_to * mask_to
        loss_sep_to = torch.sum(torch.relu(threshold - dists_to))
        loss_sep_to /= (batch_size * num_keypoints * (num_keypoints - 1) / 2)

        loss_sep = (loss_sep_fr + loss_sep_to) / 2.0

        ########### SUM UP

        loss =  loss_surf + loss_tra * 2.0 + loss_rot + Kp_dis + loss_sep
        score = (loss_tra + loss_rot).item()

        print( loss_sep.item(), loss_surf.item(), loss_rot.item(), loss_tra.item())

        return loss, score

    def ev(self, Kp_fr, Kp_to):
        ori_Kp_fr = Kp_fr
        ori_Kp_to = Kp_to
        new_r, new_t = self.estimate_pose(Kp_fr, Kp_to)
        ori_Kp_to = ori_Kp_to.unsqueeze(0)
        new_t = new_t.unsqueeze(0)
        Kp_to = torch.bmm((ori_Kp_to - new_t), new_r)

        Kp_dis = torch.mean(torch.norm((Kp_fr - Kp_to), dim=2), dim=1)

        new_t *= 1000.0
        return ori_Kp_fr, new_r.detach().cpu().numpy()[0], new_t.detach().cpu().numpy()[0], Kp_dis.item()

    def ev_org(self, Kp_fr, Kp_to, att_to):
        ori_Kp_fr = Kp_fr
        ori_Kp_to = Kp_to

        new_r, new_t = self.estimate_pose(Kp_fr, Kp_to)

        Kp_to = torch.bmm((ori_Kp_to - new_t), new_r)

        Kp_dis = torch.mean(torch.norm((Kp_fr - Kp_to), dim=2), dim=1)

        new_t *= 1000.0
        return ori_Kp_fr, new_r.detach().cpu().numpy()[0], new_t.detach().cpu().numpy()[0], Kp_dis.item(), att_to

    def ev_zero(self, Kp_fr, att_fr):
        pconf2 = self.pconf.view(1, self.num_key, 1)
        new_t = torch.sum(Kp_fr * pconf2, dim=1).view(1, 3).contiguous()

        kp_dis = torch.norm(new_t.view(-1))

        new_t *= 1000.0
        return new_t.detach().cpu().numpy()[0], att_fr, kp_dis.item()

    def inf(self, Kp_fr, Kp_to):
        ori_Kp_to = Kp_to

        new_r, new_t = self.estimate_pose(Kp_fr, Kp_to)

        Kp_to = torch.bmm((ori_Kp_to - new_t), new_r)

        Kp_dis = torch.mean(torch.norm((Kp_fr - Kp_to), dim=2), dim=1)

        new_t *= 1000.0
        return new_r.detach().cpu().numpy()[0], new_t.detach().cpu().numpy()[0], Kp_dis.item()

    def inf_zero(self, Kp_fr):
        pconf2 = self.pconf.view(1, self.num_key, 1)
        new_t = torch.sum(Kp_fr * pconf2, dim=1).view(1, 3).contiguous()

        Kp_dis = torch.norm(new_t.view(-1))

        new_t *= 1000.0
        return new_t.detach().cpu().numpy()[0], Kp_dis.item()