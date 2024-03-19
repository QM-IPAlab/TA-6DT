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

import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        k = 7
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        self.refl = nn.ReflectionPad2d(k//2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

compute_ssim_loss = SSIM().to(device)

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

    def pixel2cam(self, depth, intrinsics_inv):
        b, h, w = depth.size()
        pixel_coords = self.set_id_grid(depth)
        current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)
        cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
        return cam_coords * depth.unsqueeze(1)

    def set_id_grid(self, depth):
        b, h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)
        j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)
        ones = torch.ones(1, h, w).type_as(depth)

        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)
        return pixel_coords

    def cam2pixel2(self, cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):

        b, _, h, w = cam_coords.size()
        cam_coords_flat = cam_coords.reshape(b, 3, -1)
        if proj_c2p_rot is not None:
            pcoords = proj_c2p_rot @ cam_coords_flat
        else:
            pcoords = cam_coords_flat

        if proj_c2p_tr is not None:
            pcoords = pcoords + proj_c2p_tr

        X = pcoords[:, 0]
        Y = pcoords[:, 1]
        Z = pcoords[:, 2].clamp(min=1e-3)

        # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        X_norm = 2 * (X / Z) / (w - 1) - 1
        Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
        if padding_mode == 'zeros':
            X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
            # make sure that no point in warped image is a combinaison of im and gray
            X_norm[X_mask] = 2
            Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
            Y_norm[Y_mask] = 2

        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
        return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)

    def mean_on_mask(self, diff, valid_mask):
        mask = valid_mask.expand_as(diff)
        if mask.sum() > 100:
            mean_value = (diff * mask).sum() / mask.sum()
        else:
            mean_value = torch.tensor(0).float().to(device)
        return mean_value

    def inverse_warp(self, img, depth, ref_depth, pose, padding_mode='zeros'):
        """
        Inverse warp a source image to the target image plane.
        Args:
            img: the source image (where to sample pixels) -- [B, 3, H, W]
            depth: depth map of the target image -- [B, 1, H, W]
            ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W]
            pose: 6DoF pose parameters from target to source -- [B, 6]
            intrinsics: camera intrinsic matrix -- [B, 3, 3]
        Returns:
            projected_img: Source image warped to the target image plane
            projected_depth: sampled depth from source image
            computed_depth: computed depth of source image using the target depth
        """
        B, _, H, W = img.size()

        intrinsic = np.array([[591.01250, 0.0, 322.52500], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]])
        intrinsic = torch.from_numpy(intrinsic)
        intrinsic = intrinsic.cuda().float()
        depth = depth.float()

        cam_coords = self.pixel2cam(depth, intrinsic.inverse())
        proj_cam_to_src_pixel = intrinsic @ pose
        rot, tr = proj_cam_to_src_pixel[:, :3], proj_cam_to_src_pixel[:, -1:]
        src_pixel_coords, computed_depth = self.cam2pixel2(cam_coords, rot, tr, padding_mode)
        # projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
        # projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
        projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)
        ref_depth = ref_depth.float()
        projected_depth = F.grid_sample(ref_depth.unsqueeze(0), src_pixel_coords, padding_mode=padding_mode)

        return projected_img, projected_depth, computed_depth

    def forward(self, Kp_fr, Kp_to, img_fr_org, img_to_org, depth_fr_org, depth_to_org, mask_fr_org):


        pred_r, pred_t = self.estimate_pose(Kp_to, Kp_fr)


        img_to_org = img_to_org.float()
        img_fr_org = img_fr_org.float()
        depth_to_org = depth_to_org.float()
        depth_fr_org = depth_fr_org.float()

        pred_RT = torch.cat((torch.squeeze(pred_r), pred_t.t()), 1)
        projected_img, projected_depth, computed_depth = self.inverse_warp(img_to_org, depth_to_org, depth_fr_org, pred_RT)

        mask_fr = mask_fr_org.float()
        mask_fr = mask_fr.unsqueeze(0)
        projected_img = projected_img * mask_fr
        img_fr_org = img_fr_org.float()
        img_fr_org = img_fr_org * mask_fr


        diff_depth = (computed_depth - projected_depth).abs() / (computed_depth + projected_depth)
        diff_depth = diff_depth * mask_fr

        valid_mask_ref = (projected_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()
        valid_mask_tgt = (img_fr_org.abs().mean(dim=1, keepdim=True) > 1e-3).float()
        valid_mask = valid_mask_tgt * valid_mask_ref

        diff_color = (img_fr_org - projected_img).abs().mean(dim=1, keepdim=True)

        diff_img = (img_fr_org - projected_img).abs().clamp(0, 1)

        ssim_map = compute_ssim_loss(img_fr_org, projected_img)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

        diff_img = torch.mean(diff_img, dim=1, keepdim=True)

        photo_loss = self.mean_on_mask(diff_img, valid_mask)
        geometry_loss = self.mean_on_mask(diff_depth, valid_mask)


        loss =  photo_loss*5.0 + geometry_loss*5.0

        print(photo_loss.item(), geometry_loss.item())

        return loss

    def ev(self, Kp_fr, Kp_to, att_to):
        ori_Kp_fr = Kp_fr
        ori_Kp_to = Kp_to

        new_r, new_t = self.estimate_pose(Kp_fr, Kp_to)

        Kp_to = torch.bmm((ori_Kp_to - new_t), new_r)

        Kp_dis = torch.mean(torch.norm((Kp_fr - Kp_to), dim=2), dim=1)

        # new_t *= 1000.0
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