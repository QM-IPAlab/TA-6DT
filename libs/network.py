import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from libs.pspnet import PSPNet
import torch.distributions as tdist
import copy

from copy import deepcopy
from typing import List, Tuple

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 256, 1)

        self.all_conv1 = torch.nn.Conv1d(640, 320, 1)
        self.all_conv2 = torch.nn.Conv1d(320, 160, 1)

        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = torch.cat([pointfeat_1, pointfeat_2, x], dim=1).contiguous()  # 128 + 256 + 256

        x = F.leaky_relu(self.all_conv1(x))
        x = self.all_conv2(x)

        return x


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', [query, key]) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', [prob, value]), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self):
        super().__init__()
        feature_dim = 160
        layer_names = ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self',
                       'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross']
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


class KeyNet(nn.Module):
    def __init__(self, num_points, num_key, num_cates):
        super(KeyNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)
        self.num_cates = num_cates

        self.sm = torch.nn.Softmax(dim=2)



        self.num_key = num_key

        self.threezero = Variable(torch.from_numpy(np.array([0, 0, 0]).astype(np.float32))).cuda().view(1, 1, 3).repeat(
            1, self.num_points, 1)

        self.gnn = AttentionalGNN()

        self.conv1 = torch.nn.Conv1d(in_channels=160, out_channels=64, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        self.conv4 = torch.nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1)
        self.conv5 = torch.nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1)

        self.fc0 = torch.nn.Linear(in_features=2048, out_features=1024)
        self.fc1 = torch.nn.Linear(in_features=1024, out_features=512)
        self.fc2 = torch.nn.Linear(in_features=512, out_features=256)
        self.fc3 = torch.nn.Linear(in_features=256, out_features=128)
        self.fc4 = torch.nn.Linear(in_features=128, out_features=48)


    def forward(self, img_1, choose_1, x_1, img_2, choose_2, x_2):

        out_img_1 = self.cnn(img_1)
        out_img_2 = self.cnn(img_2)

        bs_1, di_1, _, _ = out_img_1.size()
        bs_2, di_2, _, _ = out_img_2.size()

        emb_1 = out_img_1.view(bs_1, di_1, -1)
        emb_2 = out_img_2.view(bs_2, di_2, -1)

        choose_1 = choose_1.repeat(1, di_1, 1)
        choose_2 = choose_2.repeat(1, di_2, 1)

        emb_1 = torch.gather(emb_1, 2, choose_1).contiguous()
        emb_2 = torch.gather(emb_2, 2, choose_2).contiguous()

        x_mean_1 = torch.mean(x_1, dim=1).view(bs_1, 1, -1)
        x_1 = x_1 - x_mean_1

        x_mean_2 = torch.mean(x_2, dim=1).view(bs_2, 1, -1)
        x_2 = x_2 - x_mean_2


        x_1 = x_1.transpose(2, 1).contiguous()
        ap_x_1 = self.feat(x_1, emb_1)

        x_2 = x_2.transpose(2, 1).contiguous()
        ap_x_2 = self.feat(x_2, emb_2)


        feat_x_1, feat_x_2 = self.gnn(ap_x_1, ap_x_2)

        feat_x_1 = self.conv1(feat_x_1)
        feat_x_1 = torch.nn.functional.relu(feat_x_1)
        feat_x_1 = self.conv2(feat_x_1)
        feat_x_1 = torch.nn.functional.relu(feat_x_1)
        feat_x_1 = self.conv3(feat_x_1)
        feat_x_1 = torch.nn.functional.relu(feat_x_1)
        feat_x_1 = self.conv4(feat_x_1)
        feat_x_1 = torch.nn.functional.relu(feat_x_1)
        feat_x_1 = self.conv5(feat_x_1)
        feat_x_1 = torch.nn.functional.relu(feat_x_1)
        feat_x_1 = torch.max(feat_x_1, dim=2)[0]

        feat_x_2 = self.conv1(feat_x_2)
        feat_x_2 = torch.nn.functional.relu(feat_x_2)
        feat_x_2 = self.conv2(feat_x_2)
        feat_x_2 = torch.nn.functional.relu(feat_x_2)
        feat_x_2 = self.conv3(feat_x_2)
        feat_x_2 = torch.nn.functional.relu(feat_x_2)
        feat_x_2 = self.conv4(feat_x_2)
        feat_x_2 = torch.nn.functional.relu(feat_x_2)
        feat_x_2 = self.conv5(feat_x_2)
        feat_x_2 = torch.nn.functional.relu(feat_x_2)
        feat_x_2 = torch.max(feat_x_2, dim=2)[0]


        x_2 = torch.cat((feat_x_1, feat_x_2), dim=1)

        x = self.fc0(x_2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        matched_keypoints = x.view(-1, 8, 2, 3)

        kp_1 = matched_keypoints[:, :, 0, :]
        kp_2 = matched_keypoints[:, :, 1, :]

        kp_1 = kp_1 + x_mean_1
        kp_2 = kp_2 + x_mean_2

        return kp_1, kp_2


    def eval_forward(self, img, choose, ori_x, anchor, scale, space, first):
        num_anc = len(anchor[0])
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose)
        emb = emb.repeat(1, 1, num_anc).detach()
        # print(emb.size())

        output_anchor = anchor.view(1, num_anc, 3)
        anchor_for_key = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_key, 1)
        anchor = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_points, 1)

        candidate_list = [-10 * space, 0.0, 10 * space]
        if space != 0.0:
            add_on = []
            for add_x in candidate_list:
                for add_y in candidate_list:
                    for add_z in candidate_list:
                        add_on.append([add_x, add_y, add_z])

            add_on = Variable(torch.from_numpy(np.array(add_on).astype(np.float32))).cuda().view(27, 1, 3)
        else:
            add_on = Variable(torch.from_numpy(np.array([0.0, 0.0, 0.0]).astype(np.float32))).cuda().view(1, 1, 3)

        all_kp_x = []
        all_att_choose = []
        scale_add_on = scale.view(1, 3)

        for tmp_add_on in add_on:
            tmp_add_on_scale = (tmp_add_on / scale_add_on).view(1, 1, 3).repeat(1, self.num_points, 1)
            tmp_add_on_key = (tmp_add_on / scale_add_on).view(1, 1, 3).repeat(1, self.num_key, 1)
            x = ori_x - tmp_add_on_scale

            x = x.view(1, 1, self.num_points, 3).repeat(1, num_anc, 1, 1)
            x = (x - anchor).view(1, num_anc * self.num_points, 3)

            x = x.transpose(2, 1)
            feat_x = self.feat(x, emb)
            feat_x = feat_x.transpose(2, 1)
            feat_x = feat_x.view(1, num_anc, self.num_points, 160).detach()

            loc = x.transpose(2, 1).view(1, num_anc, self.num_points, 3)
            weight = self.sm(-1.0 * torch.norm(loc, dim=3))
            weight = weight.view(1, num_anc, self.num_points, 1).repeat(1, 1, 1, 160)

            feat_x = torch.sum((feat_x * weight), dim=2).view(1, num_anc, 160)
            feat_x = feat_x.transpose(2, 1).detach()

            kp_feat = F.leaky_relu(self.kp_1(feat_x))
            kp_feat = self.kp_2(kp_feat)
            kp_feat = kp_feat.transpose(2, 1)
            kp_x = kp_feat.view(1, num_anc, self.num_key, 3)
            kp_x = (kp_x + anchor_for_key).detach()

            att_feat = F.leaky_relu(self.att_1(feat_x))
            att_feat = self.att_2(att_feat)
            att_feat = att_feat.transpose(2, 1)
            att_feat = att_feat.view(1, num_anc)
            att_x = self.sm2(att_feat).detach()

            if not first:
                att_choose = torch.argmax(att_x.view(-1))
            else:
                att_choose = Variable(torch.from_numpy(np.array([62])).long()).cuda().view(-1)

            scale_anc = scale.view(1, 1, 3).repeat(1, num_anc, 1)
            output_anchor = (output_anchor * scale_anc)

            scale_kp = scale.view(1, 1, 3).repeat(1, self.num_key, 1)
            kp_x = kp_x.view(1, num_anc, 3 * self.num_key).detach()
            kp_x = (kp_x[:, att_choose, :].view(1, self.num_key, 3) + tmp_add_on_key).detach()

            kp_x = kp_x * scale_kp

            all_kp_x.append(copy.deepcopy(kp_x.detach()))
            all_att_choose.append(copy.deepcopy(att_choose.detach()))

        return all_kp_x, all_att_choose


    def eval_one_anchor_forward(self, img_1, choose_1, x_1,  img_2, choose_2, x_2):

        out_img_1 = self.cnn(img_1)
        out_img_2 = self.cnn(img_2)

        bs_1, di_1, _, _ = out_img_1.size()
        bs_2, di_2, _, _ = out_img_2.size()

        emb_1 = out_img_1.view(bs_1, di_1, -1)
        emb_2 = out_img_2.view(bs_2, di_2, -1)

        choose_1 = choose_1.repeat(1, di_1, 1)
        choose_2 = choose_2.repeat(1, di_2, 1)

        emb_1 = torch.gather(emb_1, 2, choose_1).contiguous()
        emb_2 = torch.gather(emb_2, 2, choose_2).contiguous()

        x_mean_1 = torch.mean(x_1, dim=1).view(bs_1, 1, -1)
        x_1 = x_1 - x_mean_1

        x_mean_2 = torch.mean(x_2, dim=1).view(bs_2, 1, -1)
        x_2 = x_2 - x_mean_2

        x_1 = x_1.transpose(2, 1).contiguous()
        ap_x_1 = self.feat(x_1, emb_1)

        x_2 = x_2.transpose(2, 1).contiguous()
        ap_x_2 = self.feat(x_2, emb_2)

        feat_x_1, feat_x_2 = self.gnn(ap_x_1, ap_x_2)

        feat_x_1 = self.conv1(feat_x_1)
        feat_x_1 = torch.nn.functional.relu(feat_x_1)
        feat_x_1 = self.conv2(feat_x_1)
        feat_x_1 = torch.nn.functional.relu(feat_x_1)
        feat_x_1 = self.conv3(feat_x_1)
        feat_x_1 = torch.nn.functional.relu(feat_x_1)
        feat_x_1 = self.conv4(feat_x_1)
        feat_x_1 = torch.nn.functional.relu(feat_x_1)
        feat_x_1 = self.conv5(feat_x_1)
        feat_x_1 = torch.nn.functional.relu(feat_x_1)
        feat_x_1 = torch.max(feat_x_1, dim=2)[0]

        feat_x_2 = self.conv1(feat_x_2)
        feat_x_2 = torch.nn.functional.relu(feat_x_2)
        feat_x_2 = self.conv2(feat_x_2)
        feat_x_2 = torch.nn.functional.relu(feat_x_2)
        feat_x_2 = self.conv3(feat_x_2)
        feat_x_2 = torch.nn.functional.relu(feat_x_2)
        feat_x_2 = self.conv4(feat_x_2)
        feat_x_2 = torch.nn.functional.relu(feat_x_2)
        feat_x_2 = self.conv5(feat_x_2)
        feat_x_2 = torch.nn.functional.relu(feat_x_2)
        feat_x_2 = torch.max(feat_x_2, dim=2)[0]

        x_2 = torch.cat((feat_x_1, feat_x_2), dim=1)

        x = self.fc0(x_2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        matched_keypoints = x.view(-1, 8, 2, 3)

        kp_1 = matched_keypoints[:, :, 0, :]
        kp_2 = matched_keypoints[:, :, 1, :]

        kp_1 = kp_1 + x_mean_1
        kp_2 = kp_2 + x_mean_2

        return kp_1, kp_2
