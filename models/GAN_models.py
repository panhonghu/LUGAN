import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torch.autograd import Variable
import torch.autograd as autograd
from .networks import *


class LUGenerator(nn.Module):
    def __init__(self, A, pose_in_channel=3, angle_in_channel=11, num_scale=4, num_nodes=17, T=60):
        super(LUGenerator, self).__init__()
        self.pose_angle_encoder = PoseAngleEncoder(A=A, \
                                                   pose_in_channel=pose_in_channel, \
                                                   angle_in_channel=angle_in_channel, \
                                                   num_scale=num_scale, \
                                                   num_nodes=num_nodes
                                                  )
        self.U_CNN = CNNLayers(in_channel=T)
        self.L_CNN = CNNLayers(in_channel=T)

    def forward(self, x, angle):  # bs*3*60*17
        ## encode pose and view
        feat_pose_and_angle = self.pose_angle_encoder(x, angle)  # 80*60*68*68
        # ## compute L and U
        # Q_U = self.U_CNN(feat_pose_and_angle)
        # mask_U = torch.triu(torch.ones(3, 3), diagonal=0)
        # mask_U = torch.autograd.Variable(mask_U, requires_grad=True).to(Q_U.device)
        # Q_U = Q_U * mask_U
        # Q_L = self.L_CNN(feat_pose_and_angle)
        # mask_L = torch.tril(torch.ones(3, 3), diagonal=0)
        # mask_L = torch.autograd.Variable(mask_L, requires_grad=True).to(Q_L.device)
        # Q_L = Q_L * mask_L
        # # print('Q_L -> ', Q_L[0])
        # # print('Q_U -> ', Q_U[0])
        # # print('matmul -> ', torch.matmul(Q_L, Q_U)[0])
        # return torch.matmul(Q_L, Q_U)

        ## compute L and U
        Q_U = self.U_CNN(feat_pose_and_angle)
        return Q_U




class Discriminator(nn.Module):
    def __init__(self, A, pose_in_channel=3, angle_in_channel=11, num_scale=4, num_nodes=17, T=60):
        super(Discriminator, self).__init__()
        self.conditional_encoder = PoseAngleEncoder(A=A, \
                                                    pose_in_channel=pose_in_channel, \
                                                    angle_in_channel=angle_in_channel, \
                                                    num_scale=num_scale, \
                                                    num_nodes=num_nodes
                                                   )
        self.pose_encoder = PoseEncoder(A=A, in_channel=pose_in_channel, num_scale=num_scale, num_nodes=num_nodes)
        self.CNN_layers = CNNLayers(in_channel=T*2, return_logits=True)

    def forward(self, input_x, con_pose, con_angle):
        con_feat = self.conditional_encoder(con_pose, con_angle)  # 80*60*68*68
        input_feat = self.pose_encoder(input_x)                   # 80*60*68*68
        logits = self.CNN_layers(torch.cat((input_feat, con_feat), dim=1))
        return logits


# class Discriminator(nn.Module):
#     def __init__(self, A, pose_in_channel=3, num_scale=4, num_nodes=17, T=60):
#         super(Discriminator, self).__init__()
#         self.pose_encoder = PoseEncoder(A=A, in_channel=pose_in_channel, num_scale=num_scale, num_nodes=num_nodes)
#         self.CNN_layers = CNNLayers(in_channel=T, return_logits=True)

#     def forward(self, input_x):
#         input_feat = self.pose_encoder(input_x)  # 80*60*68*68
#         logits = self.CNN_layers(input_feat)
#         return logits




