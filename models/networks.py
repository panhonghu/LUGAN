import logging, torch
from torch import nn
import torch.nn.functional as F
from .ResGCN import ResGCN_Module


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #m.bias = None
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def zero_init_lastBN(modules):
    for m in modules:
        if isinstance(m, ResGCN_Module):
            if hasattr(m.scn, 'bn_up'):
                nn.init.constant_(m.scn.bn_up.weight, 0)
            if hasattr(m.tcn, 'bn_up'):
                nn.init.constant_(m.tcn.bn_up.weight, 0)


class PoseEncoder(nn.Module):
    def __init__(self, A, in_channel=3, num_scale=4, num_nodes=17, **kwargs):
        super(PoseEncoder, self).__init__()
        self.A = A
        self.out_channel = num_nodes*num_scale*num_scale
        self.num_scale = num_scale
        # basic blocks
        module_list = [ResGCN_Module(in_channel, 32, 'Basic', self.A, initial=True, **kwargs)]
        module_list += [ResGCN_Module(32, 32, 'Basic', A, initial=True, **kwargs)]
        # residual blocks
        module_list += [ResGCN_Module(32, 64, 'Bottleneck', A,  **kwargs)]
        module_list += [ResGCN_Module(64, 64, 'Bottleneck', A, **kwargs)]
        module_list += [ResGCN_Module(64, 128, 'Bottleneck', A, **kwargs)]
        module_list += [ResGCN_Module(128, 128, 'Bottleneck', A, **kwargs)]
        module_list += [ResGCN_Module(128, self.out_channel, 'Bottleneck', A, **kwargs)]
        self.main_stream = nn.ModuleList(module_list)
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def forward(self, x):  # bs*3*60*17
        for layer in self.main_stream:
            x = layer(x, self.A)    # bs*272*60*17
        bs, _, T, num_nodes = x.shape
        out = x.permute(0, 2, 3, 1).reshape(bs, T, num_nodes*self.num_scale, num_nodes*self.num_scale)
        return out  # bs*60*68*68


class PoseAngleEncoder(nn.Module):
    def __init__(self, A, pose_in_channel=3, angle_in_channel=11, num_scale=4, num_nodes=17, dropout=0.2, **kwargs):
        super(PoseAngleEncoder, self).__init__()
        self.A = A
        self.out_channel = num_nodes*num_scale*num_scale
        self.num_scale = num_scale
        self.num_nodes = num_nodes
        ## GCN blocks
        gcn_module_list = [ResGCN_Module(pose_in_channel, 32, 'Basic', self.A, initial=True, **kwargs)]
        gcn_module_list += [ResGCN_Module(32, 32, 'Basic', A, initial=True, **kwargs)]
        gcn_module_list += [ResGCN_Module(32, 64, 'Bottleneck', A,  **kwargs)]
        gcn_module_list += [ResGCN_Module(64, 64, 'Bottleneck', A, **kwargs)]
        gcn_module_list += [ResGCN_Module(64, 64, 'Bottleneck', A, **kwargs)]
        gcn_module_list += [ResGCN_Module(64, 128, 'Bottleneck', A, **kwargs)]
        gcn_module_list += [ResGCN_Module(128, 128, 'Bottleneck', A, **kwargs)]
        self.gcn_main_stream = nn.ModuleList(gcn_module_list)
        ## FC layers
        fc_module_list = [nn.Linear(angle_in_channel, 64)]
        fc_module_list += [nn.ReLU(inplace=False)]
        fc_module_list += [nn.Linear(64, 128)]
        fc_module_list += [nn.ReLU(inplace=False)]
        fc_module_list += [nn.Dropout(p=dropout, inplace=False)]
        self.fc_main_stream = nn.ModuleList(fc_module_list)
        ## output
        self.fc_last = nn.Linear(128+128, self.out_channel)
        ## parameter initialization
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def forward(self, x, angle):  # bs*3*60*17
        for gcn_layer in self.gcn_main_stream:
            x = gcn_layer(x, self.A)    # bs*128*60*17
        for fc_layer in self.fc_main_stream:
            angle = fc_layer(angle)     # bs*128
        bs, _, T, num_nodes = x.shape
        angle = angle.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, num_nodes)   # bs*128*60*17
        feat = torch.cat((x, angle), dim=1)            # bs*256*60*17
        out = self.fc_last(feat.permute(0, 2, 3, 1))   # bs*60*17*272
        return out.reshape(bs, T, num_nodes*self.num_scale, num_nodes*self.num_scale)  # bs*60*68*68


class CNNLayers(nn.Module):
    def __init__(self, in_channel=60, return_logits=False):
        super(CNNLayers, self).__init__()
        self.return_logits = return_logits
        # A bunch of convolutions one after another
        model = [nn.Conv2d(in_channel, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), \
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), \
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.InstanceNorm2d(512), \
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        if not self.return_logits:
            return x.squeeze(1)
        else:
            return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)



