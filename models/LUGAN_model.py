import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torch.autograd import Variable
import torch.autograd as autograd
from .GAN_models import LUGenerator, Discriminator
from .graph import Graph


class LUGAN(nn.Module):
    def __init__(self, args, logger, is_training):
        super(LUGAN, self).__init__()
        # configurations
        self.args = args
        self.logger = logger
        self.is_training = is_training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # load graph
        self.graph = Graph("coco")
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.A = A.cuda(non_blocking=True)
        self.G = LUGenerator(A=self.A, pose_in_channel=args.pose_in_channel, angle_in_channel=args.angle_in_channel, \
                             num_scale=args.num_scale, num_nodes=args.num_nodes, T=args.sequence_length)
        self.D = Discriminator(A=self.A, pose_in_channel=args.pose_in_channel, angle_in_channel=args.angle_in_channel, \
                               num_scale=args.num_scale, num_nodes=args.num_nodes, T=args.sequence_length)
        if self.is_training:
            # model input
            self.points_src, self.points_tgt = None, None
            self.angle_src, self.angle_tgt = None, None
            self.Q_src, self.Q_tgt = None, None
            self.points_src2tgt, self.points_tgt2src = None, None
            # Set up criterion and optimizers
            self.criterion_mse = torch.nn.MSELoss()
            self.criterion_l1 = torch.nn.SmoothL1Loss()
            Tensor = torch.cuda.FloatTensor if self.device=='cuda' else torch.Tensor
            self.target_real = Variable(Tensor(args.batch_size, 1).fill_(1.0), requires_grad=False)
            self.target_fake = Variable(Tensor(args.batch_size, 1).fill_(0.0), requires_grad=False)
            self.identity = torch.autograd.Variable(torch.eye(3), requires_grad=True).cuda()
            self.opt_G = torch.optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
            self.opt_D = torch.optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))


    def __train_generator(self):
        self.Q_src = self.G(self.points_src, self.angle_tgt)
        bs, _, T, num_nodes = self.points_src.shape
        ## source2target
        self.points_src2tgt =  torch.einsum("ijkl,ilm->ijkm", [self.points_src.permute(0, 2, 3, 1), self.Q_src])
        self.points_src2tgt = self.points_src2tgt.permute(3,0,1,2)
        self.points_src2tgt = (self.points_src2tgt / self.points_src2tgt[-1]).permute(1, 0, 2, 3)  # 80*3*60*17
        logits = self.D(self.points_src2tgt, self.points_src, self.angle_tgt)
        loss_GAN_G = self.criterion_mse(logits, self.target_real)
        # print('self.points_src2tgt -> ', self.points_src2tgt[0][0:2])
        # print('self.points_tgt -> ', self.points_tgt[0][0:2])
        # print('loss_GAN_G -> ', loss_GAN_G)
        return loss_GAN_G


    def __train_discriminator(self):
        ## source2target
        logits_fake = self.D(self.points_src2tgt.detach().contiguous(), self.points_src, self.angle_tgt)
        logits_real = self.D(self.points_tgt, self.points_src, self.angle_tgt)
        loss_GAN_D_fake = self.criterion_mse(logits_fake, self.target_fake)
        loss_GAN_D_real = self.criterion_mse(logits_real, self.target_real)
        # print('loss_GAN_D_fake -> ', loss_GAN_D_fake)
        # print('loss_GAN_D_real -> ', loss_GAN_D_real)
        return loss_GAN_D_fake, loss_GAN_D_real


    def train_one_epoch(self, train_loader, epoch):
        for idx, (points_src, angle_src, points_tgt, angle_tgt, _) in enumerate(train_loader):
            if self.device=='cuda':
                points_src = points_src.cuda() / 416.0   # bs*2*T*17
                points_tgt = points_tgt.cuda() / 416.0   # bs*2*T*17
                angle_src = angle_src.cuda()             # bs*11
                angle_tgt = angle_tgt.cuda()             # bs*11

            # data preprocessing for pose graph
            bs, _, T, num_nodes = points_src.shape
            assert points_src.shape==points_tgt.shape, "Check input for LUGAN!!"
            ones_padding = torch.ones(bs, 1, T, num_nodes).float()
            ones_padding = torch.autograd.Variable(ones_padding, requires_grad=True).to(points_src.device)
            self.points_src = torch.cat((points_src, ones_padding), dim=1)   # bs*3*T*17
            self.points_tgt = torch.cat((points_tgt, ones_padding), dim=1)   # bs*3*T*17
            # data preprocessing for angles
            self.angle_src = angle_src
            self.angle_tgt = angle_tgt

            ## training generator
            for i in range(self.args.G_steps):
                self.opt_G.zero_grad()
                loss_GAN_G = self.__train_generator()
                loss_GAN_G.backward()
                print('loss_GAN_G -> ', loss_GAN_G)
                self.opt_G.step()
            ## training discriminator
            for i in range(self.args.D_steps):
                self.opt_D.zero_grad()
                loss_GAN_D_fake, loss_GAN_D_real = self.__train_discriminator()
                loss_GAN_D = loss_GAN_D_fake + loss_GAN_D_real
                loss_GAN_D.backward()
                print('loss_GAN_D -> ', loss_GAN_D)
                self.opt_D.step()

            ## save good model!!
            if epoch>2 and \
               loss_GAN_G.detach().cpu()<0.8 and loss_GAN_G.detach().cpu()>0.2 and \
               loss_GAN_D.detach().cpu()<0.8 and loss_GAN_D.detach().cpu()>0.2 and \
               torch.min(self.points_src2tgt.detach())>0 and torch.max(self.points_src2tgt.detach())==1 :
                # print('loss_GAN_G -> %5f', loss_GAN_G.detach().cpu())
                # print('loss_GAN_D -> %5f', loss_GAN_D.detach().cpu())
                self.save_models(epoch, is_middle=True)
            if idx%10==0:
                self.logger.info('epoch: %d | idx: %d' % (epoch, idx))
                self.logger.info('loss_GAN_G -> %5f', loss_GAN_G.detach().cpu())
                self.logger.info('loss_GAN_D_fake -> %5f', loss_GAN_D_fake.detach().cpu())
                self.logger.info('loss_GAN_D_real -> %5f', loss_GAN_D_real.detach().cpu())
            return_G, return_D = loss_GAN_G.detach().cpu(), loss_GAN_D.detach().cpu()
        return return_G, return_D


    def save_models(self, epoch, is_middle=False):
        if os.path.exists(self.args.save_dir):
            pass
        else:
            os.mkdir(self.args.save_dir)
        if is_middle:
            path_models = self.args.model_prefix + '-middle_training' + '-epoch_' + str(epoch) + '_models.pth'
            self.logger.info("--->>> save models of epoch %d middle_training!" % (epoch))
        else:
            path_models = self.args.model_prefix + '-epoch_' + str(epoch) + '_models.pth'
            self.logger.info("--->>> save models of epoch %d !" % (epoch))
        torch.save({'G': self.G.state_dict(), \
                    'D': self.D.state_dict()}, \
                    os.path.join(self.args.save_dir, path_models))


    def load_models(self, epoch, is_middle=False):
        if is_middle:
            path_models = self.args.model_prefix + '-middle_training' + '-epoch_' + str(epoch) + '_models.pth'
            self.logger.info("--->>> load models of epoch %d middle_training!" % (epoch))
        else:
            path_models = self.args.model_prefix + '-epoch_' + str(epoch) + '_models.pth'
            self.logger.info("--->>> load models of epoch %d !" % (epoch))
        checkpoint = torch.load(os.path.join(self.args.save_dir, path_models))
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])

