import os
import sys
import time
import logging
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import dataset_factory, RandomIdentitySampler
from datasets.augmentation import *
from models import LUGAN
SEED = 1024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


parser = argparse.ArgumentParser(description="Training model on gait sequence")
# training dataset
parser.add_argument("--dataset", default="casia-b", choices=["casia-b", "outdoor-gait", "tum-gaid"])
parser.add_argument("--train_data_path", default="./data/casia-b_pose_train_valid.csv", help="Path to train data CSV")
# parser.add_argument("--train_data_path", default="./data/casia-b_pose_test.csv", help="Path to train data CSV")
parser.add_argument("--target_view", type=int, default=0, choices=[0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180], help="target view to predict")
parser.add_argument("--sequence_length", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=8)
# model parameters
parser.add_argument('--base_model', type=str, default='resgcn', help="base model for encoding pose graph")
parser.add_argument("--pose_in_channel", type=int, default=3, help="input pose channel for resgcn")
parser.add_argument("--angle_in_channel", type=int, default=11, help="input angle channel for resgcn")
parser.add_argument("--num_nodes", type=int, default=17, help="number of nodes in Pose Graph")
parser.add_argument("--num_scale", type=int, default=4, help="scale for output channel")
# training parameters
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate to train network")
parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for Adam")
parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
parser.add_argument("--G_steps", type=int, default=50, help="steps to train generator")
parser.add_argument("--D_steps", type=int, default=1, help="steps to train discriminator")
parser.add_argument("--epochs", type=int, default=10, help="epochs to train network")
parser.add_argument("--save_epoch", type=int, default=1, help="epochs to save model")
parser.add_argument('--log_dir', type=str, default='./log', help="path to save log file")
parser.add_argument('--log_file', type=str, default='log.txt', help="log file")
parser.add_argument('--save_dir', type=str, default='./save', help="path to save trained model")
parser.add_argument('--model_prefix', type=str, default='LUGAN', help="name of saved trained model")
parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')


def train_model(args, logger):
    transform_train = transforms.Compose(
        [
            RandomSelectSequence(args.sequence_length),
            ToTensor()
        ])
    dataset_class = dataset_factory(args.dataset)
    dataset_GAN = dataset_class(data_list_path=args.train_data_path,
                                angle_tgt=args.target_view,
                                sequence_length=args.sequence_length,
                                transform=transform_train,
                                )
    train_loader = torch.utils.data.DataLoader(dataset_GAN,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               shuffle=True,
                                               drop_last=True)

    ### init models
    model = LUGAN(args, logger, is_training=True)
    if torch.cuda.is_available():
        model = model.cuda()
    # print(model)

    for epoch in range(1, args.epochs+1):
        loss_G, loss_D = model.train_one_epoch(train_loader, epoch)
        # logger.info('----->>>>> training model of epoch -> %d' % epoch)
        # logger.info('loss_G -> %f' % loss_G)
        # logger.info('loss_D -> %f' % loss_D)

        ## same models
        if epoch%args.save_epoch==0:
            model.save_models(epoch)




if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    logger = get_logger(os.path.join(args.log_dir, args.log_file))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    logger.info(args)
    ## training LUGAN model
    train_model(args, logger)

