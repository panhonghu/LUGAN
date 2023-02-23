from __future__ import absolute_import
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from collections import defaultdict


class PoseDataset(Dataset):
    def __init__(self, data_list_path, angle_tgt=0, sequence_length=60, transform=None):
        super(PoseDataset, self).__init__()
        self.data_list = np.loadtxt(data_list_path, skiprows=1, dtype=str)
        self.angle_tgt = angle_tgt
        self.transform = transform
        self.sequence_length = sequence_length
        self.data_dict = {}
        for row in self.data_list:
            row = row.split(",")   # 每一行的数据 len=52 第一维是名称
            # 一个四元组  frame_num为一个视频序列的帧数
            target, frame_num = self._filename_to_target(row[0])
            if target not in self.data_dict:
                self.data_dict[target] = {}
            if len(row[1:]) != 51:
                print("Invalid pose data for: ", target, ", frame: ", frame_num)
                continue
            try:
                self.data_dict[target][frame_num] = np.array(row[1:], dtype=np.float32).reshape((-1, 3))
            except ValueError:
                print("Invalid pose data for: ", target, ", frame: ", frame_num)
                continue
        for target, sequence in self.data_dict.copy().items():  # 过滤较短的序列
            if len(sequence) < self.sequence_length + 1:
                del self.data_dict[target]
        self.targets = list(self.data_dict.keys())
        self.all_data = list(self.data_dict.values())
        self.num_classes = self.get_num_classes()
        self.dict_ids, self.pids, self.angles = self.sample_angle()

    def _filename_to_target(self, filename):
        raise NotImplemented()

    def __len__(self):   # len=7966
        return len(self.targets)

    def __getitem__(self, index):  # index是一个数组
        target = self.targets[index]
        data = np.stack(list(self.all_data[index].values()))  # T*17*3
        if self.transform is not None:
            data = self.transform(data)    # 60*17*3
        data_src = data.permute(2, 0, 1)   # 3*60*17
        sequence_id = target[0]
        cur_pid = target[1]
        cur_angle = target[-1]
        angle_one_hot_src = self.get_one_hot_label(angle=cur_angle)
        angle_one_hot_tgt = self.get_one_hot_label(angle=self.angle_tgt)
        identities = self.dict_ids[cur_pid]  # all ids of the same identity
        idx_sample = identities[self.angle_tgt]
        ## target view should not be equal to source view
        if cur_angle==self.angle_tgt or len(idx_sample)==0:
            random_idx = random.randint(0, len(self.targets)-1)
            return self.__getitem__(random_idx)  # recurrent function
        else:
            iii = random.sample(idx_sample, 1)[0]
            cur_pose = np.stack(list(self.all_data[iii].values()))  # T*17*3
            data_tgt = self.transform(cur_pose).permute(2, 0, 1)    # 3*T*17
            return data_src[0:2, :, :], angle_one_hot_src, data_tgt[0:2, :, :], angle_one_hot_tgt, sequence_id

    def get_num_classes(self):
        """
        Returns number of unique ids present in the dataset. Useful for classification networks.
        """
        if type(self.targets[0]) == int:
            classes = set(self.targets)
        else:
            classes = set([target[0] for target in self.targets])
        num_classes = len(classes)
        return num_classes

    def get_one_hot_label(self, angle=0, var=0):
        if var==-1:
            return -torch.ones(len(self.angles))
        if var==0:
            label = torch.zeros(len(self.angles))
            angles = list(self.angles)
            angles.sort()
            idx = angles.index(angle)
            label[idx] = 1
            return label

    def sample_angle(self):
        angles = []
        pids = []
        for index, (_, pid, _, _, angle) in enumerate(self.targets):
            angles.append(angle)
            pids.append(pid)
        angles = set(angles)
        pids = set(pids)
        # print(pids)
        dict_ = {}
        for p in pids:
            dict_[p] = {}
            for a in angles:
                dict_[p][a] = []
        for index, (_, pid, _, _, angle) in enumerate(self.targets):
            dict_[pid][angle].append(index)
        return dict_, pids, angles


class CasiaBPose(PoseDataset):
    mapping_walking_status = {
        'nm': 0,  # 6 normal walking
        'bg': 1,  # 2 bag carrying
        'cl': 2,  # 2 coat wearing
    }

    def _filename_to_target(self, filename):  # e.g. ./001-bg-01-000/000001.jpg
        _, sequence_id, frame = filename.split("/")
        subject_id, walking_status, sequence_num, view_angle = sequence_id.split("-")
        walking_status = self.mapping_walking_status[walking_status]
        return (
            (sequence_id, int(subject_id), int(walking_status), int(sequence_num), int(view_angle)),
            int(frame[:-4])
        )


if __name__ == '__main__':

    class RandomSelectSequence(object):
        def __init__(self, sequence_length=60):
            self.sequence_length = sequence_length
        def __call__(self, data):
            try:
                start = np.random.randint(0, data.shape[0] - self.sequence_length)
            except ValueError:
                print(data.shape[0])
                raise ValueError
            end = start + self.sequence_length
            return data[start:end]

    class ToTensor(object):
        def __call__(self, data):
            return torch.tensor(data, dtype=torch.float)

    transform_train = transforms.Compose(
        [
            RandomSelectSequence(60),
            ToTensor()
        ])
    dataset_GAN = CasiaBPose(data_list_path="../data/casia-b_pose_train_valid.csv",
                             angle_tgt=90,   # [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
                             sequence_length=60,
                             transform=transform_train,
                             )
    train_loader = torch.utils.data.DataLoader(dataset_GAN,
                                               batch_size=1,
                                               num_workers=0,
                                               pin_memory=True,
                                               shuffle=True,
                                               drop_last=True)
    for idx, (data_src, angle_one_hot_src, data_tgt, angle_one_hot_tgt, sequence_id) in enumerate(train_loader):
        # if idx%100==0:
        #     print('idx -> ', idx)
        #     print('data_src -> ', data_src.shape)
        #     print('angle_one_hot_src -> ', angle_one_hot_src.shape)
        #     print('data_tgt -> ', data_tgt.shape)
        #     print('angle_one_hot_tgt -> ', angle_one_hot_tgt.shape)

        #     print('angle_one_hot_src -> ', angle_one_hot_src)
        #     print('angle_one_hot_tgt -> ', angle_one_hot_tgt)
        for i in range(len(sequence_id)):
            pose_file = sequence_id[i] +'.json'
            if int(sequence_id[i].split("-")[0])==6:
                print(pose_file)
