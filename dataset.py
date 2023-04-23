import pickle
import numpy as np
import itertools
import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.data_utils import dct_2d
from utils.softdtw import SoftDTW


class EC3D(Dataset):

    # Preprocessing steps from Original paper:
    # https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises
    # 1 - load data from data_path
    # 2 - apply dtw to create matched pairs of sequences for correct and incorrect motion
    # 3 - targets = correct motion only, inputs = examples to train model
    # 4 - apply DCT to inputs for first dct_n(=25) co-efficients shape [57, 25] (57=19j*3d)
    # 5 - node_n stores no. joints
    # 6 - batch_ids stores indices of data samples
    def __init__(self, data_path, dct_n=25, split=0, sets=None, is_cuda=False, add_data=None):
        if sets is None:
            sets = [[0, 1], [2], [3]]
        self.dct_n = dct_n
        _, _, correct, other, _ = load_data(data_path, sets[split], add_data=add_data)
        pairs = dtw_pairs(correct, other, is_cuda=is_cuda)

        self.targets_label = [i[1] for i in pairs]
        self.inputs_label = [i[0] for i in pairs]

        self.targets = [correct[i] for i in self.targets_label]
        self.inputs_raw = [other[i] for i in self.inputs_label]

        print('Calculating DCT co-efficients')
        self.inputs = [dct_2d(torch.from_numpy(x))[:, :self.dct_n].numpy() if x.shape[1] >= self.dct_n else
                       dct_2d(torch.nn.ZeroPad2d((0, self.dct_n - x.shape[1], 0, 0))(torch.from_numpy(x))).numpy()
                       for x in self.inputs_raw]

        self.node_n = np.shape(self.inputs_raw[0])[0]
        self.batch_ids = list(range(len(self.inputs_raw)))
        self.name = "EC3D"

    def __len__(self):
        return np.shape(self.inputs)[0]

    def __getitem__(self, index):
        return self.batch_ids[index], self.inputs[index]


class EC3D_new(Dataset):
    def __init__(self, data_path, dct_n=25, split=0, sets=None, rep_dct=False, rep_3d=False, normalization=False,
                 is_cuda=False, add_data=None):
        if sets is None:
            sets = [[0, 1], [2], [3]]

        self.rep_3d = rep_3d
        self.rep_dct = rep_dct

        correct, other, correct_2d, other_2d, min_frames = load_data(data_path, sets[split], rep_3d=self.rep_3d,
                                                                     add_data=add_data)
        # corr shape [19*3, frames]
        pairs = dtw_pairs(correct_2d, other_2d, is_cuda=is_cuda)
        self.targets_label = [i[1] for i in pairs]
        self.inputs_label = [i[0] for i in pairs]

        if self.rep_3d:
            self.targets_raw = [correct[i] for i in self.targets_label]
            self.inputs_raw = [other[i] for i in self.inputs_label]

            # Resample
            self.inputs = resample(self.inputs_raw, target_frames=24)
            self.targets = resample(self.targets_raw, target_frames=24)

            if normalization:
                self.inputs = normalise(self.inputs, normalization)
                self.targets = normalise(self.targets, normalization)

        if self.rep_dct:
            self.targets_raw = [correct_2d[i] for i in self.targets_label]
            self.inputs_raw = [other_2d[i] for i in self.inputs_label]

            self.dct_n = dct_n
            self.inputs = [dct_2d(torch.from_numpy(x))[:, :self.dct_n].numpy() if x.shape[1] >= self.dct_n else
                           dct_2d(torch.nn.ZeroPad2d((0, self.dct_n - x.shape[1], 0, 0))(torch.from_numpy(x))).numpy()
                           for x in self.inputs_raw]

            # reshape
            if self.rep_dct == 3:
                self.inputs = [arr.reshape(3, 19, 25) for arr in self.inputs]

        self.batch_ids = list(range(len(self.inputs_raw)))

    def __len__(self):
        return np.shape(self.inputs)[0]

    def __getitem__(self, idx):
        return self.batch_ids[idx], self.inputs[idx]


def normalise(data, normalization):
    normalised_list = []

    if normalization == 'dataset':
        max_val = np.max([np.max(sample) for sample in data])
        min_val = np.min([np.min(sample) for sample in data])

        for sample in data:
            normalised_data = (2.0 * (sample - min_val) / (max_val - min_val)) - 1.0
            normalised_list.append(normalised_data)

    if normalization == 'sample':
        for sample in data:
            max_val = np.max(sample)
            min_val = np.min(sample)
            normalised_data = (2.0 * (sample - min_val) / (max_val - min_val)) - 1.0
            normalised_list.append(normalised_data)

    return normalised_list


def resample(data, target_frames):
    resampled_data = []
    for i, array in enumerate(data):
        num_frames = array.shape[0]
        if num_frames > target_frames:
            step_size = num_frames // target_frames
            new_array = array[range(0, num_frames, step_size)][:target_frames]
        else:
            new_array = array
        resampled_data.append(new_array)
    return resampled_data


def load_data(data_path, subs, rep_3d=False, add_data=None, is_cuda=False):
    with open(data_path, "rb") as f:
        data_gt = pickle.load(f)

    if add_data is not None:
        with open(add_data, "rb") as f:
            data = pickle.load(f)
        labels = pd.DataFrame(data['labels'], columns=['act', 'sub', 'lab', 'rep', 'cam'])
    else:
        data = data_gt
        labels = pd.DataFrame(data['labels'], columns=['act', 'sub', 'lab', 'rep', 'frame'])
        labels['cam'] = 'gt'
    # import pdb; pdb.set_trace()
    joints = list(range(15)) + [19, 21, 22, 24]

    # Duplicate labels for input (correct) and target (other)
    labels_gt = pd.DataFrame(data_gt['labels'], columns=['act', 'sub', 'lab', 'rep', 'frame'])
    labels_gt['cam'] = 'gt'

    labels[['lab', 'rep']] = labels[['lab', 'rep']].astype(int)
    labels_gt[['lab', 'rep']] = labels_gt[['lab', 'rep']].astype(int)

    # Count unique (act, sub, lab,rep) for 3 subjects in train set or 1 in test set
    subs = labels[['act', 'sub', 'lab', 'rep']].drop_duplicates().groupby('sub').count().rep[subs]

    # Create boolean masks corresponding to subs
    indices = labels['sub'].isin(subs.index)
    indices_gt = labels_gt['sub'].isin(subs.index)

    # Filter rows using boolean mask
    labels = labels[indices]
    labels_gt = labels_gt[indices_gt]

    # Separate into correct (label = 1) for targets
    lab1 = labels_gt[labels_gt['lab'] == 1].groupby(['act', 'sub', 'lab', 'rep', 'cam']).groups
    # All data labels
    labnot1 = labels.groupby(['act', 'sub', 'lab', 'rep', 'cam']).groups

    poses = data['poses'][:, :, joints]
    poses_gt = data_gt['poses'][:, :, joints]

    # shape [frames, 19, 3]
    correct = {k: poses_gt[v].transpose(0, 2, 1) for k, v in lab1.items()}
    other = {k: poses[v].transpose(0, 2, 1) for k, v in labnot1.items()}

    # Find min frames (first dim)
    corr_min = np.min([v.shape[0] for k, v in lab1.items()])
    other_min = np.min([v.shape[0] for k, v in labnot1.items()])
    min_frames = min(corr_min, other_min)

    for k, v in correct.items():
        if k[0] == 'Plank':
            data1 = v[:, :, [0, 2]]
            data2 = -v[:, :, 1]
            data2 = np.expand_dims(data2, axis=-1)
            corrected = np.dstack((data1, data2))
            correct.update({k: corrected})

    for k, v in other.items():
        if k[0] == 'Plank':
            data1 = v[:, :, [0, 2]]
            data2 = -v[:, :, 1]
            data2 = np.expand_dims(data2, axis=-1)
            corrected = np.dstack((data1, data2))
            other.update({k: corrected})
    # Flatten
    correct_2d = {k: v.reshape(-1, v.shape[1] * v.shape[2]).T for k, v in correct.items()}
    other_2d = {k: v.reshape(-1, v.shape[1] * v.shape[2]).T for k, v in other.items()}

    return correct, other, correct_2d, other_2d, min_frames


def dtw_pairs(correct, incorrect, is_cuda=False):
    pairs = []
    for act, sub in set([(k[0], k[1]) for k in incorrect.keys()]):
        ''' fetch from all sets or only training set (dataset_fetch baseline used to compare dtw_loss)'''
        correct_sub = {k: v for k, v in correct.items() if k[0] == act and k[1] == sub}  # all datasets
        # correct_sub = {k: v for k, v in correct.items() if k[0] == act and k[1] != sub}   # training sets
        incorrect_sub = {k: v for k, v in incorrect.items() if k[0] == act and k[1] == sub}
        dtw_sub = {k: {} for k in incorrect_sub.keys()}
        for i, pair in enumerate(itertools.product(incorrect_sub, correct_sub)):
            criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
            if is_cuda:
                p0 = torch.from_numpy(np.expand_dims(incorrect_sub[pair[0]].T, axis=0)).cuda()
                p1 = torch.from_numpy(np.expand_dims(correct_sub[pair[1]].T, axis=0)).cuda()
            else:
                p0 = torch.from_numpy(np.expand_dims(incorrect_sub[pair[0]].T, axis=0))
                p1 = torch.from_numpy(np.expand_dims(correct_sub[pair[1]].T, axis=0))
            dtw_sub[pair[0]][pair[1]] = (criterion(p0, p1) - 1 / 2 * (criterion(p0, p0) + criterion(p1, p1))).item()
        dtw = pd.DataFrame.from_dict(dtw_sub, orient='index').idxmin(axis=1)
        pairs = pairs + list(zip(dtw.index, dtw))
    return pairs

