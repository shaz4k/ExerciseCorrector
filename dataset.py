import pickle
import numpy as np
import itertools
import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.data_utils import dct_2d
from utils.softdtw import SoftDTW


class EC3D_V1(Dataset):
    # Using original from authors paper but added options for selecting test_subject
    def __init__(self, data_path, dct_n=25, test_subject=0, sets=None, is_cuda=False, add_data=None, test=None, return_raw=False):
        if sets is None:
            sets = [[0], [1], [2], [3]]

        train_sets = [s for i, sub_set in enumerate(sets) if i != test_subject for s in sub_set]

        train_sets = [s for i, s in enumerate(sets) if i != test_subject]
        train_sets = [subject for sublist in train_sets for subject in sublist]  # Flatten the list

        test_sets = sets[test_subject]

        if test is not None:
            train_sets = test_sets

        self.dct_n = dct_n
        _, _, correct, other, _ = load_data(data_path, train_sets, add_data=add_data, is_original=True)
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
        self.name = "EC3D Original"

        self.return_raw = return_raw

    def __len__(self):
        return np.shape(self.inputs)[0]

    def __getitem__(self, index):
        if self.return_raw:
            return self.batch_ids[index], self.inputs_raw[index]
        else:
            return self.batch_ids[index], self.inputs[index]


class EC3D_V2(Dataset):
    def __init__(self, data_path, dct_n=25, test_subject=0, sets=None, data_type='DCT', normalization='sample',
                 is_cuda=False, add_data=None, test=None):
        if sets is None:
            sets = [[0], [1], [2], [3]]

        train_sets = [s for i, sub_set in enumerate(sets) if i != test_subject for s in sub_set]

        train_sets = [s for i, s in enumerate(sets) if i != test_subject]
        train_sets = [subject for sublist in train_sets for subject in sublist]  # Flatten the list

        test_sets = sets[test_subject]

        if test is not None:
            train_sets = test_sets

        self.data_type = data_type
        correct, other, correct_2d, other_2d, min_frames = load_data(data_path, train_sets, add_data=add_data)
        pairs = dtw_pairs(correct_2d, other_2d, is_cuda=is_cuda)
        self.targets_label = [i[1] for i in pairs]
        self.inputs_label = [i[0] for i in pairs]

        if self.data_type == '3D':
            self.targets_raw = [correct[i] for i in self.targets_label]
            self.inputs_raw = [other[i] for i in self.inputs_label]

            self.inputs = process_data(self.inputs_raw, target_frames=24, normalization=normalization)
            self.targets = process_data(self.targets_raw, target_frames=24, normalization=normalization)

        if self.data_type == 'DCT':
            self.targets_raw = [correct_2d[i] for i in self.targets_label]
            self.inputs_raw = [other_2d[i] for i in self.inputs_label]
            self.dct_n = dct_n
            self.inputs = [dct_2d(torch.from_numpy(x))[:, :self.dct_n].numpy() if x.shape[1] >= self.dct_n else
                       dct_2d(torch.nn.ZeroPad2d((0, self.dct_n - x.shape[1], 0, 0))(torch.from_numpy(x))).numpy()
                       for x in self.inputs_raw]

            self.targets = self.targets_raw

        self.batch_ids = list(range(len(self.inputs_raw)))

    def __len__(self):
        return np.shape(self.inputs)[0]

    def __getitem__(self, idx):
        return self.batch_ids[idx], self.inputs[idx]


class EC3D_new(Dataset):
    def __init__(self, data_path, dct_n=25, split=0, sets=None, rep_dct=False, rep_3d=False, normalization=False,
                 is_cuda=False, add_data=None):
        if sets is None:
            sets = [[0, 1], [2], [3]]

        self.rep_3d = rep_3d
        self.rep_dct = rep_dct

        correct, other, correct_2d, other_2d, min_frames = load_data(data_path, sets[split], add_data=add_data)
        # corr shape [19*3, frames]
        pairs = dtw_pairs(correct_2d, other_2d, is_cuda=is_cuda)
        self.targets_label = [i[1] for i in pairs]
        self.inputs_label = [i[0] for i in pairs]

        if self.rep_3d:
            self.targets_raw = [correct[i] for i in self.targets_label]
            self.inputs_raw = [other[i] for i in self.inputs_label]

            self.inputs = process_data(self.inputs_raw, target_frames=24, normalization=normalization)
            self.targets = process_data(self.targets_raw, target_frames=24, normalization=normalization)

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


def repeat_final_pose(x, target_columns):
    last_pose = x[:, -1:]  # Get the final pose
    padding = last_pose.repeat(1, target_columns - x.shape[1])  # Repeat the final pose as necessary
    padded_x = torch.cat((x, padding), dim=1)  # Concatenate the original data and the padding
    return padded_x

def translate(data, joint_index):
    translation_vector = data[joint_index, :]
    translated_data = data - translation_vector
    return translated_data

def process_data(data, target_frames, normalization=None):
    processed_data = []

    if normalization == 'dataset':
        max_val = np.max([np.max(sample) for sample in data])
        min_val = np.min([np.min(sample) for sample in data])

    for sample in data:
        if normalization == 'sample':
            max_val = np.max(sample)
            min_val = np.min(sample)

        if normalization in ('dataset', 'sample'):
            sample = (2.0 * (sample - min_val) / (max_val - min_val)) - 1.0

        num_frames = sample.shape[0]
        if num_frames > target_frames:
            step_size = num_frames // target_frames
            new_sample = sample[range(0, num_frames, step_size)][:target_frames]
        else:
            new_sample = sample

        processed_data.append(new_sample)

    return processed_data


def load_data(data_path, subs, add_data=None, is_cuda=False, is_original=None):
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
    if is_original is not None:
        pass
    else:
        for dict_ in [correct, other]:
            for k, v in dict_.items():
                if k[0] == 'Plank':
                    data1 = v[:, :, [0, 2]]
                    data2 = -v[:, :, 1]
                    data2 = np.expand_dims(data2, axis=-1)
                    corrected = np.dstack((data1, data2))
                    dict_.update({k: corrected})
    # Flatten
    correct_2d = {k: v.reshape(-1, v.shape[1] * v.shape[2]).T for k, v in correct.items()}
    other_2d = {k: v.reshape(-1, v.shape[1] * v.shape[2]).T for k, v in other.items()}

    return correct, other, correct_2d, other_2d, min_frames


def dtw_pairs(correct, incorrect, is_cuda=False):
    pairs = []
    for act, sub in set([(k[0], k[1]) for k in incorrect.keys()]):
        ''' fetch from all sets or only training set (dataset_fetch baseline used to compare dtw_loss)'''
        correct_sub = {k: v for k, v in correct.items() if k[0] == act and k[1] == sub}  # all datasets
        correct_sub = dict(sorted(correct_sub.items()))
        # correct_sub = {k: v for k, v in correct.items() if k[0] == act and k[1] != sub}   # training sets
        incorrect_sub = {k: v for k, v in incorrect.items() if k[0] == act and k[1] == sub}
        incorrect_sub = dict(sorted(incorrect_sub.items()))
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

