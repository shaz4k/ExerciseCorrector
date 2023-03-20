import numpy as np
import torch
from torch.autograd import Variable
from utils.data_utils import idct, idct_2d
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from dataset import EC3D, EC3D_new

"""
    Functions for training: get_labels, dtw_loss
"""


def load_3d():
    temp_path = 'data/EC3D/tmp_3d.pickle'
    raw_data_path = 'data/EC3D/EC3D.pickle'
    is_cuda = torch.cuda.is_available()
    try:
        print('Loading saved data.')
        with open(temp_path, "rb") as f:
            data = pickle.load(f)
        data_train = data['train']
        data_test = data['test']

    except FileNotFoundError:
        print('File Not Found: Processing raw data...')
        sets = [[0, 1, 2], [], [3]]
        is_cuda = torch.cuda.is_available()
        data_train = EC3D_new(raw_data_path, sets=sets, split=0, rep_3d=True, normalization='sample',
                              is_cuda=is_cuda)
        data_test = EC3D_new(raw_data_path, sets=sets, split=2, rep_3d=True, normalization='sample',
                             is_cuda=is_cuda)
        print('Writing processed data...')
        with open(temp_path, 'wb') as f:
            pickle.dump({'train': data_train, 'test': data_test}, f)
        print('Complete')

    return data_train, data_test


def load_DCT():
    temp_path = 'data/EC3D/tmp_DCT_3CH.pickle'
    raw_data_path = 'data/EC3D/EC3D.pickle'
    is_cuda = torch.cuda.is_available()
    try:
        print('Loading saved data.')
        with open(temp_path, "rb") as f:
            data = pickle.load(f)
        data_train = data['train']
        data_test = data['test']

    except FileNotFoundError:
        print('File Not Found: Processing raw data...')
        sets = [[0, 1, 2], [], [3]]
        is_cuda = torch.cuda.is_available()
        data_train = EC3D_new(raw_data_path, sets=sets, split=0, rep_dct=3, rep_3d=False, normalization='sample',
                              is_cuda=is_cuda)
        data_test = EC3D_new(raw_data_path, sets=sets, split=2, rep_dct=3, rep_3d=False, normalization='sample',
                             is_cuda=is_cuda)
        print('Writing processed data...')
        with open(temp_path, 'wb') as f:
            pickle.dump({'train': data_train, 'test': data_test}, f)
        print('Complete')

    return data_train, data_test


def load_original():
    temp_path = 'data/EC3D/tmp_wo_val.pickle'
    raw_data_path = 'data/EC3D/EC3D.pickle'
    is_cuda = torch.cuda.is_available()
    try:
        print('Loading saved data...')
        with open(temp_path, "rb") as f:
            data = pickle.load(f)
        data_train = data['train']
        data_test = data['test']

    except FileNotFoundError:
        print('Processing raw data.')
        sets = [[0, 1, 2], [], [3]]
        is_cuda = torch.cuda.is_available()
        data_train = EC3D(raw_data_path, sets=sets, split=0, is_cuda=is_cuda)
        data_test = EC3D(raw_data_path, sets=sets, split=2, is_cuda=is_cuda)
        with open(temp_path, 'wb') as f:
            pickle.dump({'train': data_train, 'test': data_test}, f)

    return data_train, data_test


def get_labels(raw_labels, level=0):
    if level == 0:
        mapping = {'SQUAT': 0, 'Lunges': 2, 'Plank': 4}
        labels = np.zeros(len(raw_labels))
        for i, el in enumerate(raw_labels):
            if el[2] == 1:
                labels[i] = mapping[el[0]]
            else:
                labels[i] = mapping[el[0]] + 1
        return torch.from_numpy(labels).long()
    elif level == 1:
        mapping = {'SQUAT': 0, 'Lunges': 6, 'Plank': 9}
        map_label = {'SQUAT': [1, 2, 3, 4, 5, 10], 'Lunges': [1, 4, 6], 'Plank': [1, 7, 8]}
        labels = np.zeros(len(raw_labels))
        for i, el in enumerate(raw_labels):
            labels[i] = mapping[el[0]] + np.where(np.array(map_label[el[0]]) == el[2])[0].item()
        return torch.from_numpy(labels).long()


def dtw_loss_v2(predictions, targets, criterion, is_cuda=False):
    loss = 0
    for i, target in enumerate(targets):
        prediction = predictions[i]
        if isinstance(prediction, np.ndarray):
            prediction = torch.from_numpy(prediction)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        # p0 = (1, frames, 19*3 dims)
        out = prediction.unsqueeze(0)
        targ = target.reshape(target.size(0), target.size(1)*target.size(2)).unsqueeze(0)
        out = out.cuda()
        targ = targ.cuda()
        crit = criterion(out, targ) - 1 / 2 * (criterion(out, out) + criterion(targ, targ))
        loss += crit

    return loss


def dtw_loss(originals, deltas, targets, criterion, attentions=None, is_cuda=False, test=False):
    loss = 0
    preds = []
    dtw_loss_corr = []
    dtw_loss_org = []
    for i, o in enumerate(originals):

        length = o.shape[1]
        org = torch.from_numpy(o).T.unsqueeze(0)
        targ = torch.from_numpy(targets[i]).T.unsqueeze(0)

        if length > deltas[i].shape[1]:
            m = torch.nn.ZeroPad2d((0, length - deltas[i].shape[1], 0, 0))
            # delt = dct.idct_2d(m(deltas[i]).T.unsqueeze(0))
            delt = idct_2d(m(deltas[i]).T.unsqueeze(0)).cuda()
        else:
            # delt = dct.idct_2d(deltas[i, :, :length].T.unsqueeze(0))
            delt = idct_2d(deltas[i, :, :length].T.unsqueeze(0)).cuda()

        if attentions is not None:
            delt = torch.mul(delt, attentions[i].T.unsqueeze(0))

        out = org.cuda() + delt.cuda()

        if is_cuda:
            out = out.cuda()
            targ = targ.cuda()

        crit = criterion(out, targ) - 1 / 2 * (criterion(out, out) + criterion(targ, targ))
        crit_org = criterion(org.cuda(), targ) - 1 / 2 * (criterion(org.cuda(), org.cuda()) + criterion(targ, targ))
        mse = torch.nn.MSELoss()
        smoothness_loss = mse(out[:, 1:], out[:, :-1])

        dtw_loss_corr.append(crit.item())
        dtw_loss_org.append(crit_org.item())
        loss += crit + 1e-3 * smoothness_loss  # dtw_loss + smoothness
        # loss += crit  # without smoothness

        if test:
            preds.append(out[0].detach().cpu().numpy().T)

    if test:
        return loss, preds, dtw_loss_corr, dtw_loss_org
    else:
        return loss


if __name__ == '__main__':
    pass
    # targets shape = list[3, 19, 24]
    targets = [np.random.rand(3, 19, 24) for _ in range(32)]
    # predictions shape = [32, 3, 19, 24]
    predictions = np.random.rand(32, 3, 19, 24)
    from softdtw import SoftDTW

    is_cuda = torch.cuda.is_available()
    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)

    dtw_loss_v2(predictions, targets, criterion, is_cuda=is_cuda)

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    return np_mask.to()


def create_masks(src, trg, opt):
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask