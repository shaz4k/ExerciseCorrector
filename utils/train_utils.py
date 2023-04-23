import numpy as np
import torch
from torch.autograd import Variable
from utils.data_utils import idct, idct_2d
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from dataset import EC3D_V1, EC3D_V2

"""
    Functions for training: get_labels, dtw_loss
"""


def load_3d():
    temp_path = 'data/EC3D/tmp_3d_1.pickle'
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
        test_subject = 3
        is_cuda = torch.cuda.is_available()
        data_train = EC3D_V2(raw_data_path, data_type='DCT', test_subject=test_subject, is_cuda=is_cuda)
        data_test = EC3D_V2(raw_data_path, data_type='DCT', sets=[[test_subject]], is_cuda=is_cuda, test=True)
        print('Writing processed data...')
        with open(temp_path, 'wb') as f:
            pickle.dump({'train': data_train, 'test': data_test}, f)
        print('Complete')

    return data_train, data_test


def load_DCT():
    # temp_path = 'data/EC3D/tmp_DCT_3CH.pickle'
    temp_path = 'data/EC3D/tmp_DCT.pickle'
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
        # sets = [[0, 1, 2], [], [3]]
        test_subject = 3
        data_train = EC3D_V2(raw_data_path, data_type='DCT', test_subject=test_subject, is_cuda=is_cuda)
        data_test = EC3D_V2(raw_data_path, data_type='DCT', sets=[[test_subject]], is_cuda=is_cuda, test=True)
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
        test_subject = 3
        is_cuda = torch.cuda.is_available()
        data_train = EC3D_V1(raw_data_path, test_subject=test_subject, is_cuda=is_cuda)
        data_test = EC3D_V1(raw_data_path, sets=[[test_subject]], is_cuda=is_cuda, test=True)
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


def dtw_loss(originals, deltas, targets, criterion, attentions=None, is_cuda=False, test=False):
    loss = 0
    preds = []
    dtw_loss_corr = []
    dtw_loss_org = []
    for i, o in enumerate(originals):
        # o.shape = [57, len], org.shape and targ.shape = [1, seq_len, 57]
        length = o.shape[1]
        org = torch.from_numpy(o).T.unsqueeze(0)
        targ = torch.from_numpy(targets[i]).T.unsqueeze(0)

        # deltas[i].shape = [57, 25]
        if length > deltas[i].shape[1]:
            m = torch.nn.ZeroPad2d((0, length - deltas[i].shape[1], 0, 0))
            # delt = dct.idct_2d(m(deltas[i]).T.unsqueeze(0))
            delt = idct_2d(m(deltas[i]).T.unsqueeze(0)).cuda()
        else:
            # delt = dct.idct_2d(deltas[i, :, :length].T.unsqueeze(0))
            delt = idct_2d(deltas[i, :, :length].T.unsqueeze(0)).cuda()

        if attentions is not None:
            delt = torch.mul(delt, attentions[i].T.unsqueeze(0))

        # delt.shape = [1, seq_len, 57]
        out = org.cuda() + delt.cuda()
        # out = delt.cuda()

        if is_cuda:
            out = out.cuda()
            targ = targ.cuda()

        crit = criterion(out, targ) - 1 / 2 * (criterion(out, out) + criterion(targ, targ))
        normalized_crit = crit / length
        crit_org = criterion(org.cuda(), targ) - 1 / 2 * (criterion(org.cuda(), org.cuda()) + criterion(targ, targ))
        mse = torch.nn.MSELoss()
        smoothness_loss = mse(out[:, 1:], out[:, :-1])

        dtw_loss_corr.append(crit.item())
        dtw_loss_org.append(crit_org.item())
        loss += crit + 1e-3 * smoothness_loss  # dtw_loss + smoothness
        # loss += crit  # without smoothness
        # loss += normalized_crit + 1e-3 * smoothness_loss

        if test:
            preds.append(out[0].detach().cpu().numpy().T)

    if test:
        return loss, preds, dtw_loss_corr, dtw_loss_org
    else:
        return loss

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
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
