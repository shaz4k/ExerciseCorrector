import torch
import torch.nn as nn
import numpy as np

"""
    Torch DCT: dct and dct_2d
"""
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


# torch dct stuff -----------------------------------------------------------------------------------------------------#
def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

# ---------------------------------------------------------------------------------------------------------------------#

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
        crit_org =  criterion(org.cuda(), targ) - 1 / 2 * (criterion(org.cuda(), org.cuda()) + criterion(targ, targ))
        mse = torch.nn.MSELoss()
        smoothness_loss = mse(out[:,1:], out[:,:-1])

        dtw_loss_corr.append(crit.item())
        dtw_loss_org.append(crit_org.item())
        loss += crit + 1e-3 * smoothness_loss      # dtw_loss + smoothness
        # loss += crit  # without smoothness

        if test:
            preds.append(out[0].detach().cpu().numpy().T)

    if test:
        return loss, preds, dtw_loss_corr, dtw_loss_org
    else:
        return loss