import torch
import torch.nn as nn
import numpy as np
from utils import get_labels

class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def test_class(test_loader, model, is_cuda=False, level=0):
    te_l = AccumLoss()

    if level == 0:
        criterion = nn.NLLLoss(weight=torch.tensor([1, 0.5, 1, 0.5, 1, 0.5]))
    else:
        criterion = nn.NLLLoss()
    model.eval()

    correct = 0
    total = 0
    for i, (batch_id, inputs) in enumerate(test_loader):
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()
        if test_loader.dataset.name == 'EC3D':
            # import pdb; pdb.set_trace()
            labels = get_labels([test_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level).cuda()
        elif test_loader.dataset.name == 'NTU60':
            labels = torch.from_numpy(np.array([test_loader.dataset.labels[int(i)] for i in batch_id])).long().cuda()
        batch_size = inputs.shape[0]

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        # import pdb
        # pdb.set_trace()
        correct += (predicted == labels).sum().item()

        # calculate loss and backward
        loss = criterion(outputs, labels)

        # update the training loss
        te_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

        # summary = np.vstack((labels.numpy(), predicted.numpy()))
        summary = torch.stack((labels, predicted), dim=1)
        if level == 0:
            cmt = torch.zeros(6, 6, dtype=torch.int64)
        elif level == 1:
            cmt = torch.zeros(12, 12, dtype=torch.int64)
        else:
            cmt = torch.zeros(60, 60, dtype=torch.int64)
        # pdb.set_trace()
        for pp in summary:
            tl, pl = pp.tolist()
            cmt[tl, pl] = cmt[tl, pl] + 1

    return te_l.avg, 100 * correct / total, summary, cmt