import torch
import torch.nn as nn
import numpy as np
from utils.softdtw import SoftDTW
from utils.train_utils import get_labels, dtw_loss, dtw_loss_v2
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
    Training: class AccumLoss, train_class, train_corr, train_transformer, lr_decay
"""


class AccumLoss(object):
    def __init__(self):
        """
            Calculate loss accumulation across multiple batches during training
        """
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val  # current loss for single batch
        self.sum += val  # sum of all losses over batches seen so far
        self.count += n  # total no. batches seen so far
        self.avg = self.sum / self.count  # average loss over batches seen so far


def train_cnn(train_loader, model, optimizer, is_cuda, writer, epoch, level=0):
    tr_l = AccumLoss()
    if level == 0:
        # weights determined of loss function based on relative frequency of each class
        criterion = nn.NLLLoss(weight=torch.tensor([1, 0.3, 1, 0.5, 1, 1]))
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 0.3, 1, 0.5, 1, 1]))
    else:
        criterion = nn.NLLLoss()
        # criterion = nn.CrossEntropyLoss()
    model.train()
    correct = 0
    total = 0
    for i, (batch_id, inputs) in enumerate(train_loader):
        if inputs.shape[1] > 3:
            inputs = inputs.permute(0, 3, 2, 1)
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        labels = get_labels([train_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level).cuda()
        batch_size = inputs.size(0)

        # Forward
        outputs = model(inputs)
        predicted = torch.argmax(outputs.detach(), dim=1)
        # _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # calculate loss using loss function
        loss = criterion(outputs, labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().item() * batch_size, batch_size)
        # tr_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

        # Log the training loss to TensorBoard every 10 iterations (if enabled)
        if writer is not None and i % 10 == 0:
            avg_train_loss = tr_l.avg
            writer.add_scalar('Classifier/Training Loss', avg_train_loss, epoch * len(train_loader) + i)
            avg_train_acc = 100 * correct / total
            writer.add_scalar('Classifier/Training Accuracy', avg_train_acc, epoch * len(train_loader) + i)

    return tr_l.avg, 100 * correct / total


def train_class(train_loader, model, optimizer, writer, epoch, is_cuda=False, level=0):
    """
    :param train_loader:  PyTorch data loader for the training dataset
    :param model: the PyTorch model to be trained
    :param optimizer: the PyTorch optimizer to be used during training
    :param is_cuda: a boolean indicating whether to use a CUDA-enabled device
    :param level: an integer indicating the level of the dataset being used
    :return:
    """
    tr_l = AccumLoss()
    if level == 0:
        # weights determined of loss function based on relative frequency of each class
        criterion = nn.NLLLoss(weight=torch.tensor([1, 0.3, 1, 0.5, 1, 1]))
    else:
        criterion = nn.NLLLoss()
    model.train()

    correct = 0
    total = 0
    for i, (batch_id, inputs) in enumerate(train_loader):
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        labels = get_labels([train_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level).cuda()
        batch_size = inputs.size(0)

        outputs = model(inputs)
        predicted = torch.argmax(outputs.detach(), dim=1)

        # _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # calculate loss and backward
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

        # Log the training loss to TensorBoard every 10 iterations (if enabled)
        if writer is not None and i % 10 == 0:
            avg_train_loss = tr_l.avg
            writer.add_scalar('Classifier/Training Loss', avg_train_loss, epoch * len(train_loader) + i)
            avg_train_acc = 100 * correct / total
            writer.add_scalar('Classifier/Training Accuracy', avg_train_acc, epoch * len(train_loader) + i)

    return tr_l.avg, 100 * correct / total


def train_corr(train_loader, model, optimizer, epoch, writer=None, fact=None, is_cuda=False, attn=None):
    tr_l = AccumLoss()
    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.train()
    for i, (batch_id, inputs) in enumerate(train_loader):
        # targets.shape = [57, len_targ], originals.shape = [57, len_in]
        targets = [train_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [train_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        batch_size = inputs.shape[0]
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        if attn is not None:
            deltas = model(inputs)
        else:
            deltas, _ = model(inputs)

        if fact is None:
            dtw = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda)
            loss = dtw / batch_size
        else:
            dtw = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda)
            # l1 = fact * torch.sum(torch.abs(att))
            # loss = (dtw + l1) / batch_size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

        # Log the training loss to TensorBoard every 10 iterations (if enabled)
        if writer is not None and i % 10 == 0:
            avg_train_loss = tr_l.avg
            writer.add_scalar('Corrector/Training Loss', avg_train_loss, epoch * len(train_loader) + i)
        print(tr_l.avg)
    return tr_l.avg


def train_GAT(train_loader, model, optimizer, epoch, writer=None, use_adj=True, is_cuda=False):
    tr_l = AccumLoss()

    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.train()
    for i, (batch_id, inputs) in enumerate(train_loader):
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()
        targets = [train_loader.dataset.targets[int(i)] for i in batch_id]
        inputs_raw = [train_loader.dataset.inputs_raw[int(i)] for i in batch_id]

        # Create fully connected adjacency matrix
        batch_size = inputs.shape[0]
        nodes_n = inputs.shape[1]
        adj_mat = np.ones((batch_size, nodes_n, nodes_n))
        # for i in range(batch_size):
        #     np.fill_diagonal(adj_mat[i], 0)
        adj_mat = torch.from_numpy(adj_mat).float().unsqueeze(-1).to(inputs.device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward
        if use_adj:
            deltas = model(inputs, adj_mat)
        else:
            deltas = model(inputs)
        # Calculate loss
        dtw = dtw_loss(inputs_raw, deltas, targets, criterion, is_cuda=is_cuda)
        # loss = dtw / batch_size
        loss = dtw
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().data.numpy(), batch_size)
        print(tr_l.avg)

        if writer is not None:
            avg_train_loss = tr_l.avg
            writer.add_scalar('GAT_Corrector/Training Loss', avg_train_loss, epoch * len(train_loader) + i)
    return tr_l.avg



def train_transformer(train_loader, model, optimizer, epoch, writer=None, is_cuda=False):
    tr_l = AccumLoss()
    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.train()

    for i, (batch_id, inputs) in enumerate(train_loader):
        targets = [train_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [train_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()
        batch_size = inputs.shape[0]

        deltas = model(inputs)
        # Calculate loss
        dtw = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda)
        loss = dtw / batch_size

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().data.numpy() * batch_size, batch_size)
        print(tr_l.avg)
    return tr_l.avg