import torch
import torch.nn as nn
from utils.softdtw import SoftDTW
from utils.train_utils import get_labels, dtw_loss

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

        # Log the training loss to TensorBoard every 100 iterations (if enabled)
        if writer is not None and i % 100 == 0:
            avg_train_loss = tr_l.avg
            writer.add_scalar('Training Loss', avg_train_loss, epoch * len(train_loader) + i)
            avg_train_acc = 100 * correct/total
            writer.add_scalar('Training Accuracy', avg_train_acc, epoch * len(train_loader) + i)

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
        _, predicted = torch.max(outputs.data, 1)

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

        # Log the training loss to TensorBoard every 100 iterations (if enabled)
        if writer is not None and i % 100 == 0:
            avg_train_loss = tr_l.avg
            writer.add_scalar('Training Loss', avg_train_loss, epoch * len(train_loader) + i)
            avg_train_acc = 100 * correct/total
            writer.add_scalar('Training Accuracy', avg_train_acc, epoch * len(train_loader) + i)

        return tr_l.avg, 100 * correct / total


def train_corr(train_loader, model, optimizer, fact=None, is_cuda=False):
    tr_l = AccumLoss()

    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.train()
    for i, (batch_id, inputs) in enumerate(train_loader):
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()
        targets = [train_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [train_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        batch_size = inputs.shape[0]
        # import pdb; pdb.set_trace()
        # delta is dct coefficient predictions
        deltas, att = model(inputs)

        # calculate loss and backward
        if fact is None:
            dtw = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda)
            loss = dtw / batch_size
        else:
            dtw = dtw_loss(originals, deltas, targets, criterion, attentions=att, is_cuda=is_cuda)
            l1 = fact * torch.sum(torch.abs(att))
            loss = (dtw + l1) / batch_size
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

    return tr_l.avg


def train_transformer(train_loader, model, optimizer, is_cuda, level=0):
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
        # inputs = inputs.float().to(device)
        # Reshape input into seq_len, batch_size, input_size
        # inputs = inputs.permute(1, 0, 2)
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        labels = get_labels([train_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level).cuda()

        batch_size = inputs.shape[0]
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

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

        return tr_l.avg, 100 * correct / total


def lr_decay(optimizer, lr, gamma):
    """
    Decay the learning rate of the optimizer by a factor of gamma.
    """
    lr *= gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
