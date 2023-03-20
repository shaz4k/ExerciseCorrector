import pickle
import numpy as np
import random
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.train_utils import get_labels
from models import CNN_Classifier, GCN_Corrector
import torch.optim as optim
from utils.softdtw import SoftDTW
from utils.train_utils import dtw_loss
from visualisation import SkeletonVisualizer
from utils.data_utils import idct_2d
import matplotlib.pyplot as plt


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    args = checkpoint['args']

    return model, optimizer, epoch, args


def class_eval(path):
    temp_path = 'data/EC3D/tmp_3d.pickle'
    try:
        print('Loading saved data.')
        with open(temp_path, "rb") as f:
            data = pickle.load(f)
        data_test = data['test']
    except FileNotFoundError:
        print('File Not Found')
        sys.exit()

    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))
    model = CNN_Classifier(in_channels=3)
    optimizer = optim.Adam(model.parameters())
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()

    # load weights
    # path = 'runs/3D-CNN-Classifier/checkpoints/2023-03-17_01-24-56_epoch39.pt'
    model, optimizer, epoch, args = load_checkpoint(model, optimizer, path)
    # Set the model to evaluation mode
    model.eval()

    y_true = []
    y_pred = []
    classes = range(11)

    for i, (batch_id, inputs) in enumerate(test_loader):
        if inputs.shape[1] > 3:
            inputs = inputs.permute(0, 3, 2, 1)
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        labels = get_labels([test_loader.dataset.inputs_label[int(i)] for i in batch_id], level=1).cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def visualise_weights(path):
    temp_path = 'data/EC3D/tmp_3d.pickle'
    try:
        print('Loading saved data.')
        with open(temp_path, "rb") as f:
            data = pickle.load(f)
        data_test = data['test']
    except FileNotFoundError:
        print('File Not Found')
        sys.exit()

    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))
    model = CNN_Classifier(in_channels=3)
    optimizer = optim.Adam(model.parameters())
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()

    # load weights
    # path = 'runs/3D-CNN-Classifier/checkpoints/2023-03-17_01-24-56_epoch39.pt'
    model, optimizer, epoch, args = load_checkpoint(model, optimizer, path)
    # Set the model to evaluation mode
    model.eval()

    conv1_weights = model.conv2.weight.cpu().detach().numpy()
    # Normalize the weights for better visualization
    min_val, max_val = conv1_weights.min(), conv1_weights.max()
    conv1_weights_normalized = (conv1_weights - min_val) / (max_val - min_val)
    num_filters, num_input_channels = conv1_weights_normalized.shape[:2]

    # Plot the filters
    for i in range(num_filters):
        for j in range(num_input_channels):
            plt.subplot(num_filters, num_input_channels, i * num_input_channels + j + 1)
            plt.imshow(conv1_weights_normalized[i, j], cmap='gray')
            plt.axis('off')
    plt.show()


def gcn_corr_eval(path):
    temp_path = 'data/EC3D/tmp_wo_val.pickle'
    try:
        print('Loading saved data.')
        with open(temp_path, "rb") as f:
            data = pickle.load(f)
        data_test = data['test']
    except FileNotFoundError:
        print('File Not Found')
        sys.exit()

    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))
    model = GCN_Corrector()
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters())

    model, optimizer, epoch, args = load_checkpoint(model, optimizer, path)
    model.eval()
    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.eval()
    with torch.no_grad():
        for i, (batch_id, inputs) in enumerate(test_loader):
            # inputs = DCT
            if is_cuda:
                inputs = inputs.cuda().float()
            else:
                inputs = inputs.float()
            # target = 3D position
            targets = [test_loader.dataset.targets[int(i)] for i in batch_id]
            # inputs_raw = 3D position
            inputs_raw = [test_loader.dataset.inputs_raw[int(i)] for i in batch_id]
            batch_size = inputs.shape[0]
            deltas, att = model(inputs)
            outputs = inputs + deltas
            out = []
            for i, sample in enumerate(inputs_raw):
                num_frames = sample.shape[1]
                org_raw = torch.from_numpy(sample).T * 3000
                targ_raw = torch.from_numpy(targets[i]).T * 3000
                if num_frames > outputs[i].shape[1]:
                    m = torch.nn.ZeroPad2d((0, num_frames - deltas[i].shape[1], 0, 0))
                    outputs_raw = idct_2d(m(outputs[i].cpu()).T.unsqueeze(0))[0]
                else:
                    outputs_raw = idct_2d(outputs[i, :, :num_frames].cpu().T.unsqueeze(0))[0]
                out.append(outputs_raw)

        raw_inputs = [sample.reshape(3, 19, -1).T for sample in inputs_raw]


        # Plot specific still image
        for i, seq_in in enumerate(raw_inputs):
            label = data_test.inputs_label[i]
            num_frames = seq_in.shape[0]
            if label[2] == 2:
                print(label)
                seq_corr = out[i].numpy().T.reshape(3, 19, num_frames).transpose(2, 1, 0)
                viz = SkeletonVisualizer(num_frames, seq_in, seq_corr)
                viz.plot_still(25)

        # Plot all as animation
        for i, sample in enumerate(raw_inputs):
            print(data_test.inputs_label[i])
            num_frames = sample.shape[0]
            data2 = out[i].numpy().T.reshape(3, 19, num_frames).transpose(2, 1, 0)
            viz = SkeletonVisualizer(num_frames, sample, data2)
            viz.show()
            plt.show()
            plt.close()


        # for i, sample in enumerate(out):
        #     num_frames = sample.shape[0]
        #     data1 = inputs_raw[i].transpose().reshape(num_frames, 19, 3)
        #     sample = sample.cpu().numpy()
        #     data2 = sample.reshape(num_frames, 19, 3)
        #     viz = SkeletonVisualizer(num_frames, data1, data2, normalised=False)
        #     viz.show()
# ('SQUAT', 'Vidit', 2, 1, 'gt')

if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    available_eval = ['(1) Classifier', '(2) Correction']
    # class_dir = 'runs/3D-CNN-Classifier/checkpoints/2023-03-17_01-24-56_epoch39.pt'
    # class_eval(class_dir)
    gcn_dir = 'runs/GCN_Corrector/checkpoints/2023-03-17_22-19-17_epoch149.pt'
    gcn_corr_eval(gcn_dir)
    # visualise_weights(class_dir)
