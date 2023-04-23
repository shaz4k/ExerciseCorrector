import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchviz import make_dot
import random
import numpy as np
import pickle
import sys
import os

from dataset import EC3D_new
from utils.opt import Options
from models import CNN_Classifier, CNN_Classifier_v2
from train import train_cnn
from test import test_cnn
from utils.train_utils import load_3d


def load_dataset():
    temp_path = 'data/EC3D/tmp_3d.pickle'
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
        data_train = EC3D_new(arg.raw_data_path, sets=sets, split=0, rep_3d=True, normalization='sample',
                              is_cuda=is_cuda)
        data_test = EC3D_new(arg.raw_data_path, sets=sets, split=2, rep_3d=True, normalization='sample',
                             is_cuda=is_cuda)
        print('Writing processed data...')
        with open(temp_path, 'wb') as f:
            pickle.dump({'train': data_train, 'test': data_test}, f)
        print('Complete')

    return data_train, data_test


def main(arg, model_select):
    save_location = 'runs/3D-CNN-Classifier_v2'
    if model_select == 'Basic':
        print('Training Basic CNN Classifier')
        save_location = 'runs/3D-CNN-Classifier'
        model = CNN_Classifier(in_channels=3)
    if model_select == 'v2':
        print('Training CNN Classifier v2')
        save_location = 'runs/3D-CNN-Classifier_v2'
        model = CNN_Classifier_v2(in_channels=3)

    # Check CUDA
    is_cuda = torch.cuda.is_available()
    # Load model and move to CUDA device is possible
    if is_cuda:
        model.cuda()

    # Load raw data
    data_train, data_test = load_dataset()
    # data_train, data_test = load_3d()

    # Data Loader
    train_loader = DataLoader(dataset=data_train, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

    # Define the learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=arg.step_size, gamma=arg.gamma)

    # Initialise tensorboard if requested
    if not arg.record:
        start_tensorboard = input('Do you want to save Tensorboard? (y/n)\n')
        if start_tensorboard == 'y':
            arg.record = True

    if arg.record:
        print('Tensorboard enabled')
        run_id = arg.datetime
        print(f'Current run: {run_id}')

        writer = SummaryWriter(f'{save_location}/train/{run_id}')

    else:
        print('Tensorboard disabled')
        writer = None

    # Train model
    with tqdm(range(arg.epoch), desc=f'Training model', unit="epoch") as tepoch:
        for epoch in tepoch:
            tr_l, tr_acc = train_cnn(train_loader, model, optimizer, is_cuda, writer, epoch, level=1)
            if (epoch + 1) % 10 == 0:
                print(f'\nTraining_loss: {tr_l}')
                print(f'Training_acc: {tr_acc}\n')

    # Test model
    print('Starting testing...')
    with torch.no_grad():
        te_l, te_acc, preds = test_cnn(test_loader, model, is_cuda=is_cuda, level=1)
    print(f'Test Loss: {te_l}\nTest Accuracy:{te_acc}')
    if writer is not None:
        writer.add_scalar('Classifier/Test Loss', te_l)
        writer.add_scalar('Classifier/Test Accuracy', te_acc)
        writer.close()
        save_checkpoint(arg, model, save_location, preds)

    sys.exit()


def save_checkpoint(args, model, save_location, preds=None):
    checkpoint_dir = f'{save_location}/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = f'{checkpoint_dir}/{args.datetime}.pt'
    checkpoint = {
        'args': args,
        'model_state_dict': model.state_dict(),
        'results': preds
    }
    torch.save(checkpoint, checkpoint_path)


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)
    available_models = ['(1) Basic CNN Classifier', '(2) CNN Classifier v2']
    print(f'Available models: {available_models}')
    model_options = ['1', '2']
    while True:
        model_version = input('Input the number of the model you would like to train: ')
        if model_version in model_options:
            arg = Options().parse()
            if model_version == '1':
                model_select = 'Basic'
                main(arg, model_select)
            if model_version == '2':
                model_select = 'v2'
                main(arg, model_select)
        print('Please input a valid number!')
