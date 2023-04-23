import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import numpy as np
import random
from tqdm import tqdm
import sys
import os

from train import train_class, train_corr, train_transformer
from test import test_class, test_corr
from dataset import EC3D_V1, EC3D_V2
from models import Simple_GCN_Classifier, GCN_Corrector, GCN_Corrector_Attention
from utils.opt import Options
from utils.train_utils import load_original, load_DCT

"""
    Uses GCN Layers for the Classification Task
    Uses GC Residual Blocks (n=2) for the correction task
"""


def save_checkpoint(args, model, save_location, preds):
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


def train_classifier(arg):
    save_location = 'runs/GCN-Classifier'
    # Load or process the dataset
    data_train, data_test = load_original()

    # my_dct = 'data/EC3D/tmp_DCT_1CH.pickle'
    # with open(my_dct, "rb") as f:
    #     data = pickle.load(f)
    # data_train = data['train']
    # data_test = data['test']

    train_loader = DataLoader(dataset=data_train, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

    # Load model and Adam Optimizer
    model = Simple_GCN_Classifier()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

    # Define the learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=arg.step_size, gamma=arg.gamma)

    # Check Cuda
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()

    start_tensorboard = input('Do you want to save Tensorboard? (y/n)\n')

    if start_tensorboard == 'y':
        arg.record = True

    if arg.record:
        print('Tensorboard Enabled.')
        # Create a unique identifier for the run for TensorBoard
        run_id = arg.datetime
        print(f'Current run: {run_id}')

        # Initialise Tensorboard
        writer = SummaryWriter(f'{save_location}/train/{run_id}')

    else:
        print('Tensorboard disabled')
        writer = None

    print('Start training...')

    # range = 0 to epoch-1, description=Training model, unit updated every epoch to current epoch
    with tqdm(range(arg.epoch), desc=f'Training model', unit="epoch") as tepoch:
        for epoch in tepoch:
            tr_l, tr_acc = train_class(train_loader, model, optimizer, writer, epoch, is_cuda, level=1)
            if (epoch + 1) % 10 == 0:
                print(f'\nTraining_loss: {tr_l}')
                print(f'Training_acc: {tr_acc}\n')

            # Update the learning rate
            scheduler.step()

    print('Training Complete.')

    print('Start testing')

    with torch.no_grad():
        te_l, te_acc, _, _ = test_class(test_loader, model, is_cuda=is_cuda, level=1)
    print(f'Test Loss: {te_l}\n,Test Accuracy:{te_acc}')
    if writer is not None:
        writer.add_scalar('Classifier/Test Loss', te_l)
        writer.add_scalar('Classifier/Test Accuracy', te_acc)
        writer.close()
        save_checkpoint(arg, model, save_location, preds=None)


def train_corrector(arg):
    save_location = 'runs/GCN_Corrector'

    # Check Cuda
    is_cuda = torch.cuda.is_available()

    # Load or process the dataset
    print('Loading dataset..')
    data_train, data_test = load_DCT()

    print('Load complete.')

    # Data Loader
    train_loader = DataLoader(dataset=data_train, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test), shuffle=False)

    # Load model
    model = GCN_Corrector()
    if is_cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

    # Initialise tensorboard if requested
    start_tensorboard = input('Do you want to save Tensorboard? (y/n)\n')
    if start_tensorboard == 'y':
        save_name = input('What do you want to save this run as?')
        run_id = arg.datetime
        writer = SummaryWriter(f'{save_location}/train/{run_id}_{save_name}')
        print(f'{save_location}/train/{run_id}_{save_name}')
    else:
        writer = None

    print('Start training...')
    with tqdm(range(arg.epoch), desc=f'Training model', unit="epoch") as tepoch:
        for epoch in tepoch:
            tr_l = train_corr(train_loader, model, optimizer, epoch, writer, is_cuda=is_cuda)
            if (epoch + 1) % 10 == 0:
                print(f'\nTraining_loss: {tr_l}')

    print('Start testing...')
    torch.cuda.empty_cache()
    with torch.no_grad():
        te_l, preds = test_corr(test_loader, model, is_cuda=is_cuda, level=1)
    print(f'Test Loss: {te_l}')
    if writer is not None:
        writer.add_scalar('Corrector/Test Loss', te_l)
        writer.close()
        save_checkpoint(arg, model, save_location, preds)
    sys.exit()


def train_gcn_attn(arg):
    save_location = 'runs/GCN_Corrector_Attn'
    is_cuda = torch.cuda.is_available()
    data_train, data_test = load_DCT()
    train_loader = DataLoader(dataset=data_train, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test), shuffle=False)

    model = GCN_Corrector_Attention()
    if is_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

    # Initialise tensorboard if requested
    start_tensorboard = input('Do you want to save Tensorboard? (y/n)\n')
    if start_tensorboard == 'y':
        save_name = input('What do you want to save this run as?')
        run_id = arg.datetime
        writer = SummaryWriter(f'{save_location}/train/{run_id}_{save_name}')
        print(f'{save_location}/train/{run_id}_{save_name}')
    else:
        writer = None

    print('Start training...')
    with tqdm(range(arg.epoch), desc=f'Training model', unit="epoch") as tepoch:
        for epoch in tepoch:
            tr_l = train_corr(train_loader, model, optimizer, epoch, writer, is_cuda=is_cuda, attn=True)
            if (epoch + 1) % 10 == 0:
                print(f'\nTraining_loss: {tr_l}')

    print('Start testing...')
    torch.cuda.empty_cache()
    with torch.no_grad():
        te_l, preds = test_corr(test_loader, model, is_cuda=is_cuda, level=1, attn=True)
    print(f'Test Loss: {te_l}')
    if writer is not None:
        writer.add_scalar('Corrector/Test Loss', te_l)
        writer.close()
        save_checkpoint(arg, model, save_location, preds)
    sys.exit()


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)
    print('GPU Index: {}'.format(torch.cuda.current_device()))
    available_models = ['(1) GCN Classifier', '(2) GCN Corrector', '(3) GCN Corrector with Attention']
    print(f'Available models: {available_models}')
    model_options = [1, 2, 3]

    while True:
        model_version = int(input('Input the number of the model you would like to train: '))
        if model_version in model_options:
            arg = Options().parse()
            if model_version == 1:
                train_classifier(arg)
            if model_version == 2:
                train_corrector(arg)
            if model_version == 3:
                train_gcn_attn(arg)

        print('Please input valid number!')
