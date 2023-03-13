import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
import sys
import os

from train import train_class, train_corr
from test import test_class, test_corr
from dataset import EC3D
from models import Simple_GCN_Classifier, GCN_Corrector
from utils.opt import Options

"""
    Uses GCN Layers for the Classification Task
    Uses GC Residual Blocks (n=2) for the correction task
"""


def load_dataset():
    try:
        print('Loading saved data...')
        with open(arg.processed_path, "rb") as f:
            data = pickle.load(f)
        data_train = data['train']
        data_test = data['test']

    except FileNotFoundError:
        print('Processing raw data.')
        sets = [[0, 1, 2], [], [3]]
        is_cuda = torch.cuda.is_available()
        data_train = EC3D(arg.raw_data_path, sets=sets, split=0, is_cuda=is_cuda)
        data_test = EC3D(arg.raw_data_path, sets=sets, split=2, is_cuda=is_cuda)
        with open(arg.processed_path, 'wb') as f:
            pickle.dump({'train': data_train, 'test': data_test}, f)

    return data_train, data_test


def save_checkpoint(args, model, optimizer, epoch, save_location):
    checkpoint_dir = f'{save_location}/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = f'{checkpoint_dir}/{args.datetime}_epoch{epoch}.pt'
    checkpoint = {
        'args': args,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, checkpoint_path)


def train_classifier(arg):
    save_location = 'runs/GCN-Classifier'
    # Load or process the dataset
    data_train, data_test = load_dataset()

    # sets = [[0, 1, 2], [], [3]]
    # is_cuda = torch.cuda.is_available()
    # data_train = EC3D(arg.raw_data_path, sets=sets, split=0, is_cuda=is_cuda)
    # data_test = EC3D(arg.raw_data_path, sets=sets, split=2, is_cuda=is_cuda)

    print('Load complete.')

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

        # Get example data
        data_iter = iter(train_loader)
        _, example_input = next(data_iter)

        if is_cuda:
            example_input = example_input.float().cuda()
        else:
            example_input = example_input.float()

        writer.add_graph(model, example_input)
        writer.close()

    else:
        print('Tensorboard disabled')
        writer = None

    start_train = input('Do you want to start training the model? (y/n)\n')

    if start_train == 'y':

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

        start_test = input('Do you want to start testing? (y/n)\n')

        if start_test == 'y':
            print('Start testing')

            with torch.no_grad():
                te_l, te_acc, _, _ = test_class(test_loader, model, is_cuda=is_cuda, level=1)
            print(f'Test Loss: {te_l}\n,Test Accuracy:{te_acc}')
            if writer is not None:
                writer.add_scalar('Classifier/Test Loss', te_l)
                writer.add_scalar('Classifier/Test Accuracy', te_acc)
                writer.close()
        else:
            pass

        save_model = input('Would you like to save the trained model? (y/n)\n')
        if save_model == 'y':
            save_checkpoint(arg, model, optimizer, epoch, save_location)
        else:
            pass


    else:
        print('Aborted training.')
        sys.exit()

    sys.exit()


def train_corrector(arg):
    save_location = 'runs/GCN_Corrector'

    # Check Cuda
    is_cuda = torch.cuda.is_available()

    # Load or process the dataset
    print('Loading dataset..')
    data_train, data_test = load_dataset()
    print('Load complete.')

    # Data Loader
    train_loader = DataLoader(dataset=data_train, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

    # Load model
    model = GCN_Corrector()

    # Load model and move to CUDA device if possible
    if is_cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

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

        # Save parameters
        # save_parameters(arg, save_location)
        data_iter = iter(train_loader)
        _, example_input = next(data_iter)

        if is_cuda:
            example_input = example_input.float().cuda()
        else:
            example_input = example_input.float()

        writer.add_graph(model, example_input)
        writer.close()
    else:
        print('Tensorboard disabled')
        writer = None

    # Check start
    start_train = input('Do you want to train the GCN corrector? (y/n)\n')

    if start_train == 'y':
        print('Start training...')
        with tqdm(range(arg.epoch), desc=f'Training model', unit="epoch") as tepoch:
            for epoch in tepoch:
                tr_l = train_corr(train_loader, model, optimizer, writer, epoch, is_cuda=is_cuda)
                if (epoch + 1) % 10 == 0:
                    print(f'\nTraining_loss: {tr_l}')
        print('Training Complete.')

        start_test = input('Do you want to test the GCN corrector? (y/n)\n')
        if start_test == 'y':
            print('Start testing...')
            te_l, preds = test_corr(test_loader, model, is_cuda=is_cuda)
            print(f'Test Loss: {te_l}')
            if writer is not None:
                writer.add_scalar('Corrector/Test Loss', te_l)
                writer.close()
        else:
            pass

        save_model = input('Would you like to save the trained model? (y/n)\n')
        if save_model == 'y':
            save_checkpoint(arg, model, optimizer, epoch, save_location)
        else:
            pass

    else:
        print('Training aborted.')

    print('End.')
    sys.exit()


def train_class_corr(arg):
    start_train = input('Do you want to train the GCN classifier and corrector?')
    pass


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    print('GPU Index: {}'.format(torch.cuda.current_device()))
    available_models = ['(1) GCN Classifier', '(2) GCN Corrector', '(3) Combined GCN Classifier and Corrector']
    print(f'Available models: {available_models}')
    model_options = ['1', '2', '3']

    while True:
        model_version = input('Input the number of the model you would like to train: ')
        if model_version in model_options:
            arg = Options().parse()
            if model_version == '1':
                train_classifier(arg)
            if model_version == '2':
                train_corrector(arg)
            if model_version == '3':
                train_class_corr(arg)

        print('Please input valid number!')
