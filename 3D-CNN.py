import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

from dataset import EC3D_new
from utils.opt import Options
from models import CNN_Classifier, CNN_Classifier_v2
from train import train_cnn
from test import test_cnn


def main(arg):
    # Check CUDA
    is_cuda = torch.cuda.is_available()
    save_location = 'runs/3D-CNN-Classifier'

    # Load raw data
    sets = [[0, 1, 2], [], [3]]
    data_train = EC3D_new(arg.raw_data_path, sets=sets, split=0, rep_3d=True, is_cuda=is_cuda)
    data_test = EC3D_new(arg.raw_data_path, sets=sets, split=0, rep_3d=True, is_cuda=is_cuda)
    print('Load complete.')

    # Data Loader
    train_loader = DataLoader(dataset=data_train, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

    # Load model and move to CUDA device is possible
    # model = CNN_Classifier_v2(in_channels=3)
    model = CNN_Classifier(in_channels=3)
    if is_cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

    # Define the learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=arg.step_size, gamma=0.2)

    # Initialise tensorboard if requested
    start_tensorboard = input('Do you want to save Tensorboard? (y/n)\n')
    if start_tensorboard == 'y':
        arg.record = True

    if arg.record:
        print('Tensorboard enabled')
        run_id = arg.datetime
        print(f'Current run: {run_id}')

        writer = SummaryWriter(f'{save_location}/train/{run_id}')

        # Save parameters
        save_parameters(arg, save_location)
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

    # Ready to input into a CNN
    start_train = input('Would you like to start training the model? (y/n)\n')
    if start_train == 'y':

        with tqdm(range(arg.epoch), desc=f'Training model', unit="epoch") as tepoch:
            for epoch in tepoch:
                tr_l, tr_acc = train_cnn(train_loader, model, optimizer, is_cuda, writer, epoch, level=1)
                if (epoch + 1) % 10 == 0:
                    print(f'\nTraining_loss: {tr_l}')
                    print(f'Training_acc: {tr_acc}\n')

                # Update the learning rate
                scheduler.step()

        print('Training Complete.')

        start_test = input('Start testing? (y/n)\n')
        if start_test == 'y':
            with torch.no_grad():
                te_l, te_acc = test_cnn(test_loader, model, writer, is_cuda=is_cuda, level=1)
            print(f'Test Loss: {te_l}\n,Test Accuracy:{te_acc}')
        else:
            pass
    else:
        pass
    sys.exit()


def save_parameters(arg, save_location):
    parameters_dir = f'{save_location}/parameters'
    if not os.path.exists(parameters_dir):
        os.makedirs(parameters_dir)

    file_path = f'{parameters_dir}/{arg.datetime}.txt'
    with open(file_path, 'w') as f:
        f.write(f"Learning rate: {arg.lr}\n")
        f.write(f"Batch size: {arg.batch_size}\n")
        f.write(f"Number of epochs: {arg.epoch}\n")
        f.write(f"Step size: {arg.step_size}\n")
        f.write(f"Gamma: {arg.gamma}\n")


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    arg = Options().parse()
    main(arg)
