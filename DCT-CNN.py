import pickle
import sys
import os
import torch.cuda
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.opt import Options
from dataset import EC3D, EC3D_new
from models import CNN_Classifier
from train import train_cnn
from test import test_cnn


def main(arg):
    # Check CUDA
    is_cuda = torch.cuda.is_available()

    in_channels = 3
    # Directory to save data too
    if in_channels == 1:
        save_location = 'runs/DCT-CNN-Classifier-1CH'
    else:
        save_location = 'runs/DCT-CNN-Classifier-3CH'

    # Load raw data
    print('Processing raw data...')
    sets = [[0, 1, 2], [], [3]]
    data_train = EC3D_new(arg.raw_data_path, sets=sets, split=0, rep_dct=in_channels, rep_3d=False, is_cuda=is_cuda)
    data_test = EC3D_new(arg.raw_data_path, sets=sets, split=2, rep_dct=in_channels, rep_3d=False, is_cuda=is_cuda)
    print('Load complete.')

    # Data Loader
    train_loader = DataLoader(dataset=data_train, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

    # Load model and move to CUDA device is possible
    model = CNN_Classifier(in_channels=in_channels)

    if is_cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

    # Define the learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=arg.gamma)

    # Initialise tensorboard if requested
    start_tensorboard = input('Do you want to save Tensorboard? (y/n)\n')
    if start_tensorboard == 'y':
        arg.record = True

    if arg.record:
        print('Tensorboard enabled')
        run_id = arg.datetime
        print(f'Current run: {run_id}')

        writer = SummaryWriter(f'{save_location}/train/{run_id}')

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
        start_test = input('Would you like to test the trained model? (y/n)\n')

        if start_train == 'y':
            with torch.no_grad():
                te_l, te_acc = test_cnn(test_loader, model, is_cuda=is_cuda, level=1)

            print(f'Test Loss: {te_l}\nTest Accuracy:{te_acc}')
            if writer is not None:
                writer.add_scalar('Classifier/Test Loss', te_l)
                writer.add_scalar('Classifier/Test Accuracy', te_acc)
                writer.close()
        else:
            pass

        save_model = input('Would you like to save the trained model? (y/n)\n')

        if save_model == 'y':
            filename = str(arg.datetime)
            save_path = f'{save_location}/models'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), f'{save_path}/{filename}.pth')
            print(f'Save successful\nFilename: {filename}.pth')
        else:
            pass


    else:
        sys.exit()


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    arg = Options().parse()
    main(arg)
