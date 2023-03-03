import pickle
import sys
import torch.cuda
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.opt import Options
from dataset import EC3D
from models import CNN_Classifier
from train import train_cnn
from tqdm import tqdm


def main(arg):
    # Check CUDA
    is_cuda = torch.cuda.is_available()
    try:
        print('Loading saved data...')
        with open(arg.processed_path, "rb") as f:
            data = pickle.load(f)
        data_train = data['train']
        data_test = data['test']
    except FileNotFoundError:
        print('Processing raw data.')
        sets = [[0, 1, 2], [], [3]]
        data_train = EC3D(arg.raw_data_path, sets=sets, split=0, is_cuda=is_cuda)
    print('Load complete.')

    # Data Loader
    train_loader = DataLoader(dataset=data_train, batch_size=arg.batch_size, shuffle=True, drop_last=True)

    # Load model and move to CUDA device is possible
    model = CNN_Classifier()

    if is_cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

    # Define the learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Tensorboard
    start_tensorboard = input('Would you like to record with Tensorboard? (y/n)\n')

    if start_tensorboard == 'y':
        run_id = arg.datetime
        print(f'Current run: {run_id}')

        writer = SummaryWriter(f'runs/CNN_Classifier/train/{run_id}')

        data_iter = iter(train_loader)
        _, example_input = next(data_iter)

        if is_cuda:
            example_input = example_input.float().cuda()
        else:
            example_input = example_input.float()

        writer.add_graph(model, example_input)
        writer.close()
    else:
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
    else:
        sys.exit()
    print('Training Complete.')


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.set_device(0)

    arg = Options().parse()
    main(arg)
