import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train import train_class, train_corr
from test import test_class
from dataset import EC3D
from models import Simple_GCN_Classifier, GCN_Corrector
from opt import Options


def train_classifier(arg):
    start_tensorboard = input('Do you want to save Tensorboard? (y/n)\n')

    if start_tensorboard == 'y':
        # Create a unique identifier for the run for TensorBoard
        run_id = arg.datetime
        print(f'Current run: {run_id}')

        # Initialise Tensorboard
        writer_tr = SummaryWriter(f'runs/GCN_Class/train/{run_id}')
        writer_test = SummaryWriter(f'runs/GCN_Class/train/{run_id}')
    else:
        pass

    # Check Cuda
    is_cuda = torch.cuda.is_available()

    # Load or process the dataset
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

    print('Load complete.')
    train_loader = DataLoader(dataset=data_train, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

    model = Simple_GCN_Classifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

    if is_cuda:
        model.cuda()

    # save_graph = input('Do you want to save model graph to Tensorboard? (y/n)\n')
    # if save_graph == 'y':
    #     examples = iter(train_loader)
    #     batch_ids, example_targets = next(examples)
    #     writer.add_graph(model, example_targets)
    # else:
    #     pass

    start_train = input('Do you want to start training the model? (y/n)\n')

    if start_train == 'y':

        print('Start training...')

        # range = 0 to epoch-1, description=Training model, unit updated every epoch to current epoch
        with tqdm(range(arg.epoch), desc=f'Training model', unit="epoch") as tepoch:
            for epoch in tepoch:
                tr_l, tr_acc = train_class(train_loader, model, optimizer, is_cuda=is_cuda, level=1)

                if (epoch + 1) % 10 == 0:
                    print(f'\nTraining_loss: {tr_l}')
                    print(f'Traning_acc: {tr_acc}\n')

        print('Training Complete.')

        start_test = input('Do you want to start testing? (y/n)\n')

        if start_test == 'y':
            print('Start training')

            with torch.no_grad():
                te_l, te_acc, _, _ = test_class(test_loader, model, is_cuda=is_cuda, level=1)

            print(f'Test Loss: {te_l}\n,Test Accuracy:{te_acc}')

        else:
            print('Aborted testing.')

    else:
        print('Aborted training.')

def train_corrector(arg):
    start_train = input('Do you want to train the GCN corrector?')
    pass

def train_class_corr(arg):
    start_train = input('Do you want to train the GCN classifier and corrector?')
    pass



if __name__ == '__main__':
    torch.cuda.set_device(0)
    print('GPU Index: {}'.format(torch.cuda.current_device()))
    available_models = ['(1) GCN Classifier', '(2) GCN Corrector', '(3) Combined GCN Classifier and Corrector']
    # print('Model Options:\t' + "   ".join(str(x) for x in model_options))
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
