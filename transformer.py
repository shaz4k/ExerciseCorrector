import pickle
import torch.cuda
from torch.utils.data import DataLoader
from utils.opt import Options
from models import Simple_Transformer
from train import train_transformer
from tqdm import tqdm
from dataset import EC3D


def main(arg):
    # Load saved data or process the dataset
    print('Processing raw data.')
    sets = [[0, 1, 2], [], [3]]
    is_cuda = torch.cuda.is_available()
    data_train = EC3D(arg.raw_data_path, sets=sets, split=0, is_cuda=is_cuda)
    print('Load complete.')

    # Data Loader
    train_loader = DataLoader(dataset=data_train, batch_size=arg.batch_size, shuffle=True, drop_last=True)

    # Load model
    model = Simple_Transformer()

    # Check cuda
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

    # Train the model
    start_train = input('Do you want to train the Transformer? (y/n)\n')

    if start_train == 'y':
        print('Start training...')
        with tqdm(range(arg.epoch), desc=f'Training model', unit="epoch") as tepoch:
            for epoch in tepoch:
                tr_l = train_transformer(train_loader, model, optimizer, is_cuda=is_cuda)
                if (epoch + 1) % 10 == 0:
                    print(f'\nTraining_loss: {tr_l}')
        print('Training Complete.')


if __name__ == '__main__':
    torch.cuda.set_device(0)
    print('GPU Index: {}'.format(torch.cuda.current_device()))
    arg = Options().parse()
    main(arg)
