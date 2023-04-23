import pickle
import torch.cuda
from torch.utils.data import DataLoader
from utils.opt import Options
from utils.GraphAttention import GAT, GAT_Block
from utils.GraphAttention import GATv2
from utils.GraphAttention import GraphAttentionLayerV3, GATv3, GATv3_3
from utils.GraphAttention import GATv3_2
from utils.GraphAttention import Simple_GCN_Attention, GCN_Attention

from train import train_GAT
from test import test_GAT
from tqdm import tqdm
from dataset import EC3D
from utils.train_utils import load_3d, load_original, load_DCT
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import sys
import os


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


def main(arg, model_select):
    feat_dim = 25
    n_hidden = 1024
    n_heads = 4
    dropout = 0.5
    if model_select == 'GAT Only':
        print('Training GAT Only model with binary adjacency matrix')
        save_location = 'runs/GAT-Only-V1'
        use_adj = True
        model = GAT(in_features=feat_dim, n_hidden=n_hidden, n_heads=n_heads, dropout=dropout)
        # model = GAT_Block(in_features=feat_dim, n_hidden=n_hidden, n_heads=n_heads, dropout=dropout)
        # model = GATv2(in_features=feat_dim, n_hidden=n_hidden, n_heads=n_heads, dropout=dropout)
    if model_select == 'GAT V3':
        print('Training GAT only model with learnable adjacency matrix')
        save_location = 'runs/GAT-Only-V2'
        use_adj = False
        # model = GATv3(in_features=feat_dim, n_hidden=n_hidden, n_heads=n_heads, dropout=dropout)
        model = GATv3_3(in_features=feat_dim, n_hidden=n_hidden, n_heads=n_heads, dropout=dropout, share_weights=True)
        # model = GATv3_2(in_features=feat_dim, n_hidden=n_hidden, n_heads=n_heads, dropout=dropout, share_weights=True)
    if model_select == 'GCN GAT V1':
        print('Training GCN-GAT version 1')
        save_location = 'runs/GCN-GAT-V1'
        use_adj = False
        model = Simple_GCN_Attention(in_features=feat_dim, hidden_features=n_hidden, n_heads=n_heads, p_dropout=dropout)
    if model_select == 'GCN GAT V2':
        print('Training GCN-GAT Version 2')
        save_location = 'runs/GCN-GAT-V1'
        use_adj = False
        model = GCN_Attention(input_feature=feat_dim, hidden_feature=n_hidden, n_heads=n_heads, p_dropout=dropout)

    data_train, data_test = load_DCT()
    train_loader = DataLoader(dataset=data_train, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=arg.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)
    scheduler = StepLR(optimizer, gamma=0.5, step_size=10)

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()

    # Initialise tensorboard
    start_tensorboard = input('Do you want to save Tensorboard? (y/n)\n')
    if start_tensorboard == 'y':
        save_name = input('What do you want to save this run as?')
        run_id = arg.datetime
        writer = SummaryWriter(f'{save_location}/train/{run_id}_{save_name}')
        print(f'{save_location}/train/{run_id}_{save_name}')
    else:
        writer = None

    # Start train
    with tqdm(range(arg.epoch), desc=f'Training model', unit='epoch') as tepoch:
        for epoch in tepoch:
            # tr_l = train_GAT(train_loader, model, optimizer, epoch, writer=writer, is_cuda=is_cuda)
            tr_l = train_GAT(train_loader, model, optimizer, epoch, writer=writer, use_adj=use_adj, is_cuda=is_cuda)
            # tr_l = train_GAT(train_loader, model, optimizer, epoch, writer=writer, use_adj=True, is_cuda=is_cuda)
            if (epoch + 1) % 5 == 0:
                print(f'Training Loss: {tr_l}')
            scheduler.step()

    # Start test
    print('Start testing..')
    # Clear cache
    torch.cuda.empty_cache()
    te_l, preds = test_GAT(test_loader, model, use_adj=use_adj, is_cuda=is_cuda, level=1)
    print(f'Test Loss: {te_l}')
    if writer is not None:
        writer.add_scalar('GAT_Corrector/Test Loss', te_l)
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
    arg = Options().parse()
    available_models = ['(1) GAT Only', '(2) GAT w/ learnable adj matrix', '(3) Simple GCN with Attention', '(4) GCN-GAT V2']
    print(f'Available models: {available_models}')
    model_options = ['1', '2', '3', '4']
    while True:
        model_version = input('Input the number of the model you would like to train: ')
        if model_version in model_options:
            arg = Options().parse()
            if model_version == '1':
                model_select = 'GAT Only'
                main(arg, model_select)
            if model_version == '2':
                model_select = 'GAT V3'
                main(arg, model_select)
            if model_version == '3':
                model_select = 'GCN GAT V1'
                main(arg, model_select)
            if model_version == '4':
                model_select = 'GCN GAT V2'
                main(arg, model_select)

        print('Please input a valid number!')
