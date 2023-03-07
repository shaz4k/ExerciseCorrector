import pickle
import torch
from torch.utils.data import DataLoader
from utils.train_utils import get_labels
from utils.opt import Options
from dataset import EC3D_new, EC3D
from models import GCN_Corrector, Simple_GCN_Classifier
import sys
import matplotlib.pyplot as plt


def main(arg):
    # Check CUDA
    is_cuda = torch.cuda.is_available()

    name = 'temp_3d.pkl'
    save_location = 'data/EC3D'
    try:
        with open(f'{save_location}/{name}', 'rb') as f:
            data = pickle.load(f)
        data_train = data[0]
        data_test = data[1]
    except FileNotFoundError:
        # Load raw data
        sets = [[0, 1, 2], [], [3]]
        # Preprocessing
        data_train = EC3D_new(arg.raw_data_path, sets=sets, split=0, rep_3d=True, is_cuda=is_cuda)
        data_test = EC3D_new(arg.raw_data_path, sets=sets, split=0, rep_3d=True, is_cuda=is_cuda)
        print('Load complete.')
        with open(f'{save_location}/{name}', 'wb') as f:
            pickle.dump((data_train, data_test), f)

    # test sample
    example = data_test.inputs_raw[0]
    length = example.shape[2]
    test_frame = example[:, :, 0].T

    connections = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10],
                        [10, 11], [11, 17], [11, 18], [12, 13], [13, 14], [14, 15]]
    # Create figure and axis objects
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot joint coordinates and connections
    ax.scatter(test_frame[:, 0], test_frame[:, 1], test_frame[:, 2], c='r', marker='o')  # plot joint positions
    for i, j in connections:
        ax.plot([test_frame[i, 0], test_frame[j, 0]], [test_frame[i, 1], test_frame[j, 1]], [test_frame[i, 2], test_frame[j, 2]],
                'r-')  # plot connection lines

    # Set axis limits and labels
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()




if __name__ == '__main__':
    torch.cuda.set_device(0)
    arg = Options().parse()
    main(arg)
