import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataset import EC3D, EC3D_new
from utils.opt import Options


class SkeletonVisualizer:
    def __init__(self, num_frames, data1, data2=None, normalised=False):
        self.data1 = data1
        self.data2 = data2
        self.num_frames = num_frames
        self.normalised = normalised
        self.connections = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10],
                            [10, 11], [11, 17], [11, 18], [12, 13], [13, 14], [14, 15]]

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def update(self, frame):
        self.ax.clear()
        color = 'blue' if self.data2 is None else 'red'

        self.ax.scatter(self.data1[frame, :, 0], self.data1[frame, :, 1], self.data1[frame, :, 2], color=color)
        for i, j in self.connections:
            self.ax.plot([self.data1[frame, i, 0], self.data1[frame, j, 0]],
                         [self.data1[frame, i, 1], self.data1[frame, j, 1]],
                         [self.data1[frame, i, 2], self.data1[frame, j, 2]], color=color)

        if self.data2 is not None:
            self.ax.scatter(self.data2[frame, :, 0], self.data2[frame, :, 1], self.data2[frame, :, 2], color='green')
            for i, j in self.connections:
                self.ax.plot([self.data2[frame, i, 0], self.data2[frame, j, 0]],
                             [self.data2[frame, i, 1], self.data2[frame, j, 1]],
                             [self.data2[frame, i, 2], self.data2[frame, j, 2]],
                             color='green')

        if self.normalised:
            self.ax.set_xlim([-1.1, 1.1])
            self.ax.set_ylim([-1.1, 1.1])
            self.ax.set_zlim([-1.1, 1.1])
        else:
            self.ax.set_xlim([-0.5, 0.5])
            self.ax.set_ylim([-0.5, 0.5])
            self.ax.set_zlim([-0.5, 0.5])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.view_init(elev=30, azim=45)

    def show(self, save_filename=None):
        ani = FuncAnimation(self.fig, self.update, frames=self.num_frames, interval=100)
        if save_filename:
            ani.save(save_filename, writer='ffmpeg')
        else:
            plt.show()


def viz(arg):
    # Check CUDA
    is_cuda = torch.cuda.is_available()
    sets = [[0, 1, 2], [], [3]]
    data_train = EC3D_new(arg.raw_data_path, sets=sets, split=0, rep_3d=True, is_cuda=is_cuda)
    data_test = EC3D_new(arg.raw_data_path, sets=sets, split=0, rep_3d=True, is_cuda=is_cuda)
    print('Load complete.')

    for i, sample in enumerate(data_train.inputs_raw):
        data = sample.T
        num_frames = data.shape[0]
        viz = SkeletonVisualizer(num_frames, data)
        ani = viz.show()
        print(i)


def viz_from_saved():
    # Saved dataset
    with open(arg.processed_path, "rb") as f:
        data = pickle.load(f)
    data_train = data['train']
    data_test = data['test']
    viz_data = [sample.reshape(3, 19, -1).T for sample in data_test.inputs_raw]
    for i, sample in enumerate(viz_data):
        num_frames = sample.shape[0]
        print(data_test.inputs_label[i])
        viz = SkeletonVisualizer(num_frames, sample)
        viz.show()
        plt.show()
        plt.close()

def viz_from_processed():
    # Processing original dataset with original code
    sets = [[0, 1, 2], [], [3]]
    data_test_org = EC3D(arg.raw_data_path, sets=sets, split=2, is_cuda=is_cuda)
    viz_data = [sample.T for sample in data_test_org.inputs_raw]
    for i, sample in enumerate(viz_data):
        num_frames = sample.shape[0]
        print(data_test_org.inputs_label[i])
        viz = SkeletonVisualizer(num_frames, sample)
        viz.show()
        plt.show()
        plt.close()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    arg = Options().parse()

    # Check CUDA
    is_cuda = torch.cuda.is_available()

    # # Original dataset
    sets = [[0, 1, 2], [], [3]]
    data_test_new = EC3D_new(arg.raw_data_path, sets=sets, split=2, rep_3d=True, normalization='sample',
                             is_cuda=is_cuda)
    new_data = [sample for sample in data_test_new.inputs]

    for i, sample in enumerate(new_data):
        num_frames = sample.shape[0]
        label = data_test_new.inputs_label[i]
        if label[0] == 'Plank':
            viz = SkeletonVisualizer(num_frames, sample, normalised=True)
            viz.show()
            plt.show()
            plt.close()

