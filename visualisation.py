import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataset import EC3D, EC3D_new
from utils.opt import Options


class SkeletonVisualizer:
    def __init__(self, num_frames, data1, data2=None, normalised=False, colors=None):
        self.data1 = data1
        self.data2 = data2
        self.num_frames = num_frames
        self.normalised = normalised
        self.connections = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10],
                            [10, 11], [11, 17], [11, 18], [12, 13], [13, 14], [14, 15]]
        self.colors = colors if colors else {'data1': 'red', 'data2': 'green'}
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def update(self, frame):
        self.ax.clear()
        color_data1 = self.colors['data1']

        # color = 'blue' if self.data2 is None else 'red'

        self.ax.scatter(self.data1[frame, :, 0], self.data1[frame, :, 1], self.data1[frame, :, 2], color=color_data1)
        for i, j in self.connections:
            self.ax.plot([self.data1[frame, i, 0], self.data1[frame, j, 0]],
                         [self.data1[frame, i, 1], self.data1[frame, j, 1]],
                         [self.data1[frame, i, 2], self.data1[frame, j, 2]], color=color_data1)

        if self.data2 is not None:
            color_data2 = self.colors['data2']
            self.ax.scatter(self.data2[frame, :, 0], self.data2[frame, :, 1], self.data2[frame, :, 2], color=color_data2)
            for i, j in self.connections:
                self.ax.plot([self.data2[frame, i, 0], self.data2[frame, j, 0]],
                             [self.data2[frame, i, 1], self.data2[frame, j, 1]],
                             [self.data2[frame, i, 2], self.data2[frame, j, 2]],
                             color=color_data2)

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

    def plot_still(self, frame, save_filename=None):
        self.update(frame)
        if save_filename:
            plt.savefig(save_filename)
        else:
            plt.show()


def viz_from_original(arg):
    # Saved dataset
    temp_path = 'data/EC3D/tmp_wo_val.pickle'
    with open(temp_path, "rb") as f:
        data = pickle.load(f)
    data_train = data['train']
    data_test = data['test']
    input_data = [sample.reshape(3, 19, -1).T for sample in data_test.inputs_raw]
    target_data = [sample.reshape(3, 19, -1).T for sample in data_test.targets]
    for i, seq_in in enumerate(input_data):
        num_frames = seq_in.shape[0]
        print(data_test.inputs_label[i])
        print(data_test.targets_label[i])
        seq_out = target_data[i]
        viz = SkeletonVisualizer(num_frames, seq_in, seq_out)
        # viz.show()
        viz.plot_still(20)
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
    # Non-interactice Agg backend for Matplotlib
    # import matplotlib
    #
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    torch.cuda.set_device(0)
    arg = Options().parse()

    viz_from_original(arg)
    # Check CUDA
    is_cuda = torch.cuda.is_available()

    # New dataset
    sets = [[0, 1, 2], [], [3]]
    data_test_new = EC3D_new(arg.raw_data_path, sets=sets, split=2, rep_3d=True, normalization='sample',
                             is_cuda=is_cuda)
    new_data = [sample for sample in data_test_new.inputs]

    for i, sample in enumerate(new_data):
        num_frames = sample.shape[0]
        label = data_test_new.inputs_label[i]
        if label[0] == 'SQUAT':
            print(label)
            colors = {'data1': 'red'}
            viz = SkeletonVisualizer(num_frames, sample, normalised=True, colors=colors)
            viz.plot_still(12)

            # viz.show()
            # plt.show()
            # plt.close()
