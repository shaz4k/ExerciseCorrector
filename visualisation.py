import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataset import EC3D_new
from utils.opt import Options
from utils.train_utils import get_labels
from torch.utils.data import DataLoader
import sys

labels_dict = {0: ('SQUAT', 'Correct'), 1: ('SQUAT', 'Feets too wide'), 2: ('SQUAT', 'Knees inward'),
               3: ('SQUAT', 'Not low enough'), 4: ('SQUAT', 'Front bended'), 5: ('SQUAT', 'Unknown'),
               6: ('Lunges', 'Correct'), 7: ('Lunges', 'Not low enough'), 8: ('Lunges', 'Knees pass toes'),
               9: ('Plank', 'Correct'), 10: ('Plank', 'Arched back'), 11: ('Plank', 'Rolled back')}


class SkeletonVisualizer:
    def __init__(self, num_frames, data1, data2=None, text=None, normalised=False, colors=None, add_labels=None,
                 translate_to_joint=None, color_data1='red', hide_ticks=None):
        self.data1 = data1
        self.data2 = data2
        self.text = text
        self.num_frames = num_frames
        self.normalised = normalised
        self.add_labels = add_labels
        self.translate_to_joint = translate_to_joint
        self.hide_ticks = hide_ticks
        self.connections = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10],
                            [10, 11], [11, 17], [11, 18], [12, 13], [13, 14], [14, 15]]
        self.colors = colors if colors else {'data1': color_data1, 'data2': 'green'}
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(top=1.0, bottom=0.2, left=0.125, right=0.9, hspace=0.2, wspace=0.2)


    def update(self, frame):
        self.ax.clear()
        color_data1 = self.colors['data1']

        if self.translate_to_joint is not None:
            data1_frame = self.translate(self.data1[frame], self.translate_to_joint)
        else:
            data1_frame = self.data1[frame]

        self.ax.scatter(data1_frame[:, 0], data1_frame[:, 1], data1_frame[:, 2], color=color_data1, linewidth=5)
        for i, j in self.connections:
            self.ax.plot([data1_frame[i, 0], data1_frame[j, 0]],
                         [data1_frame[i, 1], data1_frame[j, 1]],
                         [data1_frame[i, 2], data1_frame[j, 2]], color=color_data1, linewidth=3)

        if self.data2 is not None:
            color_data2 = self.colors['data2']

            if self.translate_to_joint is not None:
                data2_frame = self.translate(self.data2[frame], self.translate_to_joint)
            else:
                data2_frame = self.data2[frame]

            self.ax.scatter(data2_frame[:, 0], data2_frame[:, 1], data2_frame[:, 2], color=color_data2, linewidth=5)
            for i, j in self.connections:
                self.ax.plot([data2_frame[i, 0], data2_frame[j, 0]],
                             [data2_frame[i, 1], data2_frame[j, 1]],
                             [data2_frame[i, 2], data2_frame[j, 2]],
                             color=color_data2, linewidth=3)
        self.ax.scatter(0, 0, 0, c='black', marker='x', s=50)

        # if self.normalised:
        x_min, x_max = np.min(data1_frame[:, 0]), np.max(data1_frame[:, 0])
        y_min, y_max = np.min(data1_frame[:, 1]), np.max(data1_frame[:, 1])
        z_min, z_max = np.min(data1_frame[:, 2]), np.max(data1_frame[:, 2])

        padding = 0.1

        self.ax.set_xlim([x_min - padding, x_max + padding])
        self.ax.set_ylim([y_min - padding, y_max + padding])
        self.ax.set_zlim([z_min - padding, z_max + padding])
        if self.hide_ticks is not None:
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.set_zticklabels([])

        # self.ax.view_init(elev=20, azim=50)
        self.ax.set_axis_off()
        self.fig.patch.set_visible(False)

    def add_joint_labels(self, data):
        for i in range(data.shape[0]):
            x, y, z = data[i]
            self.ax.text(x, y, z, str(i), fontsize=10, color='black', zorder=10)

    def translate(self, data, joint_index):
        translation_vector = data[joint_index, :]
        translated_data = data - translation_vector
        return translated_data

    def show(self, save_filename=None):
        ani = FuncAnimation(self.fig, self.update, frames=self.num_frames, interval=100)
        self.fig.text(0.5, 0.35, s=self.text, fontsize=12, color='black', ha='center')
        if save_filename:
            ani.save(save_filename, writer='ffmpeg')
        else:
            plt.show()

    def plot_still(self, frame, save_filename=None):
        self.update(frame)
        self.fig.text(0.5, 0.35, s=self.text, fontsize=12, color='black', ha='center')
        if save_filename:
            plt.savefig(save_filename)
        else:
            plt.show()


def viz_from_file(path, plot_still=None, plot_pairs=True, type='DCT'):
    # Saved dataset
    with open(path, "rb") as f:
        data = pickle.load(f)
    data_train = data['train']
    data_test = data['test']
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

    if type == 'Original DCT':
        input_data = [sample.reshape(3, 19, -1).T for sample in data_test.inputs_raw]
        target_data = [sample.reshape(3, 19, -1).T for sample in data_test.targets]
        normalization = False
    if type == 'DCT V2':
        input_data = [sample.T.reshape(-1, 19, 3) for sample in data_test.inputs_raw]
        target_data = [sample.T.reshape(-1, 19, 3) for sample in data_test.targets_raw]
        normalization = True
    if type == '3D':
        input_data = [sample for sample in data_test.inputs]
        target_data = [sample for sample in data_test.targets]
        normalization = True

    for batch_id, example in test_loader:
        labels = get_labels([test_loader.dataset.inputs_label[int(i)] for i in batch_id], level=1).cuda()
    labels = labels.cpu().detach().numpy()
    for i, seq_in in enumerate(input_data):
        num_frames = seq_in.shape[0]
        in_label = data_test.inputs_label[i]
        targ_label = data_test.targets_label[i]
        dataset_label = labels[i]
        dict_label = labels_dict[dataset_label]
        seq_out = target_data[i]
        col = None
        if plot_pairs is not True:
            seq_out = None
            squats = [0, 1, 2, 3, 4]
            lunge = [6, 7, 8]
            plank = [9, 10, 11]
        if dataset_label in squats:
            col = '#C73E1D'
        if dataset_label in lunge:
            col = 'orange'
            plot_still = 50

        if dataset_label in plank:
            col = '#2E86AB'

        viz = SkeletonVisualizer(num_frames, seq_in, seq_out, color_data1=col, normalised=normalization,
                             text=f"Input: {in_label}\nDataset Label: {dataset_label}\nDict Label: {dict_label}\nTarget: {targ_label}")
        if plot_still is not None:
            viz.plot_still(plot_still)
        else:
            viz.show()
        plt.show()
        plt.close()


if __name__ == '__main__':
    torch.cuda.set_device(0)

    original_dct = 'data/EC3D/tmp_wo_val.pickle'
    dct_v2 = 'data/EC3D/tmp_DCT.pickle'
    ds_3d = 'data/EC3D/tmp_3d.pickle'

    data_types = ['(1) Original DCT', '(2) DCT V2', '(3) 3D']
    data_select = ['1', '2', '3']

    print(f'Data types: {data_types}')
    while True:
        data_version = input('Input the number of the data you want to view: ')
        if data_version in data_select:
            if data_version == '1':
                path = original_dct
                type = 'Original DCT'
            if data_version == '2':
                path = dct_v2
                type = 'DCT V2'
            if data_version == '3':
                path = ds_3d
                type = '3D'
            plot_type = input('Do you want to plot incorrect and correct pairs of data? (y/n)')
            if plot_type == 'n':
                plot_type = False
            viz_from_file(path=path, plot_still=20, plot_pairs=plot_type,  type=type)
            sys.exit()
        print('Please input valid number!')
