import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SkeletonVisualizer:
    def __init__(self, num_frames, data1, data2=None):
        self.data1 = data1
        self.data2 = data2
        self.num_frames = num_frames
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

        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([-0.5, 0.5])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.view_init(elev=30, azim=45)

    def show(self, save_filename=None):
        ani = FuncAnimation(self.fig, self.update, frames=self.num_frames, interval=50)
        if save_filename:
            ani.save(save_filename, writer='ffmpeg')
        else:
            plt.show()


if __name__ == '__main__':
    print('Running example')
    saved_data = 'data/EC3D/temp_3d.pkl'
    # Loading saved data
    with open(saved_data, 'rb') as f:
        dataset = pickle.load(f)
    data_train = dataset[0]
    data_test = dataset[1]

    data1 = data_test.inputs_raw[49].T
    data2 = data_test.inputs_raw[50].T
    num_frames = min(data1.shape[0], data2.shape[0]) - 1

    # ani = plot_skeleton_animation(num_frames, data1, data2)
    # my_ani = ani
    # plt.show()

    viz = SkeletonVisualizer(num_frames, data1, data2)
    ani = viz.show(save_filename='img/test_animation.gif')
    plt.show()

