import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_skeleton_animation(num_frames, data1, data2=None):
    connections = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10],
                   [10, 11], [11, 17], [11, 18], [12, 13], [13, 14], [14, 15]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()

        # Plot first set of joint coordinates and connections
        ax.scatter(data1[frame, :, 0], data1[frame, :, 1], data1[frame, :, 2],
                   color='red')  # plot x-y-z coordinates of first skeleton
        for i, j in connections:
            ax.plot([data1[frame, i, 0], data1[frame, j, 0]],
                    [data1[frame, i, 1], data1[frame, j, 1]],
                    [data1[frame, i, 2], data1[frame, j, 2]], color='red')  # plot connection lines of first skeleton

        if data2 is not None:
            # Plot second set of joint coordinates and connections if
            ax.scatter(data2[frame, :, 0], data2[frame, :, 1], data2[frame, :, 2],
                       color='green')  # plot x-y-z coordinates of second skeleton
            for i, j in connections:
                ax.plot([data2[frame, i, 0], data2[frame, j, 0]],
                        [data2[frame, i, 1], data2[frame, j, 1]],
                        [data2[frame, i, 2], data2[frame, j, 2]],
                        color='green')  # plot connection lines of second skeleton

        # Set axis limits and labels
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set viewing angle
        ax.view_init(elev=30, azim=45)

    ani = FuncAnimation(fig, update, frames=num_frames, interval=50)

    return ani


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

    ani = plot_skeleton_animation(num_frames, data1, data2)
    my_ani = ani
    plt.show()



