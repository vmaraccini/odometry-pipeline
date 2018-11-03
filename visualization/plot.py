import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plot:
    def __init__(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

    def update(self, trajectory):
        self.ax.plot(trajectory[:, 0].flatten(),
                     trajectory[:, 1].flatten(),
                     trajectory[:, 2].flatten())

        plt.pause(0.005)
