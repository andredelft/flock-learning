# https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.spatial import KDTree

r_0 = 0.5
R = 10
N = 4

class Flock(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=50, fname=None):
        self.numpoints = numpoints
        self.birds = (np.random.random((self.numpoints, 2))-0.5)*200
        self.I = 2 * np.pi * np.random.random(self.numpoints) # instinct angle
        self.dirs = self.I
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        if fname:
            writer = animation.FFMpegWriter(fps=15)
            with writer.saving(self.fig, fname, 100):
                self.setup_plot()
                for i in range(100):
                    writer.grab_frame()
                    self.update(i)
        else:
            self.ani = animation.FuncAnimation(
                self.fig, self.update, interval=5,init_func=self.setup_plot, blit=True
            )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, = next(self.stream).T
        self.scat = self.ax.scatter(x, y, vmin=0, vmax=1, edgecolor="k")
        self.ax.axis([-100, 100, -100, 100])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def vicsek(self, no_neigbours=N, max_radius=R):
        tree = KDTree(self.birds)
        new_dirs = []
        for i,bird in enumerate(self.birds):
            neighbours = sorted(
                tree.query_ball_point(bird, max_radius),
                key = lambda j: np.linalg.norm(bird - self.birds[j])
            )[:N+1] # N nearest neigbours and itself
            new_dirs.append(
                # Circular mean of nearest neigbours (itself included)
                np.arctan2(
                    sum(np.sin(self.dirs[n]) for n in neighbours)/len(neighbours),
                    sum(np.cos(self.dirs[n]) for n in neighbours)/len(neighbours)
                )
            )
        self.dirs = new_dirs

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        while True:
            self.vicsek()
            self.birds += [[r_0 * np.cos(theta), r_0 * np.sin(theta)] for theta in self.dirs]
            # Periodic boundaries
            for i in range(self.birds.shape[0]):
                for j in range(self.birds.shape[1]):
                    if self.birds[i,j] > 100:
                        self.birds[i,j] -= 200
                    elif self.birds[i,j] < -100:
                        self.birds[i,j] += 200
            yield np.c_[self.birds[:,0], self.birds[:,1]]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


if __name__ == '__main__':
    a = Flock(200, fname="vicsek.mp4")
    # plt.show()
