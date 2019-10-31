# https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.spatial import KDTree
from os import path

r_0 = 0.3   # Stepsize
R = 10      # Sight radius (length )
N = 4

class Flock(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=50, fname=None):
        self.numpoints = numpoints
        self.birds = (np.random.random((self.numpoints, 2))-0.5)*200
        self.instincts = 2 * np.pi * np.random.random(self.numpoints) # instinct angle
        self.dirs = self.instincts # Initial flight angle
        self.stream = self.data_stream()
        self.trusts = np.random.random(self.numpoints)

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        if fname:
            writer = animation.FFMpegWriter(fps=15)
            with writer.saving(self.fig, path.join('movies',fname), 100):
                self.setup_plot()
                for i in range(400):
                    writer.grab_frame()
                    for _ in range(5):
                        self.update()
        else:
            self.ani = animation.FuncAnimation(
                self.fig, self.update, interval=5,init_func=self.setup_plot, blit=True
            )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, = next(self.stream).T
        self.scat = self.ax.scatter(x, y, vmin=0, vmax=1, c=self.trusts,
                                    cmap="coolwarm", edgecolor="k")
        self.ax.axis([-100, 100, -100, 100])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def partial_vicsek(self, no_neigbours=N, max_radius=R):
        tree = KDTree(self.birds)
        new_dirs = []
        for i,bird in enumerate(self.birds):
            if np.random.random() < self.trusts[i]:
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
            else:
                new_dirs.append(self.instincts[i])
        self.dirs = new_dirs

    def data_stream(self):
        while True:
            self.partial_vicsek()
            self.birds += [[r_0 * np.cos(theta), r_0 * np.sin(theta)] for theta in self.dirs]
            # Periodic boundaries
            for i in range(self.birds.shape[0]):
                for j in range(self.birds.shape[1]):
                    if self.birds[i,j] > 100:
                        self.birds[i,j] -= 200
                    elif self.birds[i,j] < -100:
                        self.birds[i,j] += 200
            yield np.c_[self.birds[:,0], self.birds[:,1]]

    def update(self):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


if __name__ == '__main__':
    a = Flock(50, fname="vicsek-instinct.mp4")
    plt.show()
