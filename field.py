# Inspired by https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from os import path

from birds import Birds

FIELD_DIMS = 400 * np.array([-1,1,-1,1])
PLOTSCALE = 160

class Field(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numbirds, fname=None, field_dims=FIELD_DIMS,
                 periodic=True, plotscale=PLOTSCALE):

        self.birds = Birds(numbirds, field_dims)
        self.field_dims = field_dims
        self.stream = self.data_stream()
        self.periodic = periodic

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(
            figsize = (
                abs(self.field_dims[1] - self.field_dims[0])/plotscale,
                abs(self.field_dims[3] - self.field_dims[2])/plotscale
            )
        )

        # Then setup FuncAnimation.
        if fname:
            writer = animation.FFMpegWriter(fps=15)
            with writer.saving(self.fig, path.join('movies', fname), 100):
                self.setup_plot()
                for i in range(400):
                    writer.grab_frame()
                    for _ in range(5):
                        self.update()
        else:
            self.ani = animation.FuncAnimation(
                self.fig,self.update,interval=5,init_func=self.setup_plot,blit=True
            )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, = next(self.stream).T
        self.scat = self.ax.scatter(x, y, vmin=0, vmax=1, c=np.ones(self.birds.numbirds),
                                    cmap="coolwarm", edgecolor="k")
        self.ax.axis(self.field_dims)
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        step = {
            'N': np.array([0,1]),
            'E': np.array([1,0]),
            'S': np.array([0,-1]),
            'W': np.array([-1,0])
        }

        while True:
            self.birds.update()

            for i in range(self.birds.numbirds):
                self.birds.positions[i] += step[self.birds.dirs[i]]
            print(self.birds.positions[0,1])

            # Periodic boundaries
            if self.periodic:
                for i in range(self.birds.numbirds):
                    if self.birds.positions[i,0] < self.field_dims[0]:
                        self.birds.positions[i,0] += self.field_dims[1] - self.field_dims[0]
                    elif self.birds.positions[i,0] > self.field_dims[1]:
                        self.birds.positions[i,0] -= self.field_dims[1] - self.field_dims[0]
                    if self.birds.positions[i,1] < self.field_dims[2]:
                        self.birds.positions[i,1] += self.field_dims[3] - self.field_dims[2]
                    elif self.birds.positions[i,1] > self.field_dims[3]:
                        self.birds.positions[i,1] -= self.field_dims[3] - self.field_dims[2]

            yield self.birds.positions

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,
