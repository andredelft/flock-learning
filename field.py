# Inspired by https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from os import path
from datetime import datetime

from birds import Birds

FIELD_DIMS = 400 * np.array([-1,1,-1,1])
PLOTSCALE = 160

class Field(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numbirds, sim_length=12000, record_mov=False, record_data=False,
                 field_dims=FIELD_DIMS, periodic=True, plotscale=PLOTSCALE, plot=True,
                 observe_direction = False):

        self.birds = Birds(numbirds, field_dims, observe_direction)
        self.field_dims = field_dims
        self.stream = self.data_stream()
        self.periodic = periodic
        self.record_data = record_data
        self.plot = plot
        sim_length += 1

        if record_mov: # Force plotting of data (necessary for movie)
            self.plot = True

        if not self.plot: # Force recording if there is no visualization
            self.record_data = True
        if self.record_data:
            self.data_file = f'data/{datetime.now().strftime("%Y%m%d-%H%M%S")}.npy'
            self.history = []

        if self.plot:
            # Setup the figure and axes
            self.fig, self.ax = plt.subplots(
                figsize = (
                    abs(self.field_dims[1] - self.field_dims[0])/plotscale,
                    abs(self.field_dims[3] - self.field_dims[2])/plotscale
                )
            )

        if record_mov:
            writer = animation.FFMpegWriter(fps=24)
            mov_name = f'movies/{datetime.now().strftime("%Y%m%d-%H%M%S")}.mp4'
            with writer.saving(self.fig, mov_name, 100):
                self.setup_plot()
                for i in range(sim_length):
                    writer.grab_frame()
                    for _ in range(5):
                        self.update(i)
        elif self.plot:
            self.ani = animation.FuncAnimation(
                self.fig,self.update,interval=5,init_func=self.setup_plot,blit=True
            )
            plt.show()
        else: # No visualization, only recording of data
            for i in range(sim_length):
                self.update(i)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, = next(self.stream).T
        self.scat = self.ax.scatter(
            x, y, vmin=0, vmax=1, cmap="coolwarm", edgecolor="k",
            c = [
                1 if self.birds.instincts[i] == 'E' else 0
                for i in range(self.birds.numbirds)
            ])
        self.ax.axis(self.field_dims)
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        step = {
            'N': np.array([ 0, 1]),
            'E': np.array([ 1, 0]),
            'S': np.array([ 0,-1]),
            'W': np.array([-1, 0])
        }

        tstep = 0
        while True:
            self.birds.update()

            for i in range(self.birds.numbirds):
                if self.birds.dirs[i] == 'I':
                    direction = self.birds.instincts[i]
                else:
                    direction = self.birds.dirs[i]
                self.birds.positions[i] += step[direction]

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

            if self.record_data:
                # Calculate average flight direction v of the birds and record it.
                v = np.array([0,0])
                for i in range(self.birds.numbirds):
                    if self.birds.dirs[i] == 'I':
                        direction = self.birds.instincts[i]
                    else:
                        direction = self.birds.dirs[i]
                    v += step[direction]
                v = v / self.birds.numbirds
                self.history.append(v)
                if tstep % 500 == 0:
                    if tstep == 0:
                        # initialize save file with empty array
                        np.save(self.data_file, self.history)
                        print(f'Initialized record file {self.data_file}')
                    else:
                        data = np.load(self.data_file)
                        np.save(self.data_file, np.append(data, self.history, axis = 0))
                        print(f'Recorded up to timestep {tstep}')
                    self.history = []
            tstep += 1
            yield self.birds.positions

    def update(self,i): # When used in FuncAnimation, this function needs an
                        # additional argument for some reason (hence the i)
        """Update the scatter plot."""
        data = next(self.stream)

        if self.plot:
            # Set x and y data...
            self.scat.set_offsets(data[:, :2])

            # We need to return the updated artist for FuncAnimation to draw..
            # Note that it expects a sequence of artists, thus the trailing comma.
            return self.scat,
