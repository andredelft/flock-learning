import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from os import path
import json
import pickle
from tqdm import trange
import time
from statistics import stdev

from birds import Birds
from utils import gen_rt

FIELD_DIMS = 400 * np.array([-1,1,-1,1])
PLOTSCALE = 160

class Field(object):

    def __init__(self, numbirds, sim_length = 12500, record_mov = False,
                 record_data = False, record_time = False,
                 record_quantities = [], field_dims = FIELD_DIMS,
                 periodic = True, plotscale = PLOTSCALE, plot = True,
                 comment = '', Q_every = 0, repos_every = 0,
                 record_every = 500, **kwargs):

        if record_time or 't' in record_quantities:
            self.t_prev = time.perf_counter()
            self.times = []

        self.birds = Birds(numbirds, field_dims, **kwargs)
        self.field_dims = field_dims
        self.stream = self.data_stream()
        self.periodic = periodic

        self.record_every = record_every
        if (not record_quantities) and record_data:
            # Record a default set of quantities to be recorded if no specific
            # quantities are provided but record_data == True
            if self.birds.learning_alg == 'Ried':
                record_quantities = ['v', 'policies', 'instincts']
            elif self.birds.learning_alg == 'Q':
                record_quantities = ['v', 'Delta', 'Q', 'instincts']
            elif self.birds.learning_alg == 'pol_from_Q':
                record_quantities = ['v', 'instincts']

        self.record_time = record_time or 't' in record_quantities
        self.record_v = 'v' in record_quantities
        self.record_Q = 'Q' in record_quantities
        self.record_Delta = 'Delta' in record_quantities
        self.record_policies = 'policies' in record_quantities
        self.record_instincts = 'instincts' in record_quantities
        self.record_quantities = record_quantities or record_time

        self.record_mov = record_mov
        self.Q_every = Q_every
        self.repos_every = repos_every
        self.plot = plot
        self.sim_length = sim_length + 1 # include zero, but don't count it

        self.record_tag = gen_rt()
        param_file = 'data/parameters.json'
        params = self.birds.request_params()
        params['record_every'] = self.record_every
        if self.repos_every:
            params['repos_every'] = self.repos_every
        if comment:
            params['comment'] = comment

        if record_mov: # Force plotting of data (necessary for movie)
            self.plot = True
            params['record_mov'] = True

        # Do not record parameters when no data is recorded and only a plot is
        # generated
        if not ((not record_quantities) and (not self.record_mov) and self.plot):
            if path.isfile(param_file):
                with open(param_file) as f:
                    existing_pars = json.load(f)
                with open(param_file, 'w') as f:
                    json.dump(
                        {**existing_pars, self.record_tag: params},
                        f, indent = 2
                    )
            else:
                with open(param_file, 'w') as f:
                    json.dump({self.record_tag: params}, f, indent = 2)

        # Setup the files for tracking the birds
        if record_quantities:
            self.init_record_files()

        # Setup the figure and axes
        if self.plot:
            self.fig, self.ax = plt.subplots(
                figsize = (
                    abs(self.field_dims[1] - self.field_dims[0])/plotscale,
                    abs(self.field_dims[3] - self.field_dims[2])/plotscale
                )
            )

        # Kick off the model!
        if record_mov:
            writer = animation.FFMpegWriter(fps=24)
            mov_name = f'movies/{self.record_tag}.mp4'
            print(f'Initiated movie file {self.record_tag}.mp4')
            with writer.saving(self.fig, mov_name, 100):
                self.setup_plot()
                for i in trange(self.sim_length//5, desc="Recording frames"):
                    writer.grab_frame()
                    # Record every fifth frame
                    for _ in range(5):
                        self.update(i)
        elif self.plot:
            self.ani = animation.FuncAnimation(
                self.fig, self.update, interval = 5,
                init_func = self.setup_plot, blit = True
            )
            plt.show()
        else: # No visualization, only recording of data
            for i in range(self.sim_length):
                self.update(i)

    def setup_plot(self):
        # https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
        x, y, = next(self.stream).T
        self.scat = self.ax.scatter(
            x, y, vmin = 0, vmax = 1, cmap = "coolwarm", edgecolor = "k",
            c = [
                1 if self.birds.instincts[i] == 'E' else 0
                for i in range(self.birds.numbirds)
            ]
        )
        self.ax.axis(self.field_dims)
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def init_record_files(self):
        if self.record_v:
            self.v_fname = f'data/{self.record_tag}-v.npy'
            # Initialize v-file so it always exists
            np.save(self.v_fname, np.array([]))
            self.v_history = []

        if self.record_time:
            self.t_fname = f'data/{self.record_tag}-t.npy'
            np.save(self.t_fname, np.array([]))

        if self.record_instincts:
            with open(f'data/{self.record_tag}-instincts.json','w') as f:
                json.dump(self.birds.instincts, f)

        if self.record_Q:
            if self.Q_every:
                os.mkdir(f'data/{self.record_tag}-Q')

            self.Q_fname = f'data/{self.record_tag}-Q.npy'

        if self.record_Delta:
            self.Delta_fname = f'data/{self.record_tag}-Delta.npy'
            np.save(self.Delta_fname, np.array([]))

        if self.record_policies:
            self.policy_fname = f'data/{self.record_tag}-policies.npy'

        print(f'Record files with tag {self.record_tag} initalized')

    def record(self, tstep):
        if self.record_v:
            v = self.birds.calc_v()
            self.v_history.append(v)

        if tstep % self.record_every == 0:
            if self.record_v:
                v_data = np.load(self.v_fname)
                if v_data.size == 0:
                    v_data = self.v_history
                else:
                    v_data = np.append(v_data, self.v_history, axis = 0)
                np.save(self.v_fname, v_data)
                self.v_history = []

            if self.record_Q:
                if self.Q_every and tstep % self.Q_every == 0:
                    np.save(
                        f'data/{self.record_tag}-Q/{tstep:08}.npy',
                        np.array([Q.table for Q in self.birds.Qs])
                    )
                else:
                    np.save(
                        self.Q_fname,
                        np.array([Q.table for Q in self.birds.Qs])
                    )

            if self.record_Delta:
                if self.birds.learning_alg == 'Q':
                    Delta = self.birds.calc_Delta()
                    Delta_data = np.load(self.Delta_fname)
                    Delta_data = np.append(Delta_data, Delta)
                    np.save(self.Delta_fname, Delta_data)

            if self.record_policies:
                np.save(self.policy_fname, self.birds.policies)

            if self.record_time:
                if tstep != 0:
                    # del self.times[0]
                    print(
                        f'{round(sum(self.times)/len(self.times),3)}',
                        f'± {round(stdev(self.times), 3)} s/timestep'
                    )
                t_data = np.load(self.t_fname)
                t_data = np.append(t_data, self.times, axis = 0)
                np.save(self.t_fname, t_data)

                self.times = []

            if not self.record_mov: # to not interfere with tqdm progress bar
                print(f'Recorded up to timestep {tstep}')

    def update(self,i): # When used in FuncAnimation, this function needs an
                        # additional argument for some reason (hence the i)

        data = next(self.stream)
        if self.record_time:
            t_now = time.perf_counter()
            self.times.append(t_now - self.t_prev)
            self.t_prev = t_now

        if self.plot:
            # Update the scatter plot
            self.scat.set_offsets(data[:, :2])

            # We need to return the updated artist for FuncAnimation to draw..
            # Note that it expects a sequence of artists, thus the trailing comma.
            return self.scat,

    def data_stream(self):

        tstep = 0
        while True:
            self.birds.update()

            # Periodic boundaries
            if self.periodic:
                for i in range(self.birds.numbirds):
                    if self.birds.positions[i,0] < self.field_dims[0]:
                        self.birds.positions[i,0] += (
                            self.field_dims[1] - self.field_dims[0]
                        )
                    elif self.birds.positions[i,0] > self.field_dims[1]:
                        self.birds.positions[i,0] -= (
                            self.field_dims[1] - self.field_dims[0]
                        )
                    if self.birds.positions[i,1] < self.field_dims[2]:
                        self.birds.positions[i,1] += (
                            self.field_dims[3] - self.field_dims[2]
                        )
                    elif self.birds.positions[i,1] > self.field_dims[3]:
                        self.birds.positions[i,1] -= (
                            self.field_dims[3] - self.field_dims[2]
                        )

            if self.record_quantities:
                self.record(tstep)

            if self.repos_every and tstep % self.repos_every == 0:
                self.birds.initialize_positions(self.field_dims)
                # Redo observation step with new positions
                self.birds.perform_observations()

            tstep += 1
            yield self.birds.positions
