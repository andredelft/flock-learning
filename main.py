from os import path
import json
import time
import numpy as np
from random import sample
from concurrent.futures import ProcessPoolExecutor
from glob import glob
import regex

from field import Field
from utils import gen_rt, get_rt
import plot as p

def benchmark():
    Field(
        100, sim_length = 10000, plot = False, learning_alg = 'Q',
        record_time = True, record_data = True,
        comment = 'Reference: Q, v, Delta, t'
    )
    Field(
        100, sim_length = 10000, plot = False, learning_alg = 'Q',
        record_quantities = ['t', 'v'],
        comment = 'Tracking v and t'
    )
    Field(
        100, sim_length = 10000, plot = False, learning_alg = 'Q',
        record_quantities = ['t', 'Delta'],
        comment = 'Tracking Delta and t'
    )
    Field(
        100, sim_length = 10000, plot = False, learning_alg = 'Q',
        record_quantities = ['t', 'Q'],
        comment = 'Tracking Q and t'
    )
    Field(
        100, sim_length = 10000, plot = False, learning_alg = 'Q',
        record_quantities = ['t', 'Delta', 'Q'],
        comment = 'Tracking Delta, Q and t'
    )
    Field(
        100, sim_length = 10000, plot = False, learning_alg = 'Q',
        record_quantities = ['t', 'Delta', 'Q'], record_every = 1000,
        comment = 'Tracking Delta, Q and t (record every 1000)'
    )
    Field(
        100, sim_length = 10000, plot = False, learning_alg = 'Q',
        record_quantities = ['t', 'Delta', 'Q'], record_every = 5000,
        comment = 'Tracking Delta, Q and t (record every 5000)'
    )

def load_from_Q(fpath = '', record_tag = '', data_dir = '', plot = False,
                record_data = True, Q_tables = None, params = None, comment = '',
                **kwargs):

    if fpath:

        # Default guesses for record_tag, data_dir & params
        if not record_tag:
            record_tag = get_rt(fpath)

        if not data_dir:
            data_dir = path.split(fpath)[0]

        if not params:
            with open(path.join(data_dir, 'parameters.json')) as f:
                params = json.load(f)[record_tag]

        Q_file = fpath
        params['Q_file'] = Q_file

    elif type(Q_tables) == np.ndarray and params:

        params['Q_tables'] = Q_tables

    else:
        raise Exception(
            "Not enough information provided. Either the path to a Q-table " +
            "and the associated record tag should be given or a Q-table " +
            "with parameters should be provided."
        )

    params['comment'] = comment if comment else record_tag
    params.pop('learning_alg', '')

    no_birds = params.pop('no_birds')

    # Add instincts if they are present in data_dir
    instinct_file = path.join(data_dir, f'{record_tag}-instincts.json')
    if path.isfile(instinct_file):
        with open(instinct_file) as f:
            instincts = json.load(f)
            params['instincts'] = instincts


    if 'Q_params' in params.keys():
        params.update(params.pop('Q_params'))

    # Pop some depracated or unused params
    [params.pop(key, '') for key in ['no_dirs', 'observe_direction', 'record_every']]

    # Start the simulation
    Field(
        no_birds, plot = plot, record_data = record_data,
        learning_alg = 'pol_from_Q', **params, **kwargs
    )

def load_from_Delta(Delta, data_dir = 'data', no_birds = 100, leader_frac = 0.25,
                    sim_length = 5000, **kwargs):

    params = {
        'no_birds': no_birds,
        'leader_frac': leader_frac,
        'action_space': ['V', 'I']
    }
    no_states = 3 ** 8
    no_leaders = round(no_birds * leader_frac)
    Q_tables = np.zeros([no_birds, no_states, 2])
    for i in range(no_leaders):
        for s in range(no_states):
            Q_tables[i,s,1] = 1
    for i in range(no_leaders, no_birds):
        for s in range(no_states):
            Q_tables[i,s,0] = 1
    dev_inds = sample(
        range(no_states * no_birds), int(round(Delta * no_states * no_birds))
    )
    for dev_ind in dev_inds:
        i = dev_ind // no_states
        s = dev_ind %  no_states
        Q_tables[i,s] = 1 - Q_tables[i,s] # flip the 1 and 0

    load_from_Q(
        Q_tables = Q_tables, params = params, sim_length = sim_length, **kwargs
    )

def run_Q_dirs(data_dir):
    Q_dirs = sorted(glob(path.join(data_dir,'*-Q')))
    for Q_dir in Q_dirs:
        for fpath in sorted(glob(f'{Q_dir}/*.npy')):
            record_tag = get_rt(Q_dir)
            timestep = int(regex.search('\d+', path.split(fpath)[1]).group())
            load_from_Q(fpath, record_tag, data_dir = data_dir, comment = f'{record_tag}-{timestep:>06}', sim_length = 1500)

def _mp_wrapper(indexed_pars):
    i, pars = indexed_pars
    time.sleep(5 * i) # To make sure they don't start at exactly the same time,
                      # resulting in the same record tag

    for par in ['record_data', 'plot']:
        pars.pop(par, '')

    Field(100, record_data = True, plot = False, **pars)

def run_parallel(pars, **kwargs):
    pars.update(kwargs)
    with ProcessPoolExecutor() as executor:
        list(executor.map(_mp_wrapper, enumerate(pars)))

if __name__ == '__main__':
    pass

    # 1. Start a regular simulation directly by creating an instance of the
    # `Field` class. An integer should be passed that specifies the number of
    # birds.

    # from field import Field
    # Field(100)

    # Two other ways in which a `Field` instance can run is by recording a movie:

    # Field(100, record_mov = True, sim_length = 5000)

    # or just recording the data:

    # Field(100, record_data = True, plot = False, sim_length = 5000)

    # The quantities that are recorded can be further specified using the `record_quantities` keyword.


    # 2. Start a simulation with fixed Q-tables from a given file (the output of
    # a Q-learning training phase with `record_data = True`)

    # from main import load_from_Q
    # load_from_Q(fpath = 'data/20200531/2-VI/20200531-100329-Q.npy')


    # 3. Start a simulation with fixed Q-values from a specific value of Delta

    # from main import load_from_Delta
    # load_from_Delta(0.2)


    # 4. Start multiple simulations at the same time using multiprocessing,
    # with the option of adjusting parameters in different runs:

    # pars = [
    #     {'observation_radius': value}
    #     for value in [10, 50, 100, 150]
    # ]
    # run_parallel(pars, sim_length = 10_000, comment = 'vary_obs_rad')

    # NB: There is a complication regarding multiprocessing and the python
    # random module, which sometimes results in very similar initializations.
    # This problem has not been solved yet. Cf.:
    # https://www.sicara.ai/blog/2019-01-28-how-computer-generate-random-numbers.


    # 5. If the Q-tables of a given simulation are saved regularly (using the
    # option `Q_every`), these are saved in some directory, different (short)
    # simulations can be performed for every saved Q-table. The graphs on the
    # right hand side of [this](thesis-figures/learning_params.pdf) and
    # [this](thesis-figures/lead_frac_obs_rad_discrete.pdf) figure have been
    # generated using this option.

    # data_dir = 'path/to/some/dir'
    # run_Q_dirs(data_dir)


    # 6. Run a benchmark test and create a figure with the results

    # from main import benchmark
    # benchmark()
    # p.plot_all(
    #     quantity = 't', save_as = 'benchmark.png',
    #     title = 'Benchmark test', legend = 8
    # )
