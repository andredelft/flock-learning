from os import path
import json
import time
import numpy as np
from random import sample
from concurrent.futures import ProcessPoolExecutor

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

def load_from_Q(fpath, record_tag, data_dir = 'data', plot = False,
                record_data = True, Q_tables = None, params = None, comment = '',
                **kwargs):

    if not params:
        with open(path.join(data_dir, 'parameters.json')) as f:
            params = json.load(f)[record_tag]

    params['comment'] = comment if comment else record_tag
    params.pop('learning_alg')

    no_birds = params.pop('no_birds')

    instinct_file = path.join(data_dir, f'{record_tag}-instincts.json')
    if path.isfile(instinct_file):
        with open(instinct_file) as f:
            instincts = json.load(f)
            params['instincts'] = instincts

    if type(Q_tables) == np.ndarray:
        params['Q_tables'] = Q_tables
    else:
        Q_file = fpath
        params['Q_file'] = Q_file

    if 'Q_params' in params.keys():
        params.update(params.pop('Q_params'))
    # pop some depracated or unused params
    [params.pop(key, '') for key in ['no_dirs', 'observe_direction', 'record_every']]

    Field(
        no_birds, plot = plot, record_data = record_data,
        learning_alg = 'pol_from_Q', **params, **kwargs
    )

def load_from_Delta(Delta, data_dir = 'data', no_birds = 100, leader_frac = 0.25,
                    no_dirs = 8, sim_length = 5000, **kwargs):

    params = {
        'no_birds': no_birds,
        'leader_frac': leader_frac,
        'action_space': ['V', 'I']
    }
    no_states = 3 ** no_dirs
    no_leaders = int(no_birds * leader_frac)
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

def mp_wrapper(indexed_pars):
    i, pars = indexed_pars
    time.sleep(5 * i) # To make sure they don't start at exactly the same time,
                      # resulting in the same record tag
    Field(
        100, record_data = True, plot = False, sim_length = 1_000_000,
        learning_alg = 'Q', Q_every = 10_000, record_every = 5_000, **pars
    )

if __name__ == '__main__':

    # Different ways of running the model:
    #
    # 1. Start a regular simulation by creating a Field instance. Things like
    #    recording data and parameter specifications are all handled as keyword
    #    arguments.

    # Field(
    #     100, sim_length = 10_000_000, plot = False, learning_alg = 'Q',
    #     record_quantities = ['Delta', 'Q', 'instincts'], record_every = 5000,
    #     Q_every = 50_000
    # )

    # 2. Start a simulation with fixed Q-tables from a given file (the output of
    #    a learning run)

    # load_from_Q(
    #     fname = 'data/20200501-164233-Q/1130000.npy', data_dir = 'data',
    #     record_tag = '20200501-164233', record_data = False,
    #     record_mov = False, sim_length = 15_000, plot = True
    # )

    # 3. Start a simulation with fixed Q-values from a specific value of Delta

    # load_from_Delta(0.2)

    # 4. Start multiple simulations at the same time using multiprocessing, with
    #    the option of adjusting parameters in different runs (as exemplified
    #    below with some learning pars)

    par_tweaks = [
        ('gamma', 0.0),
        ('gamma', 0.99),
        # ('alpha', 0.0),
        # ('alpha', 0.2),
        # ('alpha', 0.8),
        # ('alpha', 1),
        # ('epsilon', 0.0),
        # ('epsilon', 0.2),
        # ('epsilon', 0.6),
        # ('epsilon', 0.8),
    ]
    pars = [{par: value, 'comment': f'vary_{par}'} for par, value in par_tweaks]
    pars += [{par: value, 'gradient_reward': False, 'comment': f'vary_{par} (no gradient)'} for par, value in par_tweaks] # reference simulations
    with ProcessPoolExecutor() as executor:
        for _ in executor.map(mp_wrapper, enumerate(pars)):
            pass


    # 5. Run a benchmark test and create a figure with the results

    # benchmark()
    # p.plot_all(
    #     quantity = 't', save_as = 'benchmark.png',
    #     title = 'Benchmark test', legend = 8
    # )
