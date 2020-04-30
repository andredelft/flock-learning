from os import path
import json
import time
import numpy as np
from random import sample
from concurrent.futures import ProcessPoolExecutor

from field import Field
from utils import gen_rt

def benchmark():
    Field(
        100, plot = False, record_mov = False, learning_alg = 'Q',
        sim_length = 5000, record_data = True, comment = 'Reference'
    )
    Field(
        100, plot = False, record_mov = False, learning_alg = 'Q',
        sim_length = 5000, record_quantities = ['t', 'v'],
        comment = 'Tracking v and t'
    )
    Field(
        100, plot = False, record_mov = False, learning_alg = 'Q',
        sim_length = 5000, record_quantities = ['t', 'Delta'],
        comment = 'Tracking Delta and t'
    )
    Field(
        100, plot = False, record_mov = False, learning_alg = 'Q',
        sim_length = 5000, record_quantities = ['t', 'Delta', 'Q'],
        comment = 'Tracking Delta, Q and t'
    )
    Field(
        100, plot = False, record_mov = False, learning_alg = 'Q',
        sim_length = 5000, record_quantities = ['t', 'Delta', 'Q'],
        record_every = 1000, comment = 'Tracking Delta, Q and t (record every 1000)'
    )

def load_from_Q(record_tag = '', data_dir = 'data', plot = False,
                record_data = True, Q_tables = None, params = dict(), **kwargs):

    if not record_tag:
        record_tag = gen_rt()
    if not params:
        with open(path.join(data_dir, 'parameters.json')) as f:
            params = json.load(f)[record_tag]
        params['comment'] = record_tag
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
        Q_file = path.join(data_dir, f'{record_tag}-Q.npy')
        params['Q_file'] = Q_file
    if 'Q_params' in params.keys():
        params.update(params.pop('Q_params'))
    # pop some depracated or unused params
    [params.pop(key, '') for key in ['no_dirs', 'observe_direction', 'comment']]

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
        100, record_data = True, plot = False, sim_length = 40_000,
        learning_alg = 'Q', gradient_reward = True, **pars
    )

if __name__ == '__main__':

    # Four ways of running the model:
    #
    # 1. Start a regular simulation by creating a Field instance. Things like
    #    recording data and parameter specifications are all handled as keyword
    #    arguments.

    benchmark()

    # 2. Start a simulation with fixed Q-tables from a given file (the output of
    #    a learning run)

    # load_from_Q(
    #     record_tag, data_dir = f'data/{date}', record_data = True,
    #     record_mov = True, sim_length = 15_000, plot = False
    # )

    # 3. Start a simulation with fixed Q-values from a specific value of Delta

    # load_from_Delta(0.2)

    # 4. Start multiple simulations at the same time using multiprocessing, with
    #    the option of adjusting parameters in different runs (as exemplified
    #    below with some learning pars)

    # par_tweaks = [
    #     ('repos_every', 0),
    #     ('repos_every', 1000),
    #     ('repos_every', 5000),
    #     ('repos_every', 10_000),
    # ]
    # pars = [{par: value, 'comment': f'vary_{par}'} for par, value in par_tweaks]
    #
    # with ProcessPoolExecutor() as executor:
    #     for _ in executor.map(mp_wrapper, enumerate(pars)):
    #         pass
