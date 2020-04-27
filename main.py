from os import path
import json
import time
import numpy as np
from random import sample
from concurrent.futures import ProcessPoolExecutor

from field import Field, gen_record_tag


def load_from_Q(record_tag = '', data_dir = 'data', plot = False,
                record_data = True, Q_tables = None, params = dict(), **kwargs):

    if not record_tag:
        record_tag = gen_record_tag()
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
    for key in ['no_dirs', 'observe_direction', 'comment']:
        if key in params.keys():
            params.pop(key)
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
    dev_inds = sample(range(no_states * no_birds), int(round(Delta * no_states * no_birds)))
    for dev_ind in dev_inds:
        i = dev_ind // no_states
        s = dev_ind %  no_states
        Q_tables[i,s] = 1 - Q_tables[i,s] # flip the 1 and 0
    load_from_Q(Q_tables = Q_tables, params = params, sim_length = sim_length, **kwargs)
    # return Q_tables

def tweak_learning_params(indexed_pars):
    i, pars = indexed_pars
    time.sleep(5 * i) # To make sure they don't start at exactly the same time, resulting in the same record tag
    Field(
        100, record_data = True, plot = False, sim_length = 80_000, reward_signal = 5,
        learning_alg = 'Q', gradient_reward = True, **pars
    )

if __name__ == '__main__':

    # Field(
    #     100, record_data = True, plot = True, reward_signal = 5, record_mov = False,
    #     learning_alg = 'Q', gradient_reward = True, track_time = True
    # )

    # for record_tag in ['20200409-164937', '20200415-154211', '20200416-190443', '20200418-114214'][1:2]:
    #     date = record_tag.split('-')[0]
    #     load_from_Q(
    #         record_tag, data_dir = f'data/{date}', record_data = True, record_mov = True,
    #         sim_length = 15_000, plot = False
    #     )

    for Delta in np.linspace(0.51, 1, 50):
        load_from_Delta(Delta)

    # lp_tweaks = [
    #     ('epsilon', 0.15),
    #     ('epsilon', 0.3),
    #     #('gamma', 0.5),
    #     #('alpha', 0.5),
    # ]
    # pars = [{par: value, 'comment': f'vary_{par}'} for par, value in lp_tweaks]
    #
    # with ProcessPoolExecutor() as executor:
    #     for _ in executor.map(tweak_learning_params, enumerate(pars)):
    #         pass
