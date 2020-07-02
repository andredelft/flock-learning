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

    pars = [
        {'observation_radius': value}
        for value in [10, 50, 100, 150]
    ]
    run_parallel(pars, sim_length = 10_000, comment = 'vary_obs_rad')
        # for indexed_pars in enumerate(pars):
        #     result = executor.submit(mp_wrapper, indexed_pars)
        #     print(result)

    # Field(100, record_mov = True, sim_length = 3000, record_quantities = ['v'], eps_decr = 2000, leader_frac = 0.4)
    #action_space = ['V', 'I']
    #for :
    #    Field(
    #        100, sim_length = 5000, gradient_reward = False, reward_signal = r,
    #        action_space = ['V', 'I'], plot = False, record_mov = False, learning_alg = 'Ried', record_data = True
    #    )


    #for par in ['alpha', 'gamma', 'epsilon']:
    #for gamma in [0,0.99]:

    # values = [0, 0.5, 0.99, 0.3, 0.4, 0.1, 0.6, 0.7, 0.8, 0.9, 0.2]
    # pars = [{'gamma': gamma, 'comment': f'vary_gamma'} for gamma in values]
    #
    # for par_dict in pars:
    #     Field(
    #         100, record_data = True, plot = False, sim_length = 500_000,
    #         learning_alg = 'Q', Q_every = 10_000, record_every = 10_000, gradient_reward = False, **par_dict
    #     )

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

    #    for obs_rad in [10, 50, 100, 150]:
    #        pars.append({'observation_radius': obs_rad, 'leader_frac': lead_frac})
    #pars = [{par: value, 'comment': f'vary_{par}'} for par, value in par_tweaks]
    #pars += [{par: value, 'gradient_reward': False, 'comment': f'vary_{par} (no gradient)'} for par, value in par_tweaks] # reference simulations


    #with ProcessPoolExecutor() as executor:
    #    for _ in executor.map(mp_wrapper, enumerate(pars)):
    #        pass


    # 5. Run a benchmark test and create a figure with the results

    # benchmark()
    # p.plot_all(
    #     quantity = 't', save_as = 'benchmark.png',
    #     title = 'Benchmark test', legend = 8
    # )

    # For tonight:
    # Q_dirs = sorted(glob('data/20200525/2020052[56]-*-Q'))
    # for Q_dir in Q_dirs:
    #     for fname in sorted(glob(f'{Q_dir}/*.npy')):
    #         record_tag = get_rt(Q_dir)
    #         timestep = int(regex.search('\d+', path.split(fname)[1]).group())
    #         m.load_from_Q(fname, record_tag, data_dir = 'data/20200525', comment = f'{record_tag}-{timestep:>06}', sim_length = 1500)

    # def avg_v():
    #     plt.figure()
    #     fnames = sorted(glob('data/*-v.npy'))
    #     avg_v = []
    #     timesteps = []
    #     with open('data/parameters.json') as f:
    #         params = json.load(f)
    #     for fname in sorted(fnames):
    #         record_tag = get_rt(fname)
    #         timestep = int(params[record_tag]['comment'].split('-')[-1])
    #         print(timestep)
    #         data = np.load(fname)
    #         v = [x ** 2 + y ** 2 for x,y in data[500:]]
    #         if len(v) > 0:
    #             avg_v.append(sum(v)/len(v))
    #             timesteps.append(timestep)
    #     plt.scatter(timesteps, avg_v, marker = '.')
    #     plt.savefig(path.join(path.expanduser('~'), 'public_html', 'avg_v.png'), dpi = 300)

    # 6.

    # data_dir = 'data/20200528/1-lf_or'
    # run_Q_dirs(data_dir)
