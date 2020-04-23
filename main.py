from field import Field
from os import path
import json
import time

from concurrent.futures import ProcessPoolExecutor

def load_from_Q(record_tag, data_dir = 'data', plot = True, record_data = True, **kwargs):
    with open(path.join(data_dir, 'parameters.json')) as f:
        params = json.load(f)[record_tag]
    no_birds = params.pop('no_birds')
    instinct_file = path.join(data_dir, f'{record_tag}-instincts.json')
    with open(instinct_file) as f:
        instincts = json.load(f)
    Q_file = path.join(data_dir, f'{record_tag}-Q.npy')
    params.pop('learning_alg')
    if 'Q_params' in params.keys():
        params.update(params.pop('Q_params'))
    # pop some depracated or unused params
    for key in ['no_dirs', 'observe_direction', 'comment']:
        if key in params.keys():
            params.pop(key)
    Field(
        no_birds, plot = plot, record_data = record_data, learning_alg = 'pol_from_Q',
        Q_file = Q_file, comment = record_tag, instincts = instincts, **params, **kwargs
    )

def load_from_Delta(Delta):
    # TODO
    pass

def tweak_learning_params(indexed_pars):
    i, pars = indexed_pars
    time.sleep(5 * i) # To make sure they don't start at exactly the same time, resulting in the same record tag
    Field(
        100, record_data = True, plot = False, sim_length = 80_000, reward_signal = 5,
        learning_alg = 'Q', gradient_reward = True, **pars
    )

if __name__ == '__main__':
    # for record_tag in ['20200409-164937', '20200415-154211', '20200416-190443', '20200418-114214'][1:2]:
    #     date = record_tag.split('-')[0]
    #     load_from_Q(
    #         record_tag, data_dir = f'data/{date}', record_data = True, record_mov = True,
    #         sim_length = 15_000, plot = False
    #     )

    # Field(
    #     100, record_data = True, plot = False, sim_length = 2_000_000, reward_signal = 5,
    #     learning_alg = 'Q', gradient_reward = True, track_time = True
    # )

    # lp_tweaks = [
    #     ('epsilon', 0.1),
    #     ('epsilon', 0.2)
    #     ('gamma', 0.5),
    #     ('alpha', 0.5),
    # ]
    # pars = [{par: value, 'comment': f'vary_{par}'} for par, value in lp_tweaks]

    with ProcessPoolExecutor() as executor:
        for _ in executor.map(tweak_learning_params, enumerate(dict() for _ in range(4))):
            pass
