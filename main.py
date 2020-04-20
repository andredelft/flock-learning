from field import Field
from os import path
import json

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
    if 'no_dirs' in params:
        params.pop('no_dirs')
    Field(
        no_birds, plot = plot, record_data = record_data, learning_alg = 'pol_from_Q',
        Q_file = Q_file, comment = record_tag, instincts = instincts, **params, **kwargs
    )

if __name__ == '__main__':
    load_from_Q(
        '20200418-114214', data_dir = 'data/20200418', record_data = False, record_mov = False,
        sim_length = 10_000, plot = True
    )
    # Field(
    #     100, record_data = True, plot = False, sim_length = 2_000_000, reward_signal = 5,
    #     learning_alg = 'Q', gradient_reward = True, track_time = True
    # )
