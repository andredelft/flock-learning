# Flock-learning

This repository contains the model that has been written for the research project for my MSc in theoretical physics. The model aims to describe collective motion using Q-learning with orientation-based rewards.

In this documentation, I will explain the technical details of the model. The conceptual framework is written down in my thesis, which can be requested by [contacting me](mailto:andrevandelft@outlook.com).

## Getting started

First setup your (virtual) environment:

```shell
$ pip install -r requirements.txt
```

Simulations can then be performed in a couple of different ways:

1. Start a regular simulation directly by creating an instance of the `Field` class. An integer should be passed that specifies the number of birds.
    ```python
    >>> from field import Field
    >>> Field(100)
    ```
    Two other ways in which a `Field` instance can run is by recording a movie:
    ```python
    >>> Field(100, record_mov = True, sim_length = 5000)
    ```
    or just recording the data:
    ```python
    >>> Field(100, record_data = True, plot = False, sim_length = 5000)
    ```
    The quantities that are recorded can be further specified using the `record_quantities` keyword.

2. Start a simulation with fixed Q-tables from a given file (the output of a Q-learning training phase with `record_data = True`)

    ```python
    >>> from main import load_from_Q
    >>> load_from_Q(fpath = 'data/20200531/2-VI/20200531-100329-Q.npy')
    ```
3. Start a simulation with fixed Q-values from a specific value of Delta

    ```python
    >>> from main import load_from_Delta
    >>> load_from_Delta(0.2)
    ```

4. Start multiple simulations at the same time using multiprocessing, with the option of adjusting parameters in different runs:

    ```python
    >>> pars = [
    ...     {'observation_radius': value}
    ...     for value in [10, 50, 100, 150]
    ... ]
    >>> run_parallel(pars, sim_length = 10_000, comment = 'vary_obs_rad')
    ```
    NB: There is a complication regarding multiprocessing and the python random module, which sometimes results in very similar initializations. This problem has not been solved yet. Cf. [this blog post](https://www.sicara.ai/blog/2019-01-28-how-computer-generate-random-numbers).

5. If the Q-tables of a given simulation are saved regularly (using the option `Q_every`), these are saved in some directory, different (short) simulations can be performed for every saved Q-table. The graphs on the right hand side of [this](thesis-figures/learning_params.pdf) and [this](thesis-figures/lead_frac_obs_rad_discrete.pdf) figure have been generated using this option.

    ```python
    >>> data_dir = 'path/to/some/dir'
    >>> run_Q_dirs(data_dir)
    ```

6. Run a benchmark test and create a figure with the results

    ```python
    >>> from main import benchmark
    >>> benchmark()
    >>> p.plot_all(
    ...     quantity = 't', save_as = 'benchmark.png',
    ...     title = 'Benchmark test', legend = 8
    ... )
    ```

## All options for the `Field` and `Bird` classes

### field.Field(*numbirds, sim_length = 12500, record_mov = False, record_data = False, record_time = False, record_quantities = [], field_dims = FIELD_DIMS, periodic = True, plotscale = PLOTSCALE, plot = True, comment = '', Q_every = 0, repos_every = 0, record_every = 500, \*\*kwargs*)

* `numbirds`: Number of birds in the field (only required parameter).
* `sim_length`: Number of timesteps the simulation will be running (only works when `plot = False`)
* `record_mov`: Boolean specifiying wether to record a movie or not.
* `record_data`: Boolean specifiying wether to record data of the simulation. Default quantities will be chosen, depending on `learning_alg`. These choices can be overwritten using `record_quantities`.
* `record_quantities`: List of quantities to be recorded during a simulation. Overwrites the default options. Possible choices are:
    * `'v'`: The normalized average flight direction.
    * `'Q'`: The final Q-tables of the birds.
    * `'Delta'`: The normalized distance from the optimal policy. The optimal policy is defined as the policy in which the followers always choose to follow their neighbours (`'V'`), and the leaders always choose to follow their instinct (`'I'`). `Delta` is a measure for how far a given set of Q-tables is from this optimal policy.
    * `'policies'`: The final policies of the birds.
    * `'instincts'`: The instincts of the birds in the field. For leaders this equals `'E'` (the eastward direction), for followers any of `['N', 'S', 'W']`.
    * `'t'`: The calculation time of each timestep (for tracking the performance of the script).
* `field_dims`: The dimensions of the field.
* `periodic`: Turning the periodic boundaries on or off.
* `plotscale`: A number controlling the size of the plot.
* `plot`: Boolean specifying wether to plot the data in real time or not.
* `comment`: Some additional comment that will be included in `parameters.json`.
* `Q_every`: If non-zero, the Q-tables of the birds will be regularly saved in a separate directory in intervals specified by this parameter. Note that this requires some storage (for 100 birds, each set of Q-tables takes up 10 MB).
* `repos_every`: If non-zero, the birds will be randomly repositioned after each number of timesteps specified by this parameter.
* `record_every`: Specifies the interval for recording the data.

All other options will be passed to the `birds.Birds` instance.

### birds.Birds(*numbirds, field_dims, action_space = A, observation_space = O, leader_frac = 0.25, reward_signal = R, learning_alg = 'Q', alpha = alpha, gamma = gamma, epsilon = epsilon, Q_file = '', Q_tables = None, gradient_reward = True, observation_radius = d, instincts = [], eps_decr = 0*)

`numbirds` and `field_dims` will be inherited from the `Field` class. All other options are:

* `action_space`: The action space of the birds. Possible options are:
    * `'V'`: When choosing this option, the bird fill fly into the average flight direction of its neighbours (a Vicsek-type interaction).
    * `'I'`: When choosing this option, the bird will follow it's own instinct (one of the four cardinal directions).
    * `['N', 'E', 'S', 'W']`: When choosing one of these actions, the bird will fly into the corresponding cardinal direction. These actions represent a form of free motion of the birds.
    * `'R'`: When choosing this action, the bird will fly into a direction at random.
    For the thesis, `action_space = ['V', 'I']` is used.
* `observation_space`: The observation space of the birds. An observation is by default a tuple of numbers enumerating the number of birds flying into each possible flight direction. Should be adjusted with caution, since `Birds.perform_observations` implicitly depends on this default choice.
* `leader_frac`: The leader fraction of the birds (percentage of birds for which the instinctive drection is `'E'`).
* `reward_signal`: The maximal reward signal.
* `gradient_reward`: If `True`, rewards are given as `R \cos{\theta}`, where `R` is the value of `reward_signal`. If `False`, a discrete reward system is used, meaning that the reward is `R` only when `\theta = 0`, and `0` otherwise.
* `learning_alg`: The learning algorithm used. Possible values are:
    * `'Q'`: The Q-learning algorithm is used (details in `q_learning.py`).
    * `'Ried'`: A short term learning algorithm is used, inspired by [Ried e.a.](https://dx.doi.org/10.1371/journal.pone.0212044)
    * `'pol_from_Q'`: This starts a simulation with fixed Q-tables, which can be passed from a file with `Q_file`, or directly via `Q_tables`.
* `alpha`: The learning rate (a Q-learning parameter)
* `gamma`: The discount rate (a Q-learning parameter)
* `epsilon`: The exploration parameter (for the epsilon-greedy policy used in combination with Q-learning)
* `Q_file`: Filename of a `.npy` file with the Q-tables of the birds, used for `learning_alg = 'pol_from_Q'`. The dimension of the numpy array should equal `(numbirds, len(observation_space), len(action_space))`.
* `Q_tables`: Numpy array with the Q-tables of the birds, used for `learning_alg = 'pol_from_Q'`. The dimension of the numpy array should equal `(numbirds, len(observation_space), len(action_space))`.
* `observation_radius`: radius used for the Vicsek action (any bird within this distance will be considered a *neighbour*).
* `instincts`: Possibility of manually fixing the instincts of the birds. If `None`, a percentage of `leader_frac` will have instinct `'E'`, and the others are initalized randomly.
* `eps_decr`: If non-zero, the value of the learning parameter `epsilon` will decrease gradually over the number of timesteps specified by this option. This allows for a learning and training phase within the same run.
