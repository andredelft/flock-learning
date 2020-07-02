# Flock-learning

(This documentation is a WIP.)

## Getting started

1. Start a regular simulation directly by creating a Field instance. An integer should be passed that specifies the number of birds.
    ```pycon
    >>> from field import Field
    >>> Field(100)
    ```
    Two other ways in which a `Field` instance can run is by recording a movie:
    ```pycon
    >>> Field(100, record_mov = True, sim_length = 5000)
    ```
    or just recording the data:
    ```pycon
    >>> Field(100, record_data = True, plot = False, sim_length = 5000)
    ```
    The quantities that are recorded can be further specified using the `record_quantities` keyword.

2. Start a simulation with fixed Q-tables from a given file (the output of a Q-learning training phase with `record_data = True`)

    ```pycon
    >>> from main import load_from_Q
    >>> load_from_Q(fpath = 'data/20200531/2-VI/20200531-100329-Q.npy')
    ```
3. Start a simulation with fixed Q-values from a specific value of Delta

    ```pycon
    >>> from main import load_from_Delta
    >>> load_from_Delta(0.2)
    ```

4. Start multiple simulations at the same time using multiprocessing, with the option of adjusting parameters in different runs:

    ```pycon
    >>> pars = [
    ...     {'observation_radius': value}
    ...     for value in [10, 50, 100, 150]
    ... ]
    >>> run_parallel(pars, sim_length = 10_000, comment = 'vary_obs_rad')
    ```
