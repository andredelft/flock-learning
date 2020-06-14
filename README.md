# Flock-learning

(This documentation is a WIP.)

## Getting started

1. Start a regular simulation directly by creating a Field instance. An integer should be passed that specifies the number of birds.
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
  >>> pars = [{'obs_rad': value} for obs_rad in [10, 50, 100, 150]]
  >>> with ProcessPoolExecutor() as executor:
  ...    for _ in executor.map(mp_wrapper, enumerate(pars)):
  ...        pass
  ```
