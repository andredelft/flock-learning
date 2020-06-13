# Flock-learning

(This documentation is a WIP.)

## Different ways of running the model

1. Start a regular simulation by creating a Field instance. Things like
    recording data and parameter specifications are all handled as keyword
    arguments.

    ```python
    Field(
        100, sim_length = 100_000, plot = False, learning_alg = 'Q',
        record_quantities = ['Delta', 'Q', 'instincts'], record_every = 500,
        Q_every = 10_000
)
   ```

2. Start a simulation with fixed Q-tables from a given file (the output of
    a training phase)

    ```python
   load_from_Q(
        fname = 'data/20200501-164233-Q/1130000.npy', data_dir = 'data',
        record_tag = '20200501-164233', record_data = False,
        record_mov = False, sim_length = 15_000, plot = True
   )
   ```
3. Start a simulation with fixed Q-values from a specific value of Delta

   ```python
   load_from_Delta(0.2)
   ```

4. Start multiple simulations at the same time using multiprocessing, with
   the option of adjusting parameters in different runs (as exemplified
   below with some learning pars)

   ```python
  pars = [{'obs_rad': value} for obs_rad in [10, 50, 100, 150]]

  with ProcessPoolExecutor() as executor:
      for _ in executor.map(mp_wrapper, enumerate(pars)):
          pass
  ```
