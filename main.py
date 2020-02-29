from field import Field

if __name__ == '__main__':
    Field(
        100, periodic = True, plot = True, record_data = True, observe_direction = True,
        leader_frac = 0.25, track_birds = True, sim_length = 1000
    )
