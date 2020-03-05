from field import Field

if __name__ == '__main__':
    for N in [150,200]:
        Field(
            N, periodic = True, plot = False, record_data = True, observe_direction = True,
            track_birds = True, sim_length = 15000, comment = 'vary:no_birds'
        )
