from collections import defaultdict

import jax.random as jrandom

def choose_scaling(scaling : str):

    if scaling == "min_max":
        # scaling parameters
        moisture_max, moisture_min =  50, 0. # consider equill moisture content
        temperature_max, temperature_min = 1000., 200.
        scaled_max, scaled_min = 1, 0

        # min max scaling
        def scale_states(x, amax, amin): return (x - amin)/(amax - amin)
        def unscale_states(x, amax, amin): return amin + x*(amax - amin)
    
    elif scaling == "max":
        # scaling parameters
        moisture_max, moisture_min =  1, 0. # consider equill moisture content
        temperature_max, temperature_min = 1000., 200.
        scaled_max, scaled_min = 1, 0

        # min max scaling
        def scale_states(x, amax, amin): return x/(amax - amin)
        def unscale_states(x, amax, amin): return x*(amax - amin)

    else :
        # scaling parameters doesnot matter here
        moisture_max, moisture_min =  1., 0.
        temperature_max, temperature_min = 1., 0.
        scaled_max, scaled_min = 1e18, -1e18

        # no scaling
        def scale_states(x, amax, amin): return x
        def unscale_states(x, amax, amin): return x

    return (moisture_max, moisture_min, temperature_max, temperature_min), (scale_states, unscale_states, scaled_max, scaled_min)


def parameters_init(key, dimensions : list, scale = 0.01):

    keys = jrandom.split(key, len(dimensions) - 1)
    parameters = defaultdict(list)
    for n, m, k in zip(dimensions[1:], dimensions[:-1], keys):
        key_weights, key_bias = jrandom.split(k, 2)
        parameters["weight"].append(scale*jrandom.normal(key_weights, shape = (n, m)))
        parameters["bias"].append(scale*jrandom.normal(key_bias, shape = (n, )))

    return parameters  