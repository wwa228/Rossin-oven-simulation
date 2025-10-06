import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from typing import Tuple
import argparse
from datetime import datetime
import inspect
from pprint import pformat
import pickle

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import flatten_util, tree_util
import jax.random as jrandom
from cyipopt import minimize_ipopt

from .oven_dynamics import oven_dynamics 
from .phy_props import _mass_transfer_coefficient, _heat_transfer_coefficient, moisture_content
from .scaling import  choose_scaling
from .data_structures import Controls, Constants, OdeKwargs
from .data.targets import get_targets
from .utils import save_params, check_dir


parser = argparse.ArgumentParser("EnergyOptimization")
parser.add_argument("--products", type = str, default = "R13SSR1", help = "Multiple products that are used for training are separated by comma") # should always start with 
parser.add_argument("--iterations", type = int, default = 100, help = "The maximum number of iterations to be performed in parameter estimation")
parser.add_argument("--atol", type = float, default = 1e-8, help = "Absolute tolerance of ode solver")
parser.add_argument("--rtol", type = float, default = 1e-6, help = "Relative tolerance of ode solver")
parser.add_argument("--tol", type = float, default = 1e-5, help = "The tolerance of the optimization problem")
parser.add_argument("--mxstep", type = int, default = 10_000, help = "The maximum number of steps a ode solver takes")
parser.add_argument("--msg", type = str, default = "", help = "Sample msg that briefly describes this problem")
parser.add_argument("--load_dir", type = str, default = ".", help = "Directory to load the optimal parameters from")
parser.add_argument("--recirculation", type = int, choices = [0, 1], default = 0, help = "Whether air should be recirculation (1) or not (0)")
parser.add_argument("--samples", type = int, default = 1, help = "The number of samples used to approximate the expectation")

parser.add_argument("--id", type = str, default = "", help = "Slurm job id")
parser.add_argument("--partition", type = str, default = "", help = "The partition this job is assigned to")
parser.add_argument("--cpus", type = str, default = "1", help = "Maximum number of cpus availabe per node")

#############################################################################
## Initialization steps - initialize directories, loggers, and scaling

pargs = parser.parse_args()

# Training products
load_dir = None if pargs.load_dir == "." else pargs.load_dir # Load parameters from file
product_expt = pargs.products # "R13NSR1"
prod = product_expt[:3] # "R13"

_dir = str(datetime.now())
divider = "--"*50 # printing separater
check_dir(os.path.join("log", product_expt, _dir))
logfile = logging.FileHandler(os.path.join("log", product_expt, _dir, "solver_stats.txt"))
logger.addHandler(logfile)

ndevices = jax.local_device_count()
logger.info(f"Local device count {ndevices} \nlocal devices {jax.local_devices()}")
logger.info(f"{divider} \nEnergy Optimization of {product_expt} with warm start from {load_dir}")
logger.info(f"{divider} \nArguments")
logger.info(f"{pformat(pargs.__dict__)} \n{divider}")
ProductConstants, Target = get_targets([product_expt], [prod], logger) # get constants and log them

assert prod in ProductConstants, "All product specific information should be specified"
assert product_expt in Target, "All experiment specific information should be specified"

(moisture_max, moisture_min, temperature_max, temperature_min), (scale_states, unscale_states, _, _) = choose_scaling("max")


#############################################################################
## Create and log custom functions

scaled_moisture_content = lambda T, density_air, relative_humidity = 1 : scale_states(moisture_content(unscale_states(T, temperature_max, temperature_min), density_air, relative_humidity), moisture_max, moisture_min) # 
scaled_mass_transfer_coefficient = lambda x, p, constant: _mass_transfer_coefficient(x, p, constant)
scaled_heat_transfer_coefficient = lambda x, p, constant : _heat_transfer_coefficient(x, p, constant)
scaled_reaction_rate = lambda A, k0, T : k0 * jnp.exp( - A / 1.98 / unscale_states(T, temperature_max, temperature_min))

logger.info(f"{divider} \n{inspect.getsource(_mass_transfer_coefficient)}")
logger.info(f"{divider} \n{inspect.getsource(_heat_transfer_coefficient)}")
logger.info(f"{divider} \n{inspect.getsource(moisture_content)}")
logger.info(f"{divider} \n{inspect.getsource(scaled_reaction_rate)}")


#############################################################################
## Initialize parameters (constant for all products), constants (product specific), controls (experiment specific)

# TODO use the directory from pargs
with open("log/R13SSR4/2024-12-27 20:47:09.835171/saved_params/params", "rb") as file :
    meta_data = pickle.load(file)
    _metaparameters_deterministic = meta_data["meta_parameters"]
    solid_moisture_mean = meta_data["solid_moisture_mean"][0]
    solid_moisture_sigma = meta_data["solid_moisture_sigma"]


key = jrandom.PRNGKey(10)

parameters = _metaparameters_deterministic["parameters"]

constants = Constants(
    radius_particle = ProductConstants[prod]["radius_particle"],
    voidage = 1 - ProductConstants[prod]["density_product"] / ProductConstants[prod]["density_particle"], # "Bed porosity (1 - density of product / density of particle)"
    enthalpy_vaporization_water = ProductConstants[prod]["enthalpy_vaporization_water"],
    specific_heat_capacity_air = ProductConstants[prod]["specific_heat_capacity_air"],
    density_particle = ProductConstants[prod]["density_particle"],
    specific_heat_capacity_solid = ProductConstants[prod]["specific_heat_capacity_solid"],
    product_height = ProductConstants[prod]["product_height"],
    density_product = ProductConstants[prod]["density_product"],

    ny = ProductConstants[prod]["ny"],
    nzones = ProductConstants[prod]["nzones"],
    reverse_zone = ProductConstants[prod]["reverse_zone"],
    ntimes = ProductConstants[prod]["ntimes"], # number of measurements for given residence time
    equilibrium_moisture = scale_states(ProductConstants[prod]["equilibrium_moisture"], moisture_max, moisture_min), # equilibrium moisture content in solid

    moisture_max = moisture_max,
    moisture_min = moisture_min,
    temperature_max = temperature_max,
    temperature_min = temperature_min,

    # Equations
    scaled_moisture_content = scaled_moisture_content,
    scaled_mtc = scaled_mass_transfer_coefficient,
    scaled_htc = scaled_heat_transfer_coefficient,
    scaled_k = scaled_reaction_rate,
    density_air = lambda X, T : ProductConstants[prod]["density_air"],

    # ode kwargs
    ode_kwargs = OdeKwargs(rtol = pargs.rtol, atol = pargs.atol, mxstep = pargs.mxstep, reuse = False))

controls = Controls(inputs = 
        {
    "init_temperature_air" : scale_states(Target[product_expt].controls["init_temperature_air"], temperature_max, temperature_min), # setpoint - line loss 
    "init_velocity_air" : Target[product_expt].controls["init_velocity_air"],
    "init_moisture_air" : scale_states(Target[product_expt].controls["init_moisture_air"], moisture_max, moisture_min),
    "residence_time" : Target[product_expt].controls["residence_time"],
    "recirculation_ratio" : Target[product_expt].controls["recirculation_ratio"], # must be between 0 to 1
    })

controls_variables = {
    "init_temperature_air" : scale_states(Target[product_expt].controls["init_temperature_air"], temperature_max, temperature_min),
    "init_velocity_air" : Target[product_expt].controls["init_velocity_air"],
    "residence_time" : Target[product_expt].controls["residence_time"]
}

if pargs.recirculation : 
    controls_variables["recirculation_ratio"] = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]) # Target[product_expt].controls["recirculation_ratio"]

controls_flatten, unravel_controls = flatten_util.ravel_pytree(controls_variables)

#############################################################################
## Define the objective function

def objective(controls_flatten : jnp.ndarray, parameters : dict, solid_init : jnp.ndarray, constants : Constants, recirculation : bool) -> Tuple[jnp.ndarray]:

    ny = constants.ny
    xinit = jnp.column_stack((
        solid_init, # moisture and temperature
        jnp.zeros(shape = (ny, )) # cure 
    ))

    _controls_variable = unravel_controls(controls_flatten)
    
    _controls = Controls(inputs = 
        {
            "init_temperature_air" : _controls_variable["init_temperature_air"], # setpoint - line loss 
            "init_velocity_air" : _controls_variable["init_velocity_air"],
            "init_moisture_air" : controls["init_moisture_air"],
            "residence_time" : _controls_variable["residence_time"], # controls["residence_time"], # 
            "recirculation_ratio" : _controls_variable["recirculation_ratio"] if recirculation else controls["recirculation_ratio"], # must be between 0 to 1
        })

    parameters["reaction_rate"] = 2.33 * 10**7 # Do not optimize this yet
    parameters["activation_energy"] = 20. * 10**3 # Do not optimize this yet

    # oven_dynamics discards the provided initial condition (It is assumed that the initial condition are the values of states at t = -1)
    solution = (moisture_solid, temperature_solid, 
                cure, moisture_air, temperature_air, t_event) = oven_dynamics(xinit, parameters, _controls, constants, reverse_zone = constants.reverse_zone, nzones = constants.nzones, recirculation = recirculation)

    # Last cure values have to be in normally distributed with (mean, sigma)
    mean_cure = 0.6 # output mean and standard deviation
    mse_cost = jnp.mean((cure[-1][-1] - mean_cure)**2)

    # Calculating the energy consumption
    def CalTout(temp_air, zone): return jax.lax.cond(zone < constants.reverse_zone, lambda : jnp.mean(temp_air[:, -1]), lambda : jnp.mean(temp_air[:, 0]))

    Ta = scale_states(300, temperature_max, temperature_min) # scaled ambient temperature (300 K)
    Tout = jax.vmap(CalTout)(temperature_air, jnp.arange(1, constants.nzones + 1)) # mean outlet temperature

    if recirculation :
        energy = jnp.sum(
            (_controls["init_temperature_air"] - (_controls["recirculation_ratio"] * Tout + (1 - _controls["recirculation_ratio"]) * Ta)) * _controls["init_velocity_air"]
        )
    else : 
        energy = jnp.sum(
            (_controls["init_temperature_air"] - Ta) * _controls["init_velocity_air"]
        )

    energy *= _controls["residence_time"][0]
    total_cost = 10 * mse_cost + 0.001 * energy

    return total_cost, mse_cost, energy, xinit, *tree_util.tree_map(jnp.vstack, solution)

# get pargs.smaples of worst batches 
ny = constants.ny
xinit = jnp.array([solid_moisture_mean[0], 0.5])
keys = jrandom.split(key, 100)
solid_inits = jax.vmap(
    lambda _key : jnp.column_stack((
        jnp.tile(xinit[0], (ny, )) + solid_moisture_sigma * jrandom.normal(_key, shape = (ny, )),
        jnp.tile(xinit[1:], (ny, 1))
    ))
)(keys)

# vmap over objective. No need for recycle
sol = jax.vmap(objective, in_axes = (None, None, 0, None, None))(controls_flatten, parameters, solid_inits, constants, False)

# sort the objective function 
chosen_keys = keys[jnp.argsort(sol[0])][-pargs.samples:]

def objective_pmap(controls_flatten : jnp.ndarray, parameters : dict, solid_init : jnp.ndarray, sigma : jnp.ndarray, constants : Constants, 
        recirculation : bool, keys : jnp.ndarray) -> Tuple[jnp.ndarray] :
    # keeps the key constant for nonlinear optimization

    ny = constants.ny
    solid_inits = jax.vmap(
        lambda _key : jnp.column_stack((
            jnp.tile(solid_init[0], (ny, )) + sigma * jrandom.normal(_key, shape = (ny, )),
            jnp.tile(solid_init[1:], (ny, 1))
        ))
    )(keys)

    _cost, *solution = jax.pmap(
        objective, 
        in_axes = (None, None, 0, None, None), 
        static_broadcasted_argnums = (3, 4)
    )(controls_flatten, parameters, solid_inits, constants, recirculation)
    
    return jnp.mean(_cost), *solution


# TODO consider expectation over the objective function with respect to the initial distribution of moisture
logger.info(f"{divider} \n{inspect.getsource(objective)}")
_objective_partial = lambda p : objective_pmap(p, parameters, jnp.array([solid_moisture_mean[0], 0.5]), solid_moisture_sigma, constants, pargs.recirculation, chosen_keys)[0]
_gradient_partial = jax.grad(_objective_partial)
_hessian_partial = jax.hessian(_objective_partial) 

controls_variables_lb = {
    "init_temperature_air" : scale_states(450 * jnp.ones(5), temperature_max, temperature_min),
    "init_velocity_air" : jnp.zeros(5),
    "residence_time" : 5 * jnp.ones(1)
}

controls_variables_ub = {
    "init_temperature_air" : scale_states(600 * jnp.ones(5), temperature_max, temperature_min),
    "init_velocity_air" : 0.3 * jnp.ones(5),
    "residence_time" : jnp.array([60.])
}

if pargs.recirculation : 
    controls_variables_lb["recirculation_ratio"] = 0.1 * jnp.ones_like(controls_variables["recirculation_ratio"])
    controls_variables_ub["recirculation_ratio"] = 0.9 * jnp.ones_like(controls_variables["recirculation_ratio"])


_file = os.path.join("log", product_expt, _dir, "ipopt_output.txt")
nlp_options = {
    "tol" : pargs.tol, 
    "maxiter" : pargs.iterations, 
    "disp" : 5,
    "output_file" : _file,  
    "file_print_level" : 5,
    "print_timing_statistics" : "yes",
    "mu_strategy" : "adaptive"
    }


def _objective_partial_catch(p):
    try : 
        sol = _objective_partial(p)
    except : 
        sol = jnp.inf
    
    return sol


res = minimize_ipopt(
    _objective_partial_catch,
    jac = _gradient_partial,
    hess = _hessian_partial, # if None uses ipopts BFGS approximation
    x0 = controls_flatten,
    bounds = list(zip(flatten_util.ravel_pytree(controls_variables_lb)[0], flatten_util.ravel_pytree(controls_variables_ub)[0])),
    options = nlp_options
    )

 
# print in output file then copy the contents into the logger file
with open(_file, "r") as file:
    for line in file : logger.info(line.strip())

logger.info(f"residual {res}")

# Cure with mean moisture content
# _, *solution = objective(res.x, parameters, jnp.tile(jnp.array([solid_moisture_mean[0], 0.5]), (20, 1)), constants, pargs.recirculation)
# logger.info(f"Optimal Cure {jnp.mean(solution[3][-1])}")