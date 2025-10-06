import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from typing import Optional, List, Any, Tuple
from functools import partial
from collections import defaultdict
import argparse
from datetime import datetime
import inspect
from pprint import pformat
import pickle
import operator

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import flatten_util, tree_util
import jax.random as jrandom
from jax.example_libraries.optimizers import piecewise_constant
import numpy as np

from .oven_dynamics import oven_dynamics 
from .phy_props import _mass_transfer_coefficient, _heat_transfer_coefficient, moisture_content
from .scaling import  choose_scaling, parameters_init
from .data_structures import Controls, Parameters, Moisture, Temperature, Cost, EventTime, SimulationData, Constants, OdeKwargs, Quality, TrainingResults, DistributionData
from .optimizer import select_optimizer
from .data.targets import get_targets
from .plots import plot, plot_recirculation
from .utils import save_params, check_dir, get_coefficients, product_quality, kl_div


parser = argparse.ArgumentParser("ParameterEstimation")
parser.add_argument("--products_train", type = str, default = "R13SSR1", help = "Multiple products that are used for training are separated by comma") # should always start with 
parser.add_argument("--products_test", type = str, default = "", help = "Multiple products that are used for testing are separated by comma")
parser.add_argument("--iterations", type = int, default = 100, help = "The maximum number of iterations to be performed in parameter estimation")
parser.add_argument("--atol", type = float, default = 1e-8, help = "Absolute tolerance of ode solver")
parser.add_argument("--rtol", type = float, default = 1e-6, help = "Relative tolerance of ode solver")
parser.add_argument("--mxstep", type = int, default = 10_000, help = "The maximum number of steps a ode solver takes")
parser.add_argument("--msg", type = str, default = "", help = "Sample msg that briefly describes this problem")
parser.add_argument("--load_dir", type = str, default = ".", help = "Directory to load the optimal parameters from")
parser.add_argument("--optimize", type = int, default = 1, choices = [0, 1], help = "Perfrom parameter esitmation or not")
parser.add_argument("--recirculation", type = int, choices = [0, 1], default = 0, help = "Whether air should be recirculation (1) or not (0)")
parser.add_argument("--lr", type = float, default = 0.01, help = "The learning rate of adam optimizer")
parser.add_argument("--gen", type = int, default = 200, help = "Check for generality every epochs")

parser.add_argument("--id", type = str, default = "", help = "Slurm job id")
parser.add_argument("--partition", type = str, default = "", help = "The partition this job is assigned to")
parser.add_argument("--cpus", type = int, default = 1, help = "Maximum number of cpus availabe per node")

#############################################################################
## Initialization steps - initialize directories, loggers, and scaling

pargs = parser.parse_args()

# Training products
load_dir = None if pargs.load_dir == "." else pargs.load_dir # Load parameters from file
product_expts : List[str] = pargs.products_train.split(", ") # ["R13NSR1", "R13NSR2", "R19SSR1"]
product_dict = defaultdict(list) # make a dictionary of product specific experiments {"R13" : ["R13SSR1", "R13SSR2", "R13NSR1"], "R19" : ["R19SSR1"]}
for prod in product_expts : product_dict[prod[:3]].append(prod) 
product_list = sorted(set(prod[:3] for prod in product_expts)) # ["R13", "R19"]

# Testing products
if pargs.products_test == "" : # use training data if testing data is not provided
    product_expts_test = product_expts
    product_dict_test = product_dict
    product_list_test = product_list
else : 
    product_expts_test : List[str] = pargs.products_test.split(", ")
    product_dict_test = defaultdict(list) # make a dictionary of product specific experiments {"R13" : ["R13SSR1", "R13SSR2", "R13NSR1"], "R19" : ["R19SSR1"]}
    for prod in product_expts_test : product_dict_test[prod[:3]].append(prod) 
    product_list_test = sorted(set(prod[:3] for prod in product_expts_test)) # ["R13", "R19"]

_dir = str(datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')) # added.strftime('%Y-%m-%d %H-%M-%S.%f') because windows doesn't allow colons in file name
divider = "--"*50 # printing separater
for _product in set(product_expts).union(set(product_expts_test)) :
    check_dir(os.path.join("log", _product, _dir))
    logfile = logging.FileHandler(os.path.join("log", _product, _dir, "solver_stats.txt"))
    logger.addHandler(logfile)

ndevices = jax.local_device_count()
logger.info(f"Local device count {ndevices} \nlocal devices {jax.local_devices()}")
logger.info(f"{divider} \nOptimization of {product_expts} with warm start from {load_dir}")
logger.info(f"{divider} \nArguments")
logger.info(f"{pformat(pargs.__dict__)} \n{divider}")
ProductConstants, Target = get_targets(set([*product_expts, *product_expts_test]), set([*product_list, *product_list_test]), logger) # get constants and log them

assert all(prod in ProductConstants for prod in product_dict) and all(prod in ProductConstants for prod in product_dict_test), "All product specific information should be specified"
assert all(prod in Target for prod in product_expts) and all(prod in Target for prod in product_expts_test), "All experiment specific information should be specified"
assert all(prod in product_list for prod in product_list_test), "All products for testing should be in training list"

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

key = jrandom.PRNGKey(10)
key_mtc, key_htc, key_mean, key_sigma, key = jrandom.split(key, 5)
# All ProductOvens (R13SS, R19NS) have the same parameters
parameters = Parameters(params =
    {
        "mass_transfer_coefficient_falling" : parameters_init(key_mtc, [5, 10, 10, 1]),
        "mass_transfer_coefficient_constant" : jnp.array([160.]),
        "heat_transfer_coefficient_falling" : parameters_init(key_htc, [5, 10, 10, 1]),
        "heat_transfer_coefficient_constant" : jnp.array([.5]),
        "critical_moisture" : jnp.array([0.07403168]),
        "constant" : jnp.array([2, 5, 5, 5, 5.]), 
        "constant_jump" : jnp.array([5.]),
    })

# Each ProductOven (R13SS, R19NS) has the same values of constants
constants = [Constants(
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
    ode_kwargs = OdeKwargs(rtol = pargs.rtol, atol = pargs.atol, mxstep = pargs.mxstep, reuse = True)) for prod in product_list]

constants_test = [Constants(
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
    ode_kwargs = OdeKwargs(rtol = pargs.rtol, atol = pargs.atol, mxstep = pargs.mxstep, reuse = True)) for prod in product_list_test]

# Each experiment (R13SSR1, R13SSR2) has different control inputs
controls = tuple(Controls(inputs = 
        {
    "init_temperature_air" : jnp.vstack([scale_states(Target[_product].controls["init_temperature_air"], temperature_max, temperature_min) for _product in product_dict[prod]]), # setpoint - line loss 
    "init_velocity_air" : jnp.vstack([Target[_product].controls["init_velocity_air"] for _product in product_dict[prod]]),
    "init_moisture_air" : jnp.vstack([scale_states(Target[_product].controls["init_moisture_air"], moisture_max, moisture_min) for _product in product_dict[prod]]),
    "residence_time" : jnp.vstack([Target[_product].controls["residence_time"] for _product in product_dict[prod]]),
    "recirculation_ratio" : jnp.vstack([Target[_product].controls["recirculation_ratio"] for _product in product_dict[prod]]) , # must be between 0 to 1
    }) for prod in product_list)

controls_test = tuple(Controls(inputs = 
        {
    "init_temperature_air" : jnp.vstack([scale_states(Target[_product].controls["init_temperature_air"], temperature_max, temperature_min) for _product in product_dict_test[prod]]), # setpoint - line loss 
    "init_velocity_air" : jnp.vstack([Target[_product].controls["init_velocity_air"] for _product in product_dict_test[prod]]),
    "init_moisture_air" : jnp.vstack([scale_states(Target[_product].controls["init_moisture_air"], moisture_max, moisture_min) for _product in product_dict_test[prod]]),
    "residence_time" : jnp.vstack([Target[_product].controls["residence_time"] for _product in product_dict_test[prod]]),
    "recirculation_ratio" : jnp.vstack([Target[_product].controls["recirculation_ratio"] for _product in product_dict_test[prod]]) , # must be between 0 to 1
    }) for prod in product_list_test)


# print training and testing information
logger.info(f"{divider} \nTraining Experiments target heights")
for prod in product_list : 
    for _product in product_dict[prod] :
        logger.info(f"{_product} : {Target[_product].targets.keys()}")

logger.info("Testing experiments target heights")
for prod in product_list_test : 
    for _product in product_dict_test[prod] :
        logger.info(f"{_product} : {Target[_product].targets.keys()}")


#############################################################################
## Define the objective function

meta_parameters = {"parameters" : parameters}

# Collect training data
target_scaled, target_mask, solid_temperature_init = [], [], []
for _constants, prod in zip(constants, product_list):

    target_scaled.append(
        jnp.stack([scale_states(Target[_product_expt].targets_mask["target_map"][:_constants.ntimes*_constants.nzones - _constants.nzones + 1], temperature_max,  temperature_min) 
        for _product_expt in product_dict[prod]])
    )

    target_mask.append(jnp.vstack([Target[_product_expt].targets_mask["mask"][jnp.newaxis, :] for _product_expt in product_dict[prod]]))
    solid_temperature_init.append(jnp.vstack([jnp.sum(_tar[0])/jnp.count_nonzero(_tar[0]) for _tar in target_scaled[-1]]))

solid_moisture_mean = [scale_states(jnp.array([0.1]), moisture_max, moisture_min) for _ in product_list]
solid_moisture_sigma = jnp.array([0.003])

# Collect testing data
target_scaled_test, target_mask_test, solid_temperature_init_test = [], [], []
for _constants, prod in zip(constants_test, product_list_test):

    target_scaled_test.append(
        jnp.stack([scale_states(Target[_product_expt].targets_mask["target_map"][:_constants.ntimes*_constants.nzones - _constants.nzones + 1], temperature_max,  temperature_min) 
        for _product_expt in product_dict_test[prod]])
    )

    target_mask_test.append(jnp.vstack([Target[_product_expt].targets_mask["mask"][jnp.newaxis, :] for _product_expt in product_dict_test[prod]]))
    solid_temperature_init_test.append(jnp.vstack([jnp.sum(_tar[0])/jnp.count_nonzero(_tar[0]) for _tar in target_scaled_test[-1]]))

solid_moisture_mean_test = [scale_states(jnp.array([0.1]), moisture_max, moisture_min) for _ in product_list_test]


def _objective(meta_params : dict, solid_moisture_init : jnp.ndarray, solid_moisture_sigma : jnp.ndarray, constants : Constants, 
                target : jnp.ndarray, target_mask : jnp.ndarray, controls : Controls, solid_temperature_init : jnp.ndarray, 
                recirculation : bool, key) -> Tuple[jnp.ndarray]:
    
    ny = constants.ny
    xinit = jnp.column_stack((
        solid_moisture_init * jnp.ones(ny) + solid_moisture_sigma * jrandom.normal(key, shape = (ny, )), # moisture 
        solid_temperature_init * jnp.ones(ny), # temperature
        jnp.zeros(shape = (ny, )) # cure 
    ))

    _parameters = meta_params["parameters"]
    _parameters["reaction_rate"] = 2.33 * 10**7 # Do not optimize this yet
    _parameters["activation_energy"] = 20. * 10**3 # Do not optimize this yet

    # oven_dynamics discards the provided initial condition (It is assumed that the initial condition are the values of states at t = -1)
    solution = oven_dynamics(xinit, _parameters, controls, constants, reverse_zone = constants.reverse_zone, nzones = constants.nzones, recirculation = recirculation)
    moisture_solid, temperature_solid, cure, moisture_air, temperature_air, t_event = tree_util.tree_map(jnp.vstack, solution)

    # mse_cost = mean squared error cost. total_cost = mse_cost + regularization cost
    mse_cost = jnp.sum((temperature_solid * target_mask - target[1:])**2 * (ny - jnp.count_nonzero(target_mask)))
    total_cost = mse_cost
    return total_cost, mse_cost, xinit, *(moisture_solid, temperature_solid, cure, moisture_air, temperature_air, t_event)


logger.info(f"{divider} \n{inspect.getsource(_objective)}")


#############################################################################
## creating the optimization problem

if load_dir is not None : 
    with open(f"{load_dir}/saved_params/params", "rb") as file :
        logger.info(f"{divider} \nReading data from input file {file}")
        meta_data = pickle.load(file)
        
        meta_parameters = meta_data.get("meta_parameters", None)
        info = meta_data.get("info", None)
        solid_moisture_mean = meta_data.get("solid_moisture_mean", None)
        solid_moisture_sigma_optimal = meta_data.get("solid_moisture_sigma", None)

        def _missing(key : str) : logger.info(f"MISSING : {key} in file {file}")
        if meta_parameters is None : _missing("meta_parameters")
        if info is None : _missing("info")
        if solid_moisture_mean is None : _missing("solid_moisture_mean")
        if solid_moisture_sigma_optimal is None : _missing("solid_moisture_sigma_optimal")
        
else : info = {}

# print initial conditions
logger.info(f"{divider} \nInitial guess of parameters {pformat(meta_parameters)}")
logger.info(f"{divider} \nInitial guess of solid moisture {pformat(solid_moisture_mean)}")

meta_params_flatten, unravel_metaparams = flatten_util.ravel_pytree(meta_parameters)
_, unravel_moist = flatten_util.ravel_pytree(solid_moisture_mean)

def _objective_pmap(meta_params, *_pmap_args):
    # _pmap_args = (
    #       solid_moisture_init =   
    #       solid_moisture_sigma = 
    #       constants =             
    #       target =                
    #       target_mask =           
    #       controls =              
    #       solid_temperature_init =  
    #       recirculation =    
    #       key =      
    #   )
    # pmap over the different experiments of same product

    return jax.pmap(
        _objective, 
        in_axes = (None, None, None, None, 0, 0, 0, 0, None, None), 
        static_broadcasted_argnums = (3, 8)
    )(unravel_metaparams(meta_params), *_pmap_args)


def objective(diff_args : Tuple[jnp.ndarray, List[jnp.ndarray], jnp.ndarray], pmap_args : Tuple[jnp.ndarray], 
                key : jnp.ndarray, _product_list : List[str]) -> Tuple[jnp.ndarray, List[jnp.ndarray]] :
    
    # Function used only for training. For loop over the list of unique products 
    meta_params_flatten, moist_init, moist_sigma = diff_args
    moist_init = [moist_init[product_list.index(prod[:3])] for prod in _product_list]

    asum, cost, *solution = zip(*[
        _objective_pmap(meta_params_flatten, _args[0], moist_sigma, *_args[1:], pargs.recirculation, key)
        for _args in zip(moist_init, *pmap_args)
    ])

    _kl = 0
    for _mi in moist_init:
        _kl += kl_div(_mi, moist_sigma, 0.1, 0.005)

    asum = tree_util.tree_reduce(operator.add, tree_util.tree_map(jnp.sum, asum)) + jnp.sum(_kl)
    return asum, cost, solution

logger.info(f"{divider} \n{inspect.getsource(objective)}")
_objective_partial = lambda *_args : objective(*_args, product_list)[0]
_gradient_partial = jax.grad(_objective_partial)


class ParameterEstimationScipy():
    # class used for parameter estimation using ipopt with scipy interface
    
    def objective(self, p, args):
        return _objective_partial(p, (constants, target_scaled, target_mask, controls, solid_temperature_init), args)

    def gradient(self, p, args):
        # gradient of the objective function
        # logger.info(f"iterate {unravel(p)} \n{divider}")
        return _gradient_partial(p, (constants, target_scaled, target_mask, controls, solid_temperature_init), args)
    
    def update_args(self, args):
        # update rule for auxilary data
        _key, _subkey = jrandom.split(args)
        return _subkey
    
    def intermediate(self, p, epoch, args) -> None :
        # intermediate printing function
        
        _metap, mean, sigma = p
        _metap_unravel = unravel_metaparams(_metap)
        meantest = [mean[product_list.index(prod[:3])] for prod in product_list_test]

        total_cost, cost, solution = objective((_metap, meantest, sigma), (constants_test, target_scaled_test, target_mask_test, controls_test, solid_temperature_init_test), args, product_list_test)
        
        # print critical mositure ratio
        _critical_moisture = _metap_unravel["parameters"]["critical_moisture"]
        logger.info(f"Critical moisture ratio {_critical_moisture}")
        logger.info(f"Mass transfer coefficient constant {_metap_unravel['parameters']['mass_transfer_coefficient_constant']}")
        logger.info(f"Heat transfer coefficient constant {_metap_unravel['parameters']['heat_transfer_coefficient_constant']}")

        # printing testing conditions 
        for i, prod in enumerate(product_list_test) : 
            for j, _product in enumerate(product_dict_test[prod]) :
                logger.info(f"Inlet moisture of {_product} is {(mean[i], sigma)}(mean, sigma), mse cost {cost[i][j]}, total cost {total_cost}")
        
        
        # save intermediate results
        if epoch % pargs.gen == 0 :
            _adict = {"meta_parameters" : _metap_unravel, "solid_moisture_mean" : mean}
            _constants_test = [Constants(
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
                    reverse_zone = 10, # no reverse zone
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
                    ode_kwargs = OdeKwargs(rtol = pargs.rtol, atol = pargs.atol, mxstep = pargs.mxstep, reuse = True)) for prod in product_list_test]
            
            *_, solution_gen = objective((_metap, meantest, sigma), (_constants_test, target_scaled_test, target_mask_test, controls_test, solid_temperature_init_test), args, product_list_test)

            for i, prod in enumerate(product_list_test) : 
                
                for j, _product in enumerate(product_dict_test[prod]) :

                    time_span = jnp.arange(0, controls_test[i]["residence_time"][j, 0]*constants_test[i].nzones).flatten()
                    height_span = jnp.linspace(0, constants_test[i].product_height, constants_test[i].ny)

                    # testing plots
                    # TODO optional argument where fill between is not provided
                    plot(
                        SimulationData(
                            jnp.stack([unscale_states(solution[1][i][j], moisture_max, moisture_min)]*3), 
                            jnp.stack([unscale_states(solution[2][i][j], temperature_max, temperature_min)]*3), 
                            jnp.stack([unscale_states(solution[4][i][j], moisture_max, moisture_min)]*3), 
                            jnp.stack([unscale_states(solution[5][i][j], temperature_max, temperature_min)]*3), 
                            solution[-1][i][j],
                            None, None
                        ),
                        TrainingResults(),
                        (time_span, height_span, constants_test[i].nzones, controls_test[i]["residence_time"][j, 0]),
                        None,
                        tuple(f"{_product}" + ni for ni in ("EventSolid", "EventSolidMesh", "EventAirMesh", "EventCoeffMesh")),
                        dir = os.path.join("log", _product, _dir, str(epoch), "test")
                    )

                    # generalization plots
                    plot(
                        SimulationData(
                            jnp.stack([unscale_states(solution_gen[1][i][j], moisture_max, moisture_min)]*3), 
                            jnp.stack([unscale_states(solution_gen[2][i][j], temperature_max, temperature_min)]*3), 
                            jnp.stack([unscale_states(solution_gen[4][i][j], moisture_max, moisture_min)]*3), 
                            jnp.stack([unscale_states(solution_gen[5][i][j], temperature_max, temperature_min)]*3), 
                           solution_gen[-1][i][j],
                            None, None
                        ),
                        TrainingResults(),
                        (time_span, height_span, _constants_test[i].nzones, controls_test[i]["residence_time"][j, 0]),
                        None,
                        tuple(f"{_product}" + ni for ni in ("EventSolid", "EventSolidMesh", "EventAirMesh", "EventCoeffMesh")),
                        dir = os.path.join("log", _product, _dir, str(epoch), "gen")
                    )

                    save_params(_adict, os.path.join("log", _product, _dir, str(epoch)), "params")
        
        logger.info(f"{divider}")
        return total_cost


boundaries, values = [2000, 3000], [pargs.lr, 0.1 * pargs.lr, 0.01 * pargs.lr]
lr_schedule = piecewise_constant(boundaries, values)
logger.info(f"{divider} \nPiecewise constant learning rate with boundaries {boundaries} and values {values}")
nlp_options = {"step_size" : lr_schedule, "epochs" : pargs.iterations, "print_every_epoch" : 1, "aux_args" : key_sigma, "opt_state" : info.get("opt_state", None)}

if pargs.optimize :

    p_opt, _info = select_optimizer(ParameterEstimationScipy, (meta_params_flatten, solid_moisture_mean, solid_moisture_sigma), "sgd", 
                                nlp_options = nlp_options, use_jax = True, logger = logger)
    
    (metaparameters_optimal_flatten, solid_moisture_optimal, solid_moisture_sigma_optimal) = p_opt
    metaparameters_optimal = unravel_metaparams(p_opt[0])

    result_dict = info.get("result", defaultdict(list))
    for _key in _info["result"].keys() :
        result_dict[_key].extend(_info["result"][_key])
    
    info["result"] = result_dict # replace updated state
    info["opt_state"] = _info["opt_state"] # replace opt state

else : 

    metaparameters_optimal_flatten, solid_moisture_optimal, solid_moisture_sigma_optimal = meta_params_flatten, solid_moisture_mean, solid_moisture_sigma
    metaparameters_optimal = meta_parameters


logger.info(f"Sigma {solid_moisture_sigma_optimal}")

#############################################################################
## Saving optimal parameters

logger.info(f"{divider} \nOptimization result {pformat(info)}")
logger.info(f"{divider} \nMeta parameters optimal {pformat(metaparameters_optimal)}")
logger.info(f"{divider} \nInitial moisture content (mean, sigma) {pformat({prod : (solid_moisture_optimal[i], solid_moisture_sigma_optimal) for i, prod in enumerate(product_list)})}")

for _product in product_expts :
    logger.info(f"Saving parameters for {_product}")
    adict = {
        "meta_parameters" : metaparameters_optimal, "info" : info, 
        "solid_moisture_mean" : solid_moisture_optimal, "solid_moisture_sigma" : solid_moisture_sigma_optimal
        }
    save_params(adict, os.path.join("log", _product, _dir, "saved_params"), "params")


#############################################################################
## Simulating and plotting training experiments

result_dict = info.get("result", {})

def simulation_plots(product_expts, storage, names : Tuple):

    (_cost, _moisture, _temperature, moisture, temperature, _event_time, _quality, _constants) = storage

    for _product in product_expts :

        # get heat and mass transfer coefficients
        htc, mtc = jax.vmap(lambda *_args : get_coefficients(_args, metaparameters_optimal["parameters"], _constants[_product], 
            scaled_moisture_content, scaled_mass_transfer_coefficient, scaled_heat_transfer_coefficient))(*_moisture[_product], *_temperature[_product])

        # mask constant htc and mtc
        _mask = lambda _m, _z : np.ma.masked_where(_m >= metaparameters_optimal["parameters"]["critical_moisture"][0], _z)
        htc, mtc = np.vectorize(_mask)(_moisture[_product][0], htc), np.vectorize(_mask)(_moisture[_product][0], mtc)

        product = Target[_product]
        target = product.targets

        time_span = jnp.arange(0, product.controls["residence_time"][0]*_constants[_product].nzones).flatten()
        height_span = jnp.linspace(0, _constants[_product].product_height, _constants[_product].ny)
        target = tree_util.tree_map(lambda z : z[1:len(time_span) + 1], target)

        cost_mse = 0
        for key, value in target.items():
            cost_mse += jax.vmap(lambda _z : jnp.mean(_z[:, key] - value)**2)(temperature[_product][0])
        
        logger.info(f"{divider} \nPlotting {_product}")
        logger.info(f"Loss scaled (total cost, cost) {_product} : {_cost[_product]}")
        logger.info(f"Loss unscaled {_product} : {cost_mse}")
        logger.info(f"Product energy {_product} : {_quality[_product]['energy']}")
        logger.info(f"Product average cure index {_product} : {tree_util.tree_map(lambda _z : jnp.mean(_z, axis = -1), _quality[_product]['cure'])}")

        # plotting unscaled values
        plot(
            SimulationData(moisture[_product][0], temperature[_product][0], moisture[_product][-1], temperature[_product][-1], _event_time[_product][0], htc, mtc),
            TrainingResults(result_dict.get("training_loss", None), result_dict.get("testing_loss", None), result_dict.get("learning_rate", None)),
            (time_span, height_span, _constants[_product].nzones, product.controls["residence_time"]),
            target,
            tuple(f"{_product}" + ni for ni in names),
            dir = os.path.join("log", _product, _dir)
        )

        # plotting scaled values
        plot(
            SimulationData(_moisture[_product][0], _temperature[_product][0], _moisture[_product][-1], _temperature[_product][-1], _event_time[_product][0]),
            TrainingResults(),
            (time_span, height_span, _constants[_product].nzones, product.controls["residence_time"]),
            tree_util.tree_map(lambda _x : scale_states(_x, temperature_max, temperature_min), target),
            tuple(f"Scaled_{_product}" + ni for ni in names),
            dir = os.path.join("log", _product, _dir)
        )

def simulation_storage(product_expts, solution, storage, recirculation_ratio):
    # recirculation_ratio : shape (3, products, nzones)

    (total_cost, cost, _, moisture_solid, temperature_solid, cure, moisture_air, temperature_air, t_event) = solution
    (_cost, _moisture, _temperature, moisture, temperature, _event_time, _quality, _constants) = storage

    for j, _product in enumerate(product_expts):

        _moisture[_product] = (moisture_solid[:, j, ...], moisture_air[:, j, ...])
        _temperature[_product] = (temperature_solid[:, j, ...], temperature_air[:, j, ...])
        
        # scale back states
        _moisture_unscaled = tree_util.tree_map(lambda z : unscale_states(z, moisture_max, moisture_min), _moisture[_product])
        _temperature_unscaled = tree_util.tree_map(lambda z : unscale_states(z, temperature_max, temperature_min), _temperature[_product])
        
        moisture[_product] = _moisture_unscaled
        temperature[_product] = _temperature_unscaled
        
        _cost[_product] = (total_cost[:, j, ...], cost[:, j, ...])
        _event_time[_product] = t_event[:, j, ...]

        # get cure index and energy consumption
        nzones = _constants[_product].nzones
        _energy = jax.vmap(
            lambda ratio, *_args : product_quality(
                *tree_util.tree_map(lambda z : jnp.array_split(z, nzones), _args),
                ratio, Target[_product].controls["init_temperature_air"], 
                Target[_product].controls["init_velocity_air"], _constants[_product].reverse_zone, 25. + 273, _constants[_product].specific_heat_capacity_air
            )
        )(recirculation_ratio[:, j], _moisture_unscaled[0], _temperature_unscaled[0], _moisture_unscaled[-1], _temperature_unscaled[-1])
        _quality[_product] = {"energy" : _energy, "cure" : cure[:, j, ...]}

    return (_cost, _moisture, _temperature, moisture, temperature, _event_time, _quality, _constants)

def simulation_results(product_list : list, product_dict : dict, constants : Tuple, target_scaled : list, target_mask : list, controls : Tuple, 
                       solid_moisture_mean : list, solid_moisture_sigma : jnp.ndarray, solid_temperature_init : list, recirculation : bool, 
                       names : Tuple) :

    _cost = Cost(elbo = {}, cost = {}) # storing total cost (mse + reg) and cost (mse)
    _moisture = Moisture(states = {}) # storing scaled moisture
    _temperature = Temperature(states = {}) # storing scaled temperature
    moisture = Moisture(states = {}) # storing unscaled moisture
    temperature = Temperature(states = {}) # storing unscaled temperature
    _event_time = EventTime({})
    _constants = {expt : constants[i] for i, prod in enumerate(product_list) for expt in product_dict[prod]} # for all experiments
    _quality = Quality({})
    storage = (_cost, _moisture, _temperature, moisture, temperature, _event_time, _quality, _constants)

    for i, (prod, *_pmap_args) in enumerate(zip(product_list, constants, target_scaled, target_mask, controls, solid_temperature_init)):

        # get mean and sample trajectories
        solution = tree_util.tree_map(
            lambda *args : jnp.stack(args),
            _objective_pmap(metaparameters_optimal_flatten, solid_moisture_mean[i], jnp.zeros_like(solid_moisture_sigma), *_pmap_args, recirculation, key_sigma),
            _objective_pmap(metaparameters_optimal_flatten, solid_moisture_mean[i] + 2 * solid_moisture_sigma, jnp.zeros_like(solid_moisture_sigma), *_pmap_args, recirculation, key_sigma), 
            _objective_pmap(metaparameters_optimal_flatten, solid_moisture_mean[i] - 2 * solid_moisture_sigma, jnp.zeros_like(solid_moisture_sigma), *_pmap_args, recirculation, key_sigma), 
        ) # mean trajectory is followed by mean + 2 * sigma and mean - 2 * sigma

        _ratio = jnp.tile(_pmap_args[-2]["recirculation_ratio"], (3, 1, 1))
        storage = simulation_storage(
            product_dict[prod], solution, storage, 
            _ratio if pargs.recirculation else jnp.zeros_like(_ratio)
        )
        simulation_plots(product_dict[prod], storage, names)

    (_cost, _moisture, _temperature, moisture, temperature, _event_time, _quality, _constants) = storage

    logger.info(f"{divider} \nTotal cost {tree_util.tree_reduce(operator.add, _cost)}")
    logger.info(f"{divider} \nOptimal objective {pformat(_cost)}")
    logger.info(f"{divider} \nEvent times {pformat(_event_time)}")

    return _cost, _moisture, _temperature, moisture, temperature, _event_time, _quality


solid_moisture_optimal_mapped = [solid_moisture_optimal[product_list.index(prod[:3])] for prod in product_list]

(
    cost_train, moisture_scaled_train, temperature_scaled_train, 
    moisture_train, temperature_train, event_train, quality_train
) = simulation_results(product_list, product_dict, constants, target_scaled, target_mask, 
                       controls, solid_moisture_optimal_mapped, solid_moisture_sigma_optimal, solid_temperature_init, pargs.recirculation,
                       names = ("EventSolid", "EventSolidMesh", "EventAirMesh", "EventCoeffMesh"))


#############################################################################
## Simulating and plotting testing experiments

if not pargs.products_test == "" :

    solid_moisture_test_mapped = [solid_moisture_optimal[product_list.index(prod[:3])] for prod in product_list_test]

    (
        cost_test, moisture_scaled_test, temperature_scaled_test, 
        moisture_test, temperature_test, event_test, quality_test
    ) = simulation_results(product_list_test, product_dict_test, constants_test, target_scaled_test, target_mask_test, 
                           controls_test, solid_moisture_test_mapped, solid_moisture_sigma_optimal, solid_temperature_init_test, pargs.recirculation, 
                           names = ("EventSolid", "EventSolidMesh", "EventAirMesh", "EventCoeffMesh"))


#############################################################################