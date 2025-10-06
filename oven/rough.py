import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
import logging
from typing import Callable, Any

import jax.flatten_util
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from typing import Optional, List, Any, Tuple
from collections import defaultdict
from datetime import datetime
from pprint import pformat
import pickle
import functools

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
from jax import flatten_util, tree_util
import jax.random as jrandom
import matplotlib.pyplot as plt

from .phy_props import _mass_transfer_coefficient, _heat_transfer_coefficient, moisture_content
from .scaling import  choose_scaling, parameters_init
from .data_structures import Controls, Constants, OdeKwargs, Parameters
from .oven_dynamics import oven_dynamics 
from .data.targets import get_targets
from .utils import check_dir, product_quality

Pytree = Any

"""
# sensitivity analysis

#############################################################################
## Initialization steps - initialize directories, loggers, and scaling

# Training products
product_expts : List[str] = ["R13SSR1"]
product_dict = defaultdict(list)
for prod in product_expts : product_dict[prod[:3]].append(prod) # make a dictionary of product specific experiments {"R13" : ["R13NSR1", "R13NSR2"], "R19" : ["R19SSR1"]}
product_list = list(product_dict.keys()) # ["R13", "R19"]
prod = product_list[0]

_dir = str(datetime.now())
divider = "--"*50 # printing separater
for _product in set(product_expts):
    check_dir(os.path.join("log", _product, _dir))
    logfile = logging.FileHandler(os.path.join("log", _product, _dir, "solver_stats.txt"))
    logger.addHandler(logfile)

ndevices = jax.local_device_count()

(moisture_max, moisture_min, temperature_max, temperature_min), (scale_states, unscale_states, _, _) = choose_scaling("max")


#############################################################################
## Create and log custom functions

scaled_moisture_content = lambda T, density_air, relative_humidity = 1 : scale_states(moisture_content(unscale_states(T, temperature_max, temperature_min), density_air, relative_humidity), moisture_max, moisture_min)
scaled_mass_transfer_coefficient = lambda x, p, constant: _mass_transfer_coefficient(x, p, constant)
scaled_heat_transfer_coefficient = lambda x, p, constant : _heat_transfer_coefficient(x, p, constant)

#############################################################################
## Initialize parameters (constant for all products), constants (product specific), controls (experiment specific)

with open("log/R13SSR1/2024-09-16 10:41:37.550896/saved_params/params", "rb") as file :
    meta_data = pickle.load(file)
    parameters_optimal = meta_data["opt_params"] 
    solid_moisture_init_optimal = meta_data["solid_moisture_mean"] 
    solid_moisture_sigma_optimal = meta_data["solid_moisture_sigma"]


constants = Constants(
    radius_particle = TargetConstants[prod]["radius_particle"],
    voidage = 0.99, # 1 - TargetConstants[prod]["density_product"] / TargetConstants[prod]["density_particle"], # "Bed porosity (1 - density of product / density of particle)"
    enthalpy_vaporization_water = TargetConstants[prod]["enthalpy_vaporization_water"],
    specific_heat_capacity_air = TargetConstants[prod]["specific_heat_capacity_air"],
    density_particle = TargetConstants[prod]["density_particle"],
    specific_heat_capacity_solid = TargetConstants[prod]["specific_heat_capacity_solid"],
    product_height = TargetConstants[prod]["product_height"],
    density_product = TargetConstants[prod]["density_product"],

    ny = TargetConstants[prod]["ny"],
    nzones = TargetConstants[prod]["nzones"],
    reverse_zone = TargetConstants[prod]["reverse_zone"],
    ntimes = TargetConstants[prod]["ntimes"], # number of measurements for given residence time
    equilibrium_moisture = scale_states(TargetConstants[prod]["equilibrium_moisture"], moisture_max, moisture_min), # equilibrium moisture content in solid

    moisture_max = moisture_max,
    moisture_min = moisture_min,
    temperature_max = temperature_max,
    temperature_min = temperature_min,

    # Equations
    scaled_moisture_content = scaled_moisture_content,
    scaled_mtc = scaled_mass_transfer_coefficient,
    scaled_htc = scaled_heat_transfer_coefficient,
    density_air = lambda X, T : TargetConstants[prod]["density_air"],

    # ode kwargs
    ode_kwargs = OdeKwargs(rtol = 1e-6, atol = 1e-8, mxstep = 10_000))

controls = Controls(
    inputs = {
        "init_temperature_air" : scale_states((jnp.array([535., 535., 560., 565., 565.]) - 32)*5/9 + 273, temperature_max, temperature_min), # setpoint - line loss 
        "init_velocity_air" : jnp.array([0.097, 0.086, 0.115, 0.207, 0.207]),
        "init_moisture_air" : scale_states(jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]), moisture_max, moisture_min),
        "residence_time" : jnp.atleast_1d(jnp.round(24*60./102)),
        "recirculation_ratio" : jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]), # must be between 0 to 1
    })


#############################################################################
## Define the objective function

target_scaled = scale_states(Target[product_expts[0]].targets_mask["target_map"][:constants.ntimes*constants.nzones - constants.nzones + 1], temperature_max,  temperature_min)
target_mask = Target[product_expts[0]].targets_mask["mask"]
solid_temperature_init = jnp.sum(target_scaled[:1], axis = 1)/jnp.count_nonzero(target_scaled[0]) 
solid_moisture_mean = scale_states(jnp.array([0.1]), moisture_max, moisture_min)
solid_moisture_sigma = jnp.zeros_like(solid_moisture_mean)


#############################################################################
## Simulating and plotting training experiments

parameters_optimal_flatten, unravel = flatten_util.ravel_pytree(parameters_optimal)
key = jrandom.PRNGKey(10)
*subkeys, key = jrandom.split(key, len(product_list) + 1)


def forward_simulate(solid_moisture_init, solid_temperature_init, controls, constants, msg = ""):

    _xinit = jnp.concatenate((solid_moisture_init, solid_temperature_init))
    xinit = jnp.tile(_xinit, (constants.ny, 1))
    solution = oven_dynamics(xinit, parameters_optimal["parameters"], controls, constants, reverse_zone = constants.reverse_zone, nzones = constants.nzones, recirculation = False)
    solution = moisture_solid, temperature_solid, moisture_air, temperature_air, t_event = tree_util.tree_map(jnp.vstack, solution)

    # scale back states
    moisture_unscaled = tree_util.tree_map(lambda z : unscale_states(z, moisture_max, moisture_min), (moisture_solid, moisture_air))
    temperature_unscaled = tree_util.tree_map(lambda z : unscale_states(z, temperature_max, temperature_min), (temperature_solid, temperature_air))

    # get cure index and energy consumption
    nzones = constants.nzones
    energy, cure = product_quality(
            *tree_util.tree_map(lambda z : jnp.array_split(z, nzones), (moisture_unscaled[0], temperature_unscaled[0], moisture_unscaled[-1], temperature_unscaled[-1])),
            jnp.zeros_like(controls["recirculation_ratio"]), Target[_product].controls["init_temperature_air"], 
            Target[_product].controls["init_velocity_air"], constants.reverse_zone, 25. + 373, constants.specific_heat_capacity_air
        )
    cure = tree_util.tree_map(jnp.mean, cure)

    logger.info(f"{divider} \nMsg : {msg}")
    logger.info(f"\nSolid Initial Conditions {pformat(_xinit)}")
    logger.info(f"\nControl Inputs {pformat(controls)}")
    logger.info(f"\nEnergy {pformat(energy)}")
    logger.info(f"\nCure {pformat(cure)}")

    return *moisture_unscaled, *temperature_unscaled, energy, cure


# effect of inlet moisture on cure
*solution, energy, cure = forward_simulate(solid_moisture_mean, solid_temperature_init, controls, constants, msg = "Nominal case")
*solution1, energy1, cure1 = forward_simulate(solid_moisture_mean * 1.25, solid_temperature_init, controls, constants, msg = "Solid moisture is 1.25X")
*solution2, energy2, cure2 = forward_simulate(solid_moisture_mean * 1.5, solid_temperature_init, controls, constants, msg = "Solid moisture is 1.5X")
*solution3, energy3, cure3 = forward_simulate(solid_moisture_mean * 2, solid_temperature_init, controls, constants, msg = "Solid moisture is 2X")

with plt.style.context(["science", "notebook", "bright"]):

    fig, ax = plt.subplots(1, 2, figsize = (20, 10), gridspec_kw = {"hspace" : 0.3})

    ax[0].plot(solution[2][:, 5], label = "X")
    ax[0].plot(solution2[2][:, 5], label = "1.5X")
    ax[0].plot(solution3[2][:, 5], label = "2X")
    ax[0].set(xlabel = "Time", ylabel = "Temperature")
    ax[0].legend()

    ax[1].plot(solid_moisture_mean * jnp.array([1, 1.25, 1.5, 2]), jnp.array([cure["zone5"], cure1["zone5"], cure2["zone5"], cure3["zone5"]]), "o-")
    ax[1].set(xlabel = "Inlet solid moisture", ylabel = "Cure")

    plt.savefig(os.path.join("log", _product, _dir, "SolidMoisture"))
    plt.close()

"""

"""
# effect of inlet air temperature on cure
_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_temperature_air"] = scale_states((jnp.array([535., 535., 560., 565., 565.]) - 50 - 32)*5/9 + 273, temperature_max, temperature_min)
*solution4, _, cure4 = forward_simulate(2 * solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Temperature decreased by 50")

_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_temperature_air"] = scale_states((jnp.array([535., 535., 560., 565., 565.]) + 50 - 32)*5/9 + 273, temperature_max, temperature_min)
*solution5, _, cure5 = forward_simulate(2 * solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Temperature increased by 50")

_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_temperature_air"] = scale_states((jnp.array([535., 535., 560., 565., 565.]) +100 - 32)*5/9 + 273, temperature_max, temperature_min)
*solution6, _, cure6 = forward_simulate(2 * solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Temperature increased by 100")

with plt.style.context(["science", "notebook", "bright"]):

    fig, ax = plt.subplots(1, 2, figsize = (20, 10), gridspec_kw = {"hspace" : 0.3})

    ax[0].plot(solution3[2][:, 5], label = "Nominal")
    ax[0].plot(solution4[2][:, 5], label = "-50")
    ax[0].plot(solution5[2][:, 5], label = "+50")
    ax[0].plot(solution6[2][:, 5], label = "+100")
    ax[0].set(xlabel = "Time", ylabel = "Temperature")
    ax[0].legend()

    ax[1].plot(jnp.array([-50, 0, 50, 100]), jnp.array([cure4["zone5"], cure3["zone5"], cure5["zone5"], cure6["zone5"]]), "o-")
    ax[1].set(xlabel = "Air Temperature", ylabel = "Cure")

    plt.savefig(os.path.join("log", _product, _dir, "AirTemperature"))
    plt.close()


# effect of inlet air fan speed on cure
_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_velocity_air"] = jnp.array([0.097, 0.086, 0.115, 0.207, 0.207]) - 0.05
*solution7, _, cure7 = forward_simulate(2 * solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Fan speed decreased by 0.05")

_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_velocity_air"] = jnp.array([0.097, 0.086, 0.115, 0.207, 0.207]) + 0.05
*solution8, _, cure8 = forward_simulate(2 * solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Fan speed increased by 0.05")

_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_velocity_air"] = jnp.array([0.097, 0.086, 0.115, 0.207, 0.207]) + 0.1
*solution9, _, cure9 = forward_simulate(2 * solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Fan speed increased by 0.1")

with plt.style.context(["science", "notebook", "bright"]):

    fig, ax = plt.subplots(1, 2, figsize = (20, 10), gridspec_kw = {"hspace" : 0.3})

    ax[0].plot(solution3[2][:, 5], label = "Nominal")
    ax[0].plot(solution7[2][:, 5], label = "-0.05")
    ax[0].plot(solution8[2][:, 5], label = "+0.05")
    ax[0].plot(solution9[2][:, 5], label = "+0.1")
    ax[0].set(xlabel = "Time", ylabel = "Temperature")
    ax[0].legend()

    ax[1].plot(jnp.array([-0.05, 0, 0.05, 0.1]), jnp.array([cure7["zone5"], cure3["zone5"], cure8["zone5"], cure9["zone5"]]), "o-")
    ax[1].set(xlabel = "Fan Speed", ylabel = "Cure")

    plt.savefig(os.path.join("log", _product, _dir, "AirFanSpeed"))
    plt.close()

"""
"""
# increase temperature 
_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_temperature_air"] = scale_states((jnp.array([600., 600., 600., 600., 600.]) - 32)*5/9 + 273, temperature_max, temperature_min)
solution = forward_simulate(solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Nominal case and temperature is increased")

# increase fan speed 
_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_velocity_air"] = jnp.array([0.2, 0.15, 0.2, 0.207, 0.207])
solution = forward_simulate(solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Nominal case and air velocity is increased")


# solid inlet moisture content increase. 
solution = forward_simulate(2 * solid_moisture_mean, solid_temperature_init, controls, constants, msg = "Solid inlet moisture content is doubled")

# increase temperature 
_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_temperature_air"] = scale_states((jnp.array([600., 600., 600., 600., 600.]) - 32)*5/9 + 273, temperature_max, temperature_min)
solution = forward_simulate(2 * solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Solid inlet moisture content is doubled and temperature is increased")

# increase fan speed 
_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_velocity_air"] = jnp.array([0.2, 0.15, 0.2, 0.207, 0.207])
solution = forward_simulate(2 * solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Solid inlet moisture content is doubled and air velocity is increased")

# air inlet moisture content increase. 
_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_moisture_air"] = scale_states(jnp.array([0.02, 0.02, 0.02, 0.02, 0.02]), moisture_max, moisture_min)
solution = forward_simulate(solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Air inlet moisture content is doubled")

# increase temperature 
_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_moisture_air"] = scale_states(jnp.array([0.02, 0.02, 0.02, 0.02, 0.02]), moisture_max, moisture_min)
_controls["init_temperature_air"] = scale_states((jnp.array([600., 600., 600., 600., 600.]) - 32)*5/9 + 273, temperature_max, temperature_min)
solution = forward_simulate(solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Air inlet moisture content is doubled and temperature is increased")

# increase fan speed 
_controls = tree_util.tree_map(lambda _x : _x, controls)
_controls["init_moisture_air"] = scale_states(jnp.array([0.02, 0.02, 0.02, 0.02, 0.02]), moisture_max, moisture_min)
_controls["init_velocity_air"] = jnp.array([0.2, 0.15, 0.2, 0.207, 0.207])
solution = forward_simulate(solid_moisture_mean, solid_temperature_init, _controls, constants, msg = "Air inlet moisture content is doubled and air velocity is increased")
"""



"""
# cure as internal function
from jax.experimental.ode import odeint

def foo(x, t):
    return jnp.array([
        - 2 * x[0], 
        (1 - x[1]) * 2 * jnp.exp(-2 / x[0])
    ])

xinit = jnp.array([5., 0.])
time_span = jnp.arange(0, 5, 0.01)
solution = odeint(foo, xinit, time_span)

cure_integral = solution[:, -1]

integral = jnp.cumsum(0.01 * 2 * jnp.exp(-2 / solution[:, 0]))
cure_approx = 1 - 1/jnp.exp(integral)
"""


"""
# creating lower triangular matrix using flattened inputs

from itertools import combinations_with_replacement
from scipy.stats import multivariate_normal
import numpy as np
import pickle

def log_normal_density(mu, sigma):
    
    # Define grid for x and y values
    samples = 100
    # x = jnp.exp(jnp.linspace(mu[0] - 2 * sigma[0, 0], mu[0] + 2 * sigma[0, 0], samples))  # Avoid 0 since log-normal is undefined for x <= 0
    # y = jnp.exp(jnp.linspace(mu[1] - 2 * sigma[1, 1], mu[1] + 2 * sigma[1, 1], samples))
    x = jnp.linspace(1e-4, 1, samples)
    y = jnp.linspace(1e-4, 1, samples)
    X, Y = jnp.meshgrid(x, y)

    # Convert grid to 2D array of log-transformed values
    log_x = jnp.log(X)
    log_y = jnp.log(Y)
    pos = jnp.dstack((log_x, log_y))

    # Compute the density of the underlying normal distribution
    rv = multivariate_normal(mean = mu, cov = sigma)
    pdf = rv.pdf(pos)

    # Scale by the Jacobian determinant (to account for the log transformation)
    pdf_scaled = pdf / (X * Y)
    return X, Y, pdf_scaled

# Parameters for the underlying normal distribution
mu = jnp.array([-0.458, -0.588])  # Mean vector
sigma = jnp.array([[1.69, 4.2e-1], [4.2e-1, 1.69]])  # Covariance matrix

# Compute the log-normal density
X, Y, Z = log_normal_density(mu, sigma)


def foo(mu, sigma):

    # exchange rows and columns
    mu, sigma = np.array(mu), np.array(sigma)
    n = len(mu)
    comb = list(combinations_with_replacement(np.arange(n), 2))

    with plt.style.context(["science", "notebook", "bright"]) :
        fig, ax = plt.subplots(n, n, figsize = (40, 40))

        count = 0
        for i in range(1, n + 1):
            for j in range(n):

                if j > i - 1 : 
                    ax[i - 1, j].axis("off")
                    continue 

                _ind = comb[count]
                _sigma = sigma
                _sigma[[0, _ind[0]], [0, _ind[0]]] = _sigma[[_ind[0], 0], [_ind[0], 0]]
                _sigma[[1, _ind[1]], [1, _ind[1]]] = _sigma[[_ind[1], 1], [_ind[1], 1]]

                _mu = mu
                _mu[0], _mu[_ind[0]] = _mu[_ind[0]], _mu[0]
                _mu[1], _mu[_ind[1]] = _mu[_ind[1]], _mu[1]

                mu_cond = mu[:2]
                sigma_cond = _sigma[:2, :2] - _sigma[:2, 2:] @ np.linalg.solve(_sigma[2:, 2:], _sigma[2:, :2])
                # print(f"sigma for index {_ind} : {sigma_cond}")

                X, Y, Z = log_normal_density(mu_cond, sigma_cond)
                contour = ax[i - 1, j].contourf(X, Y, Z, levels=10, cmap="viridis")
                ax[i - 1, j].set(xlabel = "X", ylabel = "Y", title = "Contour Plot")
                ax[i - 1, j].grid(True)
                count += 1
        
        plt.tight_layout()
        plt.savefig("demo_pdf_combination")
        plt.close()


with open("log/R13SSR1/2025-01-23 20:58:03.805908/saved_params/params", "rb") as file : 
    
    meta_data = pickle.load(file)    
    controls_variables = meta_data["controls_mean"]
    controls_flatten, _ = flatten_util.ravel_pytree(controls_variables)
    controls_sigma = meta_data["controls_sigma"]

# all temperatures
mu = controls_flatten
L = controls_sigma.reshape(len(mu), -1)
sigma = L.T @ L
foo(mu[:5], sigma[:5, :5])

# TODO use truncated normal distribution instead of log normal distribution for inverse_mapping
"""

"""
# energy optimization 

#############################################################################
## Initialization steps - initialize directories, loggers, and scaling

# Training products
product_expt = "R13SSR1"
prod = product_expt[:3] # "R13"

ProductConstants, Target = get_targets([product_expt], [prod], None) # get constants and log them

(moisture_max, moisture_min, temperature_max, temperature_min), (scale_states, unscale_states, _, _) = choose_scaling("max")


#############################################################################
## Create and log custom functions

scaled_moisture_content = lambda T, density_air, relative_humidity = 1 : scale_states(moisture_content(unscale_states(T, temperature_max, temperature_min), density_air, relative_humidity), moisture_max, moisture_min)
scaled_mass_transfer_coefficient = lambda x, p, constant: _mass_transfer_coefficient(x, p, constant)
scaled_heat_transfer_coefficient = lambda x, p, constant : _heat_transfer_coefficient(x, p, constant)
scaled_reaction_rate = lambda A, k0, T : k0 * jnp.exp( - A / 1.98 / unscale_states(T, temperature_max, temperature_min))

#############################################################################
## Initialize parameters (constant for all products), constants (product specific), controls (experiment specific)

with open("log/R13SSR4/2024-12-27 20:47:09.835171/saved_params/params", "rb") as file :
    meta_data = pickle.load(file)
    _metaparameters_deterministic = meta_data["meta_parameters"]
    solid_moisture_mean = meta_data["solid_moisture_mean"][0]
    solid_moisture_sigma_optimal = meta_data["solid_moisture_sigma"][0]

parameters = _metaparameters_deterministic["parameters"]
parameters["reaction_rate"] = 2.33 * 10**7 # Do not optimize this yet
parameters["activation_energy"] = 20. * 10**3 # Do not optimize this yet

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
    ode_kwargs = OdeKwargs(rtol = 1e-4, atol = 1e-6, mxstep = 10_000))

controls = Controls(inputs = 
        {
    "init_temperature_air" : scale_states(Target[product_expt].controls["init_temperature_air"], temperature_max, temperature_min), # setpoint - line loss 
    "init_velocity_air" : Target[product_expt].controls["init_velocity_air"],
    "init_moisture_air" : scale_states(Target[product_expt].controls["init_moisture_air"], moisture_max, moisture_min),
    "residence_time" : Target[product_expt].controls["residence_time"],
    "recirculation_ratio" : Target[product_expt].controls["recirculation_ratio"], # must be between 0 to 1
    })



#############################################################################
## Simulating and plotting training experiments

key = jrandom.PRNGKey(10)
ny = constants.ny
_xinit = jnp.array([solid_moisture_mean[0], 330/800, 0])
xinit = jnp.column_stack((
        solid_moisture_mean[0] * jnp.ones(ny) + solid_moisture_sigma_optimal * jrandom.normal(key, shape = (ny, )), # moisture 
        330 / 800 * jnp.ones(ny), # temperature
        jnp.zeros(shape = (ny, )) # cure 
    ))

# oven_dynamics discards the provided initial condition (It is assumed that the initial condition are the values of states at t = -1)
solution = (moisture_solid, temperature_solid, 
            cure, moisture_air, temperature_air, t_event) = oven_dynamics(xinit, parameters, controls, constants, 
            reverse_zone = 6, nzones = 5, recirculation = False)

solution = moisture_solid, temperature_solid, cure, moisture_air, temperature_air, t_event = tree_util.tree_map(jnp.vstack, solution)

# plotting results 3d (solid)
time_span = jnp.arange(0, controls["residence_time"][0]*constants.nzones).flatten()
height_span = jnp.linspace(0, constants.product_height, constants.ny)

x, y = jnp.meshgrid(time_span, height_span)
with plt.style.context(["science", "notebook", "bright"]):
    
    fig, ax = plt.subplots(1, 2, figsize = (15, 7), )
    c = ax[0].pcolor(x, y, solution[0][:-1, :-1].T, cmap = "RdBu_r", shading = "flat")
    fig.colorbar(c, ax = ax[0])
    ax[0].set(xlabel = "Time", ylabel = "Product height", title = "Moisture")

    c = ax[1].pcolor(x, y, solution[1][:-1, :-1].T, cmap = "RdBu_r", shading = "flat")
    fig.colorbar(c, ax = ax[1])
    ax[1].set(xlabel = "Time", ylabel = "Product height", title = "Temperature")

    # plot event times
    ax[0].step(solution[-1], height_span, label = "Event Times", color = "green", linewidth = 3.)
    ax[1].step(solution[-1], height_span, label = "Event Times", color = "green", linewidth = 3.)

    # plot vertical lines for each zone
    for j in range(5 - 1):
        ax[0].axvline((j + 1)*controls["residence_time"][0], color = "k")
        ax[1].axvline((j + 1)*controls["residence_time"][0], color = "k")

    # plt.subplots_adjust(wspace = 0.5)
    fig.tight_layout()
    plt.savefig("mesh_plot_solid" + ".png")
    plt.close()
"""

    

"""
# scale back states
moisture_unscaled = tree_util.tree_map(lambda z : unscale_states(z, moisture_max, moisture_min), (moisture_solid, moisture_air))
temperature_unscaled = tree_util.tree_map(lambda z : unscale_states(z, temperature_max, temperature_min), (temperature_solid, temperature_air))

# optimal parameters without recirculation
controls["init_temperature_air"] = jnp.array([0.64687087,  0.59961392,  0.59536548,  0.64499942,  0.57503731])
controls["init_velocity_air"] = jnp.array([0.12051209,  0.13469733,  0.12111939,  0.18443222,  0.08666908])
controls["residence_time"] = jnp.array([14])

solution_optimal = oven_dynamics(xinit, parameters, controls, constants, reverse_zone = 3, nzones = 5, recirculation = False)
solution_optimal = tree_util.tree_map(jnp.vstack, solution_optimal)

# optimal parameters with recirculation
controls["init_temperature_air"] = jnp.array([0.65009977,  0.59662452,  0.61376796,  0.63898928,  0.5632435  ])
controls["init_velocity_air"] = jnp.array([0.1275417 ,  0.12163541,  0.11022095,  0.1752001 ,  0.06907675])
controls["residence_time"] = jnp.array([14.])
controls["recirculation_ratio"] = jnp.array([0.26])

solution_recir_optimal = oven_dynamics(xinit, parameters, controls, constants, reverse_zone = 3, nzones = 5, recirculation = True)
solution_recir_optimal = tree_util.tree_map(jnp.vstack, solution_recir_optimal)


# plotting results 3d (solid)
time_span = jnp.arange(0, controls["residence_time"][0]*constants.nzones).flatten()
height_span = jnp.linspace(0, constants.product_height, constants.ny)

x, y = jnp.meshgrid(time_span, height_span)
with plt.style.context(["science", "notebook", "bright"]):
    
    fig, ax = plt.subplots(1, 3, figsize = (25, 7))
    c = ax[0].pcolor(x, y, solution[2].T, cmap = "RdBu_r")
    fig.colorbar(c, ax = ax[0])
    ax[0].set(xlabel = "Time", ylabel = "Product height", title = "Initial")

    c = ax[1].pcolor(x, y, solution_optimal[2].T, cmap = "RdBu_r")
    fig.colorbar(c, ax = ax[1])
    ax[1].set(xlabel = "Time", ylabel = "Product height", title = "Optimized (without recirculation)")

    c = ax[2].pcolor(x, y, solution_recir_optimal[2].T, cmap = "RdBu_r")
    fig.colorbar(c, ax = ax[2])
    ax[2].set(xlabel = "Time", ylabel = "Product height", title = "Optimized (with recirculation)")

    # plot event times
    ax[0].step(solution[-1], height_span, label = "Event Times", color = "green", linewidth = 3.)
    ax[1].step(solution_optimal[-1], height_span, label = "Event Times", color = "green", linewidth = 3.)
    ax[2].step(solution_recir_optimal[-1], height_span, label = "Event Times", color = "green", linewidth = 3.)

    # plot vertical lines for each zone
    for j in range(constants.nzones - 1):
        ax[0].axvline((j + 1)*controls["residence_time"][0], color = "k")
        ax[1].axvline((j + 1)*controls["residence_time"][0], color = "k")
        ax[2].axvline((j + 1)*controls["residence_time"][0], color = "k")

    plt.savefig("mesh_plot_solid" + ".png")
    plt.close()

"""

## Simulating and plotting effects of recirculation
"""
def objective_recirculation_pmap_ndevices(recirculation_ratio, ndevices, _p, _moist_init, _pmap_args):
    # pmap function that takes into account maximum available devices
    *_pmap_args, controls, solid_moisture_sigma, temp_init, sub_key = _pmap_args
    np = temp_init.shape[0]
    n = recirculation_ratio.shape[0]
    unit_normal = jrandom.normal(sub_key, (1, ))

    output = []
    for _ratio in jnp.split(recirculation_ratio, list(range(ndevices//np, n, ndevices//np)), axis = 0):
        # update controls and stack them
        _controls = tree_util.tree_map(lambda _x : jnp.broadcast_to(_x, (len(_ratio), *_x.shape)), controls)
        _controls["recirculation_ratio"] = jax.vmap(lambda _x : jnp.broadcast_to(_x, _controls["recirculation_ratio"][0].shape))(_ratio)

        _output = jax.pmap(
            _objective_pmap, 
            in_axes = (None, None, None, None, 0, None, None, None, None, None),
            static_broadcasted_argnums = (1, 9)
        )(_p, *_pmap_args, _controls, solid_moisture_sigma, _moist_init, temp_init, unit_normal, True)
        output.append(_output)

    return tuple(map(jnp.vstack, zip(*output)))

def simulation_recirculation_plots(product_expts, storage, ratios):

    *_, _quality, _ = storage

    for j, _product in enumerate(product_expts) :

        plot_recirculation(
            tree_util.tree_reduce(operator.add, _quality[_product]["energy"]), 
            jnp.mean(_quality[_product]["cure"]["zone5"], axis = 1),
            ratios, 
            dir = os.path.join("log", _product, _dir)
        )

def simulation_results_recirculation(product_list : list, product_dict : dict, constants : Tuple, target_scaled : list, target_mask : list, controls : Tuple, 
                       solid_moisture_mean : list, solid_moisture_sigma : list, solid_temperature_init : list, names : Tuple) :

    _cost = Cost(elbo = {}, cost = {}) # storing total cost (mse + reg) and cost (mse)
    _moisture = Moisture(states = {}) # storing scaled moisture
    _temperature = Temperature(states = {}) # storing scaled temperature
    moisture = Moisture(states = {}) # storing unscaled moisture
    temperature = Temperature(states = {}) # storing unscaled temperature
    _event_time = EventTime({})
    _constants = {expt : constants[i] for i, prod in enumerate(product_list) for expt in product_dict[prod]} # for all experiments
    _quality = Quality({})

    storage = (_cost, _moisture, _temperature, moisture, temperature, _event_time, _quality, _constants)
    ratios = jax.vmap(lambda _x : _x * jnp.ones(5))(jnp.array([0., 0.1, 0.2, 0.4, 0.6, 0.8]))

    for i, (prod, *_pmap_args) in enumerate(zip(product_list, constants, target_scaled, target_mask, controls, solid_moisture_sigma, solid_temperature_init)):

        solution = objective_recirculation_pmap_ndevices(ratios, ndevices, metaparameters_optimal_flatten, solid_moisture_mean[i], _pmap_args)
        storage = simulation_storage(
            product_dict[prod], solid_moisture_mean[i], solid_moisture_sigma[i], solution, storage,
            jax.vmap(lambda z : jnp.tile(z, (len(product_dict[prod]), 1)))(ratios)
        )
        
        for j, _rat in enumerate(ratios[:, 0]) : 

            simulation_plots(
                product_dict[prod], 
                (*tree_util.tree_map(lambda _x : _x[j : j+1], storage[:-1]), storage[-1]), 
                tuple(ni + f"{_rat}ratio" for ni in names)
            )

        simulation_recirculation_plots(product_dict[prod], storage, ratios[:, 0])

    (_cost, _moisture, _temperature, moisture, temperature, _event_time, _quality, _constants) = storage

    logger.info(f"{divider} \nTotal cost {tree_util.tree_reduce(operator.add, _cost)}")
    logger.info(f"{divider} \nOptimal objective {pformat(_cost)}")
    logger.info(f"{divider} \nEvent times {pformat(_event_time)}")
    
    return _cost, _moisture, _temperature, moisture, temperature, _event_time, _quality

(
    cost_train_recir, moisture_scaled_recir, temperature_scaled_recir, 
    moisture_recir, temperature_recir, event_recir, quality_recir
) = simulation_results_recirculation(product_list, product_dict, constants, target_scaled, target_mask, 
                       controls, solid_moisture_optimal_mapped, solid_moisture_sigma_optimal_mapped, solid_temperature_init,
                       names = ("EventSolid", "EventSolidMesh", "EventAirMesh", "EventCoeffMesh"))
"""

