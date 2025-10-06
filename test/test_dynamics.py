import unittest
import operator
import pickle
from typing import Any, Tuple
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import tree_util, flatten_util
import jax.random as jrandom

from oven.oven_dynamics import oven_dynamics
from oven.phy_props import _mass_transfer_coefficient, _heat_transfer_coefficient, moisture_content
from oven.scaling import choose_scaling
from oven.data_structures import Constants, SimulationData, OdeKwargs, Controls, TrainingResults
from oven import plot
from oven.utils import get_coefficients


nzones = 5
reverse_zone = 3
rtol = 1e-6
atol = 1e-8
mxstep = 10_000
recirculation = False

init_velocity_air = jnp.array([0.097, 0.086, 0.115, 0.207, 0.207]) # (ft/min * constant) = m/sec
init_temperature_air = jnp.array([535., 535., 560., 565., 565.]) # F
line_speed = 102 # ft / min
init_moisture_air = jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]) 

(moisture_max, moisture_min, temperature_max, temperature_min), (scale_states, unscale_states, _, _) = choose_scaling("max")

scaled_moisture_content = lambda T, density_air, relative_humidity = 1 : scale_states(moisture_content(unscale_states(T, temperature_max, temperature_min), density_air, relative_humidity), moisture_max, moisture_min) # 
scaled_mass_transfer_coefficient = lambda x, p, constant: _mass_transfer_coefficient(x, p, constant)
scaled_heat_transfer_coefficient = lambda x, p, constant : _heat_transfer_coefficient(x, p, constant)
scaled_reaction_rate = lambda A, k0, T : k0 * jnp.exp( - A / 1.98 / unscale_states(T, temperature_max, temperature_min))


constants_reuse = Constants(
    radius_particle = 4e-6, # Radius of the particle in m,
    voidage = 1 - 0.475 * 16.018 / 2500, # "Bed porosity (1 - density of product / density of particle)"
    enthalpy_vaporization_water = 2e6, # Enthalpy of vaporization of water in
    specific_heat_capacity_air = 1047, # Specific heat capacity of the air in j/kg/k
    density_particle = 2500, # Density of the particle in kg/m3
    specific_heat_capacity_solid = 800, # Specific heat capacity of the particle in j/kg/K
    product_height = 0.1, # m
    density_product = 0.475 * 16.018, # (lb/ft3 * constant) = kg/m3

    ny = 20,
    nzones = nzones,
    reverse_zone = reverse_zone,
    ntimes = round(24 * 60 / line_speed) + 1, # residence time + 1
    equilibrium_moisture = scale_states(0.008, moisture_max, moisture_min), # Equilibrium moisture content in dry basis (kg water / kg of dry solid), 

    moisture_max = moisture_max,
    moisture_min = moisture_min,
    temperature_max = temperature_max,
    temperature_min = temperature_min,

    # Equations
    scaled_moisture_content = scaled_moisture_content,
    scaled_mtc = scaled_mass_transfer_coefficient,
    scaled_htc = scaled_heat_transfer_coefficient,
    scaled_k = scaled_reaction_rate,
    density_air = lambda X, T : 1, # Density of the air in kg/m3 

    # ode kwargs
    ode_kwargs = OdeKwargs(rtol = rtol, atol = atol, mxstep = mxstep, reuse = True))

constants = constants_reuse._replace(ode_kwargs = OdeKwargs(rtol = rtol, atol = atol, mxstep = mxstep, reuse = True))

controls = Controls(
    inputs = {
        "init_temperature_air" : scale_states((init_temperature_air - 32) * 5/9 + 273 - 60, temperature_max, temperature_min), # setpoint - line loss 
        "init_velocity_air" : init_velocity_air,
        "init_moisture_air" : scale_states(init_moisture_air, moisture_max, moisture_min),
        "residence_time" : jnp.atleast_1d(jnp.round(24 * 60. / line_speed)), # 24 ft per zone * 60 sec / linespeed
        "recirculation_ratio" : jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]), # must be between 0 to 1
    })

# Load optimal parameters
with open("log/R13SSR4/2024-12-27 20:47:09.835171/saved_params/params", "rb") as file :
    meta_data = pickle.load(file)
    parameters = meta_data["meta_parameters"]["parameters"]
    solid_moisture_mean = meta_data["solid_moisture_mean"][0]
    solid_moisture_sigma_optimal = meta_data["solid_moisture_sigma"][0]


def objective(params : dict, solid_moisture_init : jnp.ndarray, solid_moisture_sigma : jnp.ndarray, constants : Constants, 
                controls : Controls, solid_temperature_init : jnp.ndarray, recirculation : bool, key) -> Tuple[jnp.ndarray]:
    
    ny = constants.ny
    xinit = jnp.column_stack((
        solid_moisture_init * jnp.ones(ny), # + solid_moisture_sigma * jrandom.normal(key, shape = (ny, )), # moisture 
        solid_temperature_init * jnp.ones(ny), # temperature
        jnp.zeros(shape = (ny, )) # cure 
    ))

    params["reaction_rate"] = 2.33 * 10**7 # Do not optimize this yet
    params["activation_energy"] = 20. * 10**3 # Do not optimize this yet

    # oven_dynamics discards the provided initial condition (It is assumed that the initial condition are the values of states at t = -1)
    solution = oven_dynamics(xinit, params, controls, constants, reverse_zone = constants.reverse_zone, nzones = constants.nzones, recirculation = recirculation)
    solution = tree_util.tree_map(jnp.vstack, solution)
    cost = tree_util.tree_reduce(operator.add, tree_util.tree_map(lambda z : jnp.mean((z - jnp.ones_like(z))**2), solution[:-1]))

    # solution = (moisture_solid, temperature_solid, cure, moisture_air, temperature_air, t_event)
    return cost, xinit, *solution

#############################################################################

key = jrandom.PRNGKey(10)
time_span = jnp.arange(controls["residence_time"][0] * nzones).flatten()
height_span = jnp.linspace(0, constants.product_height, constants.ny)
temperature_init = scale_states(jnp.array([330.]), temperature_max, temperature_min)

solution = (
    *_, moisture_solid, temperature_solid, cure, moisture_air, temperature_air, ts_event
) = objective(parameters, solid_moisture_mean, solid_moisture_sigma_optimal, constants_reuse, controls, temperature_init, recirculation, key)


class TestDynamics(unittest.TestCase):
    
    def test_oven_dynamics(self):
        
        desired_shape = (len(time_span), len(height_span))
        self.assertTrue(tree_util.tree_reduce(operator.mul, tree_util.tree_map(lambda z : z.shape == desired_shape, (moisture_solid, temperature_solid, cure, moisture_air, temperature_air))))

    def test_plots(self):

        # mean +- 2 * sigma
        solution_meanp = objective(parameters, solid_moisture_mean + 2 * solid_moisture_sigma_optimal, solid_moisture_sigma_optimal, constants_reuse, controls, temperature_init, recirculation, key)
        solution_meanm = objective(parameters, solid_moisture_mean - 2 * solid_moisture_sigma_optimal, solid_moisture_sigma_optimal, constants_reuse, controls, temperature_init, recirculation, key)

        htc, mtc = get_coefficients(
            (moisture_solid, temperature_solid, moisture_air, temperature_air), parameters, constants_reuse, 
            scaled_moisture_content, scaled_mass_transfer_coefficient, scaled_heat_transfer_coefficient
            )

        # scale back states
        (_moisture_solid, _moisture_air, _moisture_solid_meanp, _moisture_air_meanp, _moisture_solid_meanm, _moisture_air_meanm
            )  = tree_util.tree_map(lambda z : unscale_states(z, moisture_max, moisture_min), (moisture_solid, moisture_air, solution_meanp[2], solution_meanp[5], solution_meanm[2], solution_meanm[5]))
        (_temperature_solid, _temperature_air, _temperature_solid_meanp, _temperature_air_meanp, _temperature_solid_meanm, _temperature_air_meanm
            ) = tree_util.tree_map(lambda z : unscale_states(z, temperature_max, temperature_min), (temperature_solid, temperature_air, solution_meanp[3], solution_meanp[6], solution_meanm[3], solution_meanm[6]))

        # Stack entires for values (mean +- 2 sigma)
        alist = [
            (_moisture_solid, _moisture_solid_meanp, _moisture_solid_meanm), 
            (_temperature_solid, _temperature_solid_meanp, _temperature_solid_meanm), 
            (_moisture_air, _moisture_air_meanp, _moisture_air_meanm), 
            (_temperature_air, _temperature_air_meanp, _temperature_air_meanm)
        ]

        plot(
            SimulationData(*map(jnp.stack, alist), ts_event, *tree_util.tree_map(lambda z : jnp.tile(z, (3, 1, 1)), (htc, mtc))),
            TrainingResults(),
            (time_span, height_span, nzones, controls["residence_time"][0]),
            None,
            ("EventSolid", "EventSolidMesh", "EventAirMesh", "EventCoeffMesh")
        )

    def test_gradients(self):
        
        # Gradients using reverse-mode autodiff
        obj = lambda x, p, c : objective(p, x, solid_moisture_sigma_optimal, constants, c, temperature_init, recirculation, key)[0]
        _obj_reuse = lambda x, p, c : objective(p, x, solid_moisture_sigma_optimal, constants_reuse, c, temperature_init, recirculation, key)[0]
        
        base, (rev_reuse_x, rev_reuse_p, rev_reuse_c) = jax.value_and_grad(_obj_reuse, argnums = (0, 1, 2))(solid_moisture_mean, parameters, controls)
        
        # checking gradients using finite difference
        def fd(eps):
            vars, unravel = flatten_util.ravel_pytree((solid_moisture_mean, parameters, controls))
            grads = jax.vmap(lambda v : (_obj_reuse(*unravel(vars + eps * v)) - base) / eps)(jnp.eye(len(vars)))
            return unravel(grads)

        eps = 1e-3
        fd_x, fd_p, fd_c = fd(eps)

        print("autodiff (reuse) x", rev_reuse_x)
        print("fd x", fd_x)
        print("autodiff (reuse) p", rev_reuse_p)
        print("fd p", fd_p)
        print("autodiff (reuse) c", rev_reuse_c)
        print("fd c", fd_c)
        # self.assertTrue(tree_util.tree_reduce(operator.mul, tree_util.tree_map(partial(jnp.allclose, atol = 100 * eps), rev_reuse_x, fd_x)))
        # self.assertTrue(tree_util.tree_reduce(operator.mul, tree_util.tree_map(partial(jnp.allclose, atol = 100 * eps), rev_reuse_p, fd_p)))
        # self.assertTrue(tree_util.tree_reduce(operator.mul, tree_util.tree_map(partial(jnp.allclose, atol = 100 * eps), rev_reuse_c, fd_c)))

        # Checking Hessians with respect to controls. Note that set use_inverse as False for computing higher order derivatives
        c_flat, unravel = flatten_util.ravel_pytree(controls)
        hess = jax.hessian(lambda c : obj(solid_moisture_mean, parameters, unravel(c)))(c_flat)

        def hessian_fd(eps):
            _grad = jax.grad(lambda c : obj(solid_moisture_mean, parameters, unravel(c)))
            return jax.vmap(lambda v : (_grad(c_flat + v * eps) - _grad(c_flat - v * eps)) / 2 / eps)(jnp.eye(len(c_flat)))

        hess_fd = hessian_fd(eps)
        print("Hessian autodiff", hess)
        print("Hessian fd", hess_fd)