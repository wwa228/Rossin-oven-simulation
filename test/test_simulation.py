import os
from pprint import pformat
import pickle
import operator
from typing import Tuple
from pathlib import Path



import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import flatten_util, tree_util
import jax.random as jrandom

from oven.oven_dynamics import oven_dynamics
from oven.phy_props import _mass_transfer_coefficient, _heat_transfer_coefficient, moisture_content
from oven.scaling import choose_scaling
from oven.data_structures import Controls, SimulationData, Constants, OdeKwargs, TrainingResults
from oven.plots import plot
from oven.utils import get_coefficients

nzones = 5
reverse_zone = 3
rtol = 1e-6
atol = 1e-8
mxstep = 10_000
recirculation = False

def convert_units(y):
    x = (((y - 650) / (485 - 650)) * 22000)
    w = ((1 - (y - 650) / (485 - 650)) * 32000)
    result = (x + w) / (24 * 69)
    return result

init_velocity_air = jnp.array([485, 485, 485, 540.11, 568.61])  # RPM
init_temperature_air = jnp.array([490.21, 495.07, 521.69, 554.96, 548.06])  # F
line_speed = 181.11  # ft / min
init_moisture_air = jnp.array([0.01, 0.01, 0.01, 0.01, 0.01])

converted_velocity_air = convert_units(init_velocity_air) * 5.08e-3 * 2


(moisture_max, moisture_min, temperature_max, temperature_min), (scale_states, unscale_states, _, _) = choose_scaling("max")

scaled_moisture_content = lambda T, density_air, relative_humidity=1: scale_states(
    moisture_content(unscale_states(T, temperature_max, temperature_min), density_air, relative_humidity),
    moisture_max, moisture_min
)
scaled_mass_transfer_coefficient = lambda x, p, constant: _mass_transfer_coefficient(x, p, constant)
scaled_heat_transfer_coefficient = lambda x, p, constant: _heat_transfer_coefficient(x, p, constant)
scaled_reaction_rate = lambda A, k0, T: k0 * jnp.exp(-A / 1.98 / unscale_states(T, temperature_max, temperature_min))

constants = Constants(
    radius_particle=4e-6,
    voidage=1 - 0.475 * 16.018 / 2500,
    enthalpy_vaporization_water=2e6,
    specific_heat_capacity_air=1047,
    density_particle=2500,
    specific_heat_capacity_solid=800,
    product_height=0.1,
    density_product=0.475 * 16.018,
    ny=20,
    nzones=nzones,
    reverse_zone=reverse_zone,
    ntimes=round(24 * 60 / line_speed) + 1,
    equilibrium_moisture=scale_states(0.008, moisture_max, moisture_min),
    moisture_max=moisture_max,
    moisture_min=moisture_min,
    temperature_max=temperature_max,
    temperature_min=temperature_min,
    scaled_moisture_content=scaled_moisture_content,
    scaled_mtc=scaled_mass_transfer_coefficient,
    scaled_htc=scaled_heat_transfer_coefficient,
    scaled_k=scaled_reaction_rate,
    density_air=lambda X, T: 1,
    ode_kwargs=OdeKwargs(rtol=rtol, atol=atol, mxstep=mxstep, reuse=True)
)

controls = Controls(inputs={
    "init_temperature_air": scale_states((init_temperature_air - 32) * 5 / 9 + 273 - 60, temperature_max, temperature_min),
    "init_velocity_air": converted_velocity_air,
    "init_moisture_air": scale_states(init_moisture_air, moisture_max, moisture_min),
    "residence_time": jnp.atleast_1d(jnp.round(24 * 60. / line_speed)),
    "recirculation_ratio": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
})


params_path = Path(__file__).resolve().parent / ".." / "log" / "R13SSR4" / "2024-12-27 20_47_09.835171" / "saved_params" / "params"
params_path = params_path.resolve()

with open(params_path, "rb") as file:
    meta_data = pickle.load(file)
    parameters = meta_data["meta_parameters"]["parameters"]
    solid_moisture_mean = meta_data["solid_moisture_mean"][0]
    solid_moisture_sigma_optimal = meta_data["solid_moisture_sigma"][0]

def objective(params: dict, solid_moisture_init: jnp.ndarray, solid_moisture_sigma: jnp.ndarray, constants: Constants,
              controls: Controls, solid_temperature_init: jnp.ndarray, recirculation: bool, key) -> Tuple[jnp.ndarray]:
    ny = constants.ny
    xinit = jnp.column_stack((
        solid_moisture_init * jnp.ones(ny),
        solid_temperature_init * jnp.ones(ny),
        jnp.zeros(shape=(ny,))
    ))

    params["reaction_rate"] = 2.33 * 10**7
    params["activation_energy"] = 20. * 10**3

    solution = oven_dynamics(xinit, params, controls, constants, reverse_zone=constants.reverse_zone,
                              nzones=constants.nzones, recirculation=recirculation)
    solution = tree_util.tree_map(jnp.vstack, solution)
    cost = tree_util.tree_reduce(operator.add,
                                  tree_util.tree_map(lambda z: jnp.mean((z - jnp.ones_like(z))**2), solution[:-1]))

    return cost, xinit, *solution

#############################################################################

key = jrandom.PRNGKey(10)
time_span = jnp.arange(controls["residence_time"][0] * nzones).flatten()
height_span = jnp.linspace(0, constants.product_height, constants.ny)
temperature_init = scale_states(jnp.array([330.]), temperature_max, temperature_min)

solution = (
    *_, moisture_solid, temperature_solid, cure, moisture_air, temperature_air, ts_event
) = objective(parameters, solid_moisture_mean, solid_moisture_sigma_optimal,
              constants, controls, temperature_init, recirculation, key)

solution_meanp = objective(parameters, solid_moisture_mean + 2 * solid_moisture_sigma_optimal,
                           solid_moisture_sigma_optimal, constants, controls, temperature_init, recirculation, key)
solution_meanm = objective(parameters, solid_moisture_mean - 2 * solid_moisture_sigma_optimal,
                           solid_moisture_sigma_optimal, constants, controls, temperature_init, recirculation, key)


htc, mtc = get_coefficients(
    (moisture_solid, temperature_solid, moisture_air, temperature_air), parameters, constants,
    scaled_moisture_content, scaled_mass_transfer_coefficient, scaled_heat_transfer_coefficient
)

(_moisture_solid, _moisture_air, _moisture_solid_meanp, _moisture_air_meanp, _moisture_solid_meanm, _moisture_air_meanm) = tree_util.tree_map(
    lambda z: unscale_states(z, moisture_max, moisture_min),
    (moisture_solid, moisture_air, solution_meanp[2], solution_meanp[5], solution_meanm[2], solution_meanm[5])
)

(_temperature_solid, _temperature_air, _temperature_solid_meanp, _temperature_air_meanp, _temperature_solid_meanm, _temperature_air_meanm) = tree_util.tree_map(
    lambda z: unscale_states(z, temperature_max, temperature_min),
    (temperature_solid, temperature_air, solution_meanp[3], solution_meanp[6], solution_meanm[3], solution_meanm[6])
)

alist = [
    (_moisture_solid, _moisture_solid_meanp, _moisture_solid_meanm),
    (_temperature_solid, _temperature_solid_meanp, _temperature_solid_meanm),
    (_moisture_air, _moisture_air_meanp, _moisture_air_meanm),
    (_temperature_air, _temperature_air_meanp, _temperature_air_meanm)
]

plot(
    SimulationData(*map(jnp.stack, alist), ts_event,
                   *tree_util.tree_map(lambda z: jnp.tile(z, (3, 1, 1)), (htc, mtc))),
    TrainingResults(),
    (time_span, height_span, nzones, controls["residence_time"][0]),
    None,
    ("EventSolid", "EventSolidMesh", "EventAirMesh", "EventCoeffMesh")
)
