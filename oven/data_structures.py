from typing import NamedTuple, Optional, Callable, List

import jax.numpy as jnp
from jax import tree_util

#############################################################################
## Datastructures used for parameter estimation

class Controls(NamedTuple):
    inputs : dict
    
    def __call__(self, i):
        return tree_util.tree_map(lambda x : jnp.atleast_1d(x[i]) if len(x) > 1 else x, self)
    
    def __setitem__(self, key, value):
        self.inputs[key] = value

    def __getitem__(self, key):
        return self.inputs[key]


class Parameters(NamedTuple):
    params : dict

    def __call__(self, i):
        return tree_util.tree_map(lambda x : jnp.atleast_1d(x[i]) if x.ndim == 1 and len(x) > 1 else x, self)
    
    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value


class Moisture(NamedTuple):
    states : dict

    def __getitem__(self, key):
        return self.states[key]
    
    def __setitem__(self, key, value):
        self.states[key] = value


class Temperature(Moisture):
    states : dict


class Cost(NamedTuple):
    elbo : dict
    cost : dict

    def __getitem__(self, key):
        return (self.elbo[key], self.cost[key])
    
    def __setitem__(self, key, value):
        self.cost[key] = value[1]
        self.elbo[key] = value[0]


class EventTime(NamedTuple):
    event_time : dict

    def __getitem__(self, key):
        return self.event_time[key]
    
    def __setitem__(self, key, value):
        self.event_time[key] = value


class OdeKwargs(NamedTuple):
    rtol : float
    atol : float
    mxstep : int
    reuse : bool


class Constants(NamedTuple):
    radius_particle : float
    voidage :float
    enthalpy_vaporization_water : float
    specific_heat_capacity_air : float
    product_height : float
    density_particle : float
    specific_heat_capacity_solid : float
    equilibrium_moisture : float
    moisture_max : float 
    moisture_min : float
    temperature_max : float
    temperature_min : float
    density_product : float
    # heat and mass transfer equations
    scaled_moisture_content : Callable
    scaled_mtc : Callable
    scaled_htc : Callable
    scaled_k : Callable
    density_air : Callable
    ny : int
    nzones : int
    reverse_zone : int
    ntimes : int
    ode_kwargs : OdeKwargs


#############################################################################
## Datastructures used for plotting

class SimulationData(NamedTuple):
    moisture_solid : jnp.ndarray
    temperature_solid : jnp.ndarray
    moisture_air : jnp.ndarray
    temperature_air : jnp.ndarray
    event_times : Optional[jnp.ndarray] = None
    heat_transfer_coefficient : Optional[jnp.ndarray] = None
    mass_transfer_coefficient : Optional[jnp.ndarray] = None


class TrainingResults(NamedTuple):
    training_loss : Optional[List[jnp.ndarray]] = None
    testing_loss : Optional[List[jnp.ndarray]] = None
    learning_rate : Optional[List[jnp.ndarray]] = None


class DistributionData(NamedTuple):
    mean : Optional[jnp.ndarray] = None
    mean_plus : Optional[jnp.ndarray] = None
    mean_minus : Optional[jnp.ndarray] = None
    title : Optional[str] = None


class Quality(NamedTuple):
    quality : dict

    def __getitem__(self, key):
        return self.quality[key]
    
    def __setitem__(self, key, value):
        self.quality[key] = value


#############################################################################
## Datastructures used for Targets

class Product(NamedTuple):
    controls : dict
    targets : dict
    targets_mask : dict
