from typing import Tuple
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import tree_util

from .single_zone_event import oven_zone


@partial(jax.jit, static_argnums = (3, 5))
def oven_dynamics(xinit : jnp.ndarray, parameters : dict, controls : dict, constants : dict, 
                  reverse_zone : int, nzones : int, recirculation : bool = False) -> Tuple[jnp.ndarray]:

    _oven_zone = partial(oven_zone, constant = constants, recirculation = recirculation)

    def body_func(carry, i):
        
        # reverse the states to account for reversing the direction of air flow
        _carry = jax.lax.cond(
            i >= reverse_zone, 
            lambda : tree_util.tree_map(lambda _z : _z[::-1], carry), 
            lambda : carry
        )
        solution = _oven_zone(_carry, parameters(i - 1).params, controls(i - 1).inputs, first_zone = i == 1)
        
        # reverse the states back
        _solution = (_ms, _ts, _cs, *_) = jax.lax.cond(
            i >= reverse_zone, 
            lambda : tree_util.tree_map(lambda _z : _z[..., ::-1], solution), 
            lambda : solution
        )
        return jnp.column_stack((_ms[-1], _ts[-1], _cs[-1])), _solution

    _, (*solution, _event_times) = jax.lax.scan(body_func, xinit, jnp.arange(1, nzones + 1)) 
    
    # post processing event times
    event_times = jnp.sum(jax.vmap(lambda x : jnp.clip(x, a_min = 0, a_max = controls["residence_time"]))(_event_times), axis = 0)

    return *solution, event_times # shape = [nzone, time_span, ny]