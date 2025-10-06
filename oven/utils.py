from typing import Tuple, Callable, List, Any
import functools
import operator
import os
import pickle

from typing import NamedTuple



import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import flatten_util, tree_util

Pytree = Any

class SVDResult(NamedTuple):
    u: jnp.ndarray
    s: jnp.ndarray
    vh: jnp.ndarray

def flatten_output(afunc, unravel_first_arg):
    @functools.wraps(afunc)
    def _afunc(*args):
        x, *args = args
        return jax.flatten_util.ravel_pytree(afunc(unravel_first_arg(x), *args))[0]
    return _afunc

def conjugate_gradient(f : Callable, z_guess : jnp.ndarray) -> jnp.ndarray :
    # solves root finding problem : f(z) = 0 using conjugate gradient method. 
    # Every linear solve uses conjugate gradient method and therefore only uses hvp/vhp

    vjp_f = lambda tangents, primals : jax.vjp(f, primals)[-1](tangents)[0]
    
    def body_fun(val):
        dval, _ = jax.scipy.sparse.linalg.cg(functools.partial(vjp_f, primals = val), f(val))
        return val - dval
    
    def cond_fun(val):
        return jnp.linalg.norm(f(val)) > 1e-5
    
    def scan_fun(carry, xs):
        carry = jax.lax.cond(cond_fun(carry), body_fun, lambda carry : carry, carry)
        return carry, None

    z, _ = jax.lax.scan(scan_fun, z_guess, xs = None, length = 20.)
    return z

def newton_method(f : Callable, z_guess : jnp.ndarray) -> jnp.ndarray :
    # solves root finding problem : f(z) = 0 using newtons method 
    # Every linear solve uses explicit hessian 

    # function is only reverse mode autodiff compatible
    grad_f = jax.jacrev(f)
    
    def body_fun(val):
        dval = jnp.linalg.solve(grad_f(val), f(val))
        return val - dval
    
    def cond_fun(val):
        return jnp.linalg.norm(f(val)) > 1e-5
    
    def scan_fun(carry, xs):
        carry = jax.lax.cond(cond_fun(carry), body_fun, lambda carry : carry, carry)
        return carry, None

    z, _ = jax.lax.scan(scan_fun, z_guess, xs = None, length = 20.)
    return z

def newton_method_inverse(f : Callable, z_guess : jnp.ndarray) -> Tuple[jnp.ndarray] :
    # Solves root finding problem : f(z) = 0 using newtons method 
    # SVD decomposition of the Hessian is calculated and returned for reuse in reverse-mode autodiff

    grad_f = jax.jacfwd(f)
    # https://github.com/jax-ml/jax/issues/508
    # Our case m == n. Therefore no issues
    inverse = jnp.linalg.svd(grad_f(z_guess), full_matrices = False)    

    def body_fun(val):
        val, (u, s, vh) = val
        new_val = val - vh.T @ ((u.T @ f(val)) / s)
        inverse = jnp.linalg.svd(grad_f(new_val), full_matrices = False)
        return new_val, inverse
    
    def cond_fun(val):
        val, _ = val
        return jnp.linalg.norm(f(val)) > 1e-5
    
    def scan_fun(carry, xs):
        carry = jax.lax.cond(cond_fun(carry), body_fun, lambda carry : carry, carry)
        return carry, None

    (z, inverse), _ = jax.lax.scan(scan_fun, (z_guess, inverse), xs = None, length = 20.)
    return z, inverse


@functools.partial(jax.custom_vjp, nondiff_argnums = (0, 1))
def _root_finding_rev(solver : Callable, f : Callable, z : jnp.ndarray, p : Pytree) -> jnp.ndarray :
    # Reverse mode auto diff compatible root finding problem with f : Rn -> Rn
    _f = lambda z : f(z, p)
    return solver(_f, z)

def _root_finding_rev_fwd(solver, f, z, p):
    z_star = _root_finding_rev(solver, f, z, p)
    return z_star, (z_star, p)

def _root_finding_rev_bwd(solver, f, res, gdot):
    
    z_star, p = res
    _, vjp_x = jax.vjp(lambda x : f(x, p), z_star)
    _, vjp_p = jax.vjp(lambda p : f(z_star, p), p)
    
    return None, *vjp_p(solver(lambda x : vjp_x(x)[0] + gdot, jnp.zeros_like(z_star)))

_root_finding_rev.defvjp(_root_finding_rev_fwd, _root_finding_rev_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums = (0, 1))
def _root_finding_reuse(solver : Callable, f : Callable, z : jnp.ndarray, p : Pytree) -> jnp.ndarray :
    # Reverse mode auto diff compatible root finding problem
    _f = lambda z : f(z, p) 
    return solver(_f, z)

def _root_finding_reuse_fwd(solver, f, z, p):
    z_star, inverse = _root_finding_reuse(solver, f, z, p)
    return (z_star, inverse), (z_star, p, inverse)

def _root_finding_reuse_bwd(solver, f, res, gdot):
    
    gdot, _ = gdot
    z_star, p, (u, s, vh) = res
    _, vjp_p = jax.vjp(lambda p : f(z_star, p), p)
    
    return None, *vjp_p( - ((gdot @ vh.T) / s) @ u.T)

_root_finding_reuse.defvjp(_root_finding_reuse_fwd, _root_finding_reuse_bwd)


def root_finding_rev(f : Callable, z : Pytree, p : Pytree, reuse_inverse : bool = False) -> Pytree : 
    # Reverse-mode autodiff compatible root finding problem 
    # Note that reusing inverse incorrectly predicts higher order derivatives (> 1)

    z_flat, unravel = flatten_util.ravel_pytree(z)
    _f = flatten_output(f, unravel_first_arg = unravel)
    
    z_opt = _root_finding_reuse(newton_method_inverse, _f, z_flat, p)[0] if reuse_inverse else _root_finding_rev(newton_method, _f, z_flat, p)
    return unravel(z_opt)


@functools.partial(jax.custom_jvp, nondiff_argnums = (0, 1))
def _root_finding_fwd(solver : Callable, f : Callable, z : jnp.ndarray, p : Pytree) -> Pytree :
    # Forward-mode auto diff compatible root finding problem with f : Rn -> Rn
    return solver(lambda z : f(z, p), z)

@_root_finding_fwd.defjvp
def _root_finding_fwd_fwd(solver, f, primals, tangents):
    z, p = primals
    _, pdot = tangents
    
    zstar = _root_finding_fwd(solver, f, z, p)
    # computing the jacobian is cheaper (in this case) than solving another root-finding problem
    # tangents_out = solver(lambda v : jax.jvp(lambda z, p : f(z, p), (zstar, p), (v, pdot))[-1], jnp.zeros_like(zstar))
    tangents_out = jnp.linalg.solve(jax.jacfwd(f)(zstar, p), - jax.jvp(lambda p : f(zstar, p), (p, ), (pdot, ))[-1])
    return zstar, tangents_out


@functools.partial(jax.custom_jvp, nondiff_argnums = (0, 1))
def _root_finding_fwd_reuse(solver : Callable, f : Callable, z : jnp.ndarray, p : Pytree) -> Pytree :
    # Forward mode auto diff compatible root finding problem with f : Rn -> Rn
    return solver(lambda z : f(z, p), z)

@_root_finding_fwd_reuse.defjvp
def _root_finding_fwd_reuse_fwd(solver, f, primals, tangents):
    z, p = primals
    _, pdot = tangents

    zstar, (u, s, vh) = _root_finding_fwd_reuse(solver, f, z, p)
    tangents_out = - vh.T @ ((u.T @ jax.jvp(lambda _p : f(zstar, _p), (p, ), (pdot, ))[-1]) / s)
    
    return (zstar, (u, s, vh)), (tangents_out, (jnp.zeros_like(u), jnp.zeros_like(s), jnp.zeros_like(vh)))

def root_finding_fwd(f : Callable, z : Pytree, p : Pytree, reuse_inverse : bool = False) -> Pytree : 
    # Forward- and reverse-mode autodiff compatible root finding problem. 
    # Note that reusing inverse incorrectly predicts higher order derivatives (> 1)

    z_flat, unravel = flatten_util.ravel_pytree(z)
    _f = flatten_output(f, unravel_first_arg = unravel)
    
    z_opt = _root_finding_fwd_reuse(newton_method_inverse, _f, z_flat, p)[0] if reuse_inverse else _root_finding_fwd(newton_method, _f, z_flat, p)
    return unravel(z_opt)


def step_interpolation(t : jnp.ndarray, ts : jnp.ndarray) -> int :
    # step interpolation for monotonically increasing sample points
    _ts = jnp.where(ts <= t, ts, -jnp.inf)
    ind = jnp.argmax(_ts)
    return ind

def check_dir(dir : str) -> None :

    if not os.path.exists(dir):
        os.makedirs(dir)

def save_params(obj, dir : str, filename : str) -> None :
    
    check_dir(dir)

    with open(os.path.join(dir, filename), "wb") as file:
            pickle.dump(obj, file)


# getting heat and mass transfer coefficients
def get_coefficients(states : Tuple[jnp.ndarray], parameters : dict, constants : dict, 
        moisture_content : Callable, mass_transfer_coefficient : Callable, heat_transfer_coefficient : Callable) -> Tuple[jnp.ndarray] :
    
    def _get_coefficients(x):
        ms, ts, *_ = x
        ye = moisture_content(ts, 1)
        return (jnp.where(ms < parameters["critical_moisture"], 
                            mass_transfer_coefficient((*x, ye), parameters["mass_transfer_coefficient_falling"], constants), 
                            mass_transfer_coefficient((*x, ye), parameters["mass_transfer_coefficient_constant"], constants)
                    ), 
                jnp.where(ms < parameters["critical_moisture"], 
                          heat_transfer_coefficient((*x, ye), parameters["heat_transfer_coefficient_falling"], constants),
                          heat_transfer_coefficient((*x, ye), parameters["heat_transfer_coefficient_constant"], constants)
                    )
                )

    coeffs = jax.vmap(lambda z : jax.vmap(_get_coefficients)(z))(states)
    return jax.tree_util.tree_map(jnp.squeeze, coeffs)


def product_quality(moisture_solid : List[jnp.ndarray], temperature_solid : List[jnp.ndarray], moisture_air : List[jnp.ndarray], temperature_air : List[jnp.ndarray], 
                    recirculation_ratio : jnp.ndarray, temperature_setpoint : jnp.ndarray, velocity_air : jnp.ndarray, 
                    reverse_zone : int, temperature_ambient : float, specific_heat_capacity_air : float) -> Tuple[dict] :
    
    assert (len(temperature_solid) == len(temperature_air) == len(temperature_setpoint) == len(recirculation_ratio)
            == len(velocity_air)), "All arrays should be of same length"

    # calculating energy consumtion per zone
    energy = {}
    for zone, (temp_air, temp_sp, ratio, vel) in enumerate(
        zip(temperature_air, temperature_setpoint, recirculation_ratio, velocity_air)) : 
    
        mean_temp = jnp.mean(temp_air[:, -1]) if zone + 1 < reverse_zone else jnp.mean(temp_air[:, 0])
        energy[f"zone{zone + 1}"] = (temp_sp - ratio * mean_temp + (1 - ratio) * temperature_ambient) * specific_heat_capacity_air * vel

    """
    # calculating cure index
    def traj_cure(temperature, delta_t = 1):
        _integral = jnp.cumsum(delta_t * jnp.exp(- 20 / 0.00198 / temperature) * 2.33 * 10**7)
        return 1 - (1 / jnp.exp(_integral))
    
    trajectory_cure = jax.vmap(traj_cure)(jnp.vstack(temperature_solid).T)
    cure = {f"zone{zone + 1}" : _cure[-1] for zone, _cure in zip(range(nzones), jnp.array_split(trajectory_cure.T, nzones))}
    """
    return energy
    

def kl_div(mean1, sigma1, mean2, sigma2):
    # https://statproofbook.github.io/P/norm-kl.html
    # kl divergence of two normal distributions
    # sigma = standard deviation
    return 0.5 * ((mean2 - mean1)**2 / sigma2**2 + (sigma1 / sigma2)**2 - 2 * jnp.log(sigma1 / sigma2) - 1)


def kl_div_multi(mean1, L, mean2, D):
    # https://statproofbook.github.io/P/mvn-kl.html
    # KL divergence of two multivariate normal distributions
    # Finds the KL divergence of (Q / P) with Q ~ N(mean1, L@L.T) and P ~ N(mean2, D)
    # where L is a lower triangular matrix and D is a diagonal matrix
    mean_diff = mean2 - mean1
    _diag = jnp.diag(D)
    return 0.5 * ((mean_diff / _diag) @ mean_diff + jnp.sum(jnp.diag(L @ L.T) / _diag) - 2 * jnp.log(jnp.prod(jnp.diag(L))) + jnp.log(jnp.prod(_diag)))


@functools.partial(jax.jit, static_argnums = (1, ))
def create_tril(x, n):
    # creates a lower triangular matrix using flattened array 
    # diagonal elements are exponentiated
    _splits = jnp.array_split(x, jnp.cumsum(jnp.arange(n, 1, -1)))
    return jax.tree_util.tree_reduce(lambda accum, _x : accum + jnp.diag(_x, len(_x) - n), _splits[1:], jnp.diag(jnp.exp(_splits[0])))