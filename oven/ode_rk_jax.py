from functools import partial
import operator
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax, grad
from jax import flatten_util, tree_util
from jax.experimental.ode import odeint as odeint_jax
from scipy.integrate import solve_ivp
import diffrax # to check for sensitivities

"""
https://github.com/jacobjinkelly/easy-neural-ode/blob/master/lib/ode.py
"""

def while_loop_scan(body_func, cond_func, carry, mxiter = 100):
    # customized only for odes
    def true_func(carry):
        _carry, _cond = carry
        next_carry = body_func(_carry)
        next_cond = cond_func(next_carry)
        return (next_carry, next_cond), (next_carry, next_cond) # should be the value that is recorded

    def false_func(carry):
        carry, _ = carry
        return (carry, jnp.array(0)), (carry, jnp.array(0))

    def scan_func(carry, i):
        _carry, _cond = carry
        return lax.cond(_cond, true_func, false_func, carry) 

    final, solution = lax.scan(scan_func, (carry, jnp.array(1)), None, length = mxiter)
    
    def _reached():
        # catch if mxiters are sufficient or not
        jax.debug.print("False condition was not met. Conider increasing the mxiters")
        jax.debug.breakpoint()

    lax.cond(final[-1] == 1, _reached, lambda : None)
    # "False condition was not met. Conider increasing the mxiters"
    return solution  # Do not append the initial solution. returns all trajectory and accepted/rejected steps

def interp_fit_dopri(y0, y1, k, dt):
    # Fit a polynomial to the results of a Runge-Kutta step.
    dps_c_mid = jnp.array([
        6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2,
        -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
        -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2])
    y_mid = y0 + dt*jnp.dot(dps_c_mid, k)
    return jnp.array(fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt))

def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
    a = -2.*dt*dy0 + 2.*dt*dy1 -  8.*y0 -  8.*y1 + 16.*y_mid
    b =  5.*dt*dy0 - 3.*dt*dy1 + 18.*y0 + 14.*y1 - 32.*y_mid
    c = -4.*dt*dy0 +    dt*dy1 - 11.*y0 -  5.*y1 + 16.*y_mid
    d = dt * dy0
    e = y0
    return a, b, c, d, e

def runge_kutta_step(afunc, y0, f0, t0, dt):
    # Dpri5 Butcher Table
    c = jnp.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1., 0])
    a = jnp.array([
        [1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
    ])
    b = jnp.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    b_error = jnp.array([
        35 / 384 - 1951 / 21600, 0, 500 / 1113 - 22642 / 50085,
        125 / 192 - 451 / 720, -2187 / 6784 - -12231 / 42400,
        11 / 84 - 649 / 6300, -1. / 60.
        ])

    def body_func(k, i):
        t1 = t0 + dt*c[i - 1]
        y1 = y0 + dt*jnp.dot(a[i - 1], k)
        f1 = afunc(y1, t1)
        return k.at[i].set(f1), i

    k = jnp.zeros(shape = (7, len(y0))).at[0].set(f0)
    k, _ = lax.scan(body_func, k, jnp.arange(1, 7))
    y1 = y0 + dt*jnp.dot(b, k)
    f1 = k[-1]
    y1_err = dt*jnp.dot(b_error, k)

    return y1, y1_err, k, f1

def runge_kutta45_step(afunc, y0, f0, t0, dt):
    # Dpri5 Butcher Table
    c = jnp.array([0, 1/2, 1/2, 1.])
    a = jnp.array([
        [0, 0, 0, 0],
        [1/2, 0, 0, 0],
        [0, 1/2, 0, 0],
        [0, 0, 1., 0],
    ])
    b = jnp.array([1/6, 1/3, 1/3, 1/6])
    
    def body_func(k, i):
        t1 = t0 + dt*c[i]
        y1 = y0 + dt*jnp.dot(a[i], k)
        f1 = afunc(y1, t1)
        return k.at[i].set(f1), i

    k = jnp.zeros(shape = (4, len(y0))).at[0].set(f0)
    k, _ = lax.scan(body_func, k, jnp.arange(1, 4))
    y1 = y0 + dt*jnp.dot(b, k)
    f1 = afunc(y1, dt + t0)
    
    return y1, 2, k, f1

def initial_step_size(afunc, t0, y0, order, rtol, atol, f0):
    # Algorithm from:
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    scale = atol + jnp.abs(y0) * rtol
    d0 = jnp.linalg.norm(y0 / scale)
    d1 = jnp.linalg.norm(f0 / scale)

    h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

    y1 = y0 + h0 * f0
    f1 = afunc(y1, t0 + h0)
    d2 = jnp.linalg.norm((f1 - f0) / scale) / h0

    h1 = jnp.where((d1 <= 1e-15) & (d2 <= 1e-15),
                jnp.maximum(1e-6, h0 * 1e-3),
                (0.01 / jnp.max(d1 + d2)) ** (1. / (order + 1.)))

    return jnp.minimum(100. * h0, h1)

def mean_error_ratio(error_estimate, rtol, atol, y0, y1):
    err_tol = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
    err_ratio = error_estimate / err_tol.astype(error_estimate.dtype)
    return jnp.sqrt(jnp.mean(err_ratio**2))

def optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0,
                      dfactor=0.2, order=5.0):
    """Compute optimal Runge-Kutta stepsize."""
    dfactor = jnp.where(mean_error_ratio < 1, 1.0, dfactor)

    factor = jnp.minimum(ifactor,
                        jnp.maximum(mean_error_ratio**(-1.0 / order) * safety, dfactor))
    return jnp.where(mean_error_ratio == 0, last_step * ifactor, last_step * factor)

"""
def odeint_rk_grid(afunc, xinit, time_span, parameters, dt = 0.01, mxiter = 200):
    # fixed step size rk45 solver. Forward mode and reverse mode compatible with scan based while loop
    _afunc = lambda x, t : afunc(x, t, parameters)

    def scan_func(carry, target_t):

        def cond_func(carry):
            _, _, t0 = carry
            return lax.cond(t0 < target_t, lambda : jnp.array(1), lambda : jnp.array(0))

        def body_func(carry):
            y0, f0, t0 = carry
            _dt = jnp.minimum(dt, jnp.abs(target_t - t0))
            y1, _, _, f1 = runge_kutta_step(_afunc, y0, f0, t0, _dt)
            return (y1, f1, t0 + _dt)

        state, _ = while_loop_scan(body_func, cond_func, carry, mxiter = mxiter)
        
        # return the last element of arrays in the carry tuple
        state = tree_util.tree_map(lambda x : x[-1], state)
        return state, state

    _, solution = lax.scan(scan_func, (xinit, _afunc(xinit, time_span[0]), time_span[0]), time_span[1:])

    # append initial condition    
    return jnp.vstack((xinit, solution[0]))

def odeint_rk45_grid(afunc, xinit, time_span, parameters, dt = 0.01, mxiter = 200):
    # fixed step size rk45 solver. Forward mode and reverse mode compatible with scan based while loop
    _afunc = lambda x, t : afunc(x, t, parameters)

    def scan_func(carry, target_t):

        def cond_func(carry):
            _, _, t0 = carry
            return lax.cond(t0 < target_t, lambda : jnp.array(1), lambda : jnp.array(0))

        def body_func(carry):
            y0, f0, t0 = carry
            _dt = jnp.minimum(dt, jnp.abs(target_t - t0))
            y1, _, _, f1 = runge_kutta45_step(_afunc, y0, f0, t0, _dt)
            return (y1, f1, t0 + _dt)

        state, _ = while_loop_scan(body_func, cond_func, carry, mxiter = mxiter)
        
        # return the last element of arrays in the carry tuple
        state = tree_util.tree_map(lambda x : x[-1], state)
        return state, state

    _, solution = lax.scan(scan_func, (xinit, _afunc(xinit, time_span[0]), time_span[0]), time_span[1:])

    # append initial condition    
    return jnp.vstack((xinit, solution[0]))
"""

def ravel_first_arg(f, unravel):
    return ravel_first_arg_(jax.linear_util.wrap_init(f), unravel).call_wrapped

# @jax.linear_util.transformation
def ravel_first_arg_(unravel, y_flat, *args):
    y = unravel(y_flat)
    ans = yield (y,) + args, {}
    ans_flat, _ = flatten_util.ravel_pytree(ans)
    yield ans_flat

def _odeint_grid(func, y0, t, *args, step_size):
    return _odeint_grid_wrapper(func, step_size, y0, t, *args)

@partial(jax.jit, static_argnums=(0, 1))
def _odeint_grid_wrapper(func, step_size, y0, ts, *args):
    y0, unravel = flatten_util.ravel_pytree(y0)
    func = ravel_first_arg(func, unravel)
    out, nfe = _rk4_odeint(func, step_size, y0, ts, *args)
    return jax.vmap(unravel)(out), nfe

@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _rk4_odeint(func, step_size, y0, ts, *args):
    
    func_ = lambda y, t: func(y, t, *args)

    def step_func(cur_t, dt, cur_y):
        """
        Take one step of RK4.
        """
        k1 = func_(cur_y, cur_t)
        k2 = func_(cur_y + dt * k1 / 2, cur_t + dt / 2)
        k3 = func_(cur_y + dt * k2 / 2, cur_t + dt / 2)
        k4 = func_(cur_y + dt * k3, cur_t + dt)
        return (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

    def cond_fun(carry):
        """
        Check if we've reached the last timepoint.
        """
        cur_y, cur_t, cur_nfe = carry
        return cur_t < ts[-1]

    # TODO: this doesn't work for multiple time points
    def body_fun(carry):
        """
        Take one step of RK4.
        """
        cur_y, cur_t, cur_nfe = carry
        next_t = lax.min(cur_t + step_size, ts[-1])
        dt = next_t - cur_t
        dy = step_func(cur_t, dt, cur_y)
        next_y = cur_y + dy
        new_nfe = cur_nfe + 4
        new_carry = [next_y, next_t, new_nfe]
        return new_carry

    init_t = ts[0]
    init_nfe = 0
    init_carry = [y0, init_t, init_nfe]
    y1, t1, nfe = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return jnp.concatenate((y0[None], y1[None])), nfe

def _rk4_odeint_fwd(func, step_size, y0, ts, *args):
    ys, nfe = _rk4_odeint(func, step_size, y0, ts, *args)
    return (ys, nfe), (ys, ts, args)

def _rk4_odeint_rev(func, step_size, res, g):
    ys, ts, args = res
    g, _ = g

    def aug_dynamics(augmented_state, t, *args):
        """Original system augmented with vjp_y, vjp_t and vjp_args."""
        y, y_bar, *_ = augmented_state
        # `t` here is negatice time, so we need to negate again to get back to
        # normal time. See the `odeint` invocation in `scan_fun` below.
        y_dot, vjpfun = jax.vjp(func, y, -t, *args)
        return (-y_dot, *vjpfun(y_bar))

    y_bar = g[-1]
    ts_bar = []
    t0_bar = 0.

    def scan_fun(carry, i):
        y_bar, t0_bar, args_bar = carry
        # Compute effect of moving measurement time
        t_bar = jnp.dot(func(ys[i], ts[i], *args), g[i])
        t0_bar = t0_bar - t_bar
        # Run augmented system backwards to previous observation
        _, y_bar, t0_bar, args_bar = _odeint_grid(
            aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
            jnp.array([-ts[i], -ts[i - 1]]),
            *args, step_size=step_size)[0]
        y_bar, t0_bar, args_bar = tree_util.tree_map(operator.itemgetter(1), (y_bar, t0_bar, args_bar))
        # Add gradient from current output
        y_bar = y_bar + g[i - 1]
        return (y_bar, t0_bar, args_bar), t_bar

    init_carry = (g[-1], 0., tree_util.tree_map(jnp.zeros_like, args))
    (y_bar, t0_bar, args_bar), rev_ts_bar = lax.scan(
        scan_fun, init_carry, jnp.arange(len(ts) - 1, 0, -1))
    ts_bar = jnp.concatenate([jnp.array([t0_bar]), rev_ts_bar[::-1]])
    return (y_bar, ts_bar, *args_bar)

_rk4_odeint.defvjp(_rk4_odeint_fwd, _rk4_odeint_rev)

def odeint_rk_grid(afunc, xinit, time_span, p, dt = 0.01):

    def body_func(carry, tf):
        _carry, t0 = carry
        sol, _ = _odeint_grid(afunc, _carry, jnp.array([t0, tf]), p, step_size = dt)
        return (sol[-1], tf), (sol[-1], tf)

    _, (solution, _) = jax.lax.scan(body_func, (xinit, time_span[0]), time_span[1:])
    return jnp.vstack((xinit, solution)) 


def _custom_odeint(afunc, xinit, time_span, parameters, rtol, atol, maxstep):
    # A jittable solver. Is not reverse differentiable because of lax.while_loop.
    # However, this is forward mode differentiable
    # To make it reverse mode differentiable and jittable, a custom vjp rule is defined that has scan based while loop.
    # Implements RK5 step.
    # A separate function is created so that forwad mode autodiff and forward pass in reverse mode autodiff can share the same code

    # This code is similar to 
    # https://github.com/jacobjinkelly/easy-neural-ode/blob/master/lib/ode.py
    
    _afunc = lambda x, t: afunc(x, t, parameters)

    def scan_func(carry, target_t):

        def cond_func(state):
            i, _, _, t, dt, _, _ = state
            return (t < target_t) & (i < maxstep) & (dt > 0)

        def body_func(state):
            i, y, f, t, dt, last_t, interp_coeff = state
            next_y, next_y_error, k, next_f = runge_kutta_step(_afunc, y, f, t, dt)
            next_t = t + dt
            error_ratios = mean_error_ratio(next_y_error, rtol, atol, y, next_y)
            new_interp_coeff = interp_fit_dopri(y, next_y, k, dt)
            dt = jnp.clip(lax.stop_gradient(optimal_step_size(dt, error_ratios)), a_min = 0., a_max = jnp.inf)

            new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
            old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
            return jax.lax.cond(jnp.all(error_ratios <= 1.), lambda : new, lambda : old)

        n_steps, *carry = lax.while_loop(cond_func, body_func, [0] + carry)
        _, _, t, _, last_t, interp_coeff = carry
        relative_output_time = (target_t - last_t) / (t - last_t)
        y_target = jnp.polyval(interp_coeff, relative_output_time)
        return carry, y_target

    f0 = _afunc(xinit, time_span[0])
    dt = initial_step_size(_afunc, time_span[0], xinit, 4, rtol, atol, f0)
    interp_coeff = jnp.array([xinit] * 5)
    init_carry = [xinit, f0, time_span[0], dt, time_span[0], interp_coeff]
    carry, ys = lax.scan(scan_func, init_carry, time_span[1:])
    return jnp.vstack((xinit, ys))


# The odeint function that should be called for forward mode sensitivities (optimize-then-discretize/variational) 
def odeint_rk_var_fwd(afunc, xinit, time_span, parameters, rtol = 1.4e-8, atol = 1.4e-8, mxstep = jnp.inf):
    solution = _odeint_rk_var_fwd(afunc, rtol, atol, mxstep, xinit, time_span, parameters)
    return solution

# The odeint function that should be called for forward mode sensitivities (discretize-then-optimize) 
def odeint_rk_fwd(afunc, xinit, time_span, parameters, rtol = 1.4e-8, atol = 1.4e-8, maxstep = jnp.inf):
    solution = _odeint_rk_fwd(afunc, rtol, atol, maxstep, xinit, time_span, parameters)
    return solution

# The odeint function that should be called for reverse mode sensitivities (discretize-then-optimize)
def odeint_rk_rev(afunc, xinit, time_span, parameters, rtol = 1.4e-8, atol = 1.4e-8, maxstep = jnp.inf):
    solution = _odeint_rk_rev(afunc, rtol, atol, maxstep, xinit, time_span, parameters)
    return solution

# The odeint function that uses scipy LSODA method. Forward mode sensitivites (optimize-then-discretize/variational)
# The ode solver is assumed to be black box. Therefore the internals of the solver cannot be jitted. 
def odeint_scipy_var_fwd(afunc, xinit, time_span, parameters, rtol = 1.4e-8, atol = 1.4e-8, **kwargs):
    
    afunc_jit = jax.jit(afunc) # jit compile function to speed up the calculations
    solution = _odeint_scipy_var_fwd(afunc_jit, rtol, atol, kwargs, xinit, time_span, parameters)
    return solution

# The odeint function that uses scipy LSODA method. Reverse mode sensitivities (optimize-then-discretize/variational)
# The ode solver is assumed to be black box. Therefore the internals of the solver cannnot be jitted.
def odeint_scipy_var_rev(afunc, xinit, time_span, parameters, rtol = 1.4e-8, atol = 1.4e-8, **kwargs):

    afunc_jit = jax.jit(afunc) # jit compile function to speed up the calculations
    solution = _odeint_scipy_var_rev(afunc_jit, rtol, atol, kwargs, xinit, time_span, parameters)
    return solution


# Reverse mode sensitivity calculations (discretize-then-optimize/differentiate through the ode solver)
# Reverse mode of optimize then discretize (jax default) does not give accurate results. There are other various options :
# 1. Derivatives are calculated through the ode solver. (In JAX, scan-based while loop can be used). 
#    Downside - differentiation is also performed through adaptive stepsize selection, which is unnecessary.
#    lax.stop_gradient is used to stop gradient across the optimal step size calculations
# 2. Convert the continuous time ode to discrete time by saving solution at intermediate time points. Then apply discrete time adjoint equations.
#    Downside - additional space to store all the stepsizes and states
# 3. Use checkpoints to overcome space issues in 2. Differentiate through smaller (checkpoints) time scale and then delete the computation
# 4. Use interpolation to overcome space issues (really ?) in 2. Backward pass calls this interpolation function to get the values of the states 
#    instead of integrating the ode again. 
@partial(jax.custom_vjp, nondiff_argnums = (0, 1, 2, 3))
def _odeint_rk_rev(afunc, rtol, atol, maxstep, xinit, time_span, parameters):
    return _custom_odeint(afunc, xinit, time_span, parameters, rtol, atol, maxstep)

def _odeint_rk_rev_fwd(afunc, rtol, atol, maxstep, xinit, time_span, parameters):
    solution = _odeint_rk_rev(afunc, rtol, atol, maxstep, xinit, time_span, parameters)
    return solution, (solution, time_span, parameters)

def _odeint_rk_rev_bwd(afunc, rtol, atol, maxstep, residual, g_dot):
    # This is similar to checkpointing. However, checkpoints are based on time_span
    solution, time_span, parameters = residual
    _afunc = lambda y, t : afunc(y, t, parameters)

    def get_init_cond(y0, t0):
        f0 = afunc(y0, t0, parameters)
        dt = initial_step_size(lambda y, t : afunc(y, t, parameters), t0, y0, 4, rtol, atol, f0)
        return [y0, y0, f0, t0, dt, t0, jnp.array(0)] # dont accept the first step for adjoint

    def _runge_kutta_step(afunc, y, f, t, dt, p):
        _afunc = lambda x, t : afunc(x, t, p)
        # Derivative wrt initial condition is also considered
        f = _afunc(y, t)
        return runge_kutta_step(_afunc, y, f, t, dt)

    def scan_fun(carry, i):
        
        adjoint_x, adjoint_p = carry 
        start_y, start_t, target_t, _g_dot = i
        _carry = get_init_cond(start_y, start_t)

        def cond_func(state):
            i, _, _, _, t, dt, last_t, _ = state
            _cond = (last_t < target_t) & (i < maxstep) & (dt > 0)
            return lax.cond(_cond, lambda : jnp.array(1), lambda : jnp.array(0))

        def body_func(state):
            i, y_prev, y, f, t, dt, last_t, _ = state
            next_y, next_y_error, k, next_f = runge_kutta_step(_afunc, y, f, t, dt)
            next_t = t + dt
            error_ratios = mean_error_ratio(next_y_error, rtol, atol, y, next_y)
            dt = optimal_step_size(dt, error_ratios)

            # overshooting can be permitted because last value is not ued in adjoint calculation
            new = [i + 1,      y, next_y, next_f, next_t,   dt,       t, jnp.array(1)]
            old = [i + 1, y_prev,      y,      f,      t,   dt,  last_t, jnp.array(0)]
            return jax.lax.cond(jnp.all(error_ratios <= 1.), lambda : new, lambda : old)

        # using discrete time adjoint equations
        def adjoint_cond_func(state, status):
            return jnp.logical_and(state[-1], status)

        def adjoint_scan_func(carry, i):
            adjoint_x, adjoint_p = carry
            state, status = i 

            def true_func(carry, state):
                adjoint_x, adjoint_p = carry
                _, y_prev, _, _, t, _, last_t, _ = state
                _dt = jnp.minimum(t, target_t) - last_t
                _, vjp_func = jax.vjp(lambda x, p : _runge_kutta_step(afunc, x, _, last_t, _dt, p)[0], y_prev, parameters)
                adjoint_x, _adjoint_p = vjp_func(adjoint_x)
                adjoint_p = tree_util.tree_map(operator.add, _adjoint_p, adjoint_p)
                return (adjoint_x, adjoint_p)

            def false_func(carry, state):
                return carry

            carry = lax.cond(adjoint_cond_func(state, status), true_func, false_func, carry, state)
            return carry, None

        state, status = while_loop_scan(body_func, cond_func, [0] + _carry, mxiter = 1_000)
        (adjoint_x, adjoint_p), _ = lax.scan(adjoint_scan_func, (adjoint_x, adjoint_p), (state, status), reverse = True)
        adjoint_x += _g_dot
        return (adjoint_x, adjoint_p), None 

    init_adjoint_p, init_adjoint_x = tree_util.tree_map(jnp.zeros_like, parameters), g_dot[-1]
    carry, _ = lax.scan(scan_fun, (init_adjoint_x, init_adjoint_p), (solution[:-1], time_span[:-1], time_span[1:], g_dot[:-1]), reverse = True)
    return (carry[0], None, carry[1]) 

_odeint_rk_rev.defvjp(_odeint_rk_rev_fwd, _odeint_rk_rev_bwd)


# Forward mode sensitivity calculations. (Optimize-then-discretize/variational)
@partial(jax.custom_jvp, nondiff_argnums = (0, 1, 2, 3))
def _odeint_rk_var_fwd(afunc, rtol, atol, mxstep, xinit, time_span, parameters):
    return odeint_jax(afunc, xinit, time_span, parameters, atol = atol, rtol = rtol, mxstep = mxstep)

def _odeint_rk_var_fwd_fwd(afunc, rtol, atol, mxstep, primals, tangents):
    xinit, time_span, parameters = primals
    xinit_tangent, t_tangent, parameters_tangent = tangents

    # initial conditions
    dx0_dp = jnp.zeros_like(xinit_tangent)
    dx0_dx0 = xinit_tangent

    def aug(x, t, p):
        x, dx_dx0, dx_dp = x

        return (
            *jax.jvp(lambda x : afunc(x, t, p), (x, ), (dx_dx0, )),
            jax.jvp(lambda x, p : afunc(x, t, p), (x, p), (dx_dp, parameters_tangent))[-1],
        )

    primal_out, x_tangent_out, p_tangent_out = odeint_jax(aug, (xinit, dx0_dx0, dx0_dp), time_span, parameters, atol = atol, rtol = rtol, mxstep = mxstep)
    # output should be same structure as primal output
    
    # hack for t_tangent
    # TODO incorporate t_tangent in the augmented system
    _, t_tangent_out = jax.jvp(lambda t : _custom_odeint(afunc, xinit, t, parameters, rtol, atol, mxstep), (time_span, ), (t_tangent, ))
    
    return primal_out, x_tangent_out + p_tangent_out + t_tangent_out

_odeint_rk_var_fwd.defjvp(_odeint_rk_var_fwd_fwd)


# Forward mode sensitivity calculations. (Discretize-then-optimize/differentiate through the solver)
@partial(jax.custom_jvp, nondiff_argnums = (0, 1, 2, 3))
def _odeint_rk_fwd(afunc, rtol, atol, maxstep, xinit, time_span, parameters):
    return _custom_odeint(afunc, xinit, time_span, parameters, atol = atol, rtol = rtol, maxstep = maxstep)

def _odeint_rk_fwd_fwd(afunc, rtol, atol, maxstep, primals, tangents):
    xinit, time_span, parameters = primals
    x_tangent, t_tangent, p_tangent = tangents

    primal_out, tangent_out = jax.jvp(lambda x, t, p : _custom_odeint(afunc, x, t, p, rtol, atol, maxstep), (xinit, time_span, parameters), (x_tangent, t_tangent, p_tangent))
    return primal_out, tangent_out

_odeint_rk_fwd.defjvp(_odeint_rk_fwd_fwd)


# Forward mode sensitivity calculations. (Optimize-then-discretize/variational) Scipy solver.
# Cannot perfrom discretize-then-optimize because the ode solver is not jax compatible
@partial(jax.custom_jvp, nondiff_argnums = (0, 1, 2, 3))
def _odeint_scipy_var_fwd(afunc, rtol, atol, kwargs, xinit, time_span, parameters):
    
    def solve_ivp_host(x, t, p):
        try :
            solution = solve_ivp(lambda _t, _x, _p : afunc(_x, _t, _p), [t[0], t[-1]], x, t_eval = t, args = (p, ), rtol = rtol, atol = atol, **kwargs)
        except :
            solution = np.ones((len(t), len(x)))*np.inf
        else:
            if not solution.success:
                solution = np.ones((len(t), len(x)))*np.inf
            else:
                solution = solution.y.T

        return solution

    def _odeint_scipy_callback(xinit, time_span, parameters):
        result_shape = jax.ShapeDtypeStruct((len(time_span), xinit.shape[-1]), xinit.dtype)
        return jax.pure_callback(solve_ivp_host, result_shape, xinit, time_span, parameters)
    
    return _odeint_scipy_callback(xinit, time_span, parameters)

def _odeint_scipy_var_fwd_fwd(afunc, rtol, atol, kwargs, primals, tangents):
    
    xinit, time_span, parameters = primals
    xinit_tangent, _, parameters_tangent = tangents

    def solve_ivp_host(x, t, p, p_tangent):
        solution = solve_ivp(lambda _t, _x, _p, _p_tangent : aug(_x, _t, _p, _p_tangent), [t[0], t[-1]], x, t_eval = t, args = (p, p_tangent), rtol = rtol, atol = atol, **kwargs)
        return solution.y.T

    def _odeint_scipy_callback(xinit, time_span, parameters, parameters_tangent):
        result_shape = jax.ShapeDtypeStruct((len(time_span), xinit.shape[-1]), xinit.dtype)
        return jax.pure_callback(solve_ivp_host, result_shape, xinit, time_span, parameters, parameters_tangent)

    def aug(x, t, p, p_tangent):
        x, dx_dx0, dx_dp = np.array_split(x, 3)
        return np.concatenate([
            *jax.jvp(lambda x : afunc(x, t, p), (x, ), (dx_dx0, )),
            jax.jvp(lambda x, p : afunc(x, t, p), (x, p), (dx_dp, p_tangent))[-1]
        ])

    # initial conditions
    dx0_dp = jnp.zeros_like(xinit_tangent)
    dx0_dx0 = xinit_tangent

    solution = _odeint_scipy_callback(jnp.concatenate((xinit, dx0_dx0, dx0_dp)), time_span, parameters, parameters_tangent)
    primal_out, x_tangent_out, p_tangent_out = jnp.array_split(solution, 3, axis = 1)
    
    # output should be same structure as primal output
    return primal_out, x_tangent_out + p_tangent_out 

_odeint_scipy_var_fwd.defjvp(_odeint_scipy_var_fwd_fwd)


# Reverse mode sensitivity calculations. (Optimize-then-discretize/variational)
# Cannot perform discretize-then-optimize because the ode solver is not jax compatible
@partial(jax.custom_vjp, nondiff_argnums = (0, 1, 2, 3))
def _odeint_scipy_var_rev(afunc, rtol, atol, kwargs, xinit, time_span, parameters):
    
    def solve_ivp_host(x, t, p):
        try :
            solution = solve_ivp(lambda _t, _x, _p : afunc(_x, _t, _p), [t[0], t[-1]], x, t_eval = t, args = (p, ), 
                                rtol = rtol, atol = atol, jac = afunc_jac, **kwargs)
        except :
            solution = np.ones((len(t), len(x)))*np.inf
        else:
            if not solution.success or np.isnan(solution.y).any():
                solution = np.ones((len(t), len(x)))*np.inf
            else:
                solution = solution.y.T

        return solution

    @jax.jit
    def afunc_jac(_t, _x, _p):
        # jacobian of the integrated function is used by implicit methods of solve_ivp 
        return jax.jacrev(lambda x : afunc(x, _t, _p))(_x)

    def _odeint_scipy_callback(xinit, time_span, parameters):
        result_shape = jax.ShapeDtypeStruct((len(time_span), xinit.shape[-1]), xinit.dtype)
        return jax.pure_callback(solve_ivp_host, result_shape, xinit, time_span, parameters)
    
    return _odeint_scipy_callback(xinit, time_span, parameters)

def _odeint_scipy_var_rev_fwd(afunc, rtol, atol, kwargs, xinit, time_span, parameters):
    solution = _odeint_scipy_var_rev(afunc, rtol, atol, kwargs, xinit, time_span, parameters)
    return solution, (solution, time_span, parameters)

def _odeint_scipy_var_rev_bwd(afunc, rtol, atol, kwargs, res, g_dot):
    
    solution, time_span, parameters = res
    p_flatten, p_unflatten_def = flatten_util.ravel_pytree(parameters)
    xlen =  len(solution[-1])

    # flatten integral function because scipy only accepts 1d arrays as output
    def afunc_flatten(x, t, p_flatten):
        return afunc(x, t, p_unflatten_def(p_flatten))

    def solve_ivp_host(x, t, p):
        # In parameter estimation, if forward pass is evaluated then gradient should be calculated. 
        # Therefore there is no point in returning inf
        solution = solve_ivp(lambda _t, _x, _p : aug(_x, _t, _p), [t[0], t[-1]], x, t_eval = t, args = (p, ), 
                            rtol = rtol, atol = atol, jac = aug_jac, **kwargs)
        # jax.debug.print("error message {}", solution.message)
        return solution.y.T

    def _odeint_scipy_callback(xinit, time_span, p_flatten):
        result_shape = jax.ShapeDtypeStruct((len(time_span), xinit.shape[-1]), xinit.dtype)
        return jax.pure_callback(solve_ivp_host, result_shape, xinit, time_span, p_flatten)
    
    @jax.jit
    def vjp_func(x, t, p, tangent):
        primals, _vjp_func = jax.vjp(lambda x, p : afunc_flatten(x, t, p), x, p)
        return primals, _vjp_func(tangent)

    @jax.jit
    def aug_jac(_t, _x, _p):
        # jacobian of the integrated function is used by implicit methods of solve_ivp 
        return jax.jacrev(lambda x : aug(x, _t, _p))(_x) 

    def aug(x, t, p_flatten):
        # This function is black-box and integrated using scipy. 
        # Jax compatible functions can be accelerated using jit
        x, adjoint_x, _ = jnp.array_split(x, [xlen, 2*xlen])
        primals, vjp = vjp_func(x, t, p_flatten, adjoint_x)
        return -jnp.concatenate([
            -primals,
            *vjp,
        ])

    def body_func(carry, state):
        xinit, _g_dot, t_start, t_end = state
        solution = _odeint_scipy_callback(jnp.concatenate([xinit, *carry]), jnp.array([t_start, t_end]), p_flatten)
        _, adjoint_x, adjoint_p = jnp.array_split(solution[-1], [xlen, 2*xlen])
        return (adjoint_x + _g_dot, adjoint_p), None  

    init_adjoint_p, init_adjoint_x = jnp.zeros_like(p_flatten), g_dot[-1]
    solution, _ = lax.scan(body_func, (init_adjoint_x, init_adjoint_p), (solution[1:], g_dot[:-1], time_span[1:], time_span[:-1]), reverse = True)
    
    return (solution[0], None, p_unflatten_def(solution[1]))

_odeint_scipy_var_rev.defvjp(_odeint_scipy_var_rev_fwd, _odeint_scipy_var_rev_bwd)


####################################################################################################################################
# Test cases

if __name__ == "__main__":

    def foo(x, t, p):
        # autonomous ode
        # x and t has to be jax 1d array while p can be any pytree
        return jnp.array([-p[0]*x[0], -p[1]*x[1]])

    def lotka_volterra(x, t, p):
        return jnp.array([
            p[0]*x[0] - p[1]*x[0]*x[1],
            -p[2]*x[1] + p[3]*x[0]*x[1]
        ]) 

    def kinetic(x, t, p):
        # A + B <==> C <==> B + D
        # k1 = 1, kr1 = 0.5, k2 = 2, kr2 = 1
        # Define Kinetic system ode   
        return jnp.array([
            -p[0]*x[0]*x[1] + p[1]*x[2],
            -p[0]*x[0]*x[1] + (p[1]+p[2])*x[2] - p[3]*x[1]*x[3],
            p[0]*x[0]*x[1] - (p[1]+p[2])*x[2] + p[3]*x[1]*x[3],
            p[2]*x[2] - p[3]*x[1]*x[3]
        ])


    system = lotka_volterra
    time_span = jnp.arange(0., 10., 1)

    if system == foo:
        xinit = jnp.array([2., 3.])
        p = jnp.array([1., 2., 3.])
        target = odeint_jax(system, xinit, time_span, [10, 0.5])
    elif system == lotka_volterra:
        xinit = jnp.array([1., 1.])
        p = jnp.array([3., 2., 1., 0.3])
        target = odeint_jax(system, xinit, time_span, [1.5, 1., 3., 1.])
    elif system == kinetic:
        xinit = jnp.array([5., 4., 3., 2.])
        p = jnp.array([0., 2., 1., 0.3])
        target = odeint_jax(system, xinit, time_span, [1, .5, 2., 1.])
    else :
        assert False, "Unknown system chosen"

    start = time.time()
    solution = odeint_rk_grid(system, xinit, time_span, p)
    end_1 = time.time()
    solution_actual = odeint_jax(system, xinit, time_span, p)
    end_2 = time.time()
    solution_adaptive = odeint_rk_rev(system, xinit, time_span, p)
    end_3 = time.time()

    print("solution fixed grid size", end_1 - start)
    print("solution jax implementation", end_2 - end_1)
    print("solution adaptive implementation", end_3 - end_2)

    def foo_diffrax(t, x, p):
        return system(x, t, p)

    def objective_grid(target, xinit, time_span, parameters):
        solution = odeint_rk_grid(system, xinit, time_span, parameters)
        return jnp.mean((solution - target)**2)

    def objective_actual_rev(target, xinit, time_span, parameters):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(foo_diffrax), 
            diffrax.Dopri5(),
            t0 = time_span[0],
            t1 = time_span[-1],
            dt0 = time_span[1] - time_span[0], 
            saveat = diffrax.SaveAt(ts=time_span), 
            y0 = xinit, 
            args = parameters,
            stepsize_controller = diffrax.PIDController(rtol=1.4e-8, atol=1.4e-8, dcoeff = 0.2, icoeff = 10.),
            adjoint = diffrax.RecursiveCheckpointAdjoint(),
            max_steps = 100000,
            )
        return jnp.mean((solution.ys - target)**2)

    def objective_actual_fwd(target, xinit, time_span, parameters):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(foo_diffrax), 
            diffrax.Dopri5(),
            t0 = time_span[0],
            t1 = time_span[-1],
            dt0 = time_span[1] - time_span[0], 
            saveat = diffrax.SaveAt(ts=time_span), 
            y0 = xinit, 
            args = parameters,
            stepsize_controller = diffrax.PIDController(rtol=1.4e-8, atol=1.4e-8, dcoeff = 0.2, icoeff = 10.),
            adjoint = diffrax.DirectAdjoint(),
            max_steps = 100000,
            )
        return jnp.mean((solution.ys - target)**2)

    def objective_actual_var_rev(target, xinit, time_span, parameters):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(foo_diffrax), 
            diffrax.Dopri5(),
            t0 = time_span[0],
            t1 = time_span[-1],
            dt0 = time_span[1] - time_span[0], 
            saveat = diffrax.SaveAt(ts=time_span), 
            y0 = xinit, 
            args = parameters,
            stepsize_controller = diffrax.PIDController(rtol=1.4e-8, atol=1.4e-8, dcoeff = 0.2, icoeff = 10.),
            adjoint = diffrax.BacksolveAdjoint(),
            max_steps = 100000,
            )
        return jnp.mean((solution.ys - target)**2)

    def objective_variational_fwd(target, xinit, time_span, parameters):
        solution = odeint_rk_var_fwd(system, xinit, time_span, parameters)
        return jnp.mean((solution - target)**2)

    def objective_fwd(target, xinit, time_span, parameters):
        solution = odeint_rk_fwd(system, xinit, time_span, parameters)
        return jnp.mean((solution - target)**2)

    def objective_rev(target, xinit, time_span, parameters):
        solution = odeint_rk_rev(system, xinit, time_span, parameters)
        return jnp.mean((solution - target)**2)

    def objective_variational_rev(target, xinit, time_span, parameters):
        solution = odeint_jax(system, xinit, time_span, parameters)
        return jnp.mean((solution - target)**2)

    def objective_scipy_variational_fwd(target, xinit, time_span, parameters):
        solution = odeint_scipy_var_fwd(system, xinit, time_span, parameters, method = "BDF")
        return jnp.mean((solution - target)**2)
    
    def objective_scipy_variational_rev(target, xinit, time_span, parameters):
        solution = odeint_scipy_var_rev(system, xinit, time_span, parameters, method = "BDF")
        return jnp.mean((solution - target)**2)

    
    gradient_grid_rev = grad(objective_grid, argnums = 3)(target, xinit, time_span, p)
    gradient_grid_fwd = jax.jacfwd(objective_grid, argnums = 3)(target, xinit, time_span, p)

    gradient_actual_rev_p = grad(objective_actual_rev, argnums = 3)(target, xinit, time_span, p)
    gradient_actual_rev_x = grad(objective_actual_rev, argnums = 1)(target, xinit, time_span, p)
    gradient_actual_fwd_p = jax.jacfwd(objective_actual_fwd, argnums = 3)(target, xinit, time_span, p)
    gradient_actual_fwd_x = jax.jacfwd(objective_actual_fwd, argnums = 1)(target, xinit, time_span, p)
    gradient_actual_var_rev_p = grad(objective_actual_var_rev, argnums = 3)(target, xinit, time_span, p)
    gradient_actual_var_rev_x = grad(objective_actual_var_rev, argnums = 1)(target, xinit, time_span, p)
    

    gradient_variational_rev_p = grad(objective_variational_rev, argnums = 3)(target, xinit, time_span, p)
    gradient_variational_rev_x = grad(objective_variational_rev, argnums = 1)(target, xinit, time_span, p)
    gradient_rev_p = grad(objective_rev, argnums = 3)(target, xinit, time_span, p)
    gradient_rev_x = grad(objective_rev, argnums = 1)(target, xinit, time_span, p)
    gradient_fwd_p = jax.jacfwd(objective_fwd, argnums = 3)(target, xinit, time_span, p)
    gradient_fwd_x = jax.jacfwd(objective_fwd, argnums = 1)(target, xinit, time_span, p)
    
    print("--"*50)
    gradient_scipy_var_rev_p = grad(objective_scipy_variational_rev, argnums = 3)(target, xinit, time_span, p)
    gradient_scipy_var_rev_x = grad(objective_scipy_variational_rev, argnums = 1)(target, xinit, time_span, p)

    
    def gradient_fwd(_objective, argnums = 1):

        if argnums == 1:
            _obj = lambda x : _objective(target, x, time_span, p)
            x_flatten, x_unflatten_def = flatten_util.ravel_pytree(xinit) 
            _, gradient = jax.vmap(lambda y : jax.jvp(_obj, (xinit, ), (y, )))(jnp.eye(len(x_flatten)))
            return x_unflatten_def(gradient) 
        elif argnums == 3:
            _obj = lambda p : _objective(target, xinit, time_span, p)
            p_flatten, p_unflatten_def = flatten_util.ravel_pytree(p)
            _, gradient = jax.vmap(lambda y : jax.jvp(_obj, (p, ), (y, )))(jnp.eye(len(p_flatten)))
            return p_unflatten_def(gradient) 
        else:
            assert False, f"Select correct argnums"

    
    gradient_variational_fwd_p = gradient_fwd(objective_variational_fwd, 3)
    gradient_variational_fwd_x = gradient_fwd(objective_variational_fwd, 1)
    gradient_scipy_var_fwd_p = gradient_fwd(objective_scipy_variational_fwd, 3)
    gradient_scipy_var_fwd_x = gradient_fwd(objective_scipy_variational_fwd, 1)

    def compare(_cmp, _orig, name):
        print(
            name,
            _cmp,
            "isclose",
            jnp.allclose(_cmp, _orig),
            "error",
            jnp.linalg.norm(_cmp - _orig)
        )

    
    compare(gradient_grid_rev, gradient_actual_rev_p, "gradients grid reverse p")
    compare(gradient_grid_fwd, gradient_actual_rev_p, "gradients grid forward p")
    compare(gradient_variational_fwd_p, gradient_actual_rev_p, "gradients variational forward p")
    compare(gradient_variational_fwd_x, gradient_actual_rev_x, "gradients variational forward x")
    compare(gradient_variational_rev_p, gradient_actual_rev_p, "gradients variational reverse p")
    compare(gradient_variational_rev_x, gradient_actual_rev_x, "gradients variational reverse x")
    compare(gradient_rev_p, gradient_actual_rev_p, "gradients adaptive reverse p")
    compare(gradient_rev_x, gradient_actual_rev_x, "gradients adaptive reverse x")
    compare(gradient_fwd_p, gradient_actual_rev_p, "gradients adaptive forward p")
    compare(gradient_fwd_x, gradient_actual_rev_x, "gradients adaptive forward x")
    compare(gradient_scipy_var_fwd_p, gradient_actual_rev_p, "gradients scipy variational forward p")
    compare(gradient_scipy_var_fwd_x, gradient_actual_rev_x, "gradients scipy variational forward x")
    compare(gradient_scipy_var_rev_p, gradient_actual_rev_p, "gradients scipy variational reverse p")
    compare(gradient_scipy_var_rev_x, gradient_actual_rev_x, "gradients scipy variational reverse x")

    # comparing diffrax
    print("--"*20)
    compare(gradient_actual_fwd_p, gradient_actual_rev_p, "gradient forward p")
    compare(gradient_actual_fwd_x, gradient_actual_rev_x, "gradient forward x")
    compare(gradient_actual_var_rev_p, gradient_actual_rev_p, "gradient reverse variational p")
    compare(gradient_actual_var_rev_x, gradient_actual_rev_x, "gradient reverse variational x")
    print("gradients actual (diffrax) p", gradient_actual_rev_p)
    print("gradients actual (diffrax) x", gradient_actual_rev_x)
    