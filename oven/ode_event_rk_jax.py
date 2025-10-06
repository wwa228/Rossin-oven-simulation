from functools import partial

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
from jax.interpreters import ad
from jax._src.ad_util import stop_gradient_p
from jax import tree_util, flatten_util
import diffrax

from .utils import flatten_output

# https://github.com/jax-ml/jax/issues/10994
ad.primitive_transposes[stop_gradient_p] = lambda ct, _ : [tree_util.tree_map(jnp.zeros_like, ct)]


def odeint_diffrax(afunc, rtol, atol, mxstep, xinit, time_span, parameters):
    # This is done to prevent inf values when time_span is nonincreasing (or has the same values)
    # eps = jnp.zeros_like(time_span)
    # _time_span = time_span + eps.at[0].set(-1e-16)

    _afunc = lambda t, x, p : afunc(x, t, p)
    return diffrax.diffeqsolve(
                diffrax.ODETerm(_afunc), 
                diffrax.Dopri5(),
                t0 = time_span[0], # make sure that initial conditions are at time_span[0]
                t1 = time_span[-1],
                dt0 = None, 
                saveat = diffrax.SaveAt(ts = time_span), 
                y0 = xinit, 
                args = parameters,
                stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol, pcoeff = 0.4, icoeff = 0.3, dcoeff = 0.), # pcoeff = 0.4, icoeff = 0.3, dcoeff = 0
                adjoint = diffrax.DirectAdjoint(), 
                max_steps = mxstep
        ).ys

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

def runge_kutta_step(afunc, y0, f0, t0, dt, *args):
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
        f1 = afunc(y1, t1, *args)
        return k.at[i].set(f1), i

    k = jnp.zeros(shape = (7, len(y0))).at[0].set(f0)
    k, _ = jax.lax.scan(body_func, k, jnp.arange(1, 7))
    y1 = y0 + dt*jnp.dot(b, k)
    f1 = k[-1]
    y1_err = dt*jnp.dot(b_error, k)

    return y1, y1_err, k, f1

def initial_step_size(afunc, t0, y0, order, rtol, atol, f0, *args):
    # Algorithm from:
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    scale = atol + jnp.abs(y0) * rtol
    d0 = jnp.linalg.norm(y0 / scale)
    d1 = jnp.linalg.norm(f0 / scale)

    h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

    y1 = y0 + h0 * f0
    f1 = afunc(y1, t0 + h0, *args)
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

    """
    factor = jnp.minimum(ifactor,
                        jnp.maximum(mean_error_ratio**(-1.0 / order) * safety, dfactor))
    """
    # https://github.com/jax-ml/jax/issues/14612
    factor = jnp.nanmin(jnp.array([ifactor,
                                 jnp.nanmax(jnp.array([mean_error_ratio**(-1.0 / order) * safety, dfactor]))]))

    return jnp.where(mean_error_ratio == 0, last_step * ifactor, last_step * factor)

@partial(jax.jit, static_argnums = (0, 4, 5))
def odeint_rk_grid(afunc, xinit, time_span, parameters, dt, mxiter):
    # fixed step size Dormant Prince solver. Forward mode and reverse mode compatible with scan based while loop
    # Gets value only at the last time point

    def cond_fun(carry):
        # conditions to continue
        _, _, t0 = carry
        return t0 < time_span[-1]

    def take_step(carry):
        y0, f0, t0 = carry
        _dt = jnp.minimum(dt, jnp.abs(time_span[-1] - t0))
        y1, _, _, f1 = runge_kutta_step(afunc, y0, f0, t0, _dt, parameters)
        return [y1, f1, t0 + _dt]

    def body_fun(carry, xs):
        
        next_carry = jax.lax.cond(
            cond_fun(carry), 
            take_step, 
            lambda carry : carry,
            carry
        )
        
        return next_carry, None

    solution, _ = jax.lax.scan(body_fun, [xinit, afunc(xinit, time_span[0], parameters), time_span[0]], xs = None, length = mxiter)
    # jax.debug.print("Solution computed at {}, terminal time {}", solution[-1], time_span[-1])
    return solution[0]


@partial(jax.jit, static_argnums = (0, 1, 5, 6, 7))
def _custom_odeint_event(afunc, event, xinit, time_span, parameters, rtol, atol, mxstep = None):
    # Once all events are reached the forward simulation stops and the event time is output

    # This code is similar to 
    # https://github.com/jacobjinkelly/easy-neural-ode/blob/master/lib/ode.py
    
    _afunc = lambda x, t, event_times : afunc(x, t, (event_times, parameters))
    _event_cond = lambda x, t : event(x, t) >= 0 # x - xcrit

    def scan_func(carry, target_t):

        def cond_func(state):
            # conditions to continue
            _, _, _, t, dt, *_, _events = state
            return (t < target_t) & (dt > 0) & (_events.any())

        def step_func(state):
            # body function if event has not reached yet
            i, y, f, t, dt, last_t, interp_coeff, event_times, events = state
            next_y, next_y_error, k, next_f = runge_kutta_step(_afunc, y, f, t, dt, event_times)
            next_t = t + dt
            error_ratios = mean_error_ratio(next_y_error, rtol, atol, y, next_y)
            new_interp_coeff = interp_fit_dopri(y, next_y, k, dt)
            dt = jnp.clip(optimal_step_size(dt, error_ratios), a_min = 0., a_max = jnp.inf)
            next_events = _event_cond(next_y, next_t)

            new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff, event_times, next_events]
            old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff, event_times, events]            
            return jax.lax.cond(jnp.all(error_ratios <= 1.), lambda : new, lambda : old)

        def find_event(state, new_state, counter, tol = 1e-8):
            # find event using bisection method
            
            i, prev_y, prev_f, prev_t, prev_dt, last_t, _interp_coeff, event_times, events = state
            _, next_y, _, next_t, _, _, next_interp_coeff, *_ = new_state

            _interp = lambda t : jnp.polyval(next_interp_coeff, (t - prev_t)/(next_t - prev_t))
            max_iter = jnp.ceil(jnp.log((next_t - prev_t)/tol)/jnp.log(2.))
            
            def body_func(carry):
                _cur_iter, _cur_y, _cur_t, _next_y, _next_t = carry
                mid_t = (_cur_t + _next_t)/2
                mid_y = _interp(mid_t)
                mid_events = _event_cond(mid_y, mid_t)

                left = [_cur_iter + 1, _cur_y, _cur_t, mid_y, mid_t]
                right = [_cur_iter + 1, mid_y, mid_t, _next_y, _next_t]
                
                return jax.lax.cond(mid_events[counter], lambda : right, lambda : left)

            def cond_func(carry):
                cur_iter, *_ = carry
                return cur_iter <= max_iter

            *_, _cur_t, _, _next_t = jax.lax.while_loop(cond_func, body_func, [0, prev_y, prev_t, next_y, next_t])
            
            # Make sure that the event times are unique
            _event_time = (_next_t + _cur_t) / 2
            event_times = event_times.at[counter].set(_event_time)
            # return previous state
            return [i, prev_y, prev_f, prev_t, prev_dt, last_t, _interp_coeff, event_times, events]

        def body_func(state):
            
            *_, event_times, _next_events = new_state = step_func(state)
            counters = jnp.argwhere(jnp.logical_not(_next_events) & (event_times == time_span[-1] + 1), size = xdim, fill_value = xdim).flatten()
            
            next_state = jax.lax.cond(
                counters[0] == xdim,
                lambda *args : args[1],
                find_event,
                state, new_state, counters[0]
            )
            return next_state
        
        # TODO get counter closest to heat source and whose event time is -1
        n_steps, *carry = jax.lax.while_loop(cond_func, body_func, [0] + carry)
        return carry, None

    xdim = xinit.shape[0]
    init_events = _event_cond(xinit, time_span[0])
    event_times = jnp.where(init_events, time_span[-1] + 1, time_span[0] - 1)
    f0 = _afunc(xinit, time_span[0], event_times)
    dt = initial_step_size(_afunc, time_span[0], xinit, 4, rtol, atol, f0, event_times)
    interp_coeff = jnp.array([xinit] * 5)
    init_carry = [xinit, f0, time_span[0], dt, time_span[0], interp_coeff, event_times, init_events]

    _time_span = jnp.array([time_span[0], time_span[-1]])
    carry, _ = jax.lax.scan(scan_func, init_carry, _time_span[1:])   
    
    return carry[-2] # return event_times

@partial(jax.custom_jvp, nondiff_argnums = (0, 1, 5, 6, 7))
def custom_odeint_event(afunc, event, xinit, time_span, parameters, rtol, atol, mxstep = None):
    return jax.lax.stop_gradient(_custom_odeint_event(afunc, event, xinit, time_span, parameters, rtol, atol, mxstep))

@custom_odeint_event.defjvp
def custom_odeint_event_fwd(afunc, event, rtol, atol, mxstep, primals, tangents):
    xinit, time_span, parameters = primals
    event_times = custom_odeint_event(afunc, event, xinit, time_span, parameters, rtol, atol, mxstep)
    return event_times, jnp.zeros_like(event_times)


@partial(jax.custom_vjp, nondiff_argnums = (0, 1))
def _implicit_rev(afunc, event, x, t, p):
    # Transfer function for hybrid dynamical equations. Custom vjp rules define the 
    # transfer sensitivities as given in https://ieeexplore.ieee.org/document/7831410 and https://frankschae.github.io/post/bouncing_ball/
    # afunc : vmapped function 
    # event : single trajectory function
    return x 

def _implicit_rev_fwd(afunc, event, x, t, p):
    return x, (x, t, p)

def _implicit_rev_bwd(afunc, event, res, gdot):
    
    x, t, (event_times, p) = res
    _event = lambda x, t : event(x, t)[0]    

    def _transfer_sensitivity(xconstant, xfalling, _gdot):
        de_dx = jax.jacrev(_event, argnums = 0)(xconstant, t)
        dg_dt = jnp.vdot(xconstant, de_dx)
        _v = jnp.vdot(_gdot, xfalling - xconstant) / dg_dt
        return _gdot + _v * de_dx 

    xfalling = afunc(x, t + 1e-10, (event_times, p)) 
    xconstant = afunc(x, t, (event_times, p))
    
    # vmap over event_times
    lam = jax.vmap(lambda _event_time, *args : jax.lax.cond(
        t == _event_time, 
        _transfer_sensitivity,
        lambda xconstant, xfalling, _gdot : _gdot, # do nothing
        *args
        ))(event_times, xconstant, xfalling, gdot)
    
    return lam, None, None

_implicit_rev.defvjp(_implicit_rev_fwd, _implicit_rev_bwd)

@partial(jax.custom_jvp, nondiff_argnums = (0, 1))
def _implicit_fwd(afunc, event, x, t, p):
    # Transfer function for hybrid dynamical equations. Custom vjp rules define the 
    # transfer sensitivities as given in https://ieeexplore.ieee.org/document/7831410 and https://frankschae.github.io/post/bouncing_ball/
    # afunc : vmapped function 
    # event : single trajectory function
    return x 

@_implicit_fwd.defjvp
def _implicit_fwd_bwd(afunc, event, primals, tangents):
    
    x, t, (event_times, p) = primals
    x_dot, *_ = tangents
    _event = lambda x, t : event(x, t)[0]

    def _transfer_sensitivity(xconstant, xfalling, _x_dot):
        de_dx = jax.jacrev(_event, argnums = 0)(xconstant, t)
        dg_dt = jnp.vdot(xconstant, de_dx)
        _v = jnp.vdot(_x_dot, de_dx) / dg_dt
        return _x_dot + _v * (xfalling - xconstant)

    xfalling = afunc(x, t + 1e-10, (event_times, p))
    xconstant = afunc(x, t, (event_times, p))
    
    # vmap over event_times
    lam = jax.vmap(lambda _event_time, *args : jax.lax.cond(
        t == _event_time, 
        lambda : _transfer_sensitivity(*args),
        lambda : args[-1], # do nothing
        ))(event_times, xconstant, xfalling, x_dot)
    
    return x, lam


@partial(jax.jit, static_argnums = (0, 1, 2, 6, 7, 8))
def odeint_event(afunc, event, transfer, xinit, time_span, p, rtol = 1.4e-8, atol = 1.4e-8, mxstep = 10_000):
    # vmapped function : afunc lambda x, t, p : 
    # vmapped event function : event lambda x, t : 
    # vmapped transfer function : lambda transfer_function, x, t, p : 

    # Flatten the input and the outputs of the function
    # event_times = jnp.ones_like(xinit[:, 0]) * (time_span[-1] + 1)
    
    flatten_xinit, unravel_x = flatten_util.ravel_pytree(xinit)
    afunc = flatten_output(afunc, unravel_x)
    event = flatten_output(event, unravel_x)
    _transfer = flatten_output(lambda x, t, args : transfer(_implicit_fwd, x, t, args), unravel_x)

    event_times = custom_odeint_event(afunc, event, flatten_xinit, time_span, p, rtol, atol, mxstep)

    def scan_fun(carry, loop_vars):
        event_start, event_end = loop_vars
        event_start, event_end = tree_util.tree_map(lambda _x : jnp.minimum(jnp.maximum(_x, time_span[0]), time_span[-1]), (event_start, event_end))
        
        def true_func():
            return carry, jnp.stack([jnp.zeros_like(flatten_xinit)] * len(time_span))
            
        def false_func():
            _xinit = _transfer(carry, event_start, (event_times, p))
            # _time_span = jnp.where(time_span <= event_start, event_start, jnp.where(time_span >= event_end, event_end, time_span))
            _time_span = jax.vmap(lambda _t : jax.lax.cond(
                _t <= event_start, 
                lambda : event_start, 
                lambda : jax.lax.cond(_t >= event_end, lambda : event_end, lambda : _t)
            ))(time_span)

            solution = odeint_diffrax(afunc, rtol, atol, mxstep, _xinit, _time_span, (event_times, p))
            next_carry = solution[-1]
            solution = jax.vmap(lambda _sol, _t : jax.lax.cond(
                jnp.logical_or(_t < event_start, _t >= event_end), 
                lambda : jnp.zeros_like(_sol), 
                lambda : _sol
            ))(solution, time_span)
            return next_carry, solution

        next_carry, solution = jax.lax.cond(event_start == event_end, true_func, false_func)
        return next_carry, solution

    _event_times = jnp.concatenate((time_span[:1], jnp.sort(event_times), time_span[-1:]))
    solution_final, solution_flatten = jax.lax.scan(scan_fun, flatten_xinit, (_event_times[:-1], _event_times[1:]))  
    solution_flatten = jnp.sum(solution_flatten, axis = 0)
    solution_flatten = jnp.concatenate(( solution_flatten[:-1], solution_final[jnp.newaxis] ))

    # Unravel the final solution
    solution = jax.vmap(unravel_x)(solution_flatten)
    return solution, event_times

