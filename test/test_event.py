import unittest
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import flatten_util

from oven.ode_event_rk_jax import odeint_event, _custom_odeint_event
from oven.utils import flatten_output


rtol = 1e-6
atol = 1e-8

def _foo(x, t, p): 
    _, p = p
    return jnp.array([- p[0] * x[0]**2, - p[0] * x[1]**2, 0]) # for single trajectory x

def _bar(x, t, p): 
    _, p = p
    return jnp.array([- p[0] * x[0] * x[2], - p[0] * x[1] * x[2], 0]) # for single trajectory x

def event(x, t) : return jnp.array([x[0] - x[2]]) # event function for single trajectory x

def algebraic(x, t, p):
    
    def scan_fun(carry, xs):
        carry -= 0.01 * p[0] * xs[0]
        return carry, carry

    _, sol = jax.lax.scan(scan_fun, p[1:2], xs = x)
    return sol

def afunc(x, t, args):
    # combined integration function
    event_times, p = args
    y = algebraic(x, t, p) * 0 # Proxy for air moisture
    return jax.vmap(lambda _x, _y, _event : jax.lax.cond(t <= _event, lambda : _bar(_x, t, (_y, p)), lambda : _foo(_x, t, (_y, p))))(x, y, event_times)

def transfer(trans_func, x, t, args) : return trans_func(afunc, event, x, t, args)
def event_vmap(x, t) : return jax.vmap(event, in_axes = (0, None))(x, t)


class TestEvent(unittest.TestCase):

    def test_event_time(self):

        time_span = jnp.arange(0, 1., 0.01)
        permutation = jnp.array([2, 3, 4, 9, 8, 1, 0, 6, 7, 5]) # Testing for out of order events
        # p = [dynamic parameter, yinit, xcrit]

        # Event happens for all
        p = jnp.array([2, 3, 1.5]) # event happens for all
        xinit = jnp.column_stack((jnp.arange(2, 10. * 2 + 2).reshape(-1, 2), p[-1] * jnp.ones(shape = (10, ))))
        _xinit, unravel_x = flatten_util.ravel_pytree(xinit)
        _xinit_permute, _ = flatten_util.ravel_pytree(xinit[permutation])
        
        # all have some value
        @partial(flatten_output, unravel_first_arg = unravel_x)
        def _event_vmap(x, t): return jax.vmap(lambda t, x : x[0] - 1.5, in_axes = (None, 0))(t, x)
        _event_times_happens = _custom_odeint_event(flatten_output(afunc, unravel_x), _event_vmap, _xinit, time_span, p, rtol = 1.4e-8, atol = 1.4e-8, mxstep = None)
        _event_times_happens_ooo = _custom_odeint_event(flatten_output(afunc, unravel_x), _event_vmap, _xinit_permute, time_span, p, rtol = 1.4e-8, atol = 1.4e-8, mxstep = None)
        
        self.assertTrue(jnp.allclose(_event_times_happens[permutation], _event_times_happens_ooo))
        self.assertFalse(any(_event_times_happens == time_span[0] - 1))
        self.assertFalse(any(_event_times_happens == time_span[-1] + 1))

        ########################################################################################################################
        # Event already happened
        p = jnp.array([2, 3, 25.]) # event already happened (time_span[0] ...) # gradients match exactly
        xinit = jnp.column_stack((jnp.arange(2, 10. * 2 + 2).reshape(-1, 2), p[-1] * jnp.ones(shape = (10, ))))
        _xinit, unravel_x = flatten_util.ravel_pytree(xinit)
        _xinit_permute, _ = flatten_util.ravel_pytree(xinit[permutation])

        # time_span[0] - 1
        @partial(flatten_output, unravel_first_arg = unravel_x)
        def _event_vmap(x, t): return jax.vmap(lambda t, x : x[0] - 1e10, in_axes = (None, 0))(t, x)
        _event_times_happened = _custom_odeint_event(flatten_output(afunc, unravel_x), _event_vmap, _xinit, time_span, p, rtol = 1.4e-8, atol = 1.4e-8, mxstep = None)
        _event_times_happened_ooo = _custom_odeint_event(flatten_output(afunc, unravel_x), _event_vmap, _xinit_permute, time_span, p, rtol = 1.4e-8, atol = 1.4e-8, mxstep = None)

        self.assertTrue(jnp.allclose(_event_times_happened[permutation], _event_times_happened_ooo))
        self.assertTrue(jnp.allclose(_event_times_happened, time_span[0] - 1))

        ########################################################################################################################
        # Event happens midway for some x while already happened for previous
        p = jnp.array([2, 3, 8.])
        xinit = jnp.column_stack((jnp.arange(2, 10. * 2 + 2).reshape(-1, 2), p[-1] * jnp.ones(shape = (10, ))))
        _xinit, unravel_x = flatten_util.ravel_pytree(xinit)
        _xinit_permute, _ = flatten_util.ravel_pytree(xinit[permutation])

        # some time_span[0] - 1 while others have some value
        @partial(flatten_output, unravel_first_arg = unravel_x)
        def _event_vmap(x, t): return jax.vmap(lambda t, x : x[0] - 8, in_axes = (None, 0))(t, x)
        _event_times_midhappens = _custom_odeint_event(flatten_output(afunc, unravel_x), _event_vmap, _xinit, time_span, p, rtol = 1.4e-8, atol = 1.4e-8, mxstep = None)
        _event_times_midhappens_ooo = _custom_odeint_event(flatten_output(afunc, unravel_x), _event_vmap, _xinit_permute, time_span, p, rtol = 1.4e-8, atol = 1.4e-8, mxstep = None)

        self.assertTrue(jnp.allclose(_event_times_midhappens[permutation], _event_times_midhappens_ooo))
        self.assertTrue(_event_times_midhappens[0] == time_span[0] - 1)
        self.assertTrue(_event_times_midhappens[-1] <= time_span[-1])

        ########################################################################################################################
        # Event hasnt happened yet. (Will happen in the future)
        p = jnp.array([2, 3, -1.]) 
        xinit = jnp.column_stack((jnp.arange(2, 10. * 2 + 2).reshape(-1, 2), p[-1] * jnp.ones(shape = (10, ))))
        _xinit, unravel_x = flatten_util.ravel_pytree(xinit)
        _xinit_permute, _ = flatten_util.ravel_pytree(xinit[permutation])

        # time_span[-1] + 1
        @partial(flatten_output, unravel_first_arg = unravel_x)
        def _event_vmap(x, t): return jax.vmap(lambda t, x : x[0] - 0, in_axes = (None, 0))(t, x)
        _event_times_neverhappens = _custom_odeint_event(flatten_output(afunc, unravel_x), _event_vmap, _xinit, time_span, p, rtol = 1.4e-8, atol = 1.4e-8, mxstep = None)
        _event_times_neverhappens_ooo = _custom_odeint_event(flatten_output(afunc, unravel_x), _event_vmap, _xinit_permute, time_span, p, rtol = 1.4e-8, atol = 1.4e-8, mxstep = None)

        self.assertTrue(jnp.allclose(_event_times_neverhappens[permutation], _event_times_neverhappens_ooo))
        self.assertTrue(jnp.allclose(_event_times_neverhappens, time_span[-1] + 1))

    def test_event_gradients(self):
        
        # Event happens midway for some x while already happened for others
        p = jnp.array([2, 3, 8.])
        tf = jnp.array(5.)
        xinit = jnp.column_stack((jnp.arange(2, 10. * 2 + 2).reshape(-1, 2), p[-1] * jnp.ones(shape = (10, ))))

        def obj_event(x, t, p):
            ts = jnp.linspace(0, t, 10)
            solution, _ = odeint_event(afunc, event_vmap, transfer, x, ts, p)
            return jnp.mean((solution - jnp.ones_like(solution))**2)

        print("Starting forward mode auto diff sensitivities")
        fwd_x, fwd_t, fwd_p = jax.jacfwd(obj_event, argnums = (0, 1, 2))(xinit, tf, p)

        print("Starting reverse mode auto diff sensitivities")
        rev_x, rev_t, rev_p = jax.grad(obj_event, argnums = (0, 1, 2))(xinit, tf, p)

        self.assertTrue(jnp.allclose(fwd_p, rev_p, atol = atol*10, rtol = rtol*10))
        self.assertTrue(jnp.allclose(fwd_x, rev_x, atol = atol*10, rtol = rtol*10))
        self.assertTrue(jnp.allclose(fwd_t, rev_t, atol = atol*10, rtol = rtol*10))

        print("Starting finite diff sensitivities")
        base = obj_event(xinit, tf, p)
        def fd(eps):
            vars, unravel = flatten_util.ravel_pytree((xinit, tf, p))
            grads = jax.vmap(lambda v : (obj_event(*unravel(vars + eps * v)) - base) / eps)(jnp.eye(len(vars)))
            return unravel(grads)
        
        eps = 1e-5
        fd_x, fd_t, fd_p = fd(eps)
        self.assertTrue(jnp.allclose(fwd_x, fd_x, atol = 100 * eps))
        self.assertTrue(jnp.allclose(fwd_t, fd_t, atol = 100 * eps))
        self.assertTrue(jnp.allclose(fwd_p, fd_p, atol = 100 * eps))

        # checking forward-over-reverse works
        jax.hessian(obj_event, argnums = (0, 1, 2))(xinit, tf, p)