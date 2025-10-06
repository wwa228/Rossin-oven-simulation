import unittest

import jax
import jax.numpy as jnp
from jax import flatten_util
import jax.random as jrandom

from oven.utils import root_finding_rev, root_finding_fwd


ndim = 4 # Note that in the drying example ndim over the root-finding problem is 1
f = lambda z, p : z - jnp.tanh(jnp.dot(p[0], z) + p[1])
p = (jrandom.normal(jrandom.PRNGKey(0), (ndim, ndim)) / jnp.sqrt(ndim), jrandom.normal(jrandom.PRNGKey(1), (ndim,)))
p, unravel = flatten_util.ravel_pytree(p)

def obj_rev(p, reuse):
    z_star = root_finding_rev(f, jnp.ones(ndim), unravel(p), reuse)
    return jnp.mean(z_star**2), z_star

def obj_fwd(p, reuse):
    z_star = root_finding_fwd(f, jnp.ones(ndim), unravel(p), reuse)
    return jnp.mean(z_star**2), z_star


class TestUtils(unittest.TestCase):

    def test_root_finding(self):
        
        _, z_star_rev = obj_rev(p, False)
        _, z_star_rev_reuse = obj_rev(p, True)
        self.assertTrue(jnp.allclose(z_star_rev, z_star_rev_reuse))

        _, z_star_fwd = obj_fwd(p, False)
        _, z_star_fwd_reuse = obj_fwd(p, True)
        self.assertTrue(jnp.allclose(z_star_fwd, z_star_fwd_reuse))
        self.assertTrue(jnp.allclose(z_star_fwd, z_star_rev))

    def test_root_finding_gradients(self):
        
        rev_p, _ = jax.grad(obj_rev, argnums = 0, has_aux = True)(p, False)
        rev_reuse_p, _ = jax.grad(obj_rev, argnums = 0, has_aux = True)(p, True)
        self.assertTrue(jnp.allclose(rev_p, rev_reuse_p))

        fwd_p, _ = jax.jacfwd(obj_fwd, argnums = 0, has_aux = True)(p, False)
        fwd_reuse_p, _ = jax.jacfwd(obj_fwd, argnums = 0, has_aux = True)(p, True)
        self.assertTrue(jnp.allclose(fwd_p, fwd_reuse_p))
        self.assertTrue(jnp.allclose(fwd_p, rev_p))

        # Testing with finite difference
        base, _ = obj_fwd(p, True)
        def fd(eps):
            grads = jax.vmap(lambda v : (obj_fwd(p + v * eps, True)[0]  - base) / eps)(jnp.eye(len(p)))
            return grads

        eps = 1e-5
        fd_p = fd(eps)
        self.assertTrue(jnp.allclose(fd_p, fwd_p, atol = 100 * eps))

        # Note that higher order derivatives should be computed without saving inverse
        # Hessian (fwd-over-rev) is only compatible with custom forward mode
        hess_rev, _ = jax.jacrev(jax.jacrev(obj_rev, has_aux = True), has_aux = True)(p, False)
        hess, _ = jax.hessian(obj_fwd, has_aux = True)(p, False)

        def fd(eps):
            _grad = jax.grad(lambda _p : obj_fwd(_p, True)[0])
            grads = jax.vmap(lambda v : (_grad(p + v * eps)  - _grad(p - v * eps)) / 2 / eps)(jnp.eye(len(p)))
            return grads

        hess_fd = fd(eps)
        self.assertTrue(jnp.allclose(hess_rev, hess))
        self.assertTrue(jnp.allclose(hess, hess_fd, atol = 100 * eps))
