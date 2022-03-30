import jax
import jax.numpy as jnp


def vgrad(func):
    """
    grad function for vector
    """
    return jax.grad(lambda r, gamma: jnp.sum(func(r, gamma)))


def soft_threshold(r, gamma):
    """
    soft-thresholding function
    """
    return jnp.maximum(jnp.abs(r) - gamma, 0.0) * jnp.sign(r)


def func_mmse(r, gamma):
    #(y*self.alpha2[0]/(self.alpha2[0]+tau2))*self.p[0]*self.gauss(y,(self.alpha2[0]+tau2))/((1-self.p[0])*self.gauss(y, tau2) + self.p[0]*self.gauss(y, (self.alpha2[0]+tau2)))
    pass


def df(r, gamma, func):
    """
    divergence-free function
    """
    grad_func = vgrad(func)
    return func(r, gamma) - jnp.mean(grad_func(r, gamma)) * r