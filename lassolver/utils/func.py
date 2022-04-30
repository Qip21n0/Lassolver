import jax
import jax.numpy as jnp
from scipy.stats import norm


def vgrad(func):
    """
    grad function for vector
    """
    return jax.vmap(jax.grad(func, argnums=(0)), (0, None))


def soft_threshold(r, gamma):
    """
    soft-thresholding function
    """
    return jnp.maximum(jnp.abs(r) - gamma, 0.0) * jnp.sign(r)


def func_mmse(r, gamma):
    rho = jnp.mean(soft_threshold(r, gamma) != 0)
    xi = rho**(-1) + gamma
    top = norm.pdf(r, loc=0, scale=xi**0.5) / xi
    bottom = rho * norm.pdf(r, loc=0, scale=xi**0.5) + (1 - rho) * norm.pdf(r, loc=0, scale=gamma**0.5)
    return top / bottom * r


def df(r, gamma, func):
    """
    divergence-free function
    """
    grad_func = vgrad(func)
    return func(r, gamma) - jnp.mean(grad_func(r, gamma)) * r