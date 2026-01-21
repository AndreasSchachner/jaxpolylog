# ==============================================================================
# This code is written by Andreas Schachner. Without the author's permission, this code must not be shared with anyone else or used for any other projects than those involving the author directly.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu or at a.schachner@lmu.de.
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file holds functions for polylogarithms using JAX.
# ------------------------------------------------------------------------------


# Important standard libraries
import os, sys, warnings
from functools import partial

# Important JAX libraries
import jax
from jax import custom_vjp
from jax import jit, vmap, config
import jax.numpy as jnp
from jax import Array
from numpy.typing import ArrayLike

# Enable 64 bit precision
config.update("jax_enable_x64", True)

@partial(jit, static_argnums = (2,))
def intgrand(z: complex, t: complex, s: int) -> complex:
    r"""
    **Description:**
    Integrand for the integral representation of the polylogarithm function.
    
    Args:
        z (complex): The input value(s) at which to evaluate the integrand. Can be a scalar or an array.
        t (complex): The integration variable.
        s (int): The order of the polylogarithm. Must be an integer.  
        
    Returns:
        complex: The computed integrand values.
    
    """
    return jnp.log(t)**(s-1)/(1-z*t)


@partial(custom_vjp, nondiff_argnums=(1,2,))
@partial(jit, static_argnums=(1,2,))
def jax_polylog(z: complex, s: int, p_range: int) -> complex:
    r"""
    **Description:**
    This function computes the polylogarithm of order `s` at point `z` using JAX. It supports automatic differentiation and is optimized for performance. The function is defined for integer values of `s` and can handle both real and complex inputs for `z`. 
    
    **Mathematical Definition:**
    The polylogarithm function is defined as:
    
    .. math::
        \text{Li}_s(z) = \sum_{k=1}^{\infty} \frac{z^k}{k^s}
    
    for |z| < 1 and can be analytically continued to other values of `z`.
    For integer `s`, the function can be expressed in terms of elementary functions for specific values of `s`: 
    - For `s = 1`: `Li_1(z) = -ln(1 - z)`
    - For `s = 0`: `Li_0(z) = z / (1 - z)`
    - For `s = -1`: `Li_{-1}(z) = z / (1 - z)^2`
    - For `s = -2`: `Li_{-2}(z) = z(1 + z) / (1 - z)^3`
    - For `s = -3`: `Li_{-3}(z) = z(1 + 4z + z^2) / (1 - z)^4`
    - For `s = -4`: `Li_{-4}(z) = z(1 + 11z + 11z^2 + z^3) / (1 - z)^5`
    - For `s = -5`: `Li_{-5}(z) = z(1 + 26z + 66z^2 + 26z^3 + z^4) / (1 - z)^6`
    - For `s = -6`: `Li_{-6}(z) = z(1 + 57z + 302z^2 + 302z^3 + 57z^4 + z^5) / (1 - z)^7`
    - For `s = -7`: `Li_{-7}(z) = z(1 + 120z + 1191z^2 + 2416z^3 + 1191z^4 + 120z^5 + z^6) / (1 - z)^8`
    - For `s = -8`: `Li_{-8}(z) = z(1 + 247z + 4293z^2 + 15619z^3 + 15619z^4 + 4293z^5 + 247z^6 + z^7) / (1 - z)^9`
    - For `s = -9`: `Li_{-9}(z) = z(1 + 502z + 14608z^2 + 88234z^3 + 156190z^4 + 88234z^5 + 14608z^6 + 502z^7 + z^8) / (1 - z)^10`
    - For other integer values of `s`, the function is computed using the series definition.
    
    Args:
        z (complex): The input value(s) at which to evaluate the polylogarithm. Can be a scalar or an array.
        s (int): The order of the polylogarithm. Must be an integer.
        p_range (int): The number of terms to include in the series expansion for non-predefined `s` values. Higher values increase accuracy but also computation time.
        
    Returns:
        complex: The computed polylogarithm values at the input `z`.
        
    **Example Usage:**
    ```python
    import jax.numpy as jnp
    from jaxvacua.polylogs import jax_polylog as polylog
    z = jnp.array([0.5, 0.9, 1.0])
    s = 2
    result = polylog(z, s, p_range=1000)
    print(result)
    ```
    
    """
    # Check if s is an integer
    if not isinstance(s, int):
        raise ValueError("The order 's' must be an integer.")
    # Handle special cases for specific integer values of s
    if s==1:
        return -jnp.log(1-z)
    elif s==0:
        return z/(1-z)
    elif s==-1:
        return z/(1-z)**2
    elif s==-2:
        return ((z*(1 + z))/(1 - z)**3)
    elif s==-3:
        return (z*(1 + 4*z + z**2))/(-1 + z)**4
    elif s==-4:
        return ((z*(1 + 11*z + 11*z**2 + z**3))/(1 - z)**5)
    elif s==-5:
        return (z*(1 + 26*z + 66*z**2 + 26*z**3 + z**4))/(-1 + z)**6
    elif s==-6:
        return ((z*(1 + 57*z + 302*z**2 + 302*z**3 + 57*z**4 + z**5))/(1 - z)**7)
    elif s==-7:
        return (z*(1 + 120*z + 1191*z**2 + 2416*z**3 + 1191*z**4 + 120*z**5 + z**6))/(-1 + z)**8
    elif s==-8:
        return ((z*(1 + 247*z + 4293*z**2 + 15619*z**3 + 15619*z**4 + 4293*z**5 + 247*z**6 + z**7))/(1 - z)**9)
    elif s==-9:
        return (z*(1 + 502*z + 14608*z**2 + 88234*z**3 + 156190*z**4 + 88234*z**5 + 14608*z**6 + 502*z**7 + z**8))/(-1 + z)**10
    else:
        polylog_range = jnp.arange(1,p_range)
        return jnp.sum(z**polylog_range/polylog_range**s)

def jax_polylog_fwd(z: complex,s: int,p_range: int) -> tuple:
    r"""
    **Description:**
    Forward pass for the custom VJP of the polylogarithm function.
    
    Args:
        z (complex): The input value(s) at which to evaluate the polylogarithm. Can be a scalar or an array.
        s (int): The order of the polylogarithm. Must be an integer.
        p_range (int): The number of terms to include in the series expansion for non-predefined `s` values. Higher values increase accuracy but also computation time.
        
    Returns:
        tuple: A tuple containing:
            - complex: The computed polylogarithm values at the input `z`.
            - tuple: A tuple of residuals needed for the backward pass.
    """
    
# Returns primal output and residuals to be used in backward pass by f_bwd.
    #return jax_polylog(z,s,p_range), (jax_polylog(z,s-1,p_range)/z,0,0) return jax_polylog(z,s,p_range), (jax_polylog(z,s-1,p_range)/z,0.+0*0j,0+0.*0j)
    return jax_polylog(z,s,p_range), (jax_polylog(z,s-1,p_range)/z,0.+0*0j,0+0.*0j)

def jax_polylog_bwd(s: int,p_range: int,res: tuple, g: ArrayLike) -> tuple:
    r"""
    **Description:**
    Backward pass for the custom VJP of the polylogarithm function.
    
    Args:
        s (int): The order of the polylogarithm. Must be an integer.
        p_range (int): The number of terms to include in the series expansion for non-predefined `s` values. Higher values increase accuracy but also computation time.
        res (tuple): A tuple of residuals from the forward pass.
        g (ArrayLike): The gradient of the output with respect to some scalar value.
        
    Returns:
        tuple: A tuple containing the gradient of the input `z`.
    """
# Returns the cotangent of the primal inputs and the residuals from f_fwd.
    #g is the cotangent of the output
    #res contains the residuals computed in f_fwd
    #return (g * res[0],0,0) # Derivative w.rt. z, s, p_range
    #return (g * res[0],0+0*0j,0+0.*0j) # Derivative w.rt. z, s, p_range        
    y, _,_ = res # Gets residuals computed in f_fwd
    return ((g * y+0.*0j,))

jax_polylog.defvjp(jax_polylog_fwd, jax_polylog_bwd)

jax_polylog_vmap_tmp = jax.vmap(jax_polylog,in_axes=(0,None,None))

@partial(jit, static_argnames=['s','p_range'])
def jax_polylog_vmap(z: complex,s: int,p_range: int) -> complex:
    r"""
    Vectorized version of the polylogarithm function using JAX's vmap.
    
    Args:
        z (complex): The input values at which to evaluate the polylogarithm. Must be a 1D array.
        s (int): The order of the polylogarithm. Must be an integer.
        p_range (int): The number of terms to include in the series expansion for non-predefined `s` values. Higher values increase accuracy but also computation time.
        
    Returns:
        complex: The computed polylogarithm values at the input `z`.
    """
    
    return jax_polylog_vmap_tmp(z,s,p_range)

























