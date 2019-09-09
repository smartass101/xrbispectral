import numpy as np
import numba
import xarray as xr


@numba.vectorize(nopython=True)
def abs_sq(x):
    """Return |x|^2 = Re(x)^2 + Im(x)^2
    
    Based on https://stackoverflow.com/a/37846553/4779220
    and tested to be faster than abs()**2
    """
    return x.real**2 + x.imag**2


def xr_abs_sq(x):
    """xarray (and dask-compatible) wrapper of `abs_sq`"""
    return xr.apply_ufunc(abs_sq, x, dask='allowed')