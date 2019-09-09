import pytest

import numpy as np
import xarray as xr


from xrbispectral.utils import abs_sq, xr_abs_sq

@pytest.fixture
def z_arr():
    return np.arange(2**6, dtype=complex)


def test_abs_sq(z_arr):
    np.testing.assert_allclose(abs_sq(z_arr), np.abs(z_arr)**2)
    

def test_xr_abs_sq(z_arr):
    z = xr.DataArray(z_arr).chunk()
    res = xr_abs_sq(z).compute()
    np.testing.assert_allclose(res, np.abs(z_arr)**2)