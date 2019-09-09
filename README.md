# Bispectral calculation using xarray, dask and numba

Calculates the bispectrum and bicoherence for spectra contained in `xarray.DataArray` containers using `Dask` and `Numba` for efficient memory management and parallelization.

Based on notation used in:

- Cziegler et al. Phys. Plasmas 20, 055904 (2013)
- P. Manz et al 2017 Nucl. Fusion 57 086022


## Usage

```python
from xrbispectral import calculate_bispectral

# auto-bispectral analysis of spectrum
ds_dask = calculate_bispectral(spectrum, freq_dim='f', avg_dim='t')
ds = ds_dask.compute()  # when appropriate
```