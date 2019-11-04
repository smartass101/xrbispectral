# Bispectral calculation using xarray, dask and numba

Calculates the bispectrum and bicoherence for spectra contained in `xarray.DataArray` containers using `Dask` and `Numba` for efficient memory management and parallelization.

Based on notation used in:

- Cziegler et al. Phys. Plasmas 20, 055904 (2013)
- P. Manz et al 2017 Nucl. Fusion 57 086022

## Installation

    pip install git+https://github.com/smartass101/xrbispectral

## Usage

See [this usage example Jupyter notebook](./usage_example.ipynb) for an example with 3 bi-coherent frequencies.
