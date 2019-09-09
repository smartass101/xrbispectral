import numpy as np
import xarray as xr


from .utils import xr_abs_sq


def calculate_bispectral(n_spectrum, v_spectrum=None,
                  freq_dim='frequency', avg_dim='time', avg_rolling=None,
                  noise_level=None):
    """Calculate bispectral quantites from xarray spectra using dask and numba
    
    Uses definitions as in Cziegler et al. Phys. Plasmas 20, 055904 (2013)
    
    Parameters
    ----------
    n_spectrum: xarray.DataArray
        complex spectrum of the n signal, frequencies assumed to be ordered
        must contain `freq_dim` and `avg_dim` dimensions
    v_spectrum: xarray.DataArray, optional
        v singal spectrum, defaults to n_spectrum (i.e. auto-bispectrum) none given
    freq_dim: str, optional
        name of the frequency dimention in the signals
    avg_dim: str, optional
        name of the averaging dimension, typically time, will atke mean() over it
    avg_rolling: int, optional
        if given, will use a centered rolling window of this size for the mean() averaging
    noise_level: float, optional
        below this level the power normalization in bicoherence calculation is set to 0,
        i.e.the bicoherence will be nan
        
    Returns
    -------
    bispectral_ds: xarray.Dataset
        contains dimensions:
            - `f1`: source frequency, positive
            - `f`: target frequency, positive
        chunked along the `f1` dimensions, use `.compute()` to evaluate the Dask arrays
        data vars:
            - `bispectrum`: complex
            - `n1_norm`: average power of the `n` signal in the `f1` dim
            - `n_norm`: average power of the `n` signal in the `f` dim
            - `v_norm`: average power of the `v` signal in the `f` dim
            - `norm`: product of the norms above
            - `bicoherence`: squared bicoherence = |bispectrum|^2/norm
                             nan where `norm < noise_level`
    """
    if v_spectrum is None:
        v_spectrum = n_spectrum
    # TODO test assumption of the same freq spectra
    f = n_spectrum.coords[freq_dim]
    f_min = f.where(f > 0).min().item()
    fsl = slice(f_min, None)  # use only positive f1, f
    # TODO assumes ordered freqs
    n_pos = n_spectrum.sel({freq_dim: fsl})
    n1, n = (n_pos.rename({freq_dim: fn}) for fn in ('f1', 'f'))

    # TODO should chunk to smaller in f?
    dv = v_spectrum.rename({freq_dim: 'f'}).chunk()
    # based on discussion in https://github.com/dask/dask/issues/4847
    v = xr.concat([dv.roll(f=i+1, roll_coords=False).sel(f=fsl)  # +1 as f1 > 0 strictly
                   for i in range(n1.sizes['f1'])],
                   dim=n1.f1)
    
    ds_t = xr.Dataset({
        'bispectrum': n1 * v * n.conj(),
        'n1_norm': xr_abs_sq(n1),
        'n_norm': xr_abs_sq(n),
        'v_norm': xr_abs_sq(v),
    })
    if avg_rolling is not None:
        ds = ds_t.rolling({avg_dim: avg_rolling}, center=True).mean()
    else:
        ds = ds_t.mean(dim=avg_dim)
    ds['norm'] = ds['n1_norm'] * ds['n_norm'] * ds['v_norm']
    if noise_level is None:  # TODO something better
        noise_level = np.finfo(n.dtype).eps
    ds['bicoherence'] = xr_abs_sq(ds['bispectrum']) / ds['norm'].where(ds['norm'] > noise_level)
    return ds
    