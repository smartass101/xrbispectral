import pytest
import itertools

import numpy as np
import xarray as xr
from scipy import signal


from xrbispectral.calculate import calculate_bispectral


@pytest.fixture
def sample_signal():
    N = 2**16
    x = np.arange(N)        # time
    NFFT = 32
    w = 2*np.pi*np.fft.fftfreq(NFFT) # angular frequencies
    i, j = 3, 9                      # selected frequency indices
    y = np.sin(w[i]*x) + np.sin(w[j]*x) + np.sin(w[i+j]*x) # signal
    # the spectrum of the signal is calculated for several windows (for averaging)
    f, t, s = signal.spectrogram(y, fs=1.0, window='boxcar', nperseg=NFFT, 
                                 scaling='spectrum',
                                 mode='complex', return_onesided=False, noverlap=0)
    xs = xr.DataArray(s, coords=[('f', f), ('t', t)], name='sample_spectrum')
    xs.attrs['base_fs'] = xs.f.isel(f=[i,j]).values
    return xs.sortby('f')



def test_bicoherence_calculation(sample_signal):
    ds_dask = calculate_bispectral(sample_signal, freq_dim='f', avg_dim='t')
    ds = ds_dask.compute()
    base_fs = list(sample_signal.base_fs)
    f_s = [sum(base_fs)]*2
    f_ckw = dict(f=xr.Variable('p', base_fs+f_s),
                 f1=xr.Variable('p', f_s+base_fs))
    np.testing.assert_allclose(ds.bicoherence.sel(**f_ckw), np.ones(4))
    mask = xr.full_like(ds.bicoherence, True, dtype=bool)
    mask.loc[f_ckw] = False
    assert ds.bicoherence.where(mask).isnull().all().item()
