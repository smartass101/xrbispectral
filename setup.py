import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xrbispectral",
    version="0.0.1",
    author="Ondrej Grover",
    author_email="ondrej.grover@gmail.com",
    description="bispectral analysis using xarray, dask and numba",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smartass101/xrbispectral",
    packages=setuptools.find_packages(),
    install_requires=['xarray', 'dask', 'numba'],
)