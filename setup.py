# setup.py
import setuptools
from pathlib import Path

# read the long description from your README
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="sat-tile-stack",
    version="0.1.0",
    description="Generate daily, multi-band Sentinel-2 tile time-stacks around a point from STAC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    license="MIT",
    python_requires=">=3.7",
    package_dir={"": "src"},                    # look for packages under src/
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "geopandas",
        "xarray",
        "rioxarray",
        "matplotlib",
        "dask[complete]",
        "pystac-client",
        "planetary-computer",
        "rasterio",
        "stackstac",
        "pyproj",
        "shapely",
        "scipy",
    ],
    extras_require={
        "dev": ["pytest", "flake8", "black", "mypy"],
    },
)
