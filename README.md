# sat-tile-stack

NOTE: this repo is under active development

`sat_tile_stack` is a package for constructing deep learning-ready datasets from satellite imagery.  Built on `Microsoft Planetary Computer` and `stackstac `, `sat_tile_stack` enables users to compile time-series datasets at regular cadence (e.g., daily).  Functional options include tracking cloudiness, imagery availability, and spatial mask generation at that same cadence for ease of use in attention-enabled deep learning frameworks.

## Installation and Dependencies
For now, `sat_tile_stack` can be installed via
```
pip install --upgrade --force-reinstall git+https://github.com/jharlanr/sat-tile-stack.git
```

## Example usage
Check out a brief tutorial [here](https://github.com/jharlanr/sat-tile-stack/tree/main/notebooks/cc_tutorial.ipynb)


# How to Contribute

This package is being actively developed
[here](https://github.com/jharlanr/sat-tile-stack).

If you would like to add new functionality, we welcome new contributions from
anyone as pull requests on [our Github repo](https://github.com/jharlanr/sat-tile-stack).

No contribution is too small, and we also welcome requests for new features
or bug reports. To contribute to `sat-tile-stack` this way, please 
[open an issue on GitHub](https://github.com/jharlanr/sat-tile-stack/issues).
