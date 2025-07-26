# GMUM Continual Learning Framework
![Build Status](https://img.shields.io/github/actions/workflow/status/gmum./gmumcl/ci-cd.yml) ![Version](https://img.shields.io/badge/gmumcl_ver.-0.1.0-blue) ![Python](https://img.shields.io/badge/Python-3.8-purple)

<p align="right"><img style="float: right;" src="gmum.png" alt="logo" width="100"/></p>

**Jan Miksa, Patryk Krukowski** @ *GMUM JU*

***Work in progress...***

Python 3.8 due to Continuum dependency.  

Based on: https://github.com/OneAndZero24/local-cl

## Features
- YAML Configuration
- Lightning Fabric
- Custom Plugin System for Methods
- Incremental Classifier

| Method | Status | Scenario | Status |
| ------ | -- | ------ | -- |
| Naive | ✅ | CI | ✅ |
| LwF | ✅ | DI | ✅ |
| EWC | ✅ | TI | ✅ |
| Sharpening | ✅ | II | ✅ |
| SI | ✅ | Permuted | ⭕️ |
| MAS | ✅ |  |  |
| Dreaming | ✅ |  |  |

## TODO
- [ ] Fix all composer configs
- [ ] Add more methods
- [ ] PyPI Release, Pipeline is ready

## Example
Example can be found in `tests/example_usage.py`. It demonstrates how to use the `gmumcl` package after installation.

## Acknowledgements
- Project Structure based on [template](https://github.com/sobieskibj/templates/tree/master) by Bartłomiej Sobieski
