[![Tests](https://github.com/Aegiq/lightworks/actions/workflows/tests.yml/badge.svg?event=push)](https://github.com/Aegiq/lightworks/actions/workflows/tests.yml)
[![Docs](https://github.com/Aegiq/lightworks/actions/workflows/sphinx_deploy.yml/badge.svg?event=push)](https://github.com/Aegiq/lightworks/actions/workflows/sphinx_deploy.yml)
[![Pyversions](https://img.shields.io/pypi/pyversions/lightworks.svg?style=plastic)](https://pypi.org/project/lightworks/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14925692.svg)](https://doi.org/10.5281/zenodo.14925692)


# Lightworks

Lightworks is an open-source Python SDK, designed for the encoding of linear optic circuits for application in photonic quantum computing. These circuits can be packaged with the other SDK components to create quantum jobs for execution on photonic hardware. Lightworks focuses on discrete-variable quantum computing, and can be utilized for both qubit and boson sampling paradigms.

Included within Lightworks is also an emulator, allowing users to evaluate the operation and performance of a particular configuration before hardware execution. There is a number of simulation objects, each offering a differing functionality, ranging from direct quantum state evolution to replicating the typical sampling process from a photonic system. The emulator also supports complex photonic specific noise modelling, providing a valuable insight into the effect of imperfections in photon generation, QPU programming, and detectors, on a target algorithm.

## Usage

Python versions 3.10-3.13 are supported.

Lightworks can be installed through pip using the command:

```bash
pip install lightworks
```

## Documentation

Documentation of this package is hosted at: https://aegiq.github.io/lightworks/

## Contributing

Contributions to Lightworks can be made via a pull request. If you have an idea for a feature that you'd like to implement it may be best to first raise this in the issues sections, as it may be the case that this is already in development internally or is potentially incompatible with the existing Lightworks framework.

Before contributing, please see [Contributing](https://aegiq.github.io/lightworks/contributing.html) in the documentation for more guidance on code testing & formatting requirements. 