# Lightworks

Lightworks is an open source Python SDK, designed for the encoding of jobs & algorithms before execution on photonic quantum computing hardware. It focuses on discrete-variable quantum computing, and can be utilised for both qubit-based and boson sampling paradigms.

Included within Lightworks in also an emulator, allowing users to evaluate the operation and performance of a particular configuration before hardware execution. There is a number of simulation objects, each offering a differing functionality, ranging from direct quantum state evolution to replicating the typical sampling process from a photonic system. The emulator also supports complex photonic specific noise modelling, providing a valuable insight into the effect of imperfections in photon generation, QPU programming, and detectors on a target algorithm.

Key features:
- Circuit
- State
- Emulator

## Usage

Python 3.10+ is required.

Currently the package is not hosted on pypi, this will be added once general 1.0 release is ready. For now, installation can be achieved with:

```console
pip install git+https://github.com/Aegiq/lightworks
```

## Documentation

Documentation of this package is hosted at: https://aegiq.github.io/lightworks/

## Contributing

Contributions to Lightworks can be made via a pull request. If you have an idea for a feature that you'd like to implement it may be best to first raise this in the issues sections, as it may be the case that this is already in developement internally or is potentially incompatiable with the existing Lightworks framework.

Some things to keep in mind before contributing:
1) Any pull requests should currently be made to the development branch and not main. 
2) We aim to follow the Google Python style guide (https://google.github.io/styleguide/pyguide.html) including their proposed doc strings format. 
3) The exisiting unit tests should be used to ensure the core functionality of Lightworks remains intact. Additionally, any new features should ideally include a set of tests.
4) Type hints are used throughout the code to indicate the expected inputs and return for each class and function within Lightworks. These are also used for generating the Sphinx documentation.
5) Where possible, a line limit of 80 is used across the code.
