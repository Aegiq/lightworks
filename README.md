# Lightworks

Lightworks is designed to enable the encoding of jobs on photonic quantum computing hardware, which can then either be executed by a remote QPU or locally simulated using the provided emulation tools.

Key features:
- Circuit
- Emulator

## Usage

Python 3.10+ is required.

## Documentation

Documentation of this package is hosted at: TBC

## Contributing

Contributions to Lightworks can be made via a pull request. 

Some things to keep in mind before contributing
1) Any pull requests should currently be made to the development branch and not main. 
2) We aim to follow the Google Python style guide (https://google.github.io/styleguide/pyguide.html) including their proposed doc strings format. 
3) The exisiting unit tests should be used to ensure the core functionality of Lightworks remains intact. Additionally, any new features should ideally include a set of tests.
4) Type hints are used throughout the code to indicate the expected inputs and return for each class and function within Lightworks. These are also used for generating the Sphinx documentation.
5) Ideally a line limit of 80 is used across the code.