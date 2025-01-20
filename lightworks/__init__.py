# Copyright 2024 Aegiq Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lightworks
==========

Lightworks is a high-level interface for photonic quantum computing,
enabling a range of transformations to be implemented and executed on both
remote photonic hardware and with local simulation tools.

Key objects:

    Circuit : Provides a tool for building linear optic circuits across a
        number of modes. Circuit also supports addition for combination of
        sub-circuits and can be used with built-in Parameter objects to allow
        for adjustment of the circuit configuration after creation. Circuit
        also has a display method so the created circuit can be viewed.

    State : Represents the photonic fock states which are input and output from
        the system. State objects are hashable, and so can be used as keys in
        dictionaries, and support both addition and merging to combine states
        together.

    emulator : Provides a set of local simulation tools for the testing and
        verification of outputs from a given problem. There is a number of
        different objects for simulation of the system, which provide various
        capabilities and outputs.

    qubit : Module for implementing the qubit paradigm of quantum computing on
        photonic linear optic systems.

"""

import contextlib

from . import emulator, interferometers, qubit, tomography
from .__settings import settings
from .__version import __version__
from .sdk.circuit import Parameter, ParameterDict, PhotonicCircuit, Unitary
from .sdk.state import State
from .sdk.tasks import Analyzer, Batch, Sampler, Simulator
from .sdk.utils import (
    PostSelection,
    PostSelectionFunction,
    db_loss_to_decimal,
    decimal_to_db_loss,
    random_permutation,
    random_unitary,
)
from .sdk.utils.exceptions import *
from .sdk.visualisation import Display

# If installed then also import the remote module
with contextlib.suppress(ModuleNotFoundError):
    import lightworks_remote as remote

__all__ = [
    "Analyzer",
    "Display",
    "Parameter",
    "ParameterDict",
    "PhotonicCircuit",
    "PostSelection",
    "PostSelectionFunction",
    "Sampler",
    "Simulator",
    "State",
    "Unitary",
    "db_loss_to_decimal",
    "decimal_to_db_loss",
    "emulator",
    "interferometers",
    "qubit",
    "random_permutation",
    "random_unitary",
    "settings",
    "tomography",
]
