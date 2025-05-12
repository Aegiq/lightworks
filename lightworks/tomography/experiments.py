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

import typing
from collections import UserList
from dataclasses import dataclass

from lightworks.sdk.circuit import PhotonicCircuit
from lightworks.sdk.state import State


@dataclass(slots=True, kw_only=True)
class _TomographyExperiment:
    """
    Contains the data for running a required state tomography experiment.
    """


@dataclass(slots=True, kw_only=True)
class StateTomographyExperiment(_TomographyExperiment):
    """
    Contains the data for running a required state tomography experiment.
    """

    circuit: PhotonicCircuit
    measurement_basis: str


@dataclass(slots=True, kw_only=True)
class ProcessTomographyExperiment(_TomographyExperiment):
    """
    Contains the data for running a required tomography experiment.
    """

    circuit: PhotonicCircuit
    input_state: State
    input_basis: str
    measurement_basis: str


TE = typing.TypeVar(
    "TE", StateTomographyExperiment, ProcessTomographyExperiment
)


class _TomographyList(UserList[TE]):
    """
    Base class for all list of tomography experiments.
    """

    @property
    def all_circuits(self) -> list[PhotonicCircuit]:
        """
        Returns a list of circuits corresponding to each of the tomography
        experiments in the list.
        """
        return [exp.circuit for exp in self]

    @property
    def all_measurement_basis(self) -> list[str]:
        """
        Returns a list of the measurement basis used for each tomography
        experiment.
        """
        return [exp.measurement_basis for exp in self]


class StateTomographyList(_TomographyList[StateTomographyExperiment]):
    """
    Stores a list of state tomography experiments.
    """


class ProcessTomographyList(_TomographyList[ProcessTomographyExperiment]):
    """
    Stores a list of tomography experiments.
    """

    @property
    def all_inputs(self) -> list[State]:
        """
        Returns a list of inputs corresponding to each of the tomography
        experiments in the list.
        """
        return [exp.input_state for exp in self]

    @property
    def all_input_basis(self) -> list[str]:
        """
        Returns a list of the input basis used for each tomography experiment.
        """
        return [exp.input_basis for exp in self]
