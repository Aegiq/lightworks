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


from dataclasses import dataclass

from ...emulator.components import Detector, Source
from ..circuit.photonic_compiler import CompiledPhotonicCircuit
from ..state import State
from ..utils.post_selection import PostSelectionType


@dataclass(slots=True)
class TaskData:
    """
    Base class for all task dataclasses.
    """


@dataclass(slots=True)
class AnalyzerTask(TaskData):  # noqa: D101
    circuit: CompiledPhotonicCircuit
    inputs: list[State]
    expected: dict[State, State | list[State]] | None
    post_selection: PostSelectionType


@dataclass(slots=True)
class SamplerTask(TaskData):  # noqa: D101
    circuit: CompiledPhotonicCircuit
    input_state: State
    n_samples: int
    source: Source | None
    detector: Detector | None
    post_selection: PostSelectionType
    min_detection: int
    random_seed: int | None
    sampling_mode: str


@dataclass(slots=True)
class SimulatorTask(TaskData):  # noqa: D101
    circuit: CompiledPhotonicCircuit
    inputs: list[State]
    outputs: list[State] | None