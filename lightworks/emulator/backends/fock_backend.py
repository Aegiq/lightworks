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

import numpy as np

from ...sdk.circuit.photonic_compiler import CompiledPhotonicCircuit
from ...sdk.state import State
from ...sdk.tasks.analyzer import AnalyzerTask
from ...sdk.tasks.sampler import SamplerTask
from ...sdk.tasks.simulator import SimulatorTask
from ...sdk.tasks.task import Task
from ..simulation import (
    AnalyzerRunner,
    RunnerABC,
    SamplerRunner,
    SimulatorRunner,
)
from ..utils import BackendError
from .abc_backend import BackendABC

# ruff: noqa: ARG002, D102


class FockBackend(BackendABC):
    """
    Base class for all backends. An outline of all possible functions should
    be included here.
    """

    def run(self, task: Task) -> dict:
        runner: RunnerABC
        data = task._generate_task()
        if isinstance(data, SimulatorTask):
            runner = SimulatorRunner(data, self.probability_amplitude)
        elif isinstance(data, AnalyzerTask):
            runner = AnalyzerRunner(data, self.probability)
        elif isinstance(data, SamplerTask):
            runner = SamplerRunner(data, self.full_probability_distribution)
            task.probability_distribution = runner.distribution_calculator()
        else:
            raise BackendError("Task not supported on current backend.")
        return runner.run()

    # Below defaults are defined for all possible methods in case they are
    # called without being implemented.

    def probability_amplitude(
        self,
        unitary: np.ndarray,
        input_state: list[int],
        output_state: list[int],
    ) -> complex:
        raise BackendError(
            "Current backend does not implement probability_amplitude method."
        )

    def probability(
        self,
        unitary: np.ndarray,
        input_state: list[int],
        output_state: list[int],
    ) -> float:
        raise BackendError(
            "Current backend does not implement probability method."
        )

    def full_probability_distribution(
        self, circuit: CompiledPhotonicCircuit, input_state: State
    ) -> dict:
        raise BackendError(
            "Current backend does not implement full_probability_distribution "
            "method."
        )
