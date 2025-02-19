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


from typing import Any

import numpy as np
from multimethod import multimethod
from numpy.typing import NDArray

from lightworks.emulator.simulation import (
    AnalyzerRunner,
    SamplerRunner,
    SimulatorRunner,
)
from lightworks.emulator.utils.exceptions import BackendError
from lightworks.sdk.circuit.photonic_compiler import CompiledPhotonicCircuit
from lightworks.sdk.results import (
    ProbabilityDistribution,
    Result,
    SamplingResult,
    SimulationResult,
)
from lightworks.sdk.state import State
from lightworks.sdk.tasks import Analyzer, Sampler, Simulator, Task

from .abc_backend import EmulatorBackend

# ruff: noqa: ARG002, D102


class FockBackend(EmulatorBackend):
    """
    Base class for all backends. An outline of all possible functions should
    be included here.
    """

    @multimethod
    def run(self, task: Task) -> Result[State, Any]:
        raise BackendError("Task not supported on current backend.")

    @run.register
    def run_simulator(self, task: Simulator) -> SimulationResult:
        data = task._generate_task()
        return SimulatorRunner(data, self.probability_amplitude).run()

    @run.register
    def run_analyzer(self, task: Analyzer) -> SimulationResult:
        data = task._generate_task()
        return AnalyzerRunner(data, self.probability).run()

    @run.register
    def run_sampler(self, task: Sampler) -> SamplingResult:
        data = task._generate_task()
        runner = SamplerRunner(data, self.full_probability_distribution)
        cached_results = self._check_cache(data)
        if cached_results is not None:
            runner.probability_distribution = cached_results["pdist"]
            runner.full_to_heralded = cached_results["full_to_herald"]
            task._probability_distribution = ProbabilityDistribution(
                cached_results["pdist"]
            )
        else:
            task._probability_distribution = ProbabilityDistribution(
                runner.distribution_calculator()
            )
            results = {
                "pdist": runner.probability_distribution,
                "full_to_herald": runner.full_to_heralded,
            }
            self._add_to_cache(data, results)
        return runner.run()

    # Below defaults are defined for all possible methods in case they are
    # called without being implemented. This shouldn't normally happen.

    def probability_amplitude(
        self,
        unitary: NDArray[np.complex128],
        input_state: list[int],
        output_state: list[int],
    ) -> complex:
        raise BackendError(
            "Current backend does not implement probability_amplitude method."
        )

    def probability(
        self,
        unitary: NDArray[np.complex128],
        input_state: list[int],
        output_state: list[int],
    ) -> float:
        raise BackendError(
            "Current backend does not implement probability method."
        )

    def full_probability_distribution(
        self, circuit: CompiledPhotonicCircuit, input_state: State
    ) -> dict[State, float]:
        raise BackendError(
            "Current backend does not implement full_probability_distribution "
            "method."
        )
