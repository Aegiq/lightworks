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
from multimethod import multimethod

from ...sdk.circuit.photonic_compiler import CompiledPhotonicCircuit
from ...sdk.results import (
    ProbabilityDistribution,
    SamplingResult,
    SimulationResult,
)
from ...sdk.state import State
from ...sdk.tasks.analyzer import Analyzer
from ...sdk.tasks.sampler import Sampler
from ...sdk.tasks.simulator import Simulator
from ...sdk.tasks.task import Task
from ..simulation import (
    AnalyzerRunner,
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

    @multimethod
    def run(self, task: Task) -> None:
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
