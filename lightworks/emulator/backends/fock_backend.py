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
from ...sdk.state import State
from ...sdk.tasks.analyzer import Analyzer
from ...sdk.tasks.sampler import Sampler
from ...sdk.tasks.simulator import Simulator
from ...sdk.tasks.task import Task, TaskData
from ..simulation import (
    AnalyzerRunner,
    SamplerRunner,
    SimulatorRunner,
)
from ..utils import BackendError
from .abc_backend import BackendABC
from .caching import CacheData, check_parameter_updates, get_calculation_values

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
    def run_simulator(self, task: Simulator) -> dict:
        data = task._generate_task()
        return SimulatorRunner(data, self.probability_amplitude).run()

    @run.register
    def run_analyzer(self, task: Analyzer) -> dict:
        data = task._generate_task()
        return AnalyzerRunner(data, self.probability).run()

    @run.register
    def run_sampler(self, task: Sampler) -> dict:
        data = task._generate_task()
        runner = SamplerRunner(data, self.full_probability_distribution)
        cached_results = self._check_cache(data)
        if cached_results is not None:
            runner.probability_distribution = cached_results["pdist"]
            runner.full_to_heralded = cached_results["full_to_herald"]
            task._probability_distribution = cached_results["pdist"]
        else:
            task._probability_distribution = runner.distribution_calculator()
            results = {
                "pdist": runner.probability_distribution,
                "full_to_herald": runner.full_to_heralded,
            }
            self._add_to_cache(data, results)
        return runner.run()

    def _check_cache(self, data: TaskData) -> dict | None:
        name = data.__class__.__name__
        if hasattr(self, "_cache") and name in self._cache:
            old_values = self._cache[name].values
            new_values = get_calculation_values(data)
            if not check_parameter_updates(old_values, new_values):
                return self._cache[name].results
        # Return false if cache doesn't exist or name not found
        return None

    def _add_to_cache(self, data: TaskData, results: dict) -> None:
        if not hasattr(self, "_cache"):
            self._cache = {}
        name = data.__class__.__name__
        values = get_calculation_values(data)
        self._cache[name] = CacheData(values=values, results=results)

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
