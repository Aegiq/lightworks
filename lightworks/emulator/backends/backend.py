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


from typing import TYPE_CHECKING, Any, overload

from multimethod import multimethod

from lightworks.emulator.utils.exceptions import BackendError
from lightworks.sdk.results import Result
from lightworks.sdk.tasks import Batch, Task

from .permanent import PermanentBackend
from .slos import SLOSBackend

if TYPE_CHECKING:
    from .abc_backend import EmulatorBackend


class Backend:
    """
    Provide central location for selecting and interacting with different
    simulation backends.

    Args:

        backend (str) : A string detailing the backend which is to be used.

    """

    def __init__(self, backend: str) -> None:
        self.backend = backend

    @overload
    def run(self, task: Task) -> Result[Any, Any]: ...

    @overload
    def run(self, task: Batch) -> list[Result[Any, Any]]: ...

    def run(
        self, task: Task | Batch
    ) -> Result[Any, Any] | list[Result[Any, Any]]:
        """
        Runs the provided task on the current backend.

        Args:

            task (Task|Batch) : A task or batch to run.

        Returns:

            Result|list[Result]: A dictionary like results object containing
                details of the calculated values from a task. If a batch is run
                then this will be a list of results in the same order the task
                were added to the batch.

        """
        if not isinstance(task, Task | Batch):
            raise TypeError("Object to run on the backend must be a task.")
        return self._run(task)

    @multimethod
    def _run(self, task: Task) -> Result[Any, Any]:
        if task.__class__.__name__ not in self.__backend.compatible_tasks:
            msg = (
                "Selected backend not compatible with task, supported tasks for"
                " the backend are: "
                f"{', '.join(self.__backend.compatible_tasks)}."
            )
            raise BackendError(msg)
        return self.__backend.run(task)

    @_run.register
    def _run_batch(self, task: Batch) -> list[Result[Any, Any]]:
        return [self._run(t) for t in task]

    @property
    def backend(self) -> str:
        """
        Returns the name of the currently selected backend.
        """
        return self.__backend.name

    @backend.setter
    def backend(self, value: str) -> None:
        backends = {
            "permanent": PermanentBackend,
            "slos": SLOSBackend,
        }
        if value not in backends:
            msg = (
                "Backend name not recognised, valid options are: "
                f"{', '.join(backends.keys())}."
            )
            raise ValueError(msg)
        # Initialise selected backend
        self.__backend: EmulatorBackend = backends[value]()

    def __str__(self) -> str:
        return self.backend

    def __repr__(self) -> str:
        return f"lightworks.emulator.Backend('{self.backend}')"
