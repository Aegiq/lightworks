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


from ..simulation.task import Task
from ..utils import BackendError
from .permanent import PermanentBackend
from .slos import SLOSBackend


class Backend:
    """
    Provide central location for selecting and interacting with different
    simulation backends.

    Args:

        backend (str) : A string detailing the backend which is to be used.

    """

    def __init__(self, backend: str) -> None:
        self.backend = backend

        return

    def run(self, task: "Task") -> dict:
        """
        Runs the provided task on the current backend.
        """
        if not isinstance(task, Task):
            raise TypeError("Object to run on the backend must be a task.")
        if self.backend not in task.__compatible_backends__:
            msg = (
                "Selected backend not compatible with task, supported backends "
                f"for task are: {', '.join(task.__compatible_backends__)}."
            )
            raise BackendError(msg)
        return task._run(self.__backend)

    @property
    def backend(self) -> str:
        """Stores data on the selected backend."""
        return self.__backend.name

    @backend.setter
    def backend(self, value: str) -> None:
        backends = {"permanent": PermanentBackend, "slos": SLOSBackend}
        if value not in backends:
            msg = (
                "Backend name not recognised, valid options are: "
                f"{', '.join(backends.keys())}."
            )
            raise ValueError(msg)
        self.__backend = backends[value]()  # initialise backend

    def __str__(self) -> str:
        return self.backend

    def __repr__(self) -> str:
        return f"lightworks.emulator.Backend('{self.backend}')"
