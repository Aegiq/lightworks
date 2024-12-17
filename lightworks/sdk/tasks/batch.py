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

from ..circuit import Parameter, PhotonicCircuit
from .task import Task


class Batch:
    """
    Allows for batch submission of jobs with varied parameters on each run.
    """

    def __init__(
        self,
        task: Task | None = None,
        task_args: list[list[Any]] | None = None,
        task_kwargs: dict[str, list[Any]] | None = None,
        parameters: dict[Parameter, list[Any]] | None = None,
    ) -> None:
        # Define list to store tasks
        self.__tasks: list[Task] = []
        if task is None:
            return  # Skip all setup if no task object is provided
        # Check task not already initialized.
        if not isinstance(task, type):
            raise TypeError("Provided task should not be initialized.")
        args, kwargs, params, n_vars = process_values(
            task_args, task_kwargs, parameters
        )
        # Loop over number of variations and create a task for each
        for i in range(n_vars):
            if params is not None:
                for p, v in params.items():
                    p.set(v[i])
            sub_args = (
                [copy_circuit(a[i]) for a in args] if args is not None else []
            )
            sub_kwargs = (
                {k: copy_circuit(v[i]) for k, v in kwargs.items()}
                if kwargs is not None
                else {}
            )
            self.__tasks.append(task(*sub_args, **sub_kwargs))

    def __iter__(self) -> Any:
        """Iterable to allow to loop over each task."""
        yield from self.tasks

    @property
    def tasks(self) -> list[Task]:
        """Returns all tasks contained in batch."""
        return self.__tasks

    @property
    def num(self) -> int:
        """Returns number of tasks in batch."""
        return len(self.tasks)

    def add(self, task: Task) -> None:
        """Allows for tasks to be manually added to the BatchTask."""
        if not isinstance(task, Task):
            raise TypeError("Provided task must be a Task object.")
        self.__tasks.append(task)


def copy_circuit(arg: Any) -> Any:
    """
    Checks if a provided argument is a circuit and if so makes a copy and freeze
    the circuit parameters.
    """
    if isinstance(arg, PhotonicCircuit):
        return arg.copy(freeze_parameters=True)
    return arg


def process_values(
    task_args: list[list[Any]] | None,
    task_kwargs: dict[str, list[Any]] | None,
    parameters: dict[Parameter, list[Any]] | None,
) -> tuple[
    list[Any] | None,
    dict[str, list[Any]] | None,
    dict[Parameter, list[Any]] | None,
    int,
]:
    """
    Processes args, kwargs and parameter values passed to a BatchTask.
    """
    n_vars = 0
    # Confirm each entry is list or tuple
    if task_args is not None:
        for a in task_args:
            if not isinstance(a, list | tuple):
                raise TypeError(
                    "Each value for task_args should be provided as a list."
                )
        n_vars = max([len(a) for a in task_args] + [n_vars])
    if task_kwargs is not None:
        for k, v in task_kwargs.items():
            if not isinstance(k, str):
                raise TypeError("'task_kwargs' keys should be strings.")
            if not isinstance(v, list | tuple):
                raise TypeError(
                    "Each value for task_kwargs should be provided as a "
                    "list."
                )
        n_vars = max([len(a) for a in task_kwargs.values()] + [n_vars])
    if parameters is not None:
        for p, v in parameters.items():
            if not isinstance(p, Parameter):
                raise TypeError(
                    "'parameters' keys should be Parameter objects."
                )
            if not isinstance(v, list | tuple):
                raise TypeError(
                    "Each value for parameters should be provided as a list."
                )
        n_vars = max([len(a) for a in parameters.values()] + [n_vars])
    # Then check number of values is of the correct length
    if task_args is not None:
        for i, a in enumerate(task_args):
            if len(a) == 1:
                task_args[i] = a * n_vars
            elif len(a) != n_vars:
                raise ValueError(
                    "Mismatch in number of values provided for one of the "
                    "task_args."
                )
    if task_kwargs is not None:
        for k, v in task_kwargs.items():
            if len(v) == 1:
                task_kwargs[k] = v * n_vars
            elif len(v) != n_vars:
                raise ValueError(
                    "Mismatch in number of values provided for one of the "
                    "task_kwargs."
                )
    if parameters is not None:
        for p, v in parameters.items():
            if len(v) == 1:
                parameters[p] = v * n_vars
            elif len(v) != n_vars:
                raise ValueError(
                    "Mismatch in number of values provided for one of the "
                    "parameters."
                )
    return (task_args, task_kwargs, parameters, n_vars)
