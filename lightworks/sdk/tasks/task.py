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

from abc import ABCMeta, abstractmethod

from .data import TaskData


class Task(metaclass=ABCMeta):
    """
    Base class for all tasks, which requires implementation of the run method.
    """

    __compatible_backends__: tuple[str, ...]

    @abstractmethod
    def _generate_task(self) -> TaskData:
        """
        Generates the data required to run the current task.
        """
