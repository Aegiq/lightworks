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

"""
This file contains all global settings for use across lightworks. Any new values
should be added the settings class below.
"""

from typing import Any


class _Settings:
    """
    Stores all global settings for Lightworks. After creation new attributes
    (settings) cannot be added.
    """

    __frozen: bool = False
    unitary_precision: float = 1e-10
    sampler_probability_threshold: float = 1e-9

    def __init__(self) -> None:
        self.__frozen = True

    @property
    def all(self) -> list[str]:
        return ["unitary_precision", "sampler_probability_threshold"]

    def __str__(self) -> str:
        output = ""
        for val in self.all:
            output += f"{val}: {getattr(self, val)}\n"
        return output[:-1]

    def __repr__(self) -> str:
        return f"lightworks.Settings(\n{self!s}\n)"

    def __setattr__(self, name: str, value: Any) -> None:
        if not hasattr(self, name) and self.__frozen:
            msg = f"Setting with name '{name}' does not exist."
            raise AttributeError(msg)
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        # This is only called when attribute doesn't exist, so can use this to
        # raise a custom exception
        msg = f"Setting with name '{name}' does not exist."
        raise AttributeError(msg)


settings = _Settings()
