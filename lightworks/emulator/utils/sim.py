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

from lightworks.sdk.state import State
from lightworks.sdk.utils.exceptions import PhotonNumberError


def check_photon_numbers(states: list[State], target_n: int) -> None:
    """
    Raises an exception if photon numbers are mixed when running a
    simulation.
    """
    ns = [s.n_photons for s in states]
    if min(ns) != target_n or max(ns) != target_n:
        raise PhotonNumberError(
            "Mismatch in photon numbers between some inputs/outputs, "
            "this is not currently supported in the Simulator."
        )
