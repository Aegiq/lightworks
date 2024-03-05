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

def set_statistic_type(new_type: str) -> None:
    """
    Update the sampling statistic type used in the emulator.
    
    Args:
    
        new_type (str) : The new sampling statistics to use, should be either
            bosonic or fermionic.
    
    """
    from ...emulator import __settings
    
    if new_type not in ["bosonic", "fermionic"]:
        msg = "Statistic type should have value of 'bosonic' or 'fermionic'."
        raise ValueError(msg)
    
    __settings["statistic_type"] = new_type
    
def get_statistic_type() -> str:
    """
    Retrieve the current sampling statistic type used by the emulator.
    
    Returns:
    
        str : The current sampling statistics of the emulator.
        
    """
    from ...emulator import __settings
    return __settings["statistic_type"]