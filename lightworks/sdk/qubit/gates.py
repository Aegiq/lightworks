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
Central location for importing and initializing all gates.
"""

from .single_qubit_gates import H_gate, X_gate, Y_gate, Z_gate, S_gate, T_gate
from .two_qubit_gates import CZ_gate, CNOT_gate

# Initialize all gates
H = H_gate()
X = X_gate()
Y = Y_gate()
Z = Z_gate()
S = S_gate()
T = T_gate()
CZ = CZ_gate()
CNOT = CNOT_gate()