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

from .circuit_utils import (
    add_empty_mode_to_circuit_spec,
    add_modes_to_circuit_spec,
    compress_mode_swaps,
    convert_non_adj_beamsplitters,
    unpack_circuit_spec,
)
from .conversion import db_loss_to_transmission, transmission_to_db_loss
from .exceptions import *
from .heralding_utils import add_heralds_to_state, remove_heralds_from_state
from .matrix_utils import (
    add_mode_to_unitary,
    check_unitary,
    random_permutation,
    random_unitary,
)
from .permutation_conversion import permutation_mat_from_swaps_dict
