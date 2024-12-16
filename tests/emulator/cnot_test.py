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

# ruff: noqa: E731

import pytest
from numpy import arccos, pi

from lightworks import PhotonicCircuit, PostSelection, Sampler, State
from lightworks.emulator import Backend, Detector, Source
from lightworks.qubit import CNOT

heralds = [
    lambda s: (
        s[0] == 0 and s[5] == 0 and s[1] + s[2] == 1 and s[3] + s[4] == 1
    ),
    lambda s: (s[0] + s[1] == 1 and s[2] + s[3] == 1),
]
# Including heralds
post_selection = PostSelection()
post_selection.add(0, 0)
post_selection.add(5, 0)
post_selection.add((1, 2), 1)
post_selection.add((3, 4), 1)
# Excluding heralds
post_selection2 = PostSelection()
post_selection2.add((0, 1), 1)
post_selection2.add((2, 3), 1)

post_selects = [post_selection, post_selection2]


@pytest.mark.flaky(max_runs=3)
@pytest.mark.parametrize(
    ("backend", "post_selection"),
    [
        (Backend("permanent"), heralds),
        (Backend("slos"), heralds),
        (Backend("permanent"), post_selects),
        (Backend("slos"), post_selects),
    ],
)
class TestCNOT:
    """
    Samples from a non-heralded CNOT circuit with loss, imperfect source and
    detector to check that the correct results is produced. It tests this with
    both permanent and slos backends.
    """

    def test_cnot_sample_n_inputs(self, backend, post_selection):
        """
        Checks the correct output is produced from the CNOT gate when sampling
        N inputs from the system. Note, very occasionally this test may fail
        due to the probabilistic nature of the sampler. It is only a concern if
        this happens consistently.
        """
        # Create CNOT circuit
        r = 1 / 3
        loss = 0.1
        theta = arccos(r)
        cnot_circuit = PhotonicCircuit(6)
        to_add = [
            (3, pi / 2, 0),
            (0, theta, 0),
            (2, theta, pi),
            (4, theta, 0),
            (3, pi / 2, 0),
        ]
        for m, t, p in to_add:
            cnot_circuit.bs(m, loss=loss, reflectivity=0.5)
            cnot_circuit.ps(m + 1, t)
            cnot_circuit.bs(m, loss=loss, reflectivity=0.5)
            cnot_circuit.ps(m + 1, p)
            if m in [3, 4, 3]:
                cnot_circuit.barrier()
        # Define imperfect source and detector
        source = Source(purity=0.99, brightness=0.4, indistinguishability=0.96)
        detector = Detector(efficiency=0.9, p_dark=1e-5, photon_counting=False)
        # Then define sampler with the input state |10>
        sampler = Sampler(
            cnot_circuit,
            State([0, 0, 1, 1, 0, 0]),
            20000,
            source=source,
            detector=detector,
            post_selection=post_selection[0],
            sampling_mode="input",
        )
        results = backend.run(sampler)
        # We expect the state |11> (|0,0,1,0,1,0> in mode language) with
        # reasonable fidelity, so we will assert this is measured for > 80% of
        # the total samples which met the herald condition
        eff = results[State([0, 0, 1, 0, 1, 0])] / sum(results.values())
        assert eff > 0.8

    def test_cnot_sample_n_outputs(self, backend, post_selection):
        """
        Checks the correct output is produced from the CNOT gate when sampling
        N outputs from the system. Note, very occasionally this test may fail
        due to the probabilistic nature of the sampler. It is only a concern if
        this happens consistently.
        """
        # Create CNOT circuit
        r = 1 / 3
        loss = 0.1
        theta = arccos(r)
        cnot_circuit = PhotonicCircuit(6)
        to_add = [
            (3, pi / 2, 0),
            (0, theta, 0),
            (2, theta, pi),
            (4, theta, 0),
            (3, pi / 2, 0),
        ]
        for m, t, p in to_add:
            cnot_circuit.bs(m, loss=loss, reflectivity=0.5)
            cnot_circuit.ps(m + 1, t)
            cnot_circuit.bs(m, loss=loss, reflectivity=0.5)
            cnot_circuit.ps(m + 1, p)
            if m in [3, 4, 3]:
                cnot_circuit.barrier()
        # Define imperfect source and detector
        source = Source(purity=0.99, brightness=0.4, indistinguishability=0.94)
        detector = Detector(efficiency=1, photon_counting=False)
        # Then define sampler with the input state |10>
        sampler = Sampler(
            cnot_circuit,
            State([0, 0, 1, 1, 0, 0]),
            20000,
            source=source,
            detector=detector,
            post_selection=post_selection[0],
        )
        results = backend.run(sampler)
        # We expect the state |11> (|0,0,1,0,1,0> in mode language) with
        # reasonable fidelity, so we will assert this is measured for > 80% of
        # the total samples which met the herald condition
        eff = results[State([0, 0, 1, 0, 1, 0])] / 20000
        assert eff > 0.8

    def test_cnot_sample_n_inputs_built_in(self, backend, post_selection):
        """
        Checks the correct output is produced from the built-in CNOT gate when
        sampling N inputs from the system. Note, very occasionally this test
        may fail due to the probabilistic nature of the sampler. It is only a
        concern if this happens consistently.
        """
        # Create CNOT circuit
        cnot_circuit = CNOT()
        # Define imperfect source and detector
        source = Source(purity=0.99, brightness=0.4, indistinguishability=0.94)
        detector = Detector(efficiency=0.9, p_dark=1e-5, photon_counting=False)
        # Then define sampler with the input state |10>
        sampler = Sampler(
            cnot_circuit,
            State([0, 1, 1, 0]),
            20000,
            source=source,
            detector=detector,
            post_selection=post_selection[1],
            sampling_mode="input",
        )
        results = backend.run(sampler)
        # We expect the state |11> (|0,1,0,1> in mode language) with
        # reasonable fidelity, so we will assert this is measured for > 80% of
        # the total samples which met the herald condition
        eff = results[State([0, 1, 0, 1])] / sum(results.values())
        assert eff > 0.8

    def test_cnot_sample_n_outputs_built_in(self, backend, post_selection):
        """
        Checks the correct output is produced from the built-in CNOT gate when
        sampling N outputs from the system. Note, very occasionally this test
        may fail due to the probabilistic nature of the sampler. It is only a
        concern if this happens consistently.
        """
        # Create CNOT circuit
        cnot_circuit = CNOT()
        # Define imperfect source and detector
        source = Source(purity=0.99, brightness=0.4, indistinguishability=0.94)
        detector = Detector(efficiency=1, photon_counting=False)
        # Then define sampler with the input state |10>
        sampler = Sampler(
            cnot_circuit,
            State([0, 1, 1, 0]),
            20000,
            source=source,
            detector=detector,
            post_selection=post_selection[1],
        )
        results = backend.run(sampler)
        # We expect the state |11> (|0,1,0,1> in mode language) with
        # reasonable fidelity, so we will assert this is measured for > 80% of
        # the total samples which met the herald condition
        eff = results[State([0, 1, 0, 1])] / 20000
        assert eff > 0.8
