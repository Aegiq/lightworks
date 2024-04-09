Qubit Components
================

Included within Lightworks is a set of standard photonic circuits for implementing the gates required for qubit-based paradigms of quantum computing. Included is a number of the common single and two qubit gates, these gates are intended to act on dual-rail encoded qubits. 

.. note::
    Currently, this feature is still in development. The gates provided here are designed for addition to the linear optic circuits typically used in Lightworks, and are not currently able to be used in a native qubit encoding.

Single Qubit Gates
------------------

The following single qubit gates are currently implemented:

.. list-table:: Single Qubit Gates
    :widths: 40 60
    :header-rows: 1

    * - Gate
      - Matrix
    * - H
      - .. math:: \frac{1}{\sqrt{2}}\begin{bmatrix}
                      1 & 1 \\
                      1 & 1 \\
                  \end{bmatrix}
    * - X
      - .. math:: \begin{bmatrix}
                      0 & 1 \\
                      1 & 0 \\
                  \end{bmatrix}
    * - Y
      - .. math:: \begin{bmatrix}
                      0 & -i \\
                      i & 0 \\
                  \end{bmatrix}
    * - Z
      - .. math:: \begin{bmatrix}
                      1 & 0 \\
                      0 & -1 \\
                  \end{bmatrix}
    * - S
      - .. math:: \begin{bmatrix}
                      1 & 0 \\
                      0 & i \\
                  \end{bmatrix}
    * - T
      - .. math:: \begin{bmatrix}
                      1 & 0 \\
                      0 & \exp(i\pi/4) \\
                  \end{bmatrix}

To create a new gate, it needs to be called from the qubit sub-module of Lightworks. For example, a Hadamard gate can be created with:

.. code-block:: Python

    H = lw.qubit.H()

These gates are functionally Circuits, and so can also be added and combined with the other Circuits in the same way. In the following, we create a 4 mode circuit to define two qubits (a & b) and then applying X, Y & Z gates to one qubit and H, S & T to the other.

.. code-block:: Python

    circuit = lw.Circuit(4)

    # X, Y, Z to qubit a
    circuit.add(lw.qubit.X(), 0)
    circuit.add(lw.qubit.Y(), 0)
    circuit.add(lw.qubit.Z(), 0)

    # H, S, T to qubit b
    circuit.add(lw.qubit.H(), 2)
    circuit.add(lw.qubit.S(), 2)
    circuit.add(lw.qubit.T(), 2)

    # View create circuit
    circuit.display(mode_labels = ["a0", "a1", "b0", "b1"])

.. image:: assets/single_qubit_gate_demo.svg
    :scale: 125%
    :align: center

Two Qubit Gates
---------------

All of the two qubit gates included require post-selection and/or heralding to function correctly, as well as some additional modes. The exact layout of the modes and requirements can be found in the docstrings for the chosen gate, but are also summarised in the table below. In this table, the qubit modes are also specified, where c0 and c1 are the 0 & 1 states of the control qubit respectively and t0 & t1 are the 0 & 1 states of the target qubit. Unless otherwise specified, no photons should be input on the remaining modes of the gate.

.. list-table:: Two Qubit Gates
    :widths: 20, 20, 60
    :header-rows: 1
    :align: center

    * - Gate
      - Qubit Modes
      - Post-selection/Heralding
    * - CZ
      - | c0 : 1
        | c1 : 2
        | t0 : 3
        | t1 : 4
      - Requires heralding through measurement of 0 photons on modes 0 & 5. Also need to post-select on only measuring one photon across each of the qubit modes.
    * - CNOT
      - | c0 : 1
        | c1 : 2
        | t0 : 3
        | t1 : 4
      - Requires heralding through measurement of 0 photons on modes 0 & 5. Also need to post-select on only measuring one photon across each of the qubit modes.
    * - CZ_Heralded
      - | c0 : 2
        | c1 : 3
        | t0 : 4
        | t1 : 5
      - Requires ancillary photons which should be input on modes 1 & 6 and then heralding on one photon being measured on both modes 1 & 6. Also need to herald through measurement of 0 photons on modes 0 & 7. 
    * - CNOT_Heralded
      - | c0 : 2
        | c1 : 3
        | t0 : 4
        | t1 : 5
      - Requires ancillary photons which should be input on modes 1 & 6 and then heralding on one photon being measured on both modes 1 & 6. Also need to herald through measurement of 0 photons on modes 0 & 7.

The two qubit gates can then be created in the same way as the single qubit gates. We can directly use these gates with all of the simulation objects provided within the emulator. As an example, below the heralded CNOT gate is tested with the sampler. The input :math:`\ket{1,0}` (which translates to :math:`\ket{0,1,0,1,1,0,1,0}` in mode language) is chosen.

.. code-block:: Python

    # Define cnot, input and required herald function
    cnot = lw.qubit.CNOT_Heralded()
    #                            c0 c1 t0 t1
    input_state = lw.State([0,1, 0, 1, 1, 0, 1,0])
    herald = lambda s: s[0] == 0 and s[1] == 1 and s[6] == 1 and s[7] == 0

    sampler = emulator.Sampler(cnot, input_state)
    # Then sample 10,000 times
    results = sampler.sample_N_inputs(10000, herald = herald, seed = 8)

    # View measured counts
    print(results)
    # {State(|0,1,0,1,0,1,1,0>): 615}

As expected, with the correct heralding we only measure the output state :math:`\ket{0,1,0,1,0,1,1,0}`, which corresponds to the qubit state :math:`\ket{1,1}`, demonstrating that the CNOT works as expected. Despite inputting to the system 10,000 times we only measure 615 outputs that meet the heralding conditions, this is because the heralded CNOT only has a success probability of 1/16 (= 0.0625, 615/10000 = 0.0615).

.. warning::
    Care needs to be taken when cascading two qubit gates to ensure that any post-selection and heralding criteria can still be maintained and information on this is not lost.