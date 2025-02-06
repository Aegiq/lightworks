Utilities
=========

Lightworks also has a number of utilities which aim to make using the package easier, offering random matrix generation and quantity conversion. Each utility function is detailed in a dedicated subsection.

:func:`lightworks.random_unitary`
---------------------------------

Generates an NxN random unitary matrix using the SciPy unitary_group function. There is also an optional random seed which enables the same random matrix to be generated repeatedly.

.. code-block:: Python

    # Create random 8x8 matrix 
    U = lw.random_unitary(8)

    # Create 8x8 matrix with random seed to return the same matrix
    U = lw.random_unitary(8, seed = 4)

.. note::
    Random seed values provided within Lightworks typically need to be integers.

:func:`lightworks.random_permutation`
-------------------------------------

Generates an NxN permutation matrix, which is a unitary matrix with only values of 0 & 1. This is equivalent to swapping modes in a circuit. There is also an optional random seed which enables the same random matrix to be generated repeatedly.

.. code-block:: Python

    # Create random 8x8 matrix 
    U = lw.random_permutation(8)

    # Create 8x8 matrix with random seed to return the same matrix
    U = lw.random_permutation(8, seed = 4)

:func:`lightworks.convert.db_loss_to_decimal`
---------------------------------------------

Converts a positive dB loss into a decimal loss value, which can be used with the circuit loss elements.

.. code-block:: Python

    # Convert 3dB loss into decimal
    loss = lw.db_loss_to_decimal(3)
    print(loss)
    # Output: 0.49881276637272776

:func:`lightworks.convert.decimal_to_db_loss`
---------------------------------------------

Converts a decimal loss value into a positive dB loss.

.. code-block:: Python

    # Convert 30% loss into dB value 
    loss = lw.decimal_to_db_loss(0.3)
    print(loss)
    # Output: 1.5490195998574319

:func:`lightworks.convert.qubit_to_dual_rail`
---------------------------------------------

Converts from qubit to dual-rail encoding.

.. code-block:: Python

    dr_state = lw.qubit_to_dual_rail([1, 0, 1])
    print(dr_state)
    # Output: |0,1,1,0,0,1>

:func:`lightworks.convert.dual_rail_to_qubit`
---------------------------------------------

Converts from a dual-rail encoding on pairs of adjacent modes into a qubit state.

.. code-block:: Python

    qubit_state = lw.dual_rail_to_qubit([1, 0, 0, 1, 1, 0])
    print(qubit_state)
    # Output: |0,1,0>

:func:`lightworks.convert.threshold_mapping`
--------------------------------------------

Applies a threshold mapping to the provided state.

.. code-block:: Python

    qubit_state = lw.threshold_mapping([1, 2, 0, 3, 1, 0])
    print(qubit_state)
    # Output: |1,1,0,1,1,0>


:func:`lightworks.convert.parity_mapping`
-----------------------------------------

Applies a parity mapping to the provided state, in which even & odd values are mapped to 0 & 1 respectively.

.. code-block:: Python

    qubit_state = lw.parity_mapping([1, 2, 0, 3, 1, 0])
    print(qubit_state)
    # Output: |1,0,0,1,1,0>
