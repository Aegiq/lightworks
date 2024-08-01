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

:func:`lightworks.db_loss_to_transmission`
------------------------------------------

Converts a positive dB loss into a decimal transmission value.

.. code-block:: Python

    # Convert 3dB loss into transmission
    transmission = lw.db_loss_to_transmission(3)
    print(transmission)
    # Output: 0.5011872336272722

:func:`lightworks.transmission_to_db_loss`
------------------------------------------

Converts a decimal transmission value into a positive dB loss, which can be used with the circuit loss elements.

.. code-block:: Python

    # Convert 70% transmission into loss 
    loss = lw.transmission_to_db_loss(0.7)
    print(loss)
    # Output: 1.5490195998574319
