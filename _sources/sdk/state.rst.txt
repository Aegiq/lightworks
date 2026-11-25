State
=====

In Lightworks, the :class:`State <lightworks.State>` object is used to represent the photonic Fock states which are input/output from the system. States offer some similar behaviour to lists, but with the added benefit of being compatible with dictionaries for data management. 

Creation
--------

To create a new State, a list of integers should be provided, with the integers corresponding to the number of photons on each mode. For example, the fock state :math:`\ket{1,0,2,0}` would be encoded as:

.. code-block:: Python

    new_state = lw.State([1,0,2,0])

The list used to represent the state is stored with the ``s`` attribute and can be retrieved using this, however the value of this attribute cannot be modified, and an error will be raised if this is attempted.

Usage
-----

State objects have a range of functionality designed to make them easier to view and use. The first of which is that their contents can be viewed in the form of a quantum state using the print function. 

.. code-block:: Python

    print(lw.State([1,0,2,0]))
    # Output: |1,0,2,0>

It is also possible to index the State to retrieve partial information on its value. If the index is an integer then indexing will return the integer value for that mode, alternatively if the index is a slice this will create a new State object.

.. code-block:: Python

    state = lw.State([1,0,2,0])

    print(state[0])
    # Output: 1

    print(state[1:])
    # Output: |0,2,0>

.. note::
    Indexing the state does not allow the photon numbers on each mode to be modified. For example state[0] = 1 would raise an Exception.

A number of properties can also be viewed about a state, including the number of modes used and the total photon number of the state. The number of modes can be viewed using either the ``len`` operator on the state or accessing the ``n_modes`` attribute. For photon number, the ``n_photons`` attribute is used.

.. code-block:: Python

    state = lw.State([1,0,2,0])

    print(len(state))
    # Output: 4

    print(state.n_modes)
    # Output: 4

    print(state.n_photons)
    # Output: 3

As mentioned earlier, the State object is also hashable so can be used as a key for referencing in dictionaries. Additionally, if two state objects have identical contents then they are evaluated as equivalent, so can easily be compared in this way. This also means the ``in`` operator can be used with the dictionary. 

.. code-block:: Python

    print(lw.State([1,0,2,0]) == lw.State([1,0,2,0]))
    # Output: True

    dict = {lw.State([1,0,1,0]) : 0.5,
            lw.State([0,1,0,1]) : 0.5}

    print(lw.State([1,0,1,0]) in dict)
    # Output: True

Combining States
----------------

It is also possible to combine states together, either through the use of the ``+`` operator or the ``merge`` method. When using the ``+`` operator this will join the two states together, and when using ``merge`` it will combine the photon numbers of the two states across modes - this requires that the number of modes in both states is the same.

.. code-block:: Python

    print(lw.State([1,0]) + lw.State([2,0]))
    # Output: |1,0,2,0>

    state = lw.State([1,0,2,0])
    state2 = lw.State([2,1,0,1]
    
    print(state.merge(state2))
    # Output: |3,1,2,1>