Getting Started
===============

Introduction
------------
To get started we will build a circuit to simulate the Hong-Ou-Mandel (HOM) effect with a single 50:50 beam splitter, and demonstrate the typical process flow of Lightworks.

HOM interference can be seen when two indistinguishable photons are incident on a 50:50 beam splitter. Classically, when this occurs, we would expect that 50% of the time the photons exit on the same output mode and 50% of the time they exit on a different output. Instead, we see that both photons always exit on the same output mode of the beam splitter, as is shown below.

.. figure:: assets/getting_started_hom_demo.svg
    :scale: 350%
    :align: center
    
    HOM interference occurring on a 50:50 beam splitter.

First Circuit
-------------

First we need to import Lightworks and the emulator with the following:

.. code-block:: Python

    import lightworks as lw
    from lightworks import emulator

We will configure a circuit to simulate by creating a new Circuit object. This Circuit object is one of the key components of Lightworks and is used in almost all interactions with it. When creating a circuit, the required number of circuit modes is provided, in this case 2. This number of modes is then fixed and cannot be modified without creating a new circuit.

.. code-block:: Python

    circuit = lw.Circuit(2)

If print is then used on the circuit, this will display the total number of modes.

.. code-block:: Python

    print(circuit)
    # Output: Circuit(2)

Next, we add a beam splitter to the circuit, this is achieved with the ``bs`` method. The default reflectivity of a beam splitter is 0.5 (50%), so we do not need to specify this. The value we specify in the method arguments is the first mode that the beam splitter is placed on. By default, if a second mode isn't specified then it will be set to the first mode + 1, so below we could have equivalently used ``bs(0,1)``.

.. code-block:: Python

    circuit.bs(0)

.. note::
    Like Python itself, indexing in Lightworks starts from 0, meaning this is always the first mode. So for example, in a 4 mode circuit, the modes would be referred to using the indices 0, 1, 2 & 3.

Once a Circuit has been created, we can then view it using the display method. For the circuit above this will produce the following:

.. code-block:: Python

    circuit.display()

.. image:: assets/getting_started_hom_circuit.svg
    :scale: 200%
    :align: center

Which, as expected, is a single 50:50 beam splitter (reflectivity = 0.5) across the two modes. For more information about the exact usage of circuit all of its components, visit the :doc:`sdk/circuit` section.

Initial Simulation
------------------

Once we have built a circuit, we can then move on to simulating it with the emulator. For the simulation, we will need to define the state which we wish to input into the system. This is achieved with the :doc:`sdk_reference/state` object, which takes a list of the number of photons in each mode as its input.

.. code-block:: Python

    input_state = lw.State([1,1])

The exact functionality of the State object is discussed further in the :doc:`sdk/state` section.

For this initial simulation, we will choose to use the :doc:`emulator_reference/sampler` to emulate the process of measuring photon outputs after they have propagated through the system. On creation of the Sampler, we specify the circuit and the input state to sample from. By default, it is assumed that all photons are indistinguishable, and so we do not need to set anything for this. We then use the ``sample_N_outputs`` method to generate N samples from the system, in this case choosing N = 10000. We will also specify a random seed to ensure results are reproducible, but this is optional.

.. code-block:: Python

    sampler = emulator.Sampler(circuit, input_state)
    results = sampler.sample_N_outputs(10000, seed = 1)

This produces a :doc:`emulator_reference/sampling_result` object, we can quickly view the contents of this using the print statement.

.. code-block:: Python

    print(results)
    # Output: {lightworks.State(|0,2>): 5047, lightworks.State(|2,0>): 4953}

Alternatively, we can use the ``plot`` method to quickly create a plot of the measured output states.

.. code-block:: Python

    results.plot()

.. image:: assets/getting_started_demo_plot.png
    :scale: 100%
    :align: center

As expected, in near equal numbers we measure the states :math:`\ket{2,0}` and :math:`\ket{0,2}`, with only a small variation existing due to the probabilistic nature of the system.

Distinguishable Photons
^^^^^^^^^^^^^^^^^^^^^^^

With the emulator, we can also simulate distinguishable particles, to confirm that the HOM interference is no longer present. This is possible by defining a :doc:`emulator_reference/source` to use with the Sampler, and setting the value of the source indistinguishability to 0. We can then resample from the system and plot.

.. code-block:: Python

    source = emulator.Source(indistinguishability = 0)
    sampler = emulator.Sampler(circuit, input_state, source = source)

    results = sampler.sample_N_outputs(10000, seed = 1)
    results.plot()

.. image:: assets/getting_started_demo_plot2.png
    :scale: 100%
    :align: center

From the above, we see that in the case of distinguishable photons then approximately 50% of the time the photons will go to separate outputs of the circuit. 

Next Steps
----------

To find more about the usage of the different components of Lightworks head to the :doc:`tutorials/index` and the :doc:`sdk/index` & :doc:`emulator/index` sections. Alternatively, go to the :doc:`examples/index` section to see the different ways in which Lightworks can be utilized.