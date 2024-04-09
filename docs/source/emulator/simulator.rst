Simulator
=========

The Simulator class is used to find the probability amplitudes of different states for a given input state. This means the expected output from a given input can be calculated precisely. It can be useful to do this as it enables a better understanding of the exact quantum state evolution, and also is much quicker/more accurate than sampling many times to find the output distribution.

To demonstrate the usage of the Simulator, we will first configure a Circuit for testing. To do this, we will add an arrangement of randomly configured Mach-Zhender interferometers, which acts as tuneable beam splitter, and include some additional phase shifters on one arm of the system.

.. code-block:: Python

    from random import random, seed

    circuit = lw.Circuit(4)

    seed(1) # Set random seed so circuit is reproducible
    for m in [0, 1, 0, 2]:
        circuit.add_ps(m, random())
        circuit.add_bs(m)
        circuit.add_ps(m+1, random())
        circuit.add_bs(m)

    circuit.display()

.. image:: assets/simulator_circuit.svg
    :scale: 100%
    :align: center

The Simulator can then be created, this requires the created circuit from above.

.. code-block:: Python

    sim = emulator.Simulator(circuit)

Simulation
----------

To retrieve probability amplitudes from the system the ``simulate`` method should then be used. At minimum, this requires the input(s) are specified, and optionally target outputs may also be specified. If the target outputs are not included, then all possible outputs with photon number matching the input photon number will be calculated. This introduces the condition that all inputs and outputs provided by a user must have the same photon number, if this is not the case then two separate calls to simulate must be made. When specifying a single input/output it is possible to provide these as States, but for multiple inputs and outputs then these are provided as lists. 

To find the probability amplitude between then input :math:`\ket{1,1,0,0}` and output :math:`\ket{0,0,1,1}`, the code required would therefore be:

.. code-block:: Python

    input_state = lw.State([1,1,0,0])
    output_state = lw.State([0,0,1,1])

    results = sim.simulate(input_state, output_state)

This will return a :doc:`../emulator_reference/simulation_result` object. We can access data for specific input and outputs using the ``[]`` operator on the object. The behaviour of this slightly varies depending on whether the simulation used one or multiple inputs. In the single input case it is possible to do ``[output_state]`` to get the value for a particular output, whereas for multiple inputs ``[input_state]`` will return all possible outputs for an input and ``[input_state:output_state]`` will produce a singular value for the selected combination. As only one input was used above the following are therefore equivalent:

.. code-block:: Python

    print(results[output_state])
    # Output: (0.037523401912278494-0.25889120714091474j)

    print(results[input_state, output_state])
    # Output: (0.037523401912278494-0.25889120714091474j)

Alternatively, the raw data of the results can be accessed with the array attribute.

.. code-block:: Python

    print(results.array)
    # Output: [[0.0375234-0.25889121j]]