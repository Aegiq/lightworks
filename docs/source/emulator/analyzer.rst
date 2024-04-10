Analyzer
========

The :doc:`../emulator_reference/analyzer` object provides a set of tools to enable a better understanding of the functionality of a circuit under a set of post-selection criteria. The provision of this criteria before computation enables the Analyzer to filter down the possible output states from a given target input, reducing the number of states that it needs to calculate the permanent for.

To test the Analyzer, we will use the CNOT gate from :cite:p:`ralph2002`, as this is a useful demonstration of all analyzer features. For successful operation, it requires that at the output there are no output photons measured on the upper and lower modes, and also that only one photon exists across each of the two modes used to define each qubit.

To begin, the circuit is first defined:

.. code-block:: Python

    import numpy as np

    cnot_circuit = lw.Circuit(6)

    theta = np.arccos(1/3)
    to_add = [(3, np.pi/2, 0), (0, theta, 0), (2, theta, np.pi), (4, theta, 0), 
              (3, np.pi/2, 0)]

    for m, t, p in to_add:
        cnot_circuit.add_bs(m, reflectivity = 0.5)
        cnot_circuit.add_ps(m+1, t)
        cnot_circuit.add_bs(m, reflectivity = 0.5)
        cnot_circuit.add_ps(m+1, p)

    cnot_circuit.display(mode_labels = ["a0", "c0", "c1", "t0", "t1", "a1"])

.. image:: assets/cnot_circuit.svg
    :scale: 100%
    :align: center

A new Analyzer object can then be created, with initially only the circuit being specified.

.. code-block:: Python

    analyzer = emulator.Analyzer(cnot_circuit)

We then need to add the conditions for success of the gate. As mentioned, the upper and lower modes of the circuit (a0 and a1) are reserved as ancillary modes and should have no photons at the input and output. We can then therefore use the ``set_herald`` method to condition this, stating the mode which we wish to herald from and the number of photons to herald. When defining the mode, this is technically the input mode used for heralding, and it is possible to set a different output mode, however when not specified it is assumed the input and output mode are the same.  

.. code-block:: Python

    analyzer.set_herald(0, 0)
    analyzer.set_herald(5, 0)

.. note::
    When heralds are applied the modes that the heralds are on are removed from the inputs and outputs used within the system. So for example, after adding the heralds above the inputs and outputs would only be specified across modes 1, 2, 3 & 4.

The other condition for success of the CNOT gate is that one photon is measured across the c0 & c1 modes and another across the t0 & t1 modes. These conditions can be applied with the ``set_post_selection`` method, which can be provided with functions or lambda functions to condition the measurement output. In this case the conditions would be:

.. code-block:: Python

    analyzer.set_post_selection(lambda s: s[1] + s[2] == 1)
    analyzer.set_post_selection(lambda s: s[3] + s[4] == 1)

In principle these functions could also be combined into a single function if desired.

.. warning::
    If multiple lambda functions are created and passed to ``set_post_selection`` using a loop this may create issues related to how lambda functions use out of scope variables. For more info see `here <https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result>`_.

Once the required conditions have been defined, then the Analyzer can be utilised on an input or list of States (providing these have the same photon number) with the ``analyze`` method. In this case we will use all possible qubit inputs for the system. As discussed earlier, the heralded modes are excluded from the inputs/outputs so only 4 mode States need to provided.

.. code-block:: Python

    # Store inputs in both qubit and mode language
    inputs = {"00" : lw.State([1,0,1,0]),
              "01" : lw.State([1,0,0,1]),
              "10" : lw.State([0,1,1,0]),
              "11" : lw.State([0,1,0,1])}
    states = list(inputs.values())

    # Run Analyzer
    results = analyzer.analyze(states)

The results from this can then be plotted to view the transformation. The returned :doc:`../emulator_reference/simulation_result` object contains a plotting method, but in this case it is useful to convert from mode to qubit language and plot manually.

.. code-block:: Python

    import matplotlib.pyplot as plt

    # Create new array with data
    plot_array = np.zeros((len(inputs), len(inputs)))
    for i, istate in enumerate(inputs.values()):
        for j, ostate in enumerate(inputs.values()):
            plot_array[i,j] = results[istate, ostate]

    in_labels = list(inputs.keys())
    out_labels = in_labels

    # Create image plot
    plt.figure(figsize = (7,6))
    plt.imshow(plot_array)
    plt.xticks(range(len(out_labels)), labels = out_labels)
    plt.yticks(range(len(in_labels)), labels = in_labels)
    plt.xlabel("Output")
    plt.ylabel("Input")
    plt.show()

.. image:: assets/analyzer_output.png
    :scale: 75%
    :align: center

As can be seen from the output above, the CNOT gate operates as expected, with the target qubit flipping when the control qubit is set to 1. From the Analyzer it is also possible to calculate the success rate of the system under the provided condition set. In this case we find:

.. code-block:: Python

    print(results.performance)
    # Output: 0.11111111111111126

This is ~ 1/9, the expected success rate for the gate.