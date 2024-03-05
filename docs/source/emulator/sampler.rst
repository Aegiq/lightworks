Sampler
=======

The Sampler can be used to simulate the process of running jobs with an interferometer-based photonic system, in which the same input is placed into the system many times and the output distribution measured. As this output from the system is inherently probabilistic the output distribution will tend towards the true distribution as more samples are collected. The number of samples collected from the system will depend on resource constraints, but will never be infinite, meaning there will always be some small variation from the ideal distribution. 

As sampling is how the system is utilised, it is very useful to be able to understand how the process will look like for a given job. To demonstrate sampling we will show how the 2 qubit CNOT gate from (insert reference) can be tested with the emulator. The circuit to define this gate is shown below, it occupies 6 modes where the 4 central modes c0, c1, t0 & t1 are used for qubit encoding.

.. code-block:: Python

    import numpy as np

.. code-block:: Python

    cnot_circuit = lw.Circuit(6)

    theta = np.arccos(1/3)
    to_add = [(3, np.pi/2, 0), (0, theta, 0), (2, theta, p), (4, theta, 0), 
              (3, np.pi/2, 0)]

    for m, t, p in to_add:
        cnot_circuit.add_bs(m, loss = loss, reflectivity = 0.5)
        cnot_circuit.add_ps(m+1, t)
        cnot_circuit.add_bs(m, loss = loss, reflectivity = 0.5)
        cnot_circuit.add_ps(m+1, p)

.. include display here

Note that this CNOT gate requires a set of post-selection criteria to function correctly, but this will be discussed more later. To test the CNOT gate, we will examine the output