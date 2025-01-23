Tasks
=====

Tasks are provided within the SDK as a way for defining jobs that can be executed on a particular backend.

Currently included within Lightworks are the following:

.. list-table:: Lightworks task overview
    :widths: 30 70
    :header-rows: 1

    * - Task
      - Description
    * - :doc:`../sdk_reference/tasks/simulator`
      - Generates N samples from the configured system.
    * - :doc:`../sdk_reference/tasks/sampler`
      - Calculates the complex probability amplitudes between input and output states.
    * - :doc:`../sdk_reference/tasks/analyzer`
      - Determines possible outputs and their probability based on a set of constraints placed on a system. 

Usage
-----

To create a task, the corresponding object needs to be supplied with required arguments and any optional ones that may be needed. Exact descriptions of these can be found in the task docstrings or within the reference section of this documentation. As an example, to create a new Sampler task, the following may be used.

.. code-block:: Python

    circuit = lw.PhotonicCircuit(2)
    input_state = lw.State([1, 1])
    n_samples = 10000

    sampler = lw.Sampler(circuit, input_state, n_samples)

To then add a minimum photon detection number of 2, the required code would be:

.. code-block:: Python

    sampler = lw.Sampler(circuit, input_state, n_samples, min_detection = 2)

Alternatively, this value (and any others provided to the Sampler) can be edited after creation by modifying attributes. For example:

.. code-block:: Python

    sampler.min_detection = 1

Once created, a task can then be run on a compatible backend. Demonstrations of the usage of tasks with a backend can be found in the `../emulator/index` section. 