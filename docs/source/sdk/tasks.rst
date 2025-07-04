Tasks
=====

Tasks are provided within the SDK as a way for defining jobs that can be executed on a particular backend.

Currently included within Lightworks are the following:

.. list-table:: Lightworks task overview
    :widths: 30 70
    :header-rows: 1

    * - Task
      - Description
    * - :class:`Simulator <lightworks.Simulator>`
      - Generates N samples from the configured system.
    * - :class:`Sampler <lightworks.Sampler>`
      - Calculates the complex probability amplitudes between input and output states.
    * - :class:`Analyzer <lightworks.Analyzer>`
      - Determines possible outputs and their probability based on a set of constraints placed on a system. 
    * - :class:`Batch <lightworks.Batch>`
      - Compiles a number of tasks and enables them to be run all at once.

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

Once created, a task can then be run on a compatible backend. Demonstrations of the usage of tasks with a backend can be found in the :doc:`../emulator/index` section. 

Batch tasks
-----------

It is also possible to use the ``Batch`` task to compile a number of different variations of tasks and run these on a target backend. To achieve this, the task type and the arguments to be used for each task need to be specified. To avoid confusion, each argument needs to be provided as a list, but in the case where an argument doesn't change then this can be a single valued list. Otherwise, all argument numbers need to be the same. For example, the following two configurations where the input state is varied for a target circuit, would be applicable:

.. code-block:: Python

    circ = lw.PhotonicCircuit(3)
    n_samples = 10000

    batch = lw.Batch(
        lw.Sampler, 
        task_args=[
            [circ], 
            [lw.State([1, 0, 0]), lw.State([0, 1, 0]), lw.State([0, 0, 1])], 
            [n_samples]
        ]
    )

    batch = lw.Batch(
        lw.Sampler, 
        task_args=[
            [circ, circ, circ], 
            [lw.State([1, 0, 0]), lw.State([0, 1, 0]), lw.State([0, 0, 1])], 
            [n_samples, n_samples, n_samples]
        ]
    )

Optional arguments can also introduced using the ``task_kwargs`` argument. For example, the Sampler ``min_detection`` and ``random_seed`` options could be modified using:

.. code-block:: Python

    batch = lw.Batch(
        lw.Sampler, 
        task_args=[
            [circ, circ, circ], 
            [lw.State([1, 0, 0]), lw.State([0, 1, 0]), lw.State([0, 0, 1])], 
            [n_samples]
        ],
        task_kwargs={
            "min_detection": [0, 1, 0],
            "random_seed": [10]
        }
    )

Alternatively, a batch can be created through the manual addition of tasks. To achieve this an empty batch is first created and then tasks added with the ``add`` method.

.. code-block:: Python

    batch = lw.Batch()

    task1 = lw.Sampler(lw.PhotonicCircuit(2), lw.State([1, 1]), 10000)
    task2 = lw.Sampler(lw.PhotonicCircuit(3), lw.State([1, 0, 1]), 10000)

    batch.add(task1)
    batch.add(task2)

    print(batch.num)
    # Output: 2

Once created, the batch can then be run on a backend in the same way as any other task.