SDK
===

The SDK components are those used for defining a job or algorithm which is to be computed. These components are compatible with both hardware execution and simulation through the emulator. Central to the SDK is the :doc:`circuit`, which determines the exact configuration of the QPU which is to be used, and the :doc:`state` which is used for containing quantum state data, such an inputs and outputs of a system.

Contents
--------

.. toctree::
    :maxdepth: 2

    circuit
    unitary
    parameters
    state
    qubit
    utilities