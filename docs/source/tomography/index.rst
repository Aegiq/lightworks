Tomography
==========

The tomography modules provides routines for performing state and process tomography on circuits within lightworks. It assumes that a dual-rail encoding scheme is used with pairs of adjacent photonic modes. More information about how to perform the different kinds of tomography can be found in the corresponding sections.

Contents
--------

.. toctree::
    :maxdepth: 1

    state_tomography
    process_tomography
    gate_fidelity

Overview
--------

Performing tomography in Lightworks consists of the following steps:
#. The target tomography object is created by providing the number of qubits & the circuit to examine.
#. ``get_experiments`` is used to generate all of the required experiments for the target algorithm. For state tomography, each experiment consists of a modification to the initial circuit. Process tomography is the same, but also contains specification for the input state to the system.
#. These experiments are then run by the user on the select backend with their chosen settings.
#. The created results are provided to the ``process`` method, which runs the tomography algorithm and generates a result.