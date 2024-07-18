Qubit
=====

The qubit module offers tools for programming the gate-based paradigm of quantum computing within linear optic circuits, including a set of gates and a converter for moving between quantum programming languages.

.. _qubit_usage:

Usage
-----

Some components of the Qubit module require additional dependencies, such as qiskit, and will raise exceptions if these are not installed. Missing dependencies can be added by selecting the required options when installing Lightworks. For example, to install qiskit for use with the qiskit_converter, the following command can be used:

.. code-block:: console

   (.venv) $ pip install lightworks[qiskit]

Contents
--------

.. toctree::
    :maxdepth: 2

    gates
    conversion