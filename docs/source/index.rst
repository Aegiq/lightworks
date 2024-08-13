.. Lightworks documentation master file

Lightworks |release| Documentation 
==================================

Introduction
------------

Lightworks is an open-source Python SDK, designed for the encoding of linear optic circuits for application in photonic quantum computing. It features a set of components which allow for the configuration of a target QPU to be defined, with powerful parametrisation and visualization tools for easy display and modification of created systems.

Structure
^^^^^^^^^

The core import of Lightworks is the SDK, which contains the components required to build and define a job, such as the Circuit & State objects. There is then a number of sub-packages included which offer different functionality, these are:

* emulator : Enables local simulation of circuits, including the ability to implement complex photonic-specific error models.
* qubit : For programming of qubit gates onto photonic circuits, also contains protocols for conversion of circuits from other languages.
* interferometers : Replicates process of encoding a Lightworks circuit onto a QPU, and offers error modelling capability to examine the effect of errors within this.

Where To Begin
--------------

To get started, check out the :doc:`usage` section for further information on how Lightworks can be utilised, including how to :ref:`install <installation>` the project. 

Once setup, there is then a set of tutorials for familiarization with the module, and some :doc:`examples/index` to demonstrate the different ways in which Lightworks can be applied. 

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   usage
   background
   getting_started
   tutorials/index
   sdk/index
   emulator/index
   qubit/index
   interferometers/index
   tomography/index

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   sdk_reference/index
   emulator_reference/index
   qubit_reference/index
   interferometers_reference/index
   tomography_reference/index

.. toctree::
   :maxdepth: 2
   :caption: Other
   :hidden:

   bibliography
   contributing
