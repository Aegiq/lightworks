.. Lightworks documentation master file

Lightworks |release| Documentation 
==================================

Introduction
------------

Lightworks is an open-source Python SDK, designed for the encoding of linear optic circuits for application in photonic quantum computing. It features a set of components which allow for the configuration of a target QPU to be defined, with powerful parametrisation and visualization tools for easy display and modification of created systems. Also provided within Lightworks is the emulator, which is a local simulation framework for the testing of a provided configuration before hardware execution, with the ability to account for the typical hardware errors seen within a system.

Where To Begin
--------------

To get started, check out the :doc:`usage` section for further information on how Lightworks can be utilised, including how to :ref:`install <installation>` the project. 

Once setup, there is then a set of tutorials for familiarization with the module, and some :doc:`examples/index` to demonstrate the different ways in which Lightworks can be applied. 

.. note::
   The following documentation is split into two components, the SDK and the emulator. The SDK components are those that are generally used to build and define a job, and these are directly accessible through the Lightworks import. The emulator components are the simulation tools for testing those jobs with classical computational tools, these require a separate module import.

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   usage
   background
   getting_started
   sdk/index
   emulator/index

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

.. toctree::
   :maxdepth: 2
   :caption: Bibliography
   :hidden:

   bibliography