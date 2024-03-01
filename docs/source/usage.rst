Usage
=====

Requirements
------------

The Aegiq SDK requires Python 3.10+. If not already installed, it can be downloaded from `python.org <https://www.python.org/>`_.

.. _installation:

Installation
------------

To use the Aegiq SDK, it can be installed directly through pip using the command:

.. code-block:: console

   (.venv) $ pip install aegiq_sdk

This will also install all required module dependencies for the project.

.. note::
    The SDK has been primarily designed for interaction through Jupyter notebooks, and this is where tools such as the circuit visualization will work best. Jupyter notebooks can be accessed through the Jupyter software directly or through an editor such as Visual Studio Code.

Importing the SDK
-----------------

When using the SDK, throughout the documentation the following import will be used:

.. code-block:: Python

    import aegiq_sdk as sdk

Other than that, when using the emulator it is useful to import this as:

.. code-block:: Python

    from aegiq_sdk import emulator