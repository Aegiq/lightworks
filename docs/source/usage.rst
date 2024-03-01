Usage
=====

Requirements
------------

Lightworks requires Python 3.10+. If not already installed, it can be downloaded from `python.org <https://www.python.org/>`_.

.. _installation:

Installation
------------

To use Lightworks, it can be installed directly through pip using the command:

.. code-block:: console

   (.venv) $ pip install lightworks

This will also install all required module dependencies for the project.

.. note::
    Lightworks has been primarily designed for interaction through Jupyter notebooks, and this is where tools such as the circuit visualization will work best. Jupyter notebooks can be accessed through the Jupyter software directly or through an editor such as Visual Studio Code.

Importing Lightworks
--------------------

When using Lightworks, throughout the documentation the following import will be used:

.. code-block:: Python

    import lightworks as lw

Other than that, when using the emulator it is useful to import this as:

.. code-block:: Python

    from lightworks import emulator