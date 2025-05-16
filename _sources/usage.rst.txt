Usage
=====

Requirements
------------

Lightworks requires Python 3.10 - 3.13. If not already installed, it can be downloaded from `python.org <https://www.python.org/>`_.

.. _installation:

Installation
------------

It is recommended to install Lightworks into a virtual environment, such as `venv <https://docs.python.org/3/library/venv.html/>`_. This will help to avoid conflicts with other modules. Lightworks can then be directly installed through pip with the command:

.. code-block:: console

   (.venv) $ pip install lightworks

This will also install all required dependencies for the project.

.. note::
    Lightworks has been primarily designed for interaction through Jupyter notebooks, and this is where tools such as the circuit visualization will work best. Jupyter notebooks can be accessed through the Jupyter software directly or through an IDE such as Visual Studio Code.

Importing Lightworks
--------------------

When using Lightworks, throughout the documentation the following import will be used:

.. code-block:: Python

    import lightworks as lw

Then, when using the emulator this will be imported as:

.. code-block:: Python

    from lightworks import emulator

.. note::
    These imports are not explicitly included in the majority of this documentation. 