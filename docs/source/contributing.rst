Contributing
============

To contribute to Lightworks, please fork the repo and then merge changes back via a pull request, in the first instance the target for this should be the main branch. Before opening the pull request, all unit tests should be run using `pytest <https://pytest.org/>`_ and these should all pass unless there is good justification otherwise (e.g. the new feature changes core functionality). Additionally, unit tests should be written for all new features to ensure correct functionality of these.

Code Style
----------

To ensure consistency of the Lightworks codebase the tools `ruff <https://docs.astral.sh/ruff/>`_ and `mypy <https://mypy.readthedocs.io/en/stable/>`_ are used (exact usage detailed below). Generally, when writing code for Lightworks there are a few key principles that should be followed:

#. Code should be formatted using ruff with the style configured in pyproject.toml.
#. Where possible, a line limit of 80 is used across the cdode.
#. Docstrings should be written in the `google docstring format <https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings>`_.
#. Type hints should be used in all functions across the module.

ruff
^^^^
ruff is a Python code linter and formatter which enforces a set of standards across the codebase. All code written for Lightworks should be formatted using ruff and must pass the linter with no errors. To format code, the following command should be run in the terminal from the project's root directory:

.. code-block:: console

   (.venv) $ ruff format

Alternatively your IDE may offer a ruff extension (e.g. `Ruff for VSCode <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_) which allows for automatic formatting and highlighting of issues identified by the linter.  

Once formatted, the check mode is used to lint and identify errors as below:

.. code-block:: console

   (.venv) $ ruff check

When errors are identified, these can sometimes be fixed automatically by ruff by adding the ``--fix`` flag to the command.

.. code-block:: console

   (.venv) $ ruff check --fix

Otherwise these will need to be fixed manually. The terminal output will identify the line and column location of the issues. 

.. note::
   On occasions the linter will incorrectly flag issues, in these cases a rule may be disabled for a line using the corresponding syntax, but this should be used sparingly.

mypy
^^^^
mypy enforces the usage of type hints across Lightworks and ensures the consistency of types used between functions. As with ruff, the code should be free of errors before a pull request is raised. To run mypy on the codebase the following is used:

.. code-block:: console

   (.venv) $ mypy -p lightworks

To minimise errors when writing code, all new functions should have type hints for arguments and returns. These should be as specific as allows without inhibiting functionality. Additionally, redefining the types of variables during execution should be avoided unless this is necessary for performance or functionality reasons.

.. note::
   mypy can sometimes be a little tricky to figure out, feel free to ask for assistance on this if you run into any issues.