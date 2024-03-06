Parameter & ParameterDict
=========================

As part of a Lightworks, a set of tools is included for the parametrisation of circuits. This includes the :doc:`../sdk_reference/parameter` object for storing single parameter values and the :doc:`../sdk_reference/parameter_dict` object for managing large amounts of Parameters used in a circuit.

Parameter
---------

To create a new Parameter, at minimum an initial value needs to be assigned to the object. It also possible to assign a value to the ``label`` option, which will be used when displaying a circuit which utilises this parameter. Bounds can also be included but this is detailed more later.

.. code-block:: Python

    # Create parameter with value of 1
    parameter = lw.Parameter(1)

    # Also assign label
    parameter = lw.Parameter(1, label = "P1")

The Parameter value can then be modified and retrieved with the ``set`` and ``get`` attributes respectively.

.. code-block:: Python

    print(parameter.get())
    # Output: 1

    parameter.set(2)

    print(parameter.get())
    # Output: 2

Bounds
^^^^^^

With Parameters, it is also possible to set a lower and upper bound for numeric values, which will constrain a parameter and raise an error if it is attempted to be set outside of this range. This can be useful when a Parameter may only take on certain values. To set bounds for the system this can be achieved with the ``bounds`` argument of Parameter creation, the value should be a list of the form [lower bound, upper bound]. It is also possible to leave one or both of the bounds open by assigning the value to None.

.. code-block:: Python

    # Set bounds from 0 to 2
    parameter = lw.Parameter(1, bounds = [0, 2])

    # Set lower bound to 0 and no upper bounds
    parameter = lw.Parameter(1, bounds = [0, None])

Alternatively, it is possible to set the lower and upper bounds of a Parameter after creation by modifying the ``min_bound`` and ``max_bound`` attributes. As above, it is possible to assign a value to None to remove a bound.

.. code-block:: Python

    # First create new parameter
    parameter = lw.Parameter(1)

    # Update lower bound
    parameter.min_bound = 0

    # Update upper bound
    parameter.max_bound = 2

    # Remove lower bound
    parameter.min_bound = None

Also included within Parameter is the ``had_bounds`` method, which can be used to quickly check whether a parameter has at least one bound associated with it, returning either True or False.

.. code-block:: Python

    # First create new parameter
    parameter = lw.Parameter(1)

    print(parameter.had_bounds())
    # Output: False

    # Add a bound
    parameter.min_bound = 0

    print(parameter.had_bounds())
    # Output: False

.. note::
    Parameter bounds only work with numeric values. If a Parameter is non-numeric then bounds cannot be assigned to it, and likewise if a Parameter has bounds then it cannot be modified to a non-numeric value.

ParameterDict
-------------