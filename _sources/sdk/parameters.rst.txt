Parameter & ParameterDict
=========================

As part of a Lightworks, a set of tools is included for the parametrisation of circuits. This includes the :class:`Parameter <lightworks.Parameter>` object for storing single parameter values and the :class:`ParameterDict <lightworks.ParameterDict>` object for managing large amounts of Parameters used in a circuit.

Parameter
---------

To create a new Parameter, at minimum an initial value needs to be assigned to the object. It is also possible to assign a value to the ``label`` option, which will be used when displaying a circuit which utilises this parameter. Bounds can also be included, but this is detailed more later.

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

With Parameters, it is also possible to set a lower and upper bound for numeric values, which will constrain a parameter and raise an error if it is attempted to be set outside this range. This can be useful when a Parameter may only take on certain values. To set bounds for the system this can be achieved with the ``bounds`` argument of Parameter creation, the value should be a list of the form [lower bound, upper bound]. It is also possible to leave one or both of the bounds open by assigning the value to None.

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

A ParameterDict functions very similar to a normal dictionary, but is designed specially for storing and modifying Parameters via assigned keys. On creation, it will initially be an empty dictionary.

.. code-block:: Python
    
    pd = lw.ParameterDict()

Parameters can then be added to the dictionary using the [] operator, where keys should typically be strings and the values are Parameter objects.

.. code-block:: Python

    pd["p1"] = lw.Parameter(1)
    pd["p2"] = lw.Parameter(2)

Once a Parameter has been added to the dictionary, it is possible to update the value without using the ``set`` method of the Parameter directly. Instead, the following can be used.

.. code-block:: Python

    pd["p1"] = 3

If "p1" was not an existing key in the dictionary then this would raise an exception. It can then be verified that the parameter has been updated using:

.. code-block:: Python

    print(pd["p1"].get())

.. note::
    Above, pd["p1"] returns the Parameter object itself and not its value, we therefore need to use get() to output the updated value of the object.

A Parameter added to the dictionary can also be removed through providing the associated key to the ``remove`` method.

.. code-block:: Python

    pd.remove("p1")

The ParameterDict also supports some other functionality which is similar to a normal dictionary, including the ``keys`` method to return an iterable of all keys used and ``items`` to return an iterable of keys and associated Parameter values. Alternatively, the keys used in a dictionary can be accessed from the ``params`` attribute, returning a list of values.

.. code-block:: Python

    print(pd.params)
    # Output: ["p2"]

Bounds
^^^^^^

The ParameterDict also supports some additional functionality related to bounds. The ``has_bounds`` method will check if any of the Parameters included have associated bounds, retuning either True or False. The bounds associated with all Parameters used in the dictionary can also be retrieved with ``get_bounds``, which will return a dictionary of Parameter keys and bounds. For any bounds which are set to None these will be replaced with +/- infinity (using inf from the Python math module).

.. code-block:: Python

    pd = lw.ParameterDict()

    pd["p1"] = lw.Parameter(1, bounds = [0, 2])
    pd["p2"] = lw.Parameter(2, bounds = [0, None])

    print(pd.has_bounds())
    # Output: True

    print(pd.get_bounds())
    # Output: {'p1': (0, 2), 'p2': (0, inf)}