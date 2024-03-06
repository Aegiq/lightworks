Imperfect Sampling
==================

Unfortunately, real systems are not perfect, and due to a range of imperfections relating to the source and detection modules of the system the actual results from a photonic system will vary those calculated theoretically. Luckily, we can simulate these imperfections to gain an accurate understanding of the affect they have on a target application.

.. note::
    Currently imperfections in the source and detectors are only supported in the Sampler.

Single Photon Source
--------------------

An imperfect source will affect both the rate at which samples can be generated from the system (through brightness) and introduce errors in the computation (through less than ideal purity, indistinguishability). To include an imperfect source, we create a new :doc:`../emulator_reference/source` object, and set the quantities as required. Note that all quantities are optional, and if not specified will default to their ideal values. Some examples of this are shown below.

.. code-block:: Python

    # 90% purity
    source = emulator.Source(purity = 0.9)

    # 90% indistinguishability and 95% purity
    source = emulator.Source(purity = 0.95, indistinguishability = 0.9)

    # 90% indistinguishability and 60% brightness
    source = emulator.Source(indistinguishability = 0.9, brightness = 0.6)

.. note::
    The imperfect source is simulated by a model (reference) and may not exactly match real system behaviour. It also has some limitations, such as assuming an identical mutual distinguishability between all photons.

To include the imperfect source when sampling, the Source should be included as an argument in the Sampler. This is shown below with a random unitary circuit and input state.

.. code-block:: Python

    # Create circuit and unitary
    circuit = lw.Unitary(lw.random_unitary(4))
    input_state = lw.State([1,0,1,0])

    # Then define a source to use
    source = emulator.Source(purity = 0.95, indistinguishability = 0.9)

    # And finally add this in the Sampler
    sampler = emulator.Sampler(circuit, input_state, source = source)

The system can then be sampled from in the usual way, and the source properties will also affect the calculated probability distribution assigned to the ``probability_distribution`` attribute.

.. warning::
    When using an imperfect source, it is possible to very quickly increase the number of possible inputs to the system that the Sampler has to consider. This can result in a significantly increased runtime. In some cases it may be possible to slightly mitigate this by using ``probability_threshold`` option to eliminate possible inputs which occur with very low probability.

Detectors
---------
