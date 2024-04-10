Imperfect Sampling
==================

Unfortunately, real systems are not perfect, and due to a range of imperfections relating to the source and detection modules of the system the actual results from a photonic quantum computer will vary those calculated theoretically. Luckily, we can simulate these imperfections to gain an accurate understanding of the affect they have on a target application.

.. note::
    Currently, imperfections in the source and detectors are only supported in the Sampler.

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

The system can then be sampled in the usual way, and the source properties will also affect the calculated probability distribution assigned to the ``probability_distribution`` attribute.

.. warning::
    When using an imperfect source, it is possible to very quickly increase the number of possible inputs to the system that the Sampler has to consider. This can result in a significantly increased runtime. In some cases it may be possible to slightly mitigate this by using the ``probability_threshold`` option to eliminate inputs which occur with very low probability.

Detectors
---------

There is also a number of detector imperfections that can be included, through the use of the :doc:`../emulator_reference/detector` object. This includes the efficiency, which will alter the output count rates and whether the detectors are photon number resolving, which as well as reducing count rates can also be a benefit/limitation for some applications. Also included is dark counts, which occur when the system registers a detection event where none is present, generating states that should not exist. This is provided as a probability, meaning the detector dark count rate and system clock needs to be taken into account. Currently, it is assumed that all detection channels are identical across the system. A few examples of detector usage are shown below:

.. code-block:: Python

    # Non photon number resolving detectors
    detector = emulator.Detector(photon_counting = False)

    # 80% efficiency
    detector = emulator.Detector(efficiency = 0.8)

    # Non photon-number resolving and 10^-6 dark count probability
    detector = emulator.Detector(photon_counting = False, p_dark = 1e-6)

As with the Source, the Detector is then included in the initial Sampler creation.

.. code-block:: Python

    # Create circuit and unitary
    circuit = lw.Unitary(lw.random_unitary(4))
    input_state = lw.State([1,0,1,0])

    # Then define a detector to use
    detector = emulator.Detector(photon_counting = False, p_dark = 1e-6)

    # And finally add this in the Sampler
    sampler = emulator.Sampler(circuit, input_state, detector = detector)

When the sample methods are used the detector is then applied as a post-processing step on the output state, before any heralding and post-selection options are included. It is important to note that, unlike when using a source, the detector options will not alter the values seen in the ``probability_distribution`` attribute. For example, using a Detector with ``photon_counting = False`` wouldn't produce states with a maximum of one photon per mode in the probability distribution.