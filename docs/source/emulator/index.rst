Emulator
========

The emulator is included as a sub-module within Lightworks, designed to enable the simulation of photonic systems, with a set of tools for retrieving different data about the system functionality. While using the emulator, it is also possible to include imperfections in the system, enabling an understanding of the effect that these imperfections will likely have on a chosen algorithm. All emulation tasks require specification of a backend, which the task is then run against. More info about this can be found in :doc:`backend`. 

The different simulation objects are discussed in the following sections, with an initial comparison of their functionality included below:

.. list-table:: Simulator object comparison
    :widths: 25 25 25 25
    :header-rows: 1

    * - Object
      - Input
      - Output
      - Imperfect source/detectors
    * - :doc:`simulator`
      - Circuit, Input states, Output states
      - Probability amplitudes
      - No
    * - :doc:`sampler`
      - Circuit, Input state, Post-selection/heralding rules
      - Sample counts
      - Yes
    * - :doc:`analyzer`
      - Circuit, Input state, Post-selection/heralding rules
      - Probability distribution
      - No

All simulation objects support the inclusion of heralds through the circuit object, which allows these modes to be excluded from specification of input/outputs within the emulator.

Before using the emulator, a :doc:`theory` section is included which details some of they key concepts behind simulation of indistinguishable photons in linear optic systems.

Contents
--------

.. toctree::
    :maxdepth: 2

    theory
    backend
    simulator
    sampler
    imperfect_sampling
    analyzer