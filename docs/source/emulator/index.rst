Emulator
========

The emulator is included as a sub-module within Lightworks, designed to enable the simulation of photonic systems, with a set of tools for retrieving different data about the system functionality. While using the emulator, it is also possible to include imperfections in the system, enabling an understanding of the effect that these imperfections will likely have on a chosen algorithm. The different simulation objects are discussed in the following sections, with an initial comparison of their functionality included below:

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
..    * - :doc:`quick_sampler`
..      - Circuit, Input state, Heralding rules
..      - Sample counts
..      - Only non-pnr detectors

Before using the emulator, a :doc:`theory` section is included which details some of they key concepts behind simulation of indistinguishable photons in linear optic systems.

Contents
--------

.. toctree::
    :maxdepth: 2

    theory
    simulator
    sampler
    imperfect_sampling
    analyzer
..    quick_sampler