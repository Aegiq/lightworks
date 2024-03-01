Emulator
========

This is an introduction

Below is a comparison of the functionality of the different simulation objects included with the emulator:

.. list-table:: Simulator object comparison
    :widths: 25 25 25 25
    :header-rows: 1

    * - Object
      - Input
      - Output
      - Imperfect source/detectors
    * - Simulator
      - Circuit, Input states, Output states
      - Probability amplitudes
      - No
    * - Sampler
      - Circuit, Input state, Post-selection/heralding
      - Sample counts
      - Yes
    * - Analyzer
      - Circuit, Input state, Post-selection/heralding rules
      - Probability distribution
      - No
    * - QuickSampler
      - Circuit, Input state
      - Sample counts
      - Only non-pnr detectors

Contents
--------

.. toctree::
    :maxdepth: 2

    theory
    simulator
    sampler
    imperfect_sampling
    analyzer
    quick_sampler
    fermionic_stats