Imperfect Sampling
==================

Unfortunately, real systems are not perfect, and due to a range of imperfections relating to the source and detection modules of the system the actual results from a photonic system will vary those calculated theoretically. Luckily, we can simulate these imperfections to gain an accurate understanding of the affect they have on a target application.

Single Photon Source
--------------------

.. warning::
    When using an imperfect source, it is possible to very quickly increase the number of possible inputs to the system that the Sampler has to consider. This can result in a significantly increased runtime. In some cases it may be possible to slightly mitigate this by using ``probability_threshold`` option to eliminate possible inputs which occur with very low probability.

Detectors
---------
