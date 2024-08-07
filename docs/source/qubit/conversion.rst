Circuit Conversion
==================

The Lightworks Qubit module contains methods for converting between quantum programming languages, aimed at making it easier to move between and compare them. The included conversion options are discussed below:

Qiskit
------

The :func:`lightworks.qubit.converter.qiskit_converter` function can be used to perform conversion between a provided qiskit `QuantumCircuit <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit>`_ and a lightworks :doc:`../sdk_reference/circuit`.

To use this, it should be first be imported from the converter module of Qubit. This will raise an error if the qiskit requirements are not installed, for mode info see :ref:`Qubit Usage <qubit_usage>` 

.. code-block:: Python

    from lightworks.qubit.converter import qiskit_converter
    from qiskit import QuantumCircuit

To demonstrate the converter, a qiskit QuantumCircuit should then be defined. In this case a 3 qubit circuit is chosen with a Hadamard, X-gate and 2 CNOTs.  

.. code-block:: Python

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0,2)
    qc.cx(2,1)
    qc.x(0)

    qc.draw(output="mpl")

.. image:: assets/qiskit_circuit_demo.png
    :scale: 100%
    :align: center

The Lightworks circuit is then created by providing this circuit directly to the ``qiskit_converter`` function. The created circuit can then be displayed.

.. code-block:: Python

    conv_circ = qiskit_converter(qc)

    conv_circ.display()

.. image:: assets/converted_qiskit_circuit_demo.svg
    :scale: 75%
    :align: center

From this circuit, there are a couple of things to note. The first is that the first CNOT (acting on qubits 0 & 2) has been replaced by a CNOT and two swap gates. This occurs because the photonic CNOT gate must act on adjacent modes, so the converter will automatically include swaps to ensure this is the case. In some cases this may lead to circuits which are non-optimal, and so a lower depth circuit may be possible by manually including swap components in these cases. The other thing to note is that in the second CNOT gate, the text '(1, 0)' is included in the gate label, this is used to indicate that the lower qubit modes (those representing qubit two) are acting as the control qubit and the upper modes are acting as the target qubit. These labels will always be relative to the smaller of first qubit which the gate is acting on instead of showing absolute qubit numbers.

.. note::

    The converter supports inclusion of a CCZ or CCX/Toffoli gate, but when added this must be the only multi-qubit gate in the system. Additionally, post-selection must be applied on the results to get the correct values from the system. 