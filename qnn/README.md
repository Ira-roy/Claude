# Simple Quantum Neural Network (QNN)

A minimal, dependency-light implementation of a Quantum Neural Network (also
called a variational quantum classifier). It is built entirely with NumPy —
no Qiskit, PennyLane, or other quantum SDK is required — via a small
hand-written statevector simulator.

## How it works

For each input sample `x`:

1. **Angle encoding** — classical features are loaded onto qubits with
   `RY(x_i)` rotations.
2. **Variational layers** — `n_layers` repetitions of trainable `RY`/`RZ`
   rotations on every qubit, followed by a ring of `CNOT` gates that
   entangle neighboring qubits.
3. **Measurement** — the expectation value `<Z>` of qubit 0 is measured and
   mapped from `[-1, 1]` to a probability `p = (<Z> + 1) / 2`.

The network is trained as a binary classifier with binary cross-entropy
loss. Gradients of the rotation angles are computed **exactly** (not by
finite differences or autodiff) using the **parameter-shift rule**:

```
d<Z>/dtheta = 0.5 * (<Z>(theta + pi/2) - <Z>(theta - pi/2))
```

which is then combined with the chain rule through the sigmoid-like
readout and the BCE loss.

## Files

- `quantum_neural_network.py` — the statevector simulator, gate
  definitions, and the `QuantumNeuralNetwork` class (forward pass, loss,
  parameter-shift gradient, and a `fit` training loop).
- `example_xor.py` — trains the QNN on a toy XOR-style 2D classification
  task and prints the loss/accuracy per epoch.

## Usage

```bash
pip install numpy
python example_xor.py
```

```python
from quantum_neural_network import QuantumNeuralNetwork
import numpy as np

qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=3)
params = qnn.fit(X, y, epochs=40, lr=0.4)

predictions = qnn.predict(X, params)
probability_of_class_1 = qnn.predict_proba(X[0], params)
```

## Notes

- This is a simulator meant for learning/experimentation, not a
  production-scale quantum computing library. The statevector grows as
  `2^n_qubits`, so keep `n_qubits` small (2-6 is plenty for demos).
- Swap in your own dataset by passing any `X` of shape `(n_samples,
  n_features)` (features are ideally scaled to roughly `[-pi, pi]` since
  they're used as rotation angles) and binary labels `y` of shape
  `(n_samples,)`.
