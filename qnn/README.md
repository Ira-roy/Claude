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
- `example_multi_qubit.py` — the same idea scaled up to more qubits, with
  `--n-qubits`/`--n-layers` CLI flags and a synthetic higher-dimensional
  dataset (one qubit per feature).
- `test_quantum_neural_network.py` — unit tests covering gate unitarity,
  `CNOT` correctness, statevector normalization, expectation-value bounds,
  and (most importantly) that the parameter-shift gradient matches a
  numerical finite-difference gradient of the loss.

## Usage

```bash
pip install numpy
python example_xor.py

# scale up: more qubits/features, more variational layers
python example_multi_qubit.py --n-qubits 6 --n-layers 3
```

```python
from quantum_neural_network import QuantumNeuralNetwork
import numpy as np

qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=3)
params = qnn.fit(X, y, epochs=40, lr=0.4)

predictions = qnn.predict(X, params)
probability_of_class_1 = qnn.predict_proba(X[0], params)
```

## Testing

```bash
cd qnn && python -m unittest test_quantum_neural_network.py -v
```

or, from the repo root:

```bash
python -m unittest discover -s qnn -p "test_*.py" -v
```

Only the standard library's `unittest` is required (no `pytest`), keeping
the project dependency-free beyond NumPy.

## Scaling to more qubits

`QuantumNeuralNetwork` is written generically over `n_qubits` — the
statevector simulator, gate application, and CNOT entangling ring all work
for any qubit count, not just the 2-qubit XOR demo. Each additional feature
just gets its own qubit via angle encoding.

One caveat: since the network reads out only qubit 0's `<Z>` expectation,
information from the other qubits has to be routed there through the
entangling `CNOT` layers. As `n_qubits` grows, you generally need more
`n_layers` for that routing to happen well — e.g. a 4-qubit/2-layer circuit
reaches ~98% training accuracy on the multi-qubit demo, while 6 qubits
needs 3 layers to reach 100%; 6 qubits with only 2 layers plateaus in the
70s. `example_multi_qubit.py --n-qubits <n> --n-layers <n>` makes this easy
to try out directly.

## Notes

- This is a simulator meant for learning/experimentation, not a
  production-scale quantum computing library. The statevector grows as
  `2^n_qubits`, so keep `n_qubits` small (2-8 or so) — both memory and the
  per-gradient-step cost (the parameter-shift rule evaluates the circuit
  twice per parameter per sample) grow accordingly.
- Swap in your own dataset by passing any `X` of shape `(n_samples,
  n_features)` (features are ideally scaled to roughly `[-pi, pi]` since
  they're used as rotation angles) and binary labels `y` of shape
  `(n_samples,)`.
