"""
A simple Quantum Neural Network (QNN) built from scratch with NumPy only.

No quantum SDK (Qiskit / PennyLane / etc.) is required -- this module implements
a minimal statevector simulator and a variational quantum circuit ("ansatz") on
top of it, then trains the circuit's rotation-gate angles like a neural network
using the parameter-shift rule for exact gradients.

Circuit design (per input sample x):
  1. Angle encoding:  RY(x_i) on qubit i, for each feature x_i.
  2. `n_layers` variational layers, each layer applying:
       - RY(theta) and RZ(theta) on every qubit (the trainable weights)
       - a ring of CNOT gates entangling neighboring qubits
  3. Measurement: the expectation value <Z> of qubit 0 is read out and mapped
     from [-1, 1] to a probability in [0, 1] via p = (<Z> + 1) / 2.

The network is trained as a binary classifier with binary cross-entropy loss,
using the parameter-shift rule to compute exact analytic gradients of the
expectation value with respect to each rotation angle.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Single- and two-qubit gates
# --------------------------------------------------------------------------

def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


def apply_single_qubit_gate(
    state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    """Apply a 2x2 gate to `qubit` of an n_qubits statevector."""
    arr = state.reshape([2] * n_qubits)
    arr = np.moveaxis(arr, qubit, 0)
    arr = gate @ arr.reshape(2, -1)
    arr = arr.reshape([2] + [2] * (n_qubits - 1))
    arr = np.moveaxis(arr, 0, qubit)
    return arr.reshape(-1)


def apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """Apply a CNOT gate with the given control/target qubits."""
    others = [q for q in range(n_qubits) if q not in (control, target)]
    order = [control, target] + others

    arr = state.reshape([2] * n_qubits)
    arr = np.transpose(arr, order).reshape(2, 2, -1)

    new_arr = arr.copy()
    new_arr[1, 0, :] = arr[1, 1, :]
    new_arr[1, 1, :] = arr[1, 0, :]

    new_arr = new_arr.reshape([2, 2] + [2] * len(others))
    inv_order = np.argsort(order)
    new_arr = np.transpose(new_arr, inv_order)
    return new_arr.reshape(-1)


# --------------------------------------------------------------------------
# The Quantum Neural Network
# --------------------------------------------------------------------------

class QuantumNeuralNetwork:
    """A variational quantum circuit trained as a binary classifier."""

    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 2  # RY + RZ per qubit per layer

        # Precompute which basis states have qubit 0 in state |0> (Z = +1).
        dim = 2 ** n_qubits
        indices = np.arange(dim)
        bit0 = (indices >> (n_qubits - 1)) & 1  # qubit 0 is the most-significant bit
        self._z0_eigenvalues = np.where(bit0 == 0, 1.0, -1.0)

    def init_params(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(-np.pi, np.pi, size=self.n_params)

    def _run_circuit(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        state = np.zeros(2 ** n, dtype=complex)
        state[0] = 1.0

        # Angle encoding of the classical input.
        for q in range(n):
            angle = x[q % len(x)]
            state = apply_single_qubit_gate(state, ry(angle), q, n)

        # Variational (trainable) layers.
        weights = params.reshape(self.n_layers, n, 2)
        for layer in range(self.n_layers):
            for q in range(n):
                theta_y, theta_z = weights[layer, q]
                state = apply_single_qubit_gate(state, ry(theta_y), q, n)
                state = apply_single_qubit_gate(state, rz(theta_z), q, n)
            for q in range(n):
                state = apply_cnot(state, q, (q + 1) % n, n)

        return state

    def expval_z0(self, x: np.ndarray, params: np.ndarray) -> float:
        state = self._run_circuit(x, params)
        probs = np.abs(state) ** 2
        return float(np.sum(probs * self._z0_eigenvalues))

    def predict_proba(self, x: np.ndarray, params: np.ndarray) -> float:
        return (self.expval_z0(x, params) + 1.0) / 2.0

    def predict(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        return np.array(
            [1 if self.predict_proba(x, params) >= 0.5 else 0 for x in X]
        )

    # ----------------------------------------------------------------
    # Training: binary cross-entropy loss + parameter-shift gradients
    # ----------------------------------------------------------------

    def loss(self, X: np.ndarray, y: np.ndarray, params: np.ndarray, eps: float = 1e-9) -> float:
        preds = np.array([self.predict_proba(x, params) for x in X])
        preds = np.clip(preds, eps, 1 - eps)
        return float(-np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds)))

    def gradient(self, X: np.ndarray, y: np.ndarray, params: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Exact gradient of the BCE loss via the parameter-shift rule.

        For a rotation gate exp(-i*theta/2*P), the exact gradient of an
        expectation value <Z> with respect to theta is:
            d<Z>/dtheta = 0.5 * (<Z>(theta + pi/2) - <Z>(theta - pi/2))
        The chain rule then propagates this through p = (<Z>+1)/2 and the
        binary cross-entropy loss.
        """
        shift = np.pi / 2
        grad = np.zeros_like(params)

        # Cache per-sample quantities at the current parameters.
        base_probs = np.clip(
            np.array([self.predict_proba(x, params) for x in X]), eps, 1 - eps
        )
        dbce_dp = -(y / base_probs - (1 - y) / (1 - base_probs))  # shape (n_samples,)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += shift
            params_minus = params.copy()
            params_minus[i] -= shift

            d_expval = np.array(
                [
                    0.5 * (self.expval_z0(x, params_plus) - self.expval_z0(x, params_minus))
                    for x in X
                ]
            )
            d_prob = 0.5 * d_expval
            grad[i] = np.mean(dbce_dp * d_prob)

        return grad

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: np.ndarray | None = None,
        epochs: int = 30,
        lr: float = 0.3,
        verbose: bool = True,
    ) -> np.ndarray:
        if params is None:
            params = self.init_params(seed=0)

        for epoch in range(epochs):
            grad = self.gradient(X, y, params)
            params = params - lr * grad
            if verbose:
                loss = self.loss(X, y, params)
                acc = np.mean(self.predict(X, params) == y)
                print(f"epoch {epoch + 1:3d}/{epochs}  loss={loss:.4f}  acc={acc:.2f}")

        return params
