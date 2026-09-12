"""
Test suite for the simple QNN (quantum_neural_network.py).

Uses only the standard library's `unittest` plus NumPy, matching this
project's zero-extra-dependency philosophy.

Run with:
    cd qnn && python -m unittest test_quantum_neural_network.py -v
or, from the repo root:
    python -m unittest discover -s qnn -p "test_*.py" -v
"""

import unittest

import numpy as np

from quantum_neural_network import (
    QuantumNeuralNetwork,
    apply_cnot,
    apply_single_qubit_gate,
    ry,
    rz,
)


def basis_state(index: int, n_qubits: int) -> np.ndarray:
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[index] = 1.0
    return state


class TestGates(unittest.TestCase):
    def test_ry_is_unitary(self):
        for theta in (0.0, 0.7, np.pi / 2, np.pi, -1.3):
            gate = ry(theta)
            np.testing.assert_allclose(gate @ gate.conj().T, np.eye(2), atol=1e-10)

    def test_rz_is_unitary(self):
        for theta in (0.0, 0.7, np.pi / 2, np.pi, -1.3):
            gate = rz(theta)
            np.testing.assert_allclose(gate @ gate.conj().T, np.eye(2), atol=1e-10)

    def test_ry_zero_is_identity(self):
        np.testing.assert_allclose(ry(0.0), np.eye(2), atol=1e-10)

    def test_ry_pi_flips_basis_state(self):
        # RY(pi) = [[0, -1], [1, 0]] should send |0> to (up to phase) |1>.
        n = 1
        state = apply_single_qubit_gate(basis_state(0, n), ry(np.pi), 0, n)
        probs = np.abs(state) ** 2
        np.testing.assert_allclose(probs, [0.0, 1.0], atol=1e-10)

    def test_single_qubit_gate_preserves_norm(self):
        n = 3
        rng = np.random.default_rng(0)
        state = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        state /= np.linalg.norm(state)
        for q in range(n):
            state = apply_single_qubit_gate(state, ry(0.9), q, n)
            state = apply_single_qubit_gate(state, rz(0.4), q, n)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=10)


class TestCNOT(unittest.TestCase):
    def test_cnot_control0_target1_two_qubits(self):
        n = 2
        # |10> (control=1) -> |11>
        out = apply_cnot(basis_state(2, n), control=0, target=1, n_qubits=n)
        np.testing.assert_allclose(out, basis_state(3, n), atol=1e-10)

        # |11> -> |10>
        out = apply_cnot(basis_state(3, n), control=0, target=1, n_qubits=n)
        np.testing.assert_allclose(out, basis_state(2, n), atol=1e-10)

        # |00> and |01> (control=0) are left unchanged
        for idx in (0, 1):
            out = apply_cnot(basis_state(idx, n), control=0, target=1, n_qubits=n)
            np.testing.assert_allclose(out, basis_state(idx, n), atol=1e-10)

    def test_cnot_control1_target0_two_qubits(self):
        n = 2
        # |01> (control qubit1=1) -> |11>
        out = apply_cnot(basis_state(1, n), control=1, target=0, n_qubits=n)
        np.testing.assert_allclose(out, basis_state(3, n), atol=1e-10)

    def test_cnot_with_bystander_qubit(self):
        # 3 qubits, CNOT(control=0, target=2); qubit1 is an untouched bystander.
        n = 3
        # |100> (bits q0q1q2 = 1,0,0 -> index 4) -> |101> (index 5)
        out = apply_cnot(basis_state(4, n), control=0, target=2, n_qubits=n)
        np.testing.assert_allclose(out, basis_state(5, n), atol=1e-10)

        # |110> (index 6, q0=1 control, q1=1 bystander) -> |111> (index 7)
        out = apply_cnot(basis_state(6, n), control=0, target=2, n_qubits=n)
        np.testing.assert_allclose(out, basis_state(7, n), atol=1e-10)

    def test_cnot_preserves_norm(self):
        n = 4
        rng = np.random.default_rng(1)
        state = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        state /= np.linalg.norm(state)
        state = apply_cnot(state, control=1, target=3, n_qubits=n)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=10)


class TestQuantumNeuralNetwork(unittest.TestCase):
    def test_param_count_and_init_shape(self):
        qnn = QuantumNeuralNetwork(n_qubits=3, n_layers=2)
        self.assertEqual(qnn.n_params, 3 * 2 * 2)
        params = qnn.init_params(seed=0)
        self.assertEqual(params.shape, (qnn.n_params,))
        self.assertTrue(np.all(params >= -np.pi) and np.all(params <= np.pi))

    def test_init_params_is_reproducible(self):
        qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=2)
        p1 = qnn.init_params(seed=42)
        p2 = qnn.init_params(seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_circuit_produces_normalized_state(self):
        rng = np.random.default_rng(2)
        for n_qubits in (2, 3, 4):
            qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=2)
            params = qnn.init_params(seed=1)
            x = rng.uniform(-np.pi, np.pi, size=n_qubits)
            state = qnn._run_circuit(x, params)
            self.assertAlmostEqual(np.sum(np.abs(state) ** 2), 1.0, places=10)

    def test_expval_and_proba_bounds(self):
        rng = np.random.default_rng(3)
        qnn = QuantumNeuralNetwork(n_qubits=3, n_layers=2)
        params = qnn.init_params(seed=0)
        for _ in range(10):
            x = rng.uniform(-np.pi, np.pi, size=3)
            e = qnn.expval_z0(x, params)
            self.assertGreaterEqual(e, -1.0 - 1e-9)
            self.assertLessEqual(e, 1.0 + 1e-9)
            p = qnn.predict_proba(x, params)
            self.assertGreaterEqual(p, 0.0 - 1e-9)
            self.assertLessEqual(p, 1.0 + 1e-9)

    def test_generalizes_to_more_qubits_than_features(self):
        # n_qubits=4 but only 2 input features -- angle encoding wraps around.
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=1)
        params = qnn.init_params(seed=0)
        x = np.array([0.3, -0.6])
        state = qnn._run_circuit(x, params)
        self.assertAlmostEqual(np.sum(np.abs(state) ** 2), 1.0, places=10)

    def test_predict_returns_binary_labels(self):
        qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=2)
        params = qnn.init_params(seed=0)
        X = np.array([[0.1, 0.2], [-0.3, 0.4], [1.0, -1.0]])
        preds = qnn.predict(X, params)
        self.assertEqual(preds.shape, (3,))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_gradient_matches_finite_differences(self):
        """The parameter-shift gradient should match a numerical gradient
        of the same loss function within a small tolerance."""
        rng = np.random.default_rng(4)
        qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=1)
        params = qnn.init_params(seed=0)
        X = rng.uniform(-1.0, 1.0, size=(3, 2))
        y = np.array([0.0, 1.0, 1.0])

        analytic_grad = qnn.gradient(X, y, params)

        eps = 1e-6
        numerical_grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += eps
            p_minus = params.copy()
            p_minus[i] -= eps
            numerical_grad[i] = (
                qnn.loss(X, y, p_plus) - qnn.loss(X, y, p_minus)
            ) / (2 * eps)

        np.testing.assert_allclose(analytic_grad, numerical_grad, atol=1e-4)

    def test_fit_reduces_loss_and_learns_xor(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-np.pi / 2, np.pi / 2, size=(24, 2))
        y = (np.sign(X[:, 0]) == np.sign(X[:, 1])).astype(float)

        qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=3)
        params = qnn.init_params(seed=0)
        initial_loss = qnn.loss(X, y, params)

        trained_params = qnn.fit(X, y, params=params, epochs=25, lr=0.4, verbose=False)
        final_loss = qnn.loss(X, y, trained_params)
        accuracy = np.mean(qnn.predict(X, trained_params) == y)

        self.assertLess(final_loss, initial_loss)
        self.assertGreaterEqual(accuracy, 0.8)


if __name__ == "__main__":
    unittest.main()
