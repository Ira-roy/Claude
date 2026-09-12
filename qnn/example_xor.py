"""
Demo: train the simple QNN to learn an XOR-like decision boundary.

Run with:
    python example_xor.py
"""

import numpy as np

from quantum_neural_network import QuantumNeuralNetwork


def make_xor_dataset(n_samples: int = 40, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi / 2, np.pi / 2, size=(n_samples, 2))
    # Label 1 if the two features have the same sign (an XOR-style pattern).
    y = (np.sign(X[:, 0]) == np.sign(X[:, 1])).astype(float)
    return X, y


def main():
    X, y = make_xor_dataset()

    qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=3)
    params = qnn.fit(X, y, epochs=40, lr=0.4)

    preds = qnn.predict(X, params)
    accuracy = np.mean(preds == y)
    print(f"\nFinal training accuracy: {accuracy:.2%}")

    print("\nSample predictions:")
    for x, true_label, pred in list(zip(X, y, preds))[:8]:
        proba = qnn.predict_proba(x, params)
        print(f"  x={x}, true={int(true_label)}, pred={pred}, p(1)={proba:.3f}")


if __name__ == "__main__":
    main()
