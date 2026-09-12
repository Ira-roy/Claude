"""
Demo: scale the simple QNN up to more qubits / higher-dimensional data.

`QuantumNeuralNetwork` already generalizes to any number of qubits -- the
statevector simulator and gates in `quantum_neural_network.py` are written
generically over `n_qubits`. This script exercises that by classifying a
synthetic dataset with more features than the earlier 2-qubit XOR demo,
using one qubit per feature.

Run with the default (4 qubits / 4 features):
    python example_multi_qubit.py

Or try a different qubit count (also changes the dataset's dimensionality):
    python example_multi_qubit.py --n-qubits 6 --n-layers 2
"""

import argparse

import numpy as np

from quantum_neural_network import QuantumNeuralNetwork


def make_blobs_dataset(n_features: int, n_samples: int = 60, seed: int = 0):
    """Two Gaussian blobs in `n_features` dimensions, scaled for angle encoding."""
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2

    center_a = rng.uniform(-1.0, 1.0, size=n_features)
    center_b = -center_a  # push the two classes apart

    X_a = center_a + 0.4 * rng.standard_normal((n_per_class, n_features))
    X_b = center_b + 0.4 * rng.standard_normal((n_per_class, n_features))

    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    # Shuffle and scale features into roughly [-pi/2, pi/2] for angle encoding.
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    X = np.clip(X, -2, 2) * (np.pi / 4)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train the QNN on a multi-qubit demo.")
    parser.add_argument("--n-qubits", type=int, default=4, help="number of qubits/features")
    parser.add_argument("--n-layers", type=int, default=2, help="number of variational layers")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.6)
    args = parser.parse_args()

    X, y = make_blobs_dataset(n_features=args.n_qubits)

    print(f"Training a {args.n_qubits}-qubit, {args.n_layers}-layer QNN "
          f"({args.n_qubits * args.n_layers * 2} trainable parameters)...\n")

    qnn = QuantumNeuralNetwork(n_qubits=args.n_qubits, n_layers=args.n_layers)
    params = qnn.fit(X, y, epochs=args.epochs, lr=args.lr)

    preds = qnn.predict(X, params)
    accuracy = np.mean(preds == y)
    print(f"\nFinal training accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
