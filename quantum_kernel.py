from __future__ import annotations

import numpy as np
import pennylane as qml

class ZZQuantumKernel:
    def __init__(self, n_qubits, reps=2):
        self.n_qubits = n_qubits
        self.reps = reps
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev)
        def _state(x):
            self._feature_map(x)
            return qml.state()

        self._state = _state

    def _feature_map(self, x):
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(2.0 * x[i], wires=i)
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                    qml.CNOT(wires=[i, j])

    def _encode_pennylane(self, X):
        return np.stack([np.asarray(self._state(x)) for x in X])

    def _encode_fast(self, X):
        X = np.asarray(X, float)
        n = self.n_qubits
        B = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
        Z = (2 * B - 1).astype(float)
        Zx = X @ Z.T
        Zu = (np.pi - X) @ Z.T
        phi = Zx - 0.5 * (Zu ** 2)
        return np.exp(1j * phi) / np.sqrt(2 ** n)

    def _encode(self, X):
        return self._encode_fast(X) if self.reps == 1 else self._encode_pennylane(X)

    def gram(self, X_a, X_b=None):
        S_a = self._encode(X_a)
        S_b = S_a if X_b is None else self._encode(X_b)
        overlaps = S_a @ S_b.conj().T
        return np.abs(overlaps) ** 2
