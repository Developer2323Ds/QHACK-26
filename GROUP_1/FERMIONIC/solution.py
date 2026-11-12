"""VQE ground-state energy for H_F = sum_j c_j a_j a_j^dagger (3 fermionic modes).

Jordan-Wigner gives H_Q = (c0+c1+c2)/2 * I + sum_j (c_j/2) Z_j, which is diagonal,
so the exact answer is E_min = sum_j min(c_j, 0). VQE here is a demonstration;
the closed form is used as a ground-truth check.

Requires: pip install pennylane
"""
import pennylane as qml
import pennylane.numpy as np

N_MODES = 3      # fermionic modes / wires touched by H
N_WIRES = 6      # register size (wires 3..5 idle -> 8-fold degenerate ground space)
TOL = 1e-2       # audit Section B.4


def build_fermionic_hamiltonian(coeffs):
    """H_F = sum_j c_j a_j a_j^dagger as a FermiSentence (keeps the identity term)."""
    H_F = qml.fermi.FermiSentence({})
    for j, c in enumerate(coeffs):
        # Position 0 = leftmost operator (annihilation), position 1 = creation.
        word = qml.fermi.FermiWord({(0, j): "-", (1, j): "+"})
        H_F += float(c) * word
    return H_F


def vqe_ground_energy(coeffs, steps: int = 150, lr: float = 0.1, verbose: bool = False):
    coeffs = [float(np.real(c)) for c in coeffs]  # a_j a_j^dag is Hermitian

    H_F = build_fermionic_hamiltonian(coeffs)
    H_Q = qml.jordan_wigner(H_F)
    # jordan_wigner returns complex coefficients (e.g. 2.5+0j), which makes the
    # cost complex-valued and breaks autograd. Cast to real (B.5) and rebuild
    # with qml.dot (B.6). The identity term is kept (B.4).
    h_coeffs, h_ops = H_Q.terms()
    H_Q = qml.dot([float(np.real(c)) for c in h_coeffs], h_ops)
    if verbose:
        print("  H_Q =", H_Q)

    dev = qml.device("default.qubit", wires=N_WIRES)

    @qml.qnode(dev)
    def cost(theta):
        for j in range(N_MODES):          # ansatz on wires 0..2 only
            qml.RY(theta[j], wires=j)
        return qml.expval(H_Q)

    # pi/2 avoids the zero-gradient fixed points at 0 and pi.
    theta = np.array([np.pi / 2] * N_MODES, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)
    for _ in range(steps):
        theta = opt.step(cost, theta)

    final_energy = float(cost(theta))
    exact_energy = sum(min(c, 0.0) for c in coeffs)
    assert abs(final_energy - exact_energy) < TOL, (
        f"Optimization failed: expected {exact_energy}, obtained {final_energy}"
    )
    return final_energy, np.array(theta)


if __name__ == "__main__":
    print(f"PennyLane {qml.__version__}")
    for coeffs, expected in [([-2, 2, 5], -2.0), ([7, 6, -6], -6.0)]:
        print(f"c = {coeffs}")
        energy, theta = vqe_ground_energy(coeffs, verbose=True)
        print(f"  theta  = {np.round(theta, 4)}")
        print(f"  E_VQE  = {energy:.8f}   expected {expected}   "
              f"|error| = {abs(energy - expected):.2e}")
    print("All checks passed.")
