import json
import pennylane as qp
import pennylane.numpy as np

# DO NOT CHANGE THESE VARIABLES LATER IN THE CODE

phi = np.array([0.03, 0.04, 0.07, 0.09])
source = np.array([1., 2., 1., 0.5])
dt = 0.01
h = 0.2


# 32 QSVT angles (degree-31 odd polynomial P(x) ~ C/x on [1/10.5, 1]).
# Generated with qp.poly_to_angles(coeffs, "QSVT", angle_solver="iterative") from a
# relative-error least-squares fit in the odd Chebyshev basis. For any nu, k >= 0 the
# condition number of A is below 9.47, and A is normalised by 1.05*||A||_2, so its
# singular values always lie in [1/10.5, 1/1.05]: one fixed set of angles covers every input.
phi_qsvt = np.array([
    -1.5667257048054983, 1.5626588403428359, 1.5833030875093486, 1.5535300416099935,
    1.5932333377613235, 1.5427908835204032, 1.6047286313854559, 1.5306423155746556,
    1.6173778668152559, 1.517697682307991, 1.6303571306059312, 1.505000117046386,
    1.642406457458178, 1.4940027946993757, 1.6519329516884507, 1.4863503194583911,
    1.4863503194583911, 1.6519329516884507, 1.4940027946993757, 1.642406457458178,
    1.505000117046386, 1.6303571306059312, 1.517697682307991, 1.6173778668152559,
    1.5306423155746556, 1.6047286313854559, 1.5427908835204032, 1.5932333377613235,
    1.5535300416099935, 1.5833030875093486, 1.5626588403428359, 0.0040706219893982265
])
qsvt_angles = phi_qsvt  # same angles under the name used in the challenge text

C = 0.08571428571428572  # P(x) ~ C/x on [1/10.5, 1], with C = 0.9/10.5

b = phi + dt * source  # unnormalized b = phi_old + dt * s


# Feel free to write any helper functions here

D2 = (1.0 / h**2) * np.array([[-2., 1., 0., 0.],
                              [1., -2., 1., 0.],
                              [0., 1., -2., 1.],
                              [0., 0., 1., -2.]])


def build_matrix_A(nu, k):
    """Backward Euler matrix A = (1 + k dt) I - nu dt D2 (symmetric positive-definite)."""
    return (1.0 + k * dt) * np.eye(4) - nu * dt * D2


# wire 0: LCU ancilla | wire 1: block-encoding ancilla | wires 2, 3: the 4-dim state
device = qp.device("default.qubit", wires=4)


@qp.qnode(device)
def qsvt_diffusion(nu, k):

    """
    Implements QSVT to find the concentration phi after one time step

    Args:
    - nu (float): The diffusion coefficient
    - k (float): The reaction rate

    Returns:
    - (np.array): The concentation vector on the next time step.
    """


    # Write your QSVT code for matrix inversion here
    A = build_matrix_A(nu, k)
    A_norm = A / (1.05 * np.linalg.norm(A, ord=2))  # singular values in [1/10.5, 1/1.05]

    qp.StatePrep(b / np.linalg.norm(b), wires=[2, 3])

    U_A = qp.BlockEncode(A_norm, wires=[1, 2, 3])
    projectors_pos = [qp.PCPhase(angle, dim=4, wires=[1, 2, 3]) for angle in phi_qsvt]
    projectors_neg = [qp.PCPhase(-angle, dim=4, wires=[1, 2, 3]) for angle in phi_qsvt]

    # LCU: 1/2 (QSVT(phi) + QSVT(-phi)) keeps only the real polynomial P(A_norm)
    qp.Hadamard(wires=0)
    qp.ctrl(qp.QSVT, control=0, control_values=0)(U_A, projectors_pos)
    qp.ctrl(qp.QSVT, control=0, control_values=1)(U_A, projectors_neg)
    qp.Hadamard(wires=0)

    # Return the state
    return qp.state()


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    out_state = qsvt_diffusion(*ins)[:4]
    rescaled_out= out_state * np.linalg.norm(b) / C
    output = rescaled_out/ np.linalg.norm(rescaled_out)

    return str(np.real(output).tolist())

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    fidelity = np.dot(solution_output, expected_output)
    assert fidelity >=0.99, "Your circuit doesn't look quite right "


# These are the public test cases
test_cases = [
    ('[0.2,0.5]', '[0.27959746, 0.42869397, 0.56892447, 0.64371709]'),
    ('[1.0, 1.2]', '[0.28550356, 0.4595721 , 0.59350887, 0.59584262]')
]

# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
