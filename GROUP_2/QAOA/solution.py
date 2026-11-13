import json
import pennylane as qp
import pennylane.numpy as np

# Write any helper functions you may need here

N_LAYERS = 4      # depth of QAOA required by the challenge
STEPS = 231
STEPSIZE = 0.005
INIT = 0.5        # every gamma and alpha starts at 0.5


def build_hamiltonians(edges):
    """H_cost = 3/4 sum_{(i,j) in E} (Z_i Z_j + Z_i + Z_j) - sum_{i in V} Z_i,  H_mixer = sum_{i in V} X_i."""
    vertices = sorted({v for edge in edges for v in edge})

    coeffs, ops = [], []
    for i, j in edges:
        coeffs += [0.75, 0.75, 0.75]
        ops += [qp.Z(i) @ qp.Z(j), qp.Z(i), qp.Z(j)]
    for v in vertices:
        coeffs.append(-1.0)
        ops.append(qp.Z(v))

    cost_h = qp.Hamiltonian(coeffs, ops).simplify()
    mixer_h = qp.Hamiltonian([1.0] * len(vertices), [qp.X(v) for v in vertices])
    return vertices, cost_h, mixer_h


def approximation_ratio(edges):

    """This function calculates the approximation ratio of QAOA for
    the minimum vertex cover problem

    Args:
    - Edges (list(list(int))): A list of ordered pairs of integers, representing 
    the edges of the graph for which we need to solve the minimum vertex cover problem.

    Returns:
    - np.tensor(float): The approximation ratio of QAOA for the minimum vertex cover problem for
    the input graph.
    """

    vertices, cost_h, mixer_h = build_hamiltonians(edges)
    dev = qp.device("default.qubit", wires=vertices)

    # Write your QAOA circuit and optimization routine to find the minimum expval

    def qaoa_layer(gamma, alpha):
        qp.qaoa.cost_layer(gamma, cost_h)     # exp(-i gamma H_cost)
        qp.qaoa.mixer_layer(alpha, mixer_h)   # exp(-i alpha H_mixer)

    @qp.qnode(dev)
    def cost_function(params):
        for w in vertices:
            qp.Hadamard(wires=w)              # uniform superposition |+>^n
        qp.layer(qaoa_layer, N_LAYERS, params[0], params[1])
        return qp.expval(cost_h)

    params = np.array([[INIT] * N_LAYERS, [INIT] * N_LAYERS], requires_grad=True)
    opt = qp.NesterovMomentumOptimizer(stepsize=STEPSIZE)
    for _ in range(STEPS):
        params = opt.step(cost_function, params)
    qaoa_min = cost_function(params)

    # Return the approximation ratio

    # H_cost is diagonal in the computational basis, so its exact ground energy
    # is just the smallest diagonal entry.
    true_min = np.min(np.real(np.diag(qp.matrix(cost_h, wire_order=vertices))))
    return qaoa_min / true_min


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = approximation_ratio(ins)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=5e-2
    ), "Your code does not yield the right approximation ratio"


# These are the public test cases
test_cases = [
    ('[[0, 1], [1, 2], [0, 2], [2, 3]]', '0.58160924'),
    ('[[0, 1], [1, 2], [2, 3], [3, 0]]', '0.94746734')
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
