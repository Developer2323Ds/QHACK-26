[README.md](https://github.com/user-attachments/files/34764276/README.2.md)
# QHack '26: Quantum Challenge Solutions

Our team's submissions for **QHack 2026**, a quantum computing hackathon built around hands-on problem solving with quantum algorithms, circuits, and simulation. This repository documents our work across all three challenge stages.

## Overview

The challenge was split into three groups (stages), each building on the last. For every group, this repo contains the original problem statement alongside our solution, so each stage can be read and reproduced on its own.

## Repository Structure

```
.
├── group-1/
│   ├── problem-statements/
│   └── solutions/
├── group-2/
│   ├── problem-statements/
│   └── solutions/
├── group-3/
│   ├── problem-statements/
│   └── solutions/
├── requirements.txt
└── README.md
```

Each group folder pairs the problem statement with our code, so you can read the task first and then follow our approach.

## Approach

Our general workflow for every problem:

1. **Understand the problem.** Restate the task and identify the quantum concept involved.
2. **Prototype on a simulator.** Build and test circuits locally before anything else.
3. **Optimize.** Reduce circuit depth and gate count where possible.
4. **Validate.** Check results against the expected outputs and edge cases.

## Getting Started

**Prerequisites**
- Python 3.9+
- A quantum SDK such as PennyLane (see `requirements.txt` for exact dependencies)

**Setup**

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

**Running a solution**

```bash
python group-1/solutions/<solution_file>.py
```

## Tech Stack

- Python
- PennyLane / quantum simulation tooling
- NumPy and SciPy for numerical work

## Acknowledgements

Thanks to the QHack organisers, mentors, and the wider quantum community for a challenging and rewarding event.

## Connect

Find us on LinkedIn: 
https://www.linkedin.com/in/devank-kumar-barua-2bab76388/
https://www.linkedin.com/in/prakhar-dubey-16la10vu08/
https://www.linkedin.com/in/sayak-palit-7ba240378/
https://www.linkedin.com/in/soumya-shankar-saharay-82b861416/

## License

Released under the MIT License.
