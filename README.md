This repository provides tools for constructing Quadratic Unconstrained Binary Optimization (QUBO) instances from a simple 2-class Classification Binary Neural Network (BNN) robustness verification problems. The framework converts BNN inference constraints and adversarial perturbation search into a combinatorial optimization formulation compatible with Ising machines, quantum annealers, and quantum-inspired optimization platforms.

Verification of binary neural network robustness can be formulated as a combinatorial search for an adversarial perturbation that induces misclassification. In this repository, the verification problem is reformulated as a QUBO instance, enabling the use of annealing-based optimization hardware to explore the resulting nonconvex energy landscape.

The generated QUBO captures:
  1. BNN inference constraints
  2. Perturbation budget constraints
  3. Adversarial misclassification conditions
  4. Auxiliary variables required for quadratization

The resulting QUBO can be directly used by Ising-based solvers and annealing hardware to search for adversarial perturbations that demonstrate the non-robustness of the network.
