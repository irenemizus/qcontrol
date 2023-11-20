# Summary
This is a software package aimed for numerical calculation of different quantum system types. Namely, it models molecular dynamics for small quantum systems, as well as propagation dynamics for general unitary transformations, and their behaviour under a controlled laser field excitation. The package is developed in Python, designed to be modular and easily expendable, and also supports different types of quantum systems and different methods of quantum control. The implementation has a flexible object-oriented structure, includes a simple configuration DSL allowing feeding separate configurations to specific modules with changing input parameters, an automated plotting system for making output reports and graphs, and also an automated unit-testing mechanism.

Currenttly, the package supports:
- quantum systems like:
  1. Quantum harmonic oscillator
  2. Quantum Morse-like oscillator
  3. N-level trivial quantum system
- quantum control types:
  1. Propogation without control
  2. Intuitive control type
  3. Local control type (2 variants)
  4. Optimal control type by Krotov
  5. Optimal control type for general unitary transformation
- different variants for initial laser field analytical form

# Preparing environment
Here we put all the commands necessary to create an Anaconda/Python 3 virtual environment for the project.

Tested on macOS Monterey 12.6 with Anaconda 4.12.0

```
> conda create -n newcheb numpy==1.23.1
> conda activate newcheb
> pip install jsonpath2==0.4.5
```

Every time you need to use the application, activate the environment with:
```
conda activate newcheb
```

# Examples of the modeling results

## Unitary transformation for a 2-state quantum system under an external magnetic field
A testing modeling, which solves the problem from the following Wikipedia article: https://en.wikipedia.org/wiki/Rabi_cycle, subsection "In quantum computing" (the resonance case). The transition dynamics between the two states (including the dynamics of changing of Pauli matrix expectation values), as well as the time envelope of the external magnetic field is shown on the following plots:

## A diatomic-like quantum system under a controlled laser field excitation using an optimal Krotov-like type of control
A transition from the lower to the excited stable state on the 3-d iteration of controlling procedure (which results in the accuracy value of about $10^{-6}$), as well as the modifications of the external laser field envelope during it, are shown on the following plots: