# Summary
This is a software package aimed for numerical calculation of different quantum system types. Namely, it models small quantum systems dynamics, as well as propagation dynamics for general unitary transformations, and their behaviour under a controlled laser field excitation. The package is developed in Python, designed to be modular and easily expendable, and also supports different types of quantum systems and different methods of quantum control. The implementation has a flexible object-oriented structure, includes a simple configuration DSL allowing feeding separate configurations to specific modules with changing input parameters, an automated plotting system for making output reports and graphs, and also an automated unit-testing mechanism.

Currently, the package supports:
- quantum systems like:
  1. Quantum harmonic oscillator
  2. Quantum Morse-like oscillator
  3. N-level trivial quantum system
- quantum control types:
  1. Propagation without control
  2. Intuitive control type
  3. Local control type (2 variants)
  4. Optimal control type by Krotov
  5. Optimal control type for general unitary transformation
- different variants for initial laser field analytical form

# Preparing environment
Here we put all the commands necessary to create a Python 3 virtual environment for the project.

The project requires Intel-optimized NumPy, i.e. NumPy built against Intel oneMKL, not the regular PyPI NumPy built against OpenBLAS (the latter does not provide the precision the functional tests need). There are two ways to get such an environment: a plain pip virtual environment (option 1) or an Anaconda environment (option 2, where the conda defaults channel NumPy is MKL-backed on macOS out of the box).

## Option 1: pip virtual environment

The MKL-backed NumPy wheels for macOS are not on PyPI; they are distributed on the Intel Anaconda Cloud channel (https://pypi.anaconda.org/intel). Only Python 3.8 or 3.9 can be used, as that is all the channel provides macOS wheels for.

Tested on macOS Monterey 12.6 (x86_64) with Python 3.8 and NumPy 1.22.3 (MKL 2023.2.0)

```
> python3 -m venv .venv
> .venv/bin/pip install jsonpath2==0.4.5
> # Intel-optimized NumPy (MKL) from the Intel Anaconda Cloud channel.
> # --no-deps is required: the optional accelerator deps of this wheel
> # (mkl-fft, mkl-random, mkl-umath, mkl-service) have no macOS wheels for Python 3.8
> .venv/bin/pip install --no-deps -i https://pypi.anaconda.org/intel/simple numpy==1.22.3
> # MKL runtime libraries (also pulls in tbb and intel-openmp from the same channel)
> .venv/bin/pip install -i https://pypi.anaconda.org/intel/simple mkl==2023.2.0
> # The NumPy wheel carries build-machine rpaths to the MKL libraries, so add the
> # real location of the MKL runtime to NumPy's extension modules (one-time fix):
> RPATH="$(pwd)/.venv/lib" && \
>   find .venv/lib/python3.8/site-packages/numpy -name "*.so" -exec install_name_tool -add_rpath "$RPATH" {} \;
```

The "mkl-service package failed to import" warning at NumPy import time is harmless for this setup.

Every time you need to use the application, activate the environment with:
```
source .venv/bin/activate
```

## Option 2: Anaconda environment

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

## Checking that NumPy uses MKL

To make sure NumPy really uses MKL, run:
```
python -c "import numpy; numpy.show_config()"
```
in the activated environment and check that it prints `blas_opt_info: libraries = ['mkl_rt', 'pthread']`.

# Examples of the modeling results

## Unitary transformation for a 2-state quantum system under an external magnetic field
A testing modeling, which solves the problem from the following Wikipedia article: https://en.wikipedia.org/wiki/Rabi_cycle, subsection "In quantum computing" (the resonance case). The transition dynamics between the two states (including the dynamics of changing of Pauli matrix expectation values), as well as the time envelope of the external magnetic field is shown on the following plots:

![graph1](https://raw.githubusercontent.com/irenemizus/qcontrol/master/results_to_show/fig_abs_max_pi_pulse.svg)
![graph2](https://raw.githubusercontent.com/irenemizus/qcontrol/master/results_to_show/fig_ener_pi_pulse.svg)
![graph3](https://raw.githubusercontent.com/irenemizus/qcontrol/master/results_to_show/fig_smoms_pi_pulse.svg)
![graph4](https://raw.githubusercontent.com/irenemizus/qcontrol/master/results_to_show/fig_lf_en_pi_pulse.svg)

## A diatomic-like quantum system under a controlled laser field excitation using an optimal Krotov-like type of control
A transition from the lower to the excited stable state on the 3-d iteration of controlling procedure (which results in the accuracy value of about $10^{-6}$), as well as the modifications of the external laser field envelope during it, are shown on the following plots:

![graph5](https://raw.githubusercontent.com/irenemizus/qcontrol/master/results_to_show/fig_abs_max.svg)
![graph6](https://raw.githubusercontent.com/irenemizus/qcontrol/master/results_to_show/fig_real_max.svg)
![graph7](https://raw.githubusercontent.com/irenemizus/qcontrol/master/results_to_show/fig_ener.svg)
![graph8](https://raw.githubusercontent.com/irenemizus/qcontrol/master/results_to_show/fig_gr_iter_E.svg)
