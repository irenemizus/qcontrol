import cmath
import math


def pot(x):
    """ Potential energy vector
        INPUT
        x  vector of length np defining positions of grid points
        OUTPUT
        v real vector of length np describing the potential V(X) """

    # Single harmonic potential
    v = [xi * xi / 2.0 for xi in x]
    return v


def psi_init(x, x0, p0):
    """ Initial wave function generator
        INPUT
        x    vector of length np defining positions of grid points
        x0   initial coordinate
        p0   initial momentum
        OUTPUT
        psi  complex vector of length np describing the wavefunction """

    psi = [cmath.exp(-(xi - x0) * (xi - x0) / 2.0 + 1j * p0 * xi) / pow(math.pi, 0.25) for xi in x]
    return psi