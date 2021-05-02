import cmath
import math

from phys_base import hart_to_cm, dalt_to_au


def pot(x, m, omega_0):
    """ Potential energy vector
        INPUT
        x       vector of length np defining positions of grid points
        m       reduced mass of the system
        omega_0 harmonic frequency of the system
        OUTPUT
        v       real vector of length np describing the potential V(X) """

    # stiffness coefficient
    k_s = 2 * m * omega_0 * omega_0 * dalt_to_au / hart_to_cm
    # Single harmonic potential
    v = [k_s * xi * xi / 2.0 for xi in x]
    return v


def psi_init(x, x0, p0, m, omega_0):
    """ Initial wave function generator
        INPUT
        x       vector of length np defining positions of grid points
        x0      initial coordinate
        p0      initial momentum
        m       reduced mass of the system
        omega_0 harmonic frequency of the system

        OUTPUT
        psi     complex vector of length np describing the wavefunction """

    # scaling factor
    a = math.sqrt(hart_to_cm / m / omega_0 / dalt_to_au)
    psi = [cmath.exp(-(xi - x0) * (xi - x0) / 2.0 / a / a + 1j * p0 * xi) / pow(math.pi, 0.25) / math.sqrt(a) for xi in x]
    return psi