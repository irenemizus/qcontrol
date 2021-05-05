import cmath
import math


def pot(x, m, De, a):
    """ Potential energy vector
        INPUT
        x       vector of length np defining positions of grid points
        a       scaling factor (dummy variable)
        De      dissociation energy (dummy variable)
        m       reduced mass of the system
        OUTPUT
        v       real vector of length np describing the dimensionless potential V(X) """

    # stiffness coefficient for dimensional case
#    k_s = 2.0 * m * omega_0 * omega_0 * dalt_to_au / hart_to_cm
    k_s = 1.0
    # Single dimensionless harmonic potential
    v = [k_s * xi * xi / 2.0 for xi in x]
    return v


def psi_init(x, x0, p0, m, De, a):
    """ Initial wave function generator
        INPUT
        x       vector of length np defining positions of grid points
        x0      initial coordinate
        p0      initial momentum
        m       reduced mass of the system
        a       scaling factor (dummy variable)
        De      dissociation energy (dummy variable)
        OUTPUT
        psi     complex vector of length np describing the dimensionless wavefunction """

    # scaling factor for dimensional case
#    a = math.sqrt(hart_to_cm / m / omega_0 / dalt_to_au)
    a = 0.1
    psi = [cmath.exp(-(xi - x0) * (xi - x0) / 2.0 / a / a + 1j * p0 * xi) / pow(math.pi, 0.25) / math.sqrt(a) for xi in x]
    return psi