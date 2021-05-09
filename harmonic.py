import cmath
import math

from phys_base import dalt_to_au, hart_to_cm


def pot(x, m, De, a):
    """ Potential energy vector
        INPUT
        x       vector of length np defining positions of grid points
        a       scaling factor
        De      dissociation energy (dummy variable)
        m       reduced mass of the system
        OUTPUT
        v       real vector of length np describing the dimensionless potential V(X) """

    # stiffness coefficient for dimensional case
    k_s = hart_to_cm / m / dalt_to_au / pow(a, 4.0)
#    k_s = m * omega_0 * omega_0 * dalt_to_au / hart_to_cm
#    k_s = 1.0

    # harmonic frequency  for dimensional case
    # omega_0 = hart_to_cm / dalt_to_au / m / a / a
    omega_0 = k_s * a * a
    # theoretical ground energy value
    e_0 = omega_0 / 2.0
    print("Theoretical ground energy = ", e_0)

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
        a       scaling factor
        De      dissociation energy (dummy variable)
        OUTPUT
        psi     complex vector of length np describing the dimensionless wavefunction """

    # scaling factor for dimensional case
#    a = math.sqrt(hart_to_cm / m / omega_0 / dalt_to_au)
#    a = 0.5
    psi = [cmath.exp(-(xi - x0) * (xi - x0) / 2.0 / a / a + 1j * p0 * xi) / pow(math.pi, 0.25) / math.sqrt(a) for xi in x]
    return psi