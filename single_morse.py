import cmath
import math

from phys_base import hart_to_cm, dalt_to_au
De = 20000.0


def pot(x, m, omega_0):
    """ Potential energy vector
        INPUT
        x       vector of length np defining positions of grid points
        m       reduced mass of the system
        omega_0 harmonic frequency of the system
        OUTPUT
        v       real vector of length np describing the potential V(X) """

    # scaling factor
    a = math.sqrt(2.0 * m * omega_0 * omega_0 * dalt_to_au / 2.0 / De / hart_to_cm)
    # Single morse potential
    v = [De * (1.0 - math.exp(-a * xi)) * (1.0 - math.exp(-a * xi)) for xi in x]
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
    a = math.sqrt(m * omega_0 * omega_0 * dalt_to_au / 2.0 / De / hart_to_cm)

    # anharmonicity factor
    xe = omega_0 / 4.0 / De
    y = [math.exp(-a * xi) / xe for xi in x]
    arg = 1.0 / xe - 1.0
    psi = [math.sqrt(a / math.gamma(arg)) * math.exp(-yi / 2.0) * pow(yi, float(arg / 2.0)) for yi in y]
    return psi