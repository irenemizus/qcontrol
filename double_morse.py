import math
import numpy

from phys_base import hart_to_cm, dalt_to_au


def pot(x, np, m, De, a):
    """ Potential energy vectors
        INPUT
        x       vector of length np defining positions of grid points
        np      number of grid points
        a       scaling factor
        De      dissociation energy
        m       reduced mass of the system
        OUTPUT
        v       a list of real vectors of length np describing the potentials V_u(X) and V_l(X) """

    # harmonic frequency of the system on the lower PEC
    omega_0 = a * math.sqrt(2.0 * De / hart_to_cm / m / dalt_to_au) * hart_to_cm

    # anharmonicity factor of the system on the lower PEC
    xe = omega_0 / 4.0 / De
    print("Theoretical anharmonicity factor of the system on the lower PEC = ", xe)

    # theoretical ground energy value
    e_0 = omega_0 / 2.0 * (1 - xe / 2.0)
    print("Theoretical ground energy of the system on the lower PEC = ", e_0)

    v = []
    # Lower morse potential
    Dl = De
    v_l = numpy.array([Dl * (1.0 - math.exp(-a * xi)) * (1.0 - math.exp(-a * xi)) for xi in x])
    v.append((0.0, v_l))

    # Upper morse potential
    Du = De / 2.0
    v_u = numpy.array([Du * (1.0 - math.exp(-a * xi)) * (1.0 - math.exp(-a * xi)) + Dl for xi in x])
    v.append((Dl, v_u))

    return v


def psi_init(x, np, x0, p0, m, De, a):
    """ Initial wave function generator
        INPUT
        x       vector of length np defining positions of grid points
        np      number of grid points
        x0      initial coordinate
        p0      initial momentum (dummy variable)
        m       reduced mass of the system
        a       scaling factor
        De      dissociation energy

        OUTPUT
        psi     a list of complex vectors of length np describing the wavefunctions psi_u(X) and psi_l(X) """

    # harmonic frequency of the system on the lower PEC
    omega_0 = a * math.sqrt(2.0 * De / hart_to_cm / m / dalt_to_au) * hart_to_cm

    # anharmonicity factor of the system on the lower PEC
    xe = omega_0 / 4.0 / De
    print("Theoretical anharmonicity factor of the system on the lower PEC = ", xe)

    # theoretical ground energy value
    e_0 = omega_0 / 2.0 * (1 - xe / 2.0)
    print("Theoretical ground energy of the system on the lower PEC = ", e_0)

    psi = []
    y = [math.exp(-a * (xi - x0)) / xe for xi in x]
    arg = 1.0 / xe - 1.0
    psi_l = [math.sqrt(a / math.gamma(arg)) * math.exp(-yi / 2.0) * pow(yi, float(arg / 2.0)) for yi in y]
    psi_l_np = numpy.array(psi_l)
    psi.append(psi_l_np)

    psi_u = [0.0] * np
    psi_u_np = numpy.array(psi_u)
    psi.append(psi_u_np)

    return psi