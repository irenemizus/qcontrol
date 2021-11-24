import math
import numpy

from phys_base import hart_to_cm, dalt_to_au


def pot(x, np, m, De, a, x0p, De_e, a_e, Du):
    """ Potential energy vectors
        INPUT
        x           vector of length np defining positions of grid points
        np          number of grid points (dummy variable)
        a           scaling factor of the ground state
        De          dissociation energy of the ground state
        m           reduced mass of the system
        x0p         partial shift value of the upper potential corresponding to the ground one
        De_e        dissociation energy of the excited state
        a_e         scaling factor of the excited state
        Du          energy shift between the minima of the potentials

        OUTPUT
        v       a list of real vectors of length np describing the potentials V_u(X) and V_l(X) """

    # harmonic frequency of the system on the lower PEC
    omega_0 = a * math.sqrt(2.0 * De / hart_to_cm / m / dalt_to_au) * hart_to_cm

    # anharmonicity factor of the system on the lower PEC
    xe = omega_0 / 4.0 / De
    print("Theoretical anharmonicity factor of the system on the lower PEC = ", xe)

    # theoretical ground energy value
    e_0 = omega_0 / 2.0 * (1 - xe / 2.0)
    print("Theoretical ground energy of the system (relative to the potential minimum) = ", e_0)

    v = []
    # Lower morse potential
    v_l = numpy.array([De * (1.0 - math.exp(-a * xi)) * (1.0 - math.exp(-a * xi)) for xi in x])
    v.append((0.0, v_l))

    # Upper morse potential
    v_u = numpy.array([De_e * (1.0 - math.exp(-a_e * (xi - x0p))) * (1.0 - math.exp(-a_e * (xi - x0p))) + Du for xi in x])
    v.append((Du, v_u))

    return v


def psi_init(x, np, x0, p0, m, De, a):
    """ Initial wave function generator
        INPUT
        x           vector of length np defining positions of grid points
        np          number of grid points
        x0          initial coordinate
        p0          initial momentum (dummy variable)
        m           reduced mass of the system
        a           scaling factor
        De          dissociation energy

        OUTPUT
        psi     a list of complex vectors of length np describing the wavefunctions psi_u(X) and psi_l(X) """

    # harmonic frequency of the system on the lower PEC
    omega_0 = a * math.sqrt(2.0 * De / hart_to_cm / m / dalt_to_au) * hart_to_cm

    # anharmonicity factor of the system on the lower PEC
    xe = omega_0 / 4.0 / De

    psi = []
    y = [math.exp(-a * (xi - x0)) / xe for xi in x]
    arg = 1.0 / xe - 1.0
    psi_l = numpy.array([math.sqrt(a / math.gamma(arg)) * math.exp(-yi / 2.0) * pow(yi, float(arg / 2.0)) for yi in y])
    psi.append(psi_l)

    psi_u = numpy.array([0.0] * np).astype(complex)
    psi.append(psi_u)

    return psi