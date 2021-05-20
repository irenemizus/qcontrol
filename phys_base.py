import cmath
import numpy
import copy

from math_base import points

hart_to_cm = 219474.6313708 # 1 / cm / hartree
cm_to_erg = 1.98644568e-16 # erg * cm
dalt_to_au = 1822.888486 # a.u. / D
Red_Planck_h = 1.054572e-27 # erg * s


def diff(psi, akx2, np):
    """ Calculates kinetic energy mapping carried out in momentum space
        INPUT
        psi   complex vector of length np
        akx2  complex vector of length np, = k^2/2m
        np    number of grid points
        OUTPUT
        phi   complex vector of length np describing the mapping
              of kinetic energy phi = P^2/2m psi """

    psi_freq = numpy.fft.fft(numpy.array(psi))

    phi_freq = []
    for i in range(np):
        phi_freq.append(psi_freq[i] * akx2[i])

    phi = numpy.fft.ifft(numpy.array(phi_freq))

    return phi


def hamil(psi, v, akx2, np, E):
    """ Calculates Hamiltonian mapping of vector psi
        INPUT
        psi   list of complex vectors of length np
        v     list of potential energy real vectors of length np
        akx2  complex kinetic energy vector of length np, = k^2/2m
        np    number of grid points
        E     a complex value of external laser field
        OUTPUT
        phi = H psi list of complex vectors of length np """

    phi = []
    # diagonal terms
    # kinetic energy mapping for the lower state
    phi_dl = diff(psi[0], akx2, np)
    # potential energy mapping and accumulation phi_l = H psi_l
    for i in range(np):
        phi_dl[i] += v[0][1][i] * psi[0][i]

    # kinetic energy mapping for the upper state
    phi_du = diff(psi[1], akx2, np)
    # potential energy mapping and accumulation phi_u = H psi_u
    for i in range(np):
        phi_du[i] += v[1][1][i] * psi[1][i]

    # adding non-diagonal terms
    phi_l = []
    for i in range(np):
        phi_l.append(phi_dl[i] - E * psi[1][i])
    phi.append(phi_l)

    phi_u = []
    for i in range(np):
        phi_u.append(phi_du[i] - E.conjugate() * psi[0][i])
    phi.append(phi_u)

    return phi


def residum(psi, v, akx2, xp, np, edges, E):
    """ Scaled and normalized mapping phi = ( O - xp I ) phi
        INPUT
        psi   list of complex vectors of length np
        v     list of potential energy vectors of length np
        xp    sampling interpolation point
        np    number of grid points (must be a power of 2)
        edges upper and lower limits of energy spectra for the lower and upper states
              edges = [emax_l, emin_l, emax_u, emin_u]
        E     a complex value of external laser field
        OUTPUT
        phi  list of complex vectors of length np
             the operator is normalized from -2 to 2 resulting in:
             phi = 4.O / (emax - emin) * H phi - 2.0 (emax + emin) / (emax - emin) * I phi - xp I phi """

    hpsi = hamil(psi, v, akx2, np, E)

    phi = []
    # changing the range from -2 to 2
    # for the lower state
    phi_l = []
    for i in range(np):
        hpsi[0][i] = 2.0 * (2.0 * hpsi[0][i] / (edges[0] - edges[1]) - (edges[0] + edges[1]) * psi[0][i] / (edges[0] - edges[1]))
        phi_l.append(hpsi[0][i] - xp * psi[0][i])
    phi.append(phi_l)

    # for the upper state
    phi_u = []
    for i in range(np):
        hpsi[1][i] = 2.0 * (2.0 * hpsi[1][i] / (edges[2] - edges[3]) - (edges[2] + edges[3]) * psi[1][i] / (edges[2] - edges[3]))
        phi_u.append(hpsi[1][i] - xp * psi[1][i])
    phi.append(phi_u)

    return phi


def func(z, t):
    """ The function to be interpolated
        INPUT
        z     real coordinate parameter
        t     real time parameter (dimensionless)
        OUTPUT
        func  value of the function (complex)
        func = f (z, t) """

    return cmath.exp(-1j * z * t)


def prop(psi, t_sc_l, t_sc_u, nch, np, v, akx2, edges, E):
    """ Propagation subroutine using Newton interpolation
        P(O) psi = dv(1) psi + dv2 (O - x1 I) psi + dv3 (O - x2)(O - x1 I) psi + ...
        INPUT
        psi  list of complex vectors of length np describing wavefunctions
             at the beginning of interval
        t_sc_l time interval for the lower state (normalized by the reduced Planck constant)
        t_sc_u time interval for the upper state (normalized by the reduced Planck constant)
        nch  order of interpolation polynomial (must be a power of 2 if
             reorder is necessary)
        np   number of grid points (must be a power of 2)
        v    list of potential energy vectors of length np
        akx2 kinetic energy vector of length np
        edges upper and lower limits of energy spectra for the lower and upper states
              edges = [emax_l, emin_l, emax_u, emin_u]
        E     a complex value of external laser field

        OUTPUT
        psi  list of complex vectors of length np
             describing the propagated wavefunction
             phi(t) = exp(-iHt) psi(0) """

    # interpolation points and divided difference coefficients
    xp, dv_l = points(nch, t_sc_l, func)
    xp, dv_u = points(nch, t_sc_u, func)

    # auxiliary vector used for recurrence
    phi = []
    phi = copy.deepcopy(psi)

    # accumulating first term
    psi[0] = [el * dv_l[0] for el in psi[0]]
    psi[1] = [el * dv_u[0] for el in psi[1]]

    # recurrence loop
    for j in range(nch - 1):
        # mapping by scaled operator of phi
        phi = residum(phi, v, akx2, xp[j], np, edges, E)

        # accumulation of Newtonian's interpolation
        for i in range(np):
            psi[0][i] += dv_l[j + 1] * phi[0][i]
            psi[1][i] += dv_u[j + 1] * phi[1][i]

    psi[0] = [el * cmath.exp(-1j * 2.0 * t_sc_l * (edges[0] + edges[1]) / (edges[0] - edges[1])) for el in psi[0]]
    psi[1] = [el * cmath.exp(-1j * 2.0 * t_sc_u * (edges[2] + edges[3]) / (edges[2] - edges[3])) for el in psi[1]]

    return psi