import cmath
import numpy

from math_base import points

hart_to_cm = 219474.6313708 # 1 / cm / hartree
cm_to_erg = 1.98644568e-16 # erg * cm
dalt_to_au = 1822.888486 # a.u. / D
Red_Planck_h = 1.054572e-27 # erg * s

def test_fft():
    test = []
    for i in range(128):
        test.append(i / 128 + 0j)

    spectr = numpy.fft.fft(numpy.array(test))
    test2 = numpy.fft.ifft(spectr)
    pass

test_fft()


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


def hamil(psi, v, akx2, np):
    """ Calculates Hamiltonian mapping of vector psi
        INPUT
        psi   complex vector of length np
        v     potential energy real vector of length np
        akx2  complex kinetic energy vector of length np, = k^2/2m
        np    number of grid points
        OUTPUT
        phi = H psi complex vector of length np """

    # kinetic energy mapping
    phi = diff(psi, akx2, np)

#    for i in range(np):
#        print(i, v[i] * psi[i].real)

    # potential energy mapping and accumulation phi = H psi
    for i in range(np):
        phi[i] += v[i] * psi[i]

    return phi


def residum(psi, v, akx2, xp, np, emax):
    """ Scaled and normalized mapping phi = ( O - xp I ) phi
        INPUT
        psi  complex vector of length np
        v    potential energy of length np
        xp   sampling interpolation point
        np   number of grid points (must be a power of 2)
        emax range of energy spectrum
        OUTPUT
        phi  complex vector of length np
             the operator is normalized from -2 to 2 resulting in:
             phi = (4.O / emax - 2I) phi - xp I phi	(emin = 0) """

    hpsi = hamil(psi, v, akx2, np)

    # changing the range from -2 to 2
    phi = []
    for i in range(np):
        hpsi[i] = 2.0 * (2.0 * hpsi[i] / emax - psi[i])
        phi.append(hpsi[i] - xp * psi[i])

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


def prop(psi, t_sc, nch, np, v, akx2, emax):
    """ Propagation subroutine using Newton interpolation
        P(O) psi = dv(1) psi + dv2 (O - x1 I) psi + dv3 (O - x2)(O - x1 I) psi + ...
        INPUT
        psi  complex vector of length np describing wavefunction
             at the beginning of interval
        t_sc time interval (normalized by the reduced Planck constant)
        nch  order of interpolation polynomial (must be a power of 2 if
             reorder is necessary)
        np   number of grid points (must be a power of 2)
        v    potential energy vector of length np
        akx2 kinetic energy vector of length np
        OUTPUT
        psi  complex vector of length np
             describing the propagated wavefunction
             phi(t) = exp(-iHt) psi(0) """

    # interpolation points and divided difference coefficients
    xp, dv = points(nch, t_sc, func)
    dvr = [abs(el) for el in dv]
    # auxiliary vector used for recurrence
    phi = []
    phi[:] = psi[:]

    # accumulating first term
    psi = [el * dv[0] for el in psi]

    # recurrence loop
    for j in range(nch - 1):
        # mapping by scaled operator of phi
        phi = residum(phi, v, akx2, xp[j], np, emax)

        # accumulation of Newtonian's interpolation
        for i in range(np):
            psi[i] += dv[j + 1] * phi[i]

    psi = [el * cmath.exp(-1j * 2.0 * t_sc) for el in psi]

    return psi