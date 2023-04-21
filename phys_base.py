import cmath

import numpy
import copy

from numpy.typing import NDArray

import math_base
from psi_basis import Psi

hart_to_cm = numpy.float64(219474.6313708) # 1 / cm / hartree
cm_to_erg = numpy.float64(1.98644568e-16) # erg * cm
dalt_to_au = numpy.float64(1822.888486) # a.u. / D
Red_Planck_h = numpy.float64(1.054572e-27) # erg * s
Hz_to_cm = numpy.float64(3.33564095e-11) # s / cm


def diff_cpu(psi, akx2, np):
    """ Calculates kinetic energy mapping carried out in momentum space
        INPUT
        psi   complex vector of length np
        akx2  complex vector of length np, = k^2/2m
        np    number of grid points

        OUTPUT
        phi   complex vector of length np describing the mapping
              of kinetic energy phi = P^2/2m psi """

    assert psi.size == np
    assert akx2.size == np

    psi_freq = numpy.fft.fft(psi)
    phi_freq = numpy.multiply(psi_freq, akx2)
    phi = numpy.fft.ifft(phi_freq)

    return phi


def hamil_cpu(psi, v, akx2, np, ntriv):
    """ Calculates the simplest one-dimensional Hamiltonian mapping of vector psi
        INPUT
        psi     list of complex vectors of length np
        v       list of potential energy real vectors of length np
        akx2    complex kinetic energy vector of length np, = k^2/2m
        np      number of grid points
        ntriv   constant parameter; 1 -- an ordinary non-trivial diatomic-like system
                                    0 -- a trivial 2-level system
                                   -1 -- a trivial n-level system with angular momentum Hamiltonian and
                                         with external laser field augmented inside a Jz term
                                   -2 -- a trivial n-level system with angular momentum Hamiltonian and
                                         with external laser field augmented inside a Jx term

        OUTPUT
        phi = H psi list of complex vectors of length np """

    assert psi.size == np
    assert v.size == np
    assert akx2.size == np

    # kinetic energy mapping
    if ntriv == 1:
        phi = diff_cpu(psi, akx2, np)

        # potential energy mapping and accumulation phi_l = H psi_l
        vpsi = numpy.multiply(v, psi)
        numpy.add(phi, vpsi, out=phi)
    else:
        phi = numpy.multiply(v, psi)

    return phi


def residum_cpu(psi: Psi, hamil2D, xp, np, emin, emax, E, eL, E_full, orig):
    """ Scaled and normalized mapping phi = ( O - xp I ) phi
        INPUT
        psi         PsiBasis element (current wavefunction)
        hamil2D     hamil2D object (hpsi = H psi)
        xp          sampling interpolation point
        np          number of grid points (must be a power of 2)
        emax, emin  upper and lower limits of energy spectra
        E           a real value of external laser field
        eL          a laser field energy shift = h * nu_L / 2.0
        E_full      a complex value of external laser field
        orig        a boolean parameter that depends
                    if an original form of the Hamiltonian should be used (orig = True) or
                    the shifted real version (orig = False -- by default)

        OUTPUT
        phi  PsiBasis element
             the operator is normalized from -2 to 2 resulting in:
             phi = 4.O / (emax - emin) * H psi - 2.0 (emax + emin) / (emax - emin) * I psi - xp I psi """

    for i in range(len(psi.f)):
        assert psi.f[i].size == np

    phi: Psi = Psi(lvls=len(psi.f))

    hpsi = hamil2D(orig, psi=psi, E=E, eL=eL, E_full=E_full)

    # changing the range from -2 to 2
    coef1 = 4.0 / (emax - emin)
    coef2 = 2.0 * (emax + emin) / (emax - emin)
    for n in range(len(psi.f)):
        hpsi.f[n] *= coef1
        tmp = psi.f[n] * coef2
        numpy.subtract(hpsi.f[n], tmp, out=hpsi.f[n])

        phi_n = psi.f[n] * (-xp)
        numpy.add(phi_n, hpsi.f[n], out=phi_n)

        phi.f[n] = phi_n

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


def prop_cpu(psi: Psi, hamil2D, t_sc, nch, np, emin, emax, E, eL, E_full, orig):
    """ Propagation subroutine using Newton interpolation
        P(O) psi = dv0 psi + dv1 (O - x0 I) psi + dv2 (O - x1 I)(O - x0 I) psi + ...
        INPUT
        psi         PsiBasis element (current wavefunction at the beginning of interval)
        hamil2D     hamil2D object (hpsi = H psi)
        t_sc        time interval (normalized by the reduced Planck constant)
        nch         order of interpolation polynomial (must be a power of 2 if
                    reorder is necessary)
        np          number of grid points (must be a power of 2)
        emax, emin  upper and lower limits of energy spectra
        E           a real value of external laser field
        eL          a laser field energy shift = h * nu_L / 2.0
        E_full      a complex value of external laser field
        orig        a boolean parameter that depends
                    if an original form of the Hamiltonian should be used (orig = True) or
                    the shifted real version (orig = False -- by default)

        OUTPUT
        psi  PsiBasis element describing the propagated wavefunction
             phi(t) = exp(-iHt) psi(0) """

    for i in range(len(psi.f)):
        assert psi.f[i].size == np

    # interpolation points and divided difference coefficients
    xp, dv = math_base.points(nch, t_sc, func)

    # auxiliary vector used for recurrence
    phi = copy.deepcopy(psi)

    # accumulating first term
    for n in range(len(psi.f)):
        psi.f[n] *= dv[0]

    # recurrence loop
    for j in range(nch - 1):
        # mapping by scaled operator of phi
        phi = residum_cpu(psi=phi, hamil2D=hamil2D, xp=xp[j], np=np, emin=emin, emax=emax, E=E, eL=eL, E_full=E_full, orig=orig)

        # accumulation of Newtonian's interpolation
        for n in range(len(psi.f)):
            phidv = phi.f[n] * dv[j + 1]
            numpy.add(psi.f[n], phidv, out=psi.f[n])

    coef = cmath.exp(-1j * 2.0 * t_sc * (emax + emin) / (emax - emin))
    for n in range(len(psi.f)):
        psi.f[n] *= coef

    return psi


class ExpectationValues:
    def __init__(self, x, x2, p, p2):
        self.x = x
        self.x2 = x2
        self.p = p
        self.p2 = p2


class SigmaExpectationValues:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def exp_vals_calc(psi: Psi, x, akx2, dx, np, m, ntriv):
    """ Calculation of expectation values <x>, <x^2>, <p>, <p^2>
        INPUT
        psi     list of complex vectors of length np describing wavefunctions
        x       vector of length np defining positions of grid points
        akx2    kinetic energy vector of length np
        dx      coordinate step of the problem
        np      number of grid points (must be a power of 2)
        m       reduced mass of the system
        ntriv       constant parameter; 1 -- an ordinary non-trivial diatomic-like system
                                        0 -- a trivial 2-level system
                                       -1 -- a trivial n-level system with angular momentum Hamiltonian and
                                             with external laser field augmented inside a Jz term
                                       -2 -- a trivial n-level system with angular momentum Hamiltonian and
                                             with external laser field augmented inside a Jx term

        OUTPUT
        moms  list of complex vectors of length np """

    nlevs = len(psi.f)
    momx: NDArray[numpy.complex128] = numpy.zeros(nlevs, dtype=numpy.complex128)
    momx2: NDArray[numpy.complex128] = numpy.zeros(nlevs, dtype=numpy.complex128)
    momp: NDArray[numpy.complex128] = numpy.zeros(nlevs, dtype=numpy.complex128)
    momp2: NDArray[numpy.complex128] = numpy.zeros(nlevs, dtype=numpy.complex128)

    if ntriv == 1:
        # for x
        for n in range(nlevs):
            momx[n] = math_base.cprod2(psi.f[n], x, dx, np)

        # for x^2
        x2 = numpy.multiply(x, x)
        for n in range(nlevs):
            momx2[n] = math_base.cprod2(psi.f[n], x2, dx, np)

        # for p^2
        for n in range(nlevs):
            phi_kin = diff_cpu(psi.f[n], akx2, np)
            phi_p2 = phi_kin * (2.0 * m)
            momp2[n] = math_base.cprod(psi.f[n], phi_p2, dx, np)

        # for p
        akx = math_base.initak(np, dx, 1, ntriv)
        akx_mul = hart_to_cm / (-1j) / dalt_to_au
        akx *= akx_mul

        for n in range(nlevs):
            phip = diff_cpu(psi.f[n], akx, np)
            momp[n] = math_base.cprod(psi.f[n], phip, dx, np)

    else:
        # for x
        for n in range(nlevs):
            momx[n] = math_base.cprod2(psi.f[n], x, dx, np)

        # for x^2
        x2 = numpy.multiply(x, x)
        for n in range(nlevs):
            momx2[n] = math_base.cprod2(psi.f[n], x2, dx, np)

    return ExpectationValues(momx, momx2, momp, momp2)


def exp_sigma_vals_calc(psi: Psi, dx, np, ntriv):
    """ Calculation of expectation values <sigma_x>, <sigma_y>, <sigma_z> for a two-level system
        INPUT
        psi     list of complex vectors of length np describing wavefunctions
        dx      coordinate step of the problem
        np      number of grid points (must be a power of 2)
        ntriv       constant parameter; 1 -- an ordinary non-trivial diatomic-like system
                                        0 -- a trivial 2-level system
                                       -1 -- a trivial n-level system with angular momentum Hamiltonian and
                                             with external laser field augmented inside a Jz term
                                       -2 -- a trivial n-level system with angular momentum Hamiltonian and
                                             with external laser field augmented inside a Jx term

        OUTPUT
        smoms  list of complex vectors of length np """

    for i in range(len(psi.f)):
        assert psi.f[i].size == np

    nlevs = len(psi.f)
    smomx = numpy.float64(0.0)
    smomy = numpy.float64(0.0)
    smomz = numpy.float64(0.0)

    if ntriv != 1 and nlevs == 2:
        smomx = 2.0 * (math_base.cprod(psi.f[0], psi.f[1], dx, np)).real
        smomy = 2.0 * (math_base.cprod(psi.f[0], psi.f[1], dx, np)).imag
        smomz = math_base.cprod(psi.f[0], psi.f[0], dx, np) - math_base.cprod(psi.f[1], psi.f[1], dx, np)

    return SigmaExpectationValues(smomx, smomy, smomz)
