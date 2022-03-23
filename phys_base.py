import cmath

import numpy
import copy

import math_base

import pyopencl.array as cla
import pyopencl as cl
import pyopencl.elementwise as cle
#import pyopencl.clmath
from pyvkfft.fft import fftn, ifftn

from psi_basis import Psi

hart_to_cm = 219474.6313708 # 1 / cm / hartree
cm_to_erg = 1.98644568e-16 # erg * cm
dalt_to_au = 1822.888486 # a.u. / D
Red_Planck_h = 1.054572e-27 # erg * s
Hz_to_cm = 3.33563492e-11 # s / cm

cl_context = cl._cl.Context()
complex_mult = cle.ElementwiseKernel(cl_context,
                                     "cdouble_t *x, cdouble_t *y",
                                     "x[i] = cdouble_mul(x[i], y[i])",
                                     "complex_mult")
cq = cl.CommandQueue(cl_context)


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


def diff_gpu(psi, akx2, np):
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

    buf_dev = cla.to_device(cq, psi)
    akx2_dev = cla.to_device(cq, akx2)

    fftn(buf_dev)
    complex_mult(buf_dev, akx2_dev)
    ifftn(buf_dev)

    phi = buf_dev.get()

    return phi


def hamil(psi: numpy.ndarray, v, akx2, np):
    """ Calculates the simplest one-dimensional Hamiltonian mapping of vector psi
        INPUT
        psi   list of complex vectors of length np
        v     list of potential energy real vectors of length np
        akx2  complex kinetic energy vector of length np, = k^2/2m
        np    number of grid points
        OUTPUT
        phi = H psi list of complex vectors of length np """

    assert psi.size == np
    assert v.size == np
    assert akx2.size == np

    # kinetic energy mapping
    phi = diff_cpu(psi, akx2, np)

    # potential energy mapping and accumulation phi_l = H psi_l
    vpsi = numpy.multiply(v, psi)
    numpy.add(phi, vpsi, out=phi)

    return phi


def hamil2D_orig(psi: list[numpy.ndarray], v, akx2, np, E_full):
    """ Calculates two-dimensional Hamiltonian mapping of vector psi (without energy shifting)
        INPUT
        psi    list of complex vectors of length np
        v      list of potential energy real vectors of length np
        akx2   complex kinetic energy vector of length np, = k^2/2m
        np     number of grid points
        E_full a real value of external laser field
        OUTPUT
        phi = H psi list of complex vectors of length np """

    for i in range(len(psi)):
        assert psi[i].size == np
        assert v[i][1].size == np
    assert akx2.size == np

    phi = []
    # diagonal terms
    # ground state 1D Hamiltonian mapping for the lower state
    phi_dl = hamil(psi[0], v[0][1], akx2, np)

    # excited state 1D Hamiltonian mapping for the upper state
    phi_du = hamil(psi[1], v[1][1], akx2, np)

    # adding non-diagonal terms
    psiE_u = psi[1] * E_full
    phi_l = numpy.subtract(phi_dl, psiE_u)
    phi.append(phi_l)

    psiE_d = psi[0] * E_full.conjugate()
    phi_u = numpy.subtract(phi_du, psiE_d)
    phi.append(phi_u)

    return phi


def hamil2D(psi, v, akx2, np, E, eL):
    """ Calculates two-dimensional Hamiltonian mapping of vector psi
        INPUT
        psi   list of complex vectors of length np
        v     list of potential energy real vectors of length np
        akx2  complex kinetic energy vector of length np, = k^2/2m
        np    number of grid points
        E     a real value of external laser field
        eL    a laser field energy shift
        OUTPUT
        phi = H psi list of complex vectors of length np """

    for i in range(len(psi)):
        assert psi[i].size == np
        assert v[i][1].size == np
    assert akx2.size == np

    phi = []
    # diagonal terms
    # ground state 1D Hamiltonian mapping for the lower state
    phi_dl = hamil(psi[0], v[0][1], akx2, np)
    # adding of the laser field energy shift
    psieL_d = psi[0] * eL
    numpy.add(phi_dl, psieL_d, out=phi_dl)

    # excited state 1D Hamiltonian mapping for the upper state
    phi_du = hamil(psi[1], v[1][1], akx2, np)
    # adding of the laser field energy shift
    psieL_u = psi[1] * eL
    numpy.subtract(phi_du, psieL_u, out=phi_du)

    # adding non-diagonal terms
    psiE_u = psi[1] * E
    phi_l = numpy.subtract(phi_dl, psiE_u)
    phi.append(phi_l)

    psiE_d = psi[0] * E
    phi_u = numpy.subtract(phi_du, psiE_d)
    phi.append(phi_u)

    return phi


def residum(psi, v, akx2, xp, np, emin, emax, E, eL):
    """ Scaled and normalized mapping phi = ( O - xp I ) phi
        INPUT
        psi         list of complex vectors of length np
        v           list of potential energy vectors of length np
        xp          sampling interpolation point
        np          number of grid points (must be a power of 2)
        emax, emin  upper and lower limits of energy spectra
        E           a complex value of external laser field
        eL          a laser field energy shift
        OUTPUT
        phi  list of complex vectors of length np
             the operator is normalized from -2 to 2 resulting in:
             phi = 4.O / (emax - emin) * H psi - 2.0 (emax + emin) / (emax - emin) * I psi - xp I psi """

    for i in range(len(psi)):
        assert psi[i].size == np
        assert v[i][1].size == np
    assert akx2.size == np

    hpsi = hamil2D(psi, v, akx2, np, E, eL)

    phi = []
    # changing the range from -2 to 2
    for n in range(len(psi)):
        coef1 = 4.0 / (emax - emin)
        coef2 = 2.0 * (emax + emin) / (emax - emin)
        hpsi[n] *= coef1
        tmp = psi[n] * coef2
        numpy.subtract(hpsi[n], tmp, out=hpsi[n])

        phi_n = psi[n] * (-xp)
        numpy.add(phi_n, hpsi[n], out=phi_n)

        phi.append(phi_n)

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


def prop(psi, t_sc, nch, np, v, akx2, emin, emax, E, eL):
    """ Propagation subroutine using Newton interpolation
        P(O) psi = dv(1) psi + dv2 (O - x1 I) psi + dv3 (O - x2)(O - x1 I) psi + ...
        INPUT
        psi         list of complex vectors of length np describing wavefunctions
                    at the beginning of interval
        t_sc        time interval (normalized by the reduced Planck constant)
        nch         order of interpolation polynomial (must be a power of 2 if
                    reorder is necessary)
        np          number of grid points (must be a power of 2)
        v           list of potential energy vectors of length np
        akx2        kinetic energy vector of length np
        emax, emin  upper and lower limits of energy spectra
        E           a complex value of external laser field
        eL          a laser field energy shift

        OUTPUT
        psi  list of complex vectors of length np
             describing the propagated wavefunction
             phi(t) = exp(-iHt) psi(0) """

    for i in range(len(psi)):
        assert psi[i].size == np
        assert v[i][1].size == np
    assert akx2.size == np

    # interpolation points and divided difference coefficients
    xp, dv = math_base.points(nch, t_sc, func)

    # auxiliary vector used for recurrence
    phi = copy.deepcopy(psi)

    # accumulating first term
    for n in range(len(psi)):
        psi[n] *= dv[0]

    # recurrence loop
    for j in range(nch - 1):
        # mapping by scaled operator of phi
        phi = residum(phi, v, akx2, xp[j], np, emin, emax, E, eL)

        # accumulation of Newtonian's interpolation
        for n in range(len(psi)):
            phidv = phi[n] * dv[j + 1]
            numpy.add(psi[n], phidv, out=psi[n])

    coef = cmath.exp(-1j * 2.0 * t_sc * (emax + emin) / (emax - emin))
    for n in range(len(psi)):
        psi[n] *= coef

    return psi


class ExpectationValues():
    def __init__(self, x_l, x_u, x2_l, x2_u, p_l, p_u, p2_l, p2_u):
        self.x_l = x_l
        self.x_u = x_u
        self.x2_l = x2_l
        self.x2_u = x2_u
        self.p_l = p_l
        self.p_u = p_u
        self.p2_l = p2_l
        self.p2_u = p2_u


def exp_vals_calc(psi: Psi, x, akx2, dx, np, m):
    """ Calculation of expectation values <x>, <x^2>, <p>, <p^2>
        INPUT
        psi     list of complex vectors of length np describing wavefunctions
        x       vector of length np defining positions of grid points
        akx2    kinetic energy vector of length np
        dx      coordinate step of the problem
        np      number of grid points (must be a power of 2)
        m       reduced mass of the system
        OUTPUT
        moms  list of complex vectors of length np """

    # for x
    momx_l = math_base.cprod2(psi.f[0], x, dx, np)
    momx_u = math_base.cprod2(psi.f[1], x, dx, np)

    # for x^2
    x2 = numpy.multiply(x, x)
    momx2_l = math_base.cprod2(psi.f[0], x2, dx, np)
    momx2_u = math_base.cprod2(psi.f[1], x2, dx, np)

    # for p^2
    phi_kin_l = diff_cpu(psi.f[0], akx2, np)
    phi_p2_l = phi_kin_l * (2.0 * m)
    momp2_l = math_base.cprod(psi.f[0], phi_p2_l, dx, np)

    phi_kin_u = diff_cpu(psi.f[1], akx2, np)
    phi_p2_u = phi_kin_u * (2.0 * m)
    momp2_u = math_base.cprod(psi.f[1], phi_p2_u, dx, np)

    # for p
    akx = math_base.initak(np, dx, 1)
    akx_mul = hart_to_cm / (-1j) / dalt_to_au
    akx *= akx_mul

    phip_l = diff_cpu(psi.f[0], akx, np)
    momp_l = math_base.cprod(psi.f[0], phip_l, dx, np)

    phip_u = diff_cpu(psi.f[1], akx, np)
    momp_u = math_base.cprod(psi.f[1], phip_u, dx, np)

    return ExpectationValues(momx_l, momx_u, momx2_l, momx2_u, momp_l, momp_u, momp2_l, momp2_u)

