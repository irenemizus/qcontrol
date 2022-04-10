import cmath
import sys
import scipy
from datetime import datetime

import numpy
import copy

import math_base

import pyopencl.array as cla
import pyopencl as cl
import pyopencl.elementwise as cle
#import pyopencl.clmath
from pyvkfft.fft import fftn, ifftn, _prepare_transform, _get_fft_app
from pyvkfft.opencl import _vkfft_opencl, VkFFTApp

from psi_basis import Psi
from test_tools import TableComparer

hart_to_cm = 219474.6313708 # 1 / cm / hartree
cm_to_erg = 1.98644568e-16 # erg * cm
dalt_to_au = 1822.888486 # a.u. / D
Red_Planck_h = 1.054572e-27 # erg * s
Hz_to_cm = 3.33563492e-11 # s / cm

cl_context = cl._cl.Context()
cq = cl.CommandQueue(cl_context)


class FastElementwiseKernel:
    def __init__(self, queue: cl.CommandQueue, eltKernel: cle.ElementwiseKernel):
        self.queue = queue
        self.eltKernel = eltKernel
        self.kernel, self.arg_descrs = eltKernel.get_kernel(use_range=False)
        self.max_wg_size = self.kernel.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE,
            queue.device)

    def __call__(self, invocation_args):
        gs, ls = invocation_args[0]._get_sizes(self.queue, self.max_wg_size)
        #invocation_args = [buf_dev, akx2_dev, buf_dev.size]
        self.kernel(self.queue, gs, ls, *invocation_args, wait_for=[])


complex_mult_f = FastElementwiseKernel(cq, cle.ElementwiseKernel(cl_context,
                                     "cdouble_t *x, cdouble_t *y",
                                     "x[i] = cdouble_mul(x[i], y[i])",
                                     "complex_mult"))

complex_double_mult_add_f = FastElementwiseKernel(cq, cle.ElementwiseKernel(cl_context,
                                     "double *v, cdouble_t *psi, cdouble_t *phi",
                                     "phi[i] = cdouble_add(cdouble_mul(psi[i], cdouble_new(v[i], 0.0)), phi[i])",
                                     "complex_double_mult_add"))

prop_recurr_part_cl_f = FastElementwiseKernel(cq, cle.ElementwiseKernel(cl_context,
                                     "cdouble_t *psi, cdouble_t *phi, cdouble_t dvj",
                                     "psi[i] = cdouble_add(cdouble_mul(phi[i], dvj), psi[i])",
                                     "prop_recurr_part_cl"))

hamil2D_half_cl_f = FastElementwiseKernel(cq, cle.ElementwiseKernel(cl_context,
                                   "cdouble_t *phi_d, cdouble_t *psi_d, cdouble_t *psi_nd, double eL, double E",
                                   "phi_d[i] = "
                                   "    cdouble_sub("
                                   "        cdouble_sub("
                                   "            phi_d[i], "
                                   "            cdouble_mul("
                                   "                psi_d[i], cdouble_new(eL, 0.0)"
                                   "            )"
                                   "        ), "
                                   "        cdouble_mul("
                                   "            psi_nd[i], cdouble_new(E, 0.0)"
                                   "        )"
                                   "    )",
                                   "hamil2D_half_cl"))

residum_half_cl_f = FastElementwiseKernel(cq, cle.ElementwiseKernel(cl_context,
                                   "cdouble_t *phi_d, cdouble_t *psi_d, double coef1, double coef2, double xp",
                                   "phi_d[i] = "
                                   "    cdouble_sub("
                                   "        cdouble_sub("
                                   "            cdouble_mul("
                                   "                phi_d[i], cdouble_new(coef1, 0.0)"
                                   "            ),"
                                   "            cdouble_mul("
                                   "                psi_d[i], cdouble_new(coef2, 0.0)"
                                   "            )"
                                   "        ), "
                                   "        cdouble_mul("
                                   "            psi_d[i], cdouble_new(xp, 0.0)"
                                   "        )"
                                   "    )",
                                   "residum_half_cl"))


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
#    psi_freq = scipy.fft.fft(psi, n=None, axis=-1, norm=None, overwrite_x=False, workers=2, plan=None)
    phi_freq = numpy.multiply(psi_freq, akx2)
#    phi = scipy.fft.ifft(phi_freq, n=None, axis=-1, norm=None, overwrite_x=True, workers=2, plan=None)
    phi = numpy.fft.ifft(phi_freq)

    # numpy.set_printoptions(threshold=sys.maxsize)
    # print(f"psi = {psi}")
    # print(f"akx2 = {akx2}")
    # print(f"np = {np}")
    # print(f"phi = {phi}")

    return phi

#app = None
fftAppInplace: VkFFTApp = None
fftAppOutplace: VkFFTApp = None

def diff_gpu(psi_dev: cla.Array, akx2_dev: cla.Array, np):
    """ Calculates kinetic energy mapping carried out in momentum space
        INPUT
        psi   complex vector of length np
        akx2  complex vector of length np, = k^2/2m
        np    number of grid points
        OUTPUT
        phi   complex vector of length np describing the mapping
              of kinetic energy phi = P^2/2m psi """

    #buf_dev = cla.to_device(cq, psi.astype(numpy.complex128))
    #akx2_dev = cla.to_device(cq, akx2.astype(numpy.complex128))
    #buf_dev = fftn(psi_dev)
    # dest = cla.empty_like(psi_dev)
    # src = psi_dev
    # cl_queue = cq
    # backend, inplace, dest, cl_queue = _prepare_transform(src, dest, cl_queue, False)

    # global app
    # if app is None:
    #     app = _get_fft_app(backend, src.shape, src.dtype, inplace, ndim=None, axes=None, norm=1, cuda_stream=None, cl_queue=cl_queue)
    # res = _vkfft_opencl.fft(app, int(psi_dev.data.int_ptr), int(dest.data.int_ptr), int(cl_queue.int_ptr))
    # if res != 0:
    #     raise Exception(f"We have a problem... res={res}")
    # buf_dev = dest

    #complex_mult(buf_dev, akx2_dev)
    #gs, ls = buf_dev._get_sizes(cq, complex_mult_kernel_max_wg_size)
    #complex_mult_kernel(cq, gs, ls, *invocation_args, wait_for=[])

    global fftAppOutplace
    if fftAppOutplace is None:
        fftAppOutplace = VkFFTApp(psi_dev.shape, psi_dev.dtype, cq, ndim=None, inplace=False, norm=1,
                 r2c=False, dct=False, axes=None)

    buf_dev = cla.empty_like(psi_dev)
    fftAppOutplace.fft(psi_dev, buf_dev)

    complex_mult_f([ buf_dev, akx2_dev, buf_dev.size ])

    global fftAppInplace
    if fftAppInplace is None:
        fftAppInplace = VkFFTApp(psi_dev.shape, psi_dev.dtype, cq, ndim=None, inplace=True, norm=1,
                 r2c=False, dct=False, axes=None)

    fftAppInplace.ifft(buf_dev)

    #phi = buf_dev.get()

    return buf_dev



def hamil_cpu(psi, v, akx2, np):
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


def hamil_gpu(psi_dev: cla.Array, v_dev: cla.Array, akx2_dev: cla.Array, np):
    """ Calculates the simplest one-dimensional Hamiltonian mapping of vector psi
        INPUT
        psi   list of complex vectors of length np
        v     list of potential energy real vectors of length np
        akx2  complex kinetic energy vector of length np, = k^2/2m
        np    number of grid points
        OUTPUT
        phi = H psi list of complex vectors of length np """

    #psi_dev = cla.to_device(cq, psi.astype(numpy.complex128))
    #akx2_dev = cla.to_device(cq, akx2.astype(numpy.complex128))
    #v_dev = cla.to_device(cq, v.astype(numpy.float64))

    # kinetic energy mapping
    res = diff_gpu(psi_dev, akx2_dev, np)

    # potential energy mapping and accumulation phi_l = H psi_l
    #vpsi_dev = numpy.multiply(v, psi)
    #numpy.add(phi, vpsi, out=phi)
    complex_double_mult_add_f([v_dev, psi_dev, res, v_dev.size])

    return res #.get()


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
    phi_dl = hamil_cpu(psi[0], v[0][1], akx2, np)

    # excited state 1D Hamiltonian mapping for the upper state
    phi_du = hamil_cpu(psi[1], v[1][1], akx2, np)

    # adding non-diagonal terms
    psiE_u = psi[1] * E_full
    phi_l = numpy.subtract(phi_dl, psiE_u)
    phi.append(phi_l)

    psiE_d = psi[0] * E_full.conjugate()
    phi_u = numpy.subtract(phi_du, psiE_d)
    phi.append(phi_u)

    return phi


def hamil2D_cpu(psi, v, akx2, np, E, eL):
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

    # diagonal terms
    # ground state 1D Hamiltonian mapping for the lower state
    phi_dl = hamil_cpu(psi[0], v[0][1], akx2, np)
    phi_du = hamil_cpu(psi[1], v[1][1], akx2, np)

    # adding of the laser field energy shift
    psieL_d = psi[0] * eL
    numpy.add(phi_dl, psieL_d, out=phi_dl)

    # excited state 1D Hamiltonian mapping for the upper state

    # adding of the laser field energy shift
    psieL_u = psi[1] * eL
    numpy.subtract(phi_du, psieL_u, out=phi_du)

    # adding non-diagonal terms
    psiE_u = psi[1] * E
    phi_l = numpy.subtract(phi_dl, psiE_u)

    psiE_d = psi[0] * E
    phi_u = numpy.subtract(phi_du, psiE_d)

    return [phi_l, phi_u]


def hamil2D_gpu(psi_dev, v_dev, akx2_dev, np, E, eL):
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

    # psi0_dev = cla.to_device(cq, psi[0].astype(numpy.complex128))
    # psi1_dev = cla.to_device(cq, psi[1].astype(numpy.complex128))
    # akx2_dev = cla.to_device(cq, akx2.astype(numpy.complex128))
    # v01_dev = cla.to_device(cq, v[0][1].astype(numpy.float64))
    # v11_dev = cla.to_device(cq, v[1][1].astype(numpy.float64))

    # diagonal terms
    # ground state 1D Hamiltonian mapping for the lower state
    phi_dl_dev = hamil_gpu(psi_dev[0], v_dev[0], akx2_dev, np)
    phi_du_dev = hamil_gpu(psi_dev[1], v_dev[1], akx2_dev, np)

    hamil2D_half_cl_f([phi_dl_dev, psi_dev[0], psi_dev[1], eL, E, phi_dl_dev.size])
    hamil2D_half_cl_f([phi_du_dev, psi_dev[1], psi_dev[0], eL, E, phi_du_dev.size])

    return [phi_dl_dev, phi_du_dev]


def residum_cpu(psi, v, akx2, xp, np, emin, emax, E, eL):
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

    hpsi = hamil2D_cpu(psi, v, akx2, np, E, eL)

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


def residum_gpu(psi_dev, v_dev, akx2_dev, xp, np, emin, emax, E, eL):
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

    # for i in range(len(psi)):
    #     assert psi[i].size == np
    #     assert v[i][1].size == np
    # assert akx2.size == np

    # psi_dev = []
    # psi_dev.append(cla.to_device(cq, psi[0].astype(numpy.complex128)))
    # psi_dev.append(cla.to_device(cq, psi[1].astype(numpy.complex128)))
    # akx2_dev = cla.to_device(cq, akx2.astype(numpy.complex128))
    #
    # v_dev = []
    # v_dev.append(cla.to_device(cq, v[0][1].astype(numpy.float64)))
    # v_dev.append(cla.to_device(cq, v[1][1].astype(numpy.float64)))

    hpsi_dev = hamil2D_gpu(psi_dev, v_dev, akx2_dev, np, E, eL)

    coef1 = 4.0 / (emax - emin)
    coef2 = 2.0 * (emax + emin) / (emax - emin)

    phi_dev = [ hpsi_dev[0], hpsi_dev[1] ]

    # changing the range from -2 to 2
    residum_half_cl_f([phi_dev[0], psi_dev[0], coef1, coef2, xp, phi_dev[0].size])
    residum_half_cl_f([phi_dev[1], psi_dev[1], coef1, coef2, xp, phi_dev[1].size])

    #phi = [ phi_dev[0].get(), phi_dev[1].get() ]
    return phi_dev


def func(z, t):
    """ The function to be interpolated
        INPUT
        z     real coordinate parameter
        t     real time parameter (dimensionless)
        OUTPUT
        func  value of the function (complex)
        func = f (z, t) """

    return cmath.exp(-1j * z * t)


def prop_cpu(psi, t_sc, nch, np, v, akx2, emin, emax, E, eL):
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
        phi = residum_cpu(phi, v, akx2, xp[j], np, emin, emax, E, eL)

        # accumulation of Newtonian's interpolation
        for n in range(len(psi)):
            phidv = phi[n] * dv[j + 1]
            numpy.add(psi[n], phidv, out=psi[n])

    coef = cmath.exp(-1j * 2.0 * t_sc * (emax + emin) / (emax - emin))
    for n in range(len(psi)):
        psi[n] *= coef

    return psi

already_there = False

def prop_gpu(psi, t_sc, nch, np, v, akx2, emin, emax, E, eL):
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

    time_before = datetime.now()
    phi_dev = []
    phi_dev.append(cla.to_device(cq, phi[0].astype(numpy.complex128)))
    phi_dev.append(cla.to_device(cq, phi[1].astype(numpy.complex128)))

    psi_dev = []
    psi_dev.append(cla.to_device(cq, psi[0].astype(numpy.complex128)))
    psi_dev.append(cla.to_device(cq, psi[1].astype(numpy.complex128)))
    akx2_dev = cla.to_device(cq, akx2.astype(numpy.complex128))

    v_dev = []
    v_dev.append(cla.to_device(cq, v[0][1].astype(numpy.float64)))
    v_dev.append(cla.to_device(cq, v[1][1].astype(numpy.float64)))

    time_middle = datetime.now()
    # recurrence loop
    k = None
    for j in range(nch - 1):
        # mapping by scaled operator of phi
        phi_dev = residum_gpu(phi_dev, v_dev, akx2_dev, xp[j], np, emin, emax, E, eL)

        # accumulation of Newtonian's interpolation
        #for n in range(len(psi)):
        #    phidv = phi[n] * dv[j + 1]
        #    numpy.add(psi[n], phidv, out=psi[n])
        prop_recurr_part_cl_f([psi_dev[0], phi_dev[0], dv[j + 1], psi_dev[0].size])
        prop_recurr_part_cl_f([psi_dev[1], phi_dev[1], dv[j + 1], psi_dev[1].size])

    time_before_get = datetime.now()
    psi = [ psi_dev[0].get(), psi_dev[1].get() ]

    time_after = datetime.now()

    coef = cmath.exp(-1j * 2.0 * t_sc * (emax + emin) / (emax - emin))
    for n in range(len(psi)):
        psi[n] *= coef

    dt_1 = time_middle - time_before
    dt_1_ms = dt_1.microseconds / 1000
    dt_2 = time_before_get - time_middle
    dt_2_ms = dt_2.microseconds / 1000
    dt_3 = time_after - time_before_get
    dt_3_ms = dt_3.microseconds / 1000

    print(
        "milliseconds per step: dt_1 = " + str(dt_1_ms) + ", dt_2 =  " + str(dt_2_ms) + ", dt_3 =  " + str(dt_3_ms)
    )
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

