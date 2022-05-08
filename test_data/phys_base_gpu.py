import pyopencl.array as cla
import pyopencl as cl
import pyopencl.elementwise as cle
from pyvkfft.opencl import VkFFTApp

from datetime import datetime
import copy
import numpy
import cmath

import math_base
import phys_base

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

    return buf_dev


def hamil_gpu(psi_dev: cla.Array, v_dev: cla.Array, akx2_dev: cla.Array, np):
    """ Calculates the simplest one-dimensional Hamiltonian mapping of vector psi
        INPUT
        psi   list of complex vectors of length np
        v     list of potential energy real vectors of length np
        akx2  complex kinetic energy vector of length np, = k^2/2m
        np    number of grid points
        OUTPUT
        phi = H psi list of complex vectors of length np """

    # kinetic energy mapping
    res = diff_gpu(psi_dev, akx2_dev, np)

    # potential energy mapping and accumulation phi_l = H psi_l
    complex_double_mult_add_f([v_dev, psi_dev, res, v_dev.size])

    return res


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

    # diagonal terms
    # ground state 1D Hamiltonian mapping for the lower state
    phi_dl_dev = hamil_gpu(psi_dev[0], v_dev[0], akx2_dev, np)
    phi_du_dev = hamil_gpu(psi_dev[1], v_dev[1], akx2_dev, np)

    hamil2D_half_cl_f([phi_dl_dev, psi_dev[0], psi_dev[1], eL, E, phi_dl_dev.size])
    hamil2D_half_cl_f([phi_du_dev, psi_dev[1], psi_dev[0], eL, E, phi_du_dev.size])

    return [phi_dl_dev, phi_du_dev]


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

    hpsi_dev = hamil2D_gpu(psi_dev, v_dev, akx2_dev, np, E, eL)

    coef1 = 4.0 / (emax - emin)
    coef2 = 2.0 * (emax + emin) / (emax - emin)

    phi_dev = [ hpsi_dev[0], hpsi_dev[1] ]

    # changing the range from -2 to 2
    residum_half_cl_f([phi_dev[0], psi_dev[0], coef1, coef2, xp, phi_dev[0].size])
    residum_half_cl_f([phi_dev[1], psi_dev[1], coef1, coef2, xp, phi_dev[1].size])

    return phi_dev


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
    xp, dv = math_base.points(nch, t_sc, phys_base.func)

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
    for j in range(nch - 1):
        # mapping by scaled operator of phi
        phi_dev = residum_gpu(phi_dev, v_dev, akx2_dev, xp[j], np, emin, emax, E, eL)

        # accumulation of Newtonian's interpolation
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
