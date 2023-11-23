import cmath
import math

import numpy
from numpy.typing import NDArray

import math_base
from config import TaskRootConfiguration
from propagation import PropagationSolver
from psi_basis import PsiBasis

from phys_base import dalt_to_au, hart_to_cm


class _LaserFields:
    @staticmethod
    def zero(E0, t, t0, sigma):
        return numpy.float64(0.0)

    @staticmethod
    def const(E0, t, t0, sigma):
        return E0

    @staticmethod
    def laser_field_gauss(E0, t, t0, sigma):
        """ Calculates envelope of external laser field pulse energy
            INPUT
            E0      amplitude value of the laser field energy envelope
            t0      initial time, when the laser field is switched on
            sigma   scaling parameter of the laser field envelope
            t       current time value
            OUTPUT
            E       complex value of current external laser field  """

        E = E0 * math.exp(-(t - t0) * (t - t0) / 2.0 / sigma / sigma)

        return E

    @staticmethod
    def laser_field_sqrsin(E0, t, t0, sigma):
        """ Calculates envelope of external laser field pulse energy
            INPUT
            E0      amplitude value of the laser field energy envelope
            t0      initial time, when the laser field is switched on
            sigma   scaling parameter of the laser field envelope
            t       current time value
            OUTPUT
            E       complex value of current external laser field  """

        E = E0 * math.sin(2.0 * math.pi * (t - t0) / sigma) * math.sin(2.0 * math.pi * (t - t0) / sigma)

        return E

    @staticmethod
    def laser_field_maxwell(E0, t, t0, sigma):
        """ Calculates envelope of external laser field pulse energy
            INPUT
            E0      amplitude value of the laser field energy envelope
            t0      initial time, when the laser field is switched on
            sigma   scaling parameter of the laser field envelope
            t       current time value
            OUTPUT
            E       complex value of current external laser field  """

        E = E0 * t * t * math.exp(-(t - t0) * (t - t0) / 2.0 / sigma / sigma) / sigma / sigma

        return E

class _LaserFieldsHighFrequencyPart:
    @staticmethod
    def cexp(nu, freq_mult, t, pcos, w_list):
        """ Calculates a high-frequency part of external laser field pulse energy
            INPUT
            nu           basic laser field frequency
            freq_mult    basic laser field frequency multiplier
            t            current time value
            pcos         maximum frequency multiplier of the cos/sin set (dummy variable)
            w_list       list of partial amplitudes for the cos/sin set (dummy variable)
            OUTPUT
            E_omega      complex value of a high-frequency part for current external laser field  """

        E_omega = numpy.complex128(cmath.exp(1j * 2.0 * math.pi * nu * freq_mult * t))

        return E_omega

    @staticmethod
    def cos(nu, freq_mult, t, pcos, w_list):
        """ Calculates a high-frequency part of external laser field pulse energy
            INPUT
            nu           basic laser field frequency
            freq_mult    basic laser field frequency multiplier
            t            current time value
            pcos         maximum frequency multiplier of the cos/sin set (if applicable)
            w_list       list of partial amplitudes for the cos/sin set (dummy variable)
            OUTPUT
            E_omega      complex value of a high-frequency part for current external laser field  """

        E_omega = numpy.float64(math.cos(2.0 * math.pi * pcos * nu * freq_mult * t))

        return E_omega

    @staticmethod
    def sin(nu, freq_mult, t, pcos, w_list):
        """ Calculates a high-frequency part of external laser field pulse energy
            INPUT
            nu           basic laser field frequency
            freq_mult    basic laser field frequency multiplier
            t            current time value
            pcos         maximum frequency multiplier of the cos/sin set (if applicable)
            w_list       list of partial amplitudes for the cos/sin set (dummy variable)
            OUTPUT
            E_omega      complex value of a high-frequency part for current external laser field  """

        E_omega = numpy.float64(math.sin(2.0 * math.pi * pcos * nu * freq_mult * t))

        return E_omega

    @staticmethod
    def cos_set(nu, freq_mult, t, pcos, w_list):
        """ Calculates a high-frequency part of external laser field pulse energy
            INPUT
            nu           basic laser field frequency
            freq_mult    basic laser field frequency multiplier
            t            current time value
            pcos         maximum frequency multiplier of the cos/sin set (if applicable)
            w_list       list of (2 * pcos - 1) partial amplitudes for the cos/sin set (if applicable)
            OUTPUT
            E_omega      complex value of a high-frequency part for current external laser field  """

        E_omega = w_list[0] * math.cos(2.0 * math.pi * nu * freq_mult * t)
        i = -1
        for p in range(2, math.floor(pcos) + 1):
            i += 2
            E_omega += w_list[i] * math.cos(2.0 * math.pi * nu * freq_mult * t * p)
            E_omega += w_list[i + 1] * math.cos(2.0 * math.pi * nu * freq_mult * t / p)

        return numpy.float64(E_omega)

    @staticmethod
    def sin_set(nu, freq_mult, t, pcos, w_list):
        """ Calculates a high-frequency part of external laser field pulse energy
            INPUT
            nu           basic laser field frequency
            freq_mult    basic laser field frequency multiplier
            t            current time value
            pcos         maximum frequency multiplier of the cos/sin set (if applicable)
            w_list       list of (2 * pcos - 1) partial amplitudes for the cos/sin set (if applicable)
            OUTPUT
            E_omega      complex value of a high-frequency part for current external laser field  """

        E_omega = w_list[0] * math.sin(2.0 * math.pi * nu * freq_mult * t)
        for p in range(1, math.floor(pcos) + 1):
            E_omega += w_list[p] * math.sin(2.0 * math.pi * nu * freq_mult * t * (2 * p + 1))
            #E_omega += w_list[p] * math.sin(2.0 * math.pi * nu * freq_mult * t * (p + 1))   #Tmp!

        return numpy.float64(E_omega)


class _F_type:
    @staticmethod
    def F_sm(gc_vec, nb):
        """ Calculates the corresponding functional for the unitary transformation task
            INPUT
            nb      number of basis vectors used
            gc_vect list of the distances from the goal at the end of propagation for each basis vector
            OUTPUT
            F       complex value of the corresponding functional  """

        F = numpy.complex128(0)
        for vect in range(nb):
            for vect1 in range(nb):
                F -= gc_vec[vect] * gc_vec[vect1].conjugate()

        return F

    @staticmethod
    def F_re(gc_vec, nb):
        """ Calculates the corresponding functional for the unitary transformation task
            INPUT
            nb      number of basis vectors used
            gc_vect list of the distances from the goal at the end of propagation for each basis vector
            OUTPUT
            F       complex value of the corresponding functional  """

        F = numpy.complex128(0)
        for vect in range(nb):
            F -= gc_vec[vect].real

        return F

    @staticmethod
    def F_ss(gc_vec, nb):
        """ Calculates the corresponding functional for the unitary transformation task
            INPUT
            nb      number of basis vectors used
            gc_vect list of the distances from the goal at the end of propagation for each basis vector
            OUTPUT
            F       complex value of the corresponding functional  """

        F = numpy.complex128(0)
        for vect in range(nb):
            F -= gc_vec[vect] * gc_vec[vect].conjugate()

        return F

class _aF_type:
    @staticmethod
    def a_sm(psi_init, chi_init, dx, nb, nlevs, np):
        """ Calculates 'a' coefficient, which corresponds to the given functional for the unitary transformation task
            INPUT
            nb          number of basis vectors used
            nlevs       number of levels used
            np          number of grid points
            dx          coordinate grid step
            psi_init    initial wavefunctions
            chi_init    wavefunctions from the previous backward propagation at the initial time point t = 0
            OUTPUT
            a           list of complex values of the corresponding 'a' coefficients  """

        a = numpy.zeros(nb, dtype=numpy.complex128)
        for veca in range(nb):
            for vect in range(nb):
                for n in range(nlevs):
                    a[veca] += math_base.cprod(psi_init.psis[vect].f[n], chi_init.psis[vect].f[n], dx, np)

        return a

    @staticmethod
    def a_re(psi_init, chi_init, dx, nb, nlevs, np):
        """ Calculates 'a' coefficient, which corresponds to the given functional for the unitary transformation task
            INPUT
            nb          number of basis vectors used
            nlevs       number of levels used
            np          number of grid points
            dx          coordinate grid step
            psi_init    initial wavefunctions
            chi_init    wavefunctions from the previous backward propagation at the initial time point t = 0
            OUTPUT
            a           list of complex values of the corresponding 'a' coefficients  """

        a = numpy.zeros(nb, dtype=numpy.complex128)
        for veca in range(nb):
            a[veca] = numpy.complex128(0.5)

        return a

    @staticmethod
    def a_ss(psi_init, chi_init, dx, nb, nlevs, np):
        """ Calculates 'a' coefficient, which corresponds to the given functional for the unitary transformation task
            INPUT
            nb          number of basis vectors used
            nlevs       number of levels used
            np          number of grid points
            dx          coordinate grid step
            psi_init    initial wavefunctions
            chi_init    wavefunctions from the previous backward propagation at the initial time point t = 0
            OUTPUT
            a           list of complex values of the corresponding 'a' coefficients  """

        a = numpy.zeros(nb, dtype=numpy.complex128)
        for vect in range(nb):
            for n in range(nlevs):
                a[vect] += math_base.cprod(psi_init.psis[vect].f[n], chi_init.psis[vect].f[n], dx, np)

        return a

"""
This class contains all the possible wavefunction types

This class is "module-private". It may/should be used only among Task Manager implementations
If you want to call a wavefunction implementation from somewhere except a Task Manager, you 
are probably wrong :) 
"""


class _PsiFunctions:
    @staticmethod
    def zero(np):
        return numpy.zeros(np, dtype=numpy.complex128)

    @staticmethod
    def one(x, np, x0, p0, m, De, a, L):
        return numpy.array([1.0 / math.sqrt(L)] * np).astype(numpy.complex128)

    @staticmethod
    def harmonic(x, np, x0, p0, m, De, a, L):
        """ Initial wave function generator
            INPUT
            x           vector of length np defining positions of grid points
            L           spatial range of the problem
            np          number of grid points
            x0          initial coordinate
            p0          initial momentum
            m           reduced mass of the system
            a           scaling factor
            De          dissociation energy (dummy variable)

            OUTPUT
            psi     complex vector of length np describing the dimensionless wavefunction """

        psi = numpy.array(
            [cmath.exp(-(xi - x0) * (xi - x0) / 2.0 / a / a + 1j * p0 * xi) / pow(math.pi, 0.25) / math.sqrt(a) for xi
             in x]).astype(numpy.complex128)

        return psi

    @staticmethod
    def morse(x, np, x0, p0, m, De, a, L):
        """ Initial wave function generator
            INPUT
            x           vector of length np defining positions of grid points
            L           spatial range of the problem
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

        y = [math.exp(-a * (xi - x0)) / xe for xi in x]
        arg = 1.0 / xe - 1.0
        psi = numpy.array(
            [math.sqrt(a / math.gamma(arg)) * math.exp(-yi / 2.0) * pow(yi, numpy.float64(arg / 2.0)) for yi in y]).astype(numpy.complex128)

        return psi


"""
The implementations of this interface set up the task. That includes defining the starting
conditions, the goal, the potential, and all the possible other parameters necessary to define the task.

For example: The calculating and fitting lasses should NOT be aware of if they are solving 
Morse or Harmonic problem. It lays on this class. 
"""


class TaskManager:
    def __init__(self, conf_task: TaskRootConfiguration):
        if conf_task.wf_type == TaskRootConfiguration.WaveFuncType.MORSE:
            print("Morse wavefunctions are used")
            self.psi_init_impl = _PsiFunctions.morse
        elif conf_task.wf_type == TaskRootConfiguration.WaveFuncType.HARMONIC:
            print("Harmonic wavefunctions are used")
            self.psi_init_impl = _PsiFunctions.harmonic
        elif conf_task.wf_type == TaskRootConfiguration.WaveFuncType.CONST:
            print("Constant wavefunctions are used")
            self.psi_init_impl = _PsiFunctions.one
        else:
            raise RuntimeError("Impossible case in the WaveFuncType class")

        if conf_task.fitter.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.ZERO:
            print("Calculation without laser field")
            self.lf_init_guess = _LaserFields.zero
        elif conf_task.fitter.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.CONST:
            print("Constant type of initial guess for the laser field envelope is used")
            self.lf_init_guess = _LaserFields.const
        elif conf_task.fitter.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.GAUSS:
            print("Gaussian type of initial guess for the laser field envelope is used")
            self.lf_init_guess = _LaserFields.laser_field_gauss
        elif conf_task.fitter.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.SQRSIN:
            print("Squared sinus type of initial guess for the laser field envelope is used")
            self.lf_init_guess = _LaserFields.laser_field_sqrsin
        elif conf_task.fitter.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.MAXWELL:
            print("Maxwell distribution-like type of initial guess for the laser field envelope is used")
            self.lf_init_guess = _LaserFields.laser_field_maxwell
        else:
            raise RuntimeError("Impossible case in the InitGuess class")

        if conf_task.fitter.init_guess_hf == TaskRootConfiguration.FitterConfiguration.InitGuessHf.EXP:
            if not conf_task.task_type == TaskRootConfiguration.TaskType.FILTERING and \
               not conf_task.task_type == TaskRootConfiguration.TaskType.SINGLE_POT:
                print("Exponential high-frequency part of initial guess for the laser field is used")
            self.lf_init_guess_hf = _LaserFieldsHighFrequencyPart.cexp
        elif conf_task.fitter.init_guess_hf == TaskRootConfiguration.FitterConfiguration.InitGuessHf.COS:
            print("Cos-like high-frequency part of initial guess for the laser field with frequency multiplier "
                  "'pcos' = %f is used" % conf_task.fitter.pcos)
            self.lf_init_guess_hf = _LaserFieldsHighFrequencyPart.cos
        elif conf_task.fitter.init_guess_hf == TaskRootConfiguration.FitterConfiguration.InitGuessHf.SIN:
            print("Sin-like high-frequency part of initial guess for the laser field with frequency multiplier "
                  "'pcos' = %f is used" % conf_task.fitter.pcos)
            self.lf_init_guess_hf = _LaserFieldsHighFrequencyPart.sin
        elif conf_task.fitter.init_guess_hf == TaskRootConfiguration.FitterConfiguration.InitGuessHf.COS_SET:
            print("A sequence of cos-type terms with 'pcos' = %f as the high-frequency part of initial guess "
                  "for the laser field is used. The maximum frequency multiplier equal to floor(pcos) will be used"
                  % conf_task.fitter.pcos)
            if not conf_task.fitter.pcos > 1.0:
                raise ValueError("The maximum frequency multiplier in the high frequency part of the laser field, "
                                 "'pcos', has to be > 1")
            self.lf_init_guess_hf = _LaserFieldsHighFrequencyPart.cos_set
        elif conf_task.fitter.init_guess_hf == TaskRootConfiguration.FitterConfiguration.InitGuessHf.SIN_SET:
            print("A sequence of sin-type terms with 'pcos' = %f as the high-frequency part of initial guess "
                  "for the laser field is used. The maximum frequency multiplier equal to floor(pcos) will be used"
                  % conf_task.fitter.pcos)
            if not conf_task.fitter.pcos > 1.0:
                raise ValueError("The maximum frequency multiplier in the high frequency part of the laser field, "
                                 "'pcos', has to be > 1")
            self.lf_init_guess_hf = _LaserFieldsHighFrequencyPart.sin_set
        else:
            raise RuntimeError("Impossible case in the InitGuessHf class")

        if conf_task.hamil_type == TaskRootConfiguration.HamilType.NTRIV:
            print("Non-trivial type of the Hamiltonian is used")
            self.ntriv = 1
            if not conf_task.fitter.init_guess_hf == TaskRootConfiguration.FitterConfiguration.InitGuessHf.EXP:
                raise RuntimeError("For a non-trivial Hamiltonian an exponential high-frequency part of initial guess for the laser field has to be used!")
        elif conf_task.hamil_type == TaskRootConfiguration.HamilType.BH_MODEL:
            if conf_task.fitter.lf_aug_type == TaskRootConfiguration.FitterConfiguration.LfAugType.Z:
                print("Bose-Hubbard Hamiltonian with external laser field augmented inside a Jz term is used")
                self.ntriv = -1
            elif conf_task.fitter.lf_aug_type == TaskRootConfiguration.FitterConfiguration.LfAugType.X:
                print("Bose-Hubbard Hamiltonian with external laser field augmented inside a Jx term is used")
                self.ntriv = -2
            else:
                raise RuntimeError("Impossible case in the LfAugType class")
        elif conf_task.hamil_type == TaskRootConfiguration.HamilType.TWO_LEVELS:
            print("Simple trivial two-levels type of the Hamiltonian is used")
            self.ntriv = 0
            if not conf_task.fitter.nb == 2:
                raise RuntimeError("Number of basis vectors 'nb' for 'hamil_type' = 'two_levels' has to be equal to 2!")
        else:
            raise RuntimeError("Impossible case in the HamilType class")

        if conf_task.fitter.F_type == TaskRootConfiguration.FitterConfiguration.FType.SM:
            print("The 'squared module' type of the functional (F_sm) is used")
            self.F_type = _F_type.F_sm
            self.aF_type = _aF_type.a_sm
            self.F_goal = -conf_task.fitter.nb * conf_task.fitter.nb
        elif conf_task.fitter.F_type == TaskRootConfiguration.FitterConfiguration.FType.RE:
            print("The 'real' type of the functional (F_re) is used")
            self.F_type = _F_type.F_re
            self.aF_type = _aF_type.a_re
            self.F_goal = -conf_task.fitter.nb
        elif conf_task.fitter.F_type == TaskRootConfiguration.FitterConfiguration.FType.SS:
            print("The 'state-to-state' type of the functional (F_ss) is used")
            self.F_type = _F_type.F_ss
            self.aF_type = _aF_type.a_ss
            self.F_goal = -conf_task.fitter.nb
        else:
            raise RuntimeError("Impossible FType for the unitary transformation task")

        print(f"Number of %d-level basis vectors 'nb' = %d is used" % (conf_task.fitter.nlevs, conf_task.fitter.nb))

        self.conf_task = conf_task
        self.init_dir = PropagationSolver.Direction.FORWARD

        if self.conf_task.fitter.propagation.nu_L_auto:
            self.nu = numpy.float64(1.0 / 2.0 / self.conf_task.T)
        else:
            self.nu = self.conf_task.fitter.propagation.nu_L

    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb, nlevs):
        raise NotImplementedError()

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb, nlevs):
        raise NotImplementedError()

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du):
        raise NotImplementedError()

    def laser_field(self, E0, t, t0, sigma):
        return self.lf_init_guess(E0, t, t0, sigma)

    def laser_field_hf(self, freq_mult, t, pcos, w_list):
        return self.lf_init_guess_hf(self.nu, freq_mult, t, pcos, w_list)

    def F_type(self, gc_vec, nb):
        return self.F_type(gc_vec, nb)

    def aF_type(self, psi_init, chi_init, dx, nb, nlevs, np):
        return self.aF_type(psi_init, chi_init, dx, nb, nlevs, np)


class HarmonicSingleStateTaskManager(TaskManager):
    def __init__(self, conf_task: TaskRootConfiguration):
        super().__init__(conf_task)

    @staticmethod
    def _pot_level1(x, m, a):
        v = []
        # stiffness coefficient for dimensional case on the lower PEC
        k_s = hart_to_cm / m / dalt_to_au / pow(a, 4.0)
        # k_s = m * omega_0 * omega_0 * dalt_to_au / hart_to_cm
        # scaling factor for dimensional case on the lower PEC
        # a = math.sqrt(hart_to_cm / m / omega_0 / dalt_to_au)

        # harmonic frequency for dimensional case on the lower PEC
        omega_0 = k_s * a * a

        # theoretical ground energy value
        e_0 = omega_0 / 2.0
        print("Theoretical ground energy for the harmonic oscillator (relative to the potential minimum) = ", e_0)

        # Lower harmonic potential
        v_l = numpy.array([k_s * xi * xi / 2.0 for xi in x])
        v.append((numpy.float64(0.0), v_l))
        return v

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du):
        """ Potential energy vector
            INPUT
            x       vector of length np defining positions of grid points
            np      number of grid points
            a       scaling factor
            De      dissociation energy (dummy variable)
            m       reduced mass of the system
            x0p     partial shift value of the upper potential corresponding to the ground one (dummy variable)
            De_e    dissociation energy of the excited state (dummy variable)
            a_e     scaling factor of the excited state (dummy variable)
            Du      energy shift between the minima of the potentials (dummy variable)

            OUTPUT
            v       real vector of length np describing the dimensionless potential V(X) """

        # Lower harmonic potential
        v = HarmonicSingleStateTaskManager._pot_level1(x, m, a)

        # Upper harmonic potential
        v_u = numpy.array([0.0] * np).astype(numpy.float64)
        v.append((numpy.float64(0.0), v_u))

        return v

    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb, nlevs) -> PsiBasis:
        psi_init_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_init_obj.psis[vec].f[0] = self.psi_init_impl(x, np, x0, p0, m, De, a, L)
            psi_init_obj.psis[vec].f[1] = _PsiFunctions.zero(np)
        return psi_init_obj

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb, nlevs) -> PsiBasis:
        psi_goal_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_goal_obj.psis[vec].f[0] = _PsiFunctions.harmonic(x, np, x0, p0, m, De, a, L)
            psi_goal_obj.psis[vec].f[1] = _PsiFunctions.zero(np)
        return psi_goal_obj


class HarmonicMultipleStateTaskManager(HarmonicSingleStateTaskManager):
    def __init__(self, conf_task: TaskRootConfiguration):
        super().__init__(conf_task)

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du):
        """ Potential energy vector
            INPUT
            x       vector of length np defining positions of grid points
            np      number of grid points
            a       scaling factor
            De      dissociation energy (dummy variable)
            m       reduced mass of the system
            x0p     partial shift value of the upper potential corresponding to the ground one (dummy variable)
            De_e    dissociation energy of the excited state (dummy variable)
            a_e     scaling factor of the excited state (dummy variable)
            Du      energy shift between the minima of the potentials (dummy variable)

            OUTPUT
            v       real vector of length np describing the dimensionless potential V(X) """

        # Lower harmonic potential
        v = HarmonicSingleStateTaskManager._pot_level1(x, m, a)

        # stiffness coefficient for dimensional case on the upper PEC
        k_s_u = hart_to_cm / m / dalt_to_au / pow(a_e, 4.0)

        # Upper harmonic potential
        v_u = numpy.array([k_s_u * (xi - x0p) * (xi - x0p) / 2.0 + Du for xi in x])
        v.append((Du, v_u))

        return v

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb, nlevs) -> PsiBasis:
        psi_goal_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_goal_obj.psis[vec].f[0] = _PsiFunctions.zero(np)
            psi_goal_obj.psis[vec].f[1] = _PsiFunctions.harmonic(x, np, x0p + x0, p0, m, De_e, a_e, L)
        return psi_goal_obj


class MorseSingleStateTaskManager(TaskManager):
    def __init__(self, conf_task: TaskRootConfiguration):
        super().__init__(conf_task)

    @staticmethod
    def _pot_level1(x, m, De, a):
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
        v.append((numpy.float64(0.0), v_l))

        return v

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du):
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

        # Lower morse potential
        v = MorseSingleStateTaskManager._pot_level1(x, m, De, a)

        # Upper morse potential
        v_u = numpy.array([0.0] * np).astype(numpy.float64)
        v.append((numpy.float64(0.0), v_u))

        return v

    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb, nlevs) -> PsiBasis:
        psi_init_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_init_obj.psis[vec].f[0] = self.psi_init_impl(x, np, x0, p0, m, De, a, L)
            psi_init_obj.psis[vec].f[1] = _PsiFunctions.zero(np)
        return psi_init_obj

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb, nlevs) -> PsiBasis:
        psi_goal_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_goal_obj.psis[vec].f[0] = _PsiFunctions.morse(x, np, x0, p0, m, De, a, L)
            psi_goal_obj.psis[vec].f[1] = _PsiFunctions.zero(np)
        return psi_goal_obj


class MorseMultipleStateTaskManager(MorseSingleStateTaskManager):
    def __init__(self, conf_task: TaskRootConfiguration):
        super().__init__(conf_task)

    @staticmethod
    def _pot(x, np, m, De, a, x0p, De_e, a_e, Du):
        # Lower morse potential
        v = MorseSingleStateTaskManager._pot_level1(x, m, De, a)

        # Upper morse potential
        v_u = numpy.array(
            [De_e * (1.0 - math.exp(-a_e * (xi - x0p))) * (1.0 - math.exp(-a_e * (xi - x0p))) + Du for xi in x])
        v.append((Du, v_u))

        return v

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du):
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

        return self._pot(x, np, m, De, a, x0p, De_e, a_e, Du)

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb, nlevs) -> PsiBasis:
        psi_goal_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_goal_obj.psis[vec].f[0] = _PsiFunctions.zero(np)
            psi_goal_obj.psis[vec].f[1] = _PsiFunctions.morse(x, np, x0p + x0, p0, m, De_e, a_e, L)
        return psi_goal_obj


class MultipleStateUnitTransformTaskManager(MorseMultipleStateTaskManager):
    def __init__(self, conf_task: TaskRootConfiguration):
        super().__init__(conf_task)

    @staticmethod
    def _quant_fourier_transform(nb):
        omega = cmath.exp(1j * 2.0 * math.pi / nb)
        F = numpy.zeros((nb, nb), dtype=numpy.complex128)
        for v1 in range(nb):
            for v2 in range(nb):
                F.itemset((v1, v2), omega**(v1 * v2) / math.sqrt(nb))

        return numpy.matrix(F)

    @staticmethod
    def _matrix_PsiBasis_mult(F: numpy.matrix, psi: PsiBasis, nb, np):
        phi: PsiBasis = PsiBasis(nb, nb)
        for bv in range(nb):
            for gl in range(nb):
                phi_gl = numpy.zeros(np, dtype=numpy.complex128)
                for il in range(nb):
                    F_psi_el_mult = F.item(gl, il) * psi.psis[bv].f[il]
                    phi_gl = numpy.add(phi_gl, F_psi_el_mult)
                phi.psis[bv].f[gl] = phi_gl

        return phi

    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb, nlevs) -> PsiBasis:
        if nb == nlevs:
            psi_init_obj = PsiBasis(nb, nb)

            for vect in range(nb):
                for vect1 in range(nb):
                    if vect1 == vect:
                        psi_init_obj.psis[vect].f[vect1] = self.psi_init_impl(x, np, x0, p0, m, De, a, L)
                    else:
                        psi_init_obj.psis[vect].f[vect1] = _PsiFunctions.zero(np)
        elif nb == 1 and nlevs == 2:
            psi_init_obj = PsiBasis(nb)
            psi_init_obj.psis[0].f[0] = self.psi_init_impl(x, np, x0, p0, m, De, a, L)
            psi_init_obj.psis[0].f[1] = _PsiFunctions.zero(np)
        else:
            raise RuntimeError("Impossible Task Type")

        return psi_init_obj

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb, nlevs) -> PsiBasis:
        if nb == nlevs:
            F = self._quant_fourier_transform(nb)
            psi = self.psi_init(x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb, nlevs)
            phi = self._matrix_PsiBasis_mult(F, psi, nb, np)
        elif nb == 1 and nlevs == 2:
            phi = PsiBasis(nb)
            # phi.psis[0].f[0] = _PsiFunctions.zero(np)
            # phi.psis[0].f[1] = self.psi_init_impl(x, np, x0, p0, m, De, a, L)

            phi.psis[0].f[0] = self.psi_init_impl(x, np, x0, p0, m, De, a, L) / math.sqrt(2.0)
            phi.psis[0].f[1] = self.psi_init_impl(x, np, x0, p0, m, De, a, L) / math.sqrt(2.0)
        else:
            raise RuntimeError("Impossible Task Type")

        return phi

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du):
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

        v = []

        if not self.ntriv:
            # Lower potential
            D_l = -Du / 2.0
            v_l = numpy.array([D_l] * np)
            v.append((D_l, v_l))

            # Upper potential
            D_u = Du + D_l
            v_u = numpy.array([D_u] * np)
            v.append((D_u, v_u))

        elif self.ntriv < 0:
            U = self.conf_task.fitter.propagation.U # U ~ 1 / cm
            W = self.conf_task.fitter.propagation.W  # W ~ 1 / cm
            Emax = self.conf_task.fitter.propagation.E0 * self.conf_task.fitter.Em
            l = (self.conf_task.fitter.nlevs - 1) / 2.0

            # Maximum and minimum energies achieved during the calculation
            if self.ntriv == -1:
                vmax = 2.0 * U * l**2 + 2.0 * Emax * l
                vmin = -2.0 * Emax * l
            elif self.ntriv == -2:
                vmax = 2.0 * l * (U + W * l)
                if W != 0.0 and self.conf_task.fitter.nlevs >= U / W + 1:
                    vmin = -U * U / W / 2.0
                else:
                    vmin = 2.0 * l * (-U + W * l)
            else:
                raise RuntimeError("Impossible case in the LfAugType class")

            for n in range(self.conf_task.fitter.nlevs):
                vmax_list = numpy.array([vmax] * np)
                v.append((vmin, vmax_list))
        else:
            raise RuntimeError("Unsupported type of potential!")

        return v


def create(conf_task: TaskRootConfiguration):
    if conf_task.task_type == TaskRootConfiguration.TaskType.FILTERING or \
       conf_task.task_type == TaskRootConfiguration.TaskType.SINGLE_POT:

        task_manager_imp: TaskManager
        if conf_task.fitter.nlevs != 2:
            raise RuntimeError(
                "Number of levels in basis vectors 'nlevs' for 'task_type' = 'single_pot' and 'filtering' should be equal to 2!")
        if conf_task.hamil_type != TaskRootConfiguration.HamilType.NTRIV:
            raise RuntimeError("For 'task_type' = 'single_pot' and 'filtering' the Hamiltonian type 'hamil_type' = 'ntriv' should be specified!")
        if conf_task.fitter.nb != 1:
            raise RuntimeError("Number of basis vectors 'nb' for 'task_type' = 'single_pot' and 'filtering' should be equal to 1!")
        if conf_task.pot_type == TaskRootConfiguration.PotentialType.MORSE:
            print("Morse potentials are used")
            task_manager_imp = MorseSingleStateTaskManager(conf_task)
        elif conf_task.pot_type == TaskRootConfiguration.PotentialType.HARMONIC:
            print("Harmonic potentials are used")
            task_manager_imp = HarmonicSingleStateTaskManager(conf_task)
        else:
            raise RuntimeError("Impossible PotentialType")
    else:
        if conf_task.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
            task_manager_imp = MultipleStateUnitTransformTaskManager(conf_task)
            task_manager_imp.init_dir = PropagationSolver.Direction.BACKWARD
            if conf_task.pot_type == TaskRootConfiguration.PotentialType.NONE:
                print("No potentials is used")
                if conf_task.fitter.hf_hide:
                    raise RuntimeError("'hf_hide' parameter for 'none' potential type should be set to 'false'!")
            else:
                raise RuntimeError("Impossible PotentialType for the unitary transformation task")
        else:
            if conf_task.fitter.nb != 1:
                raise RuntimeError(
                        "Number of basis vectors 'nb' for 'task_type' = 'trans_wo_control', 'intuitive_control', 'local_control_population', "
                        "'local_control_projection', and 'optimal_control_krotov' should be equal to 1!")
            if conf_task.fitter.nlevs != 2:
                raise RuntimeError(
                        "Number of levels in basis vectors 'nlevs' for 'task_type' = 'trans_wo_control', 'intuitive_control', 'local_control_population', "
                        "'local_control_projection', and 'optimal_control_krotov' should be equal to 2!")
            if conf_task.pot_type == TaskRootConfiguration.PotentialType.MORSE:
                print("Morse potentials are used")
                task_manager_imp = MorseMultipleStateTaskManager(conf_task)
            elif conf_task.pot_type == TaskRootConfiguration.PotentialType.HARMONIC:
                print("Harmonic potentials are used")
                task_manager_imp = HarmonicMultipleStateTaskManager(conf_task)
            elif conf_task.pot_type == TaskRootConfiguration.PotentialType.NONE:
                print("No potentials are used")
                if conf_task.fitter.hf_hide:
                    raise RuntimeError("'hf_hide' parameter for 'none' potential type should be set to 'false'!")
                task_manager_imp = MultipleStateUnitTransformTaskManager(conf_task)
            else:
                raise RuntimeError("Impossible PotentialType")

    return task_manager_imp
