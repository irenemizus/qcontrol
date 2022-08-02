import cmath
import math

import numpy

import phys_base
from config import TaskRootConfiguration
from propagation import PropagationSolver
from psi_basis import PsiBasis

from phys_base import dalt_to_au, hart_to_cm


class _LaserFields:
    @staticmethod
    def zero(E0, t, t0, sigma):
        return 0.0

    @staticmethod
    def laser_field_gauss(E0, t, t0, sigma):
        """ Calculates envelope of external laser field pulse energy
            INPUT
            E0      amplitude value of the laser field energy envelope
            t0      initial time, when the laser field is switched on
            sigma   scaling parameter of the laser field envelope
            nu_L    basic frequency of the laser field
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
            nu_L    basic frequency of the laser field
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
            nu_L    basic frequency of the laser field
            t       current time value
            OUTPUT
            E       complex value of current external laser field  """

        E = E0 * t * t * math.exp(-(t - t0) * (t - t0) / 2.0 / sigma / sigma) / sigma / sigma

        return E

class _LaserFieldsHighFrequencyPart:
    @staticmethod
    def cexp(nu_L, t, pcos):
        """ Calculates a high-frequency part of external laser field pulse energy
            INPUT
            nu_L    basic frequency of the laser field
            t       current time value
            pcos    maximum frequency multiplier of the cos set (if applicable)
            OUTPUT
            E_omega complex value of a high-frequency part for current external laser field  """

        E_omega = cmath.exp(1j * 2.0 * math.pi * nu_L * t)

        return E_omega

    @staticmethod
    def cos(nu_L, t, pcos):
        """ Calculates a high-frequency part of external laser field pulse energy
            INPUT
            nu_L    basic frequency of the laser field
            t       current time value
            pcos    maximum frequency multiplier of the cos set (if applicable)
            OUTPUT
            E_omega complex value of a high-frequency part for current external laser field  """

        E_omega = math.cos(2.0 * math.pi * pcos * nu_L * t)

        return E_omega

    @staticmethod
    def cos_set(nu_L, t, pcos):
        """ Calculates a high-frequency part of external laser field pulse energy
            INPUT
            nu_L    basic frequency of the laser field
            t       current time value
            pcos    maximum frequency multiplier of the cos set (if applicable)
            OUTPUT
            E_omega complex value of a high-frequency part for current external laser field  """

        #E_omega = 0.0
        #for p in range(-pcos, pcos + 1):
        #    E_omega += math.cos(2.0**(p + 1) * math.pi * nu_L * t)

        E_omega = math.cos(2.0 * math.pi * nu_L * t)
        for p in range(2, math.floor(pcos) + 1):
            E_omega += math.cos(2.0 * math.pi * nu_L * t * p)
            E_omega += math.cos(2.0 * math.pi * nu_L * t / p)

        return E_omega


"""
This class contains all the possible wavefunction types

This class is "module-private". It may/should be used only among Task Manager implementations
If you want to call a wavefunction implementation from somewhere except a Task Manager, you 
are probably wrong :) 
"""


class _PsiFunctions:
    @staticmethod
    def zero(np):
        return numpy.array([0.0] * np).astype(complex)

    @staticmethod
    def one(x, np, x0, p0, m, De, a, L):
        return numpy.array([1.0 / math.sqrt(L)] * np).astype(complex)

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
             in x]).astype(complex)

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
            [math.sqrt(a / math.gamma(arg)) * math.exp(-yi / 2.0) * pow(yi, float(arg / 2.0)) for yi in y]).astype(complex)

        return psi


"""
The implementations of this interface set up the task. That includes defining the starting
conditions, the goal, the potential, and all the possible other parameters necessary to define the task.

For example: The calculating and fitting lasses should NOT be aware of if they are solving 
Morse or Harmonic problem. It lays on this class. 
"""


class TaskManager:
    def __init__(self, wf_type: TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: TaskRootConfiguration.FitterConfiguration):
        if wf_type == TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType.MORSE:
            print("Morse wavefunctions are used")
            self.psi_init_impl = _PsiFunctions.morse
        elif wf_type == TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType.HARMONIC:
            print("Harmonic wavefunctions are used")
            self.psi_init_impl = _PsiFunctions.harmonic
        elif wf_type == TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType.CONST:
            print("Constant wavefunctions are used")
            self.psi_init_impl = _PsiFunctions.one
        else:
            raise RuntimeError("Impossible case in the WaveFuncType class")

        if conf_fitter.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.ZERO:
            print("Calculation without laser field")
            self.lf_init_guess = _LaserFields.zero
        elif conf_fitter.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.GAUSS:
            print("Gaussian type of initial guess for the laser field envelope is used")
            self.lf_init_guess = _LaserFields.laser_field_gauss
        elif conf_fitter.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.SQRSIN:
            print("Squared sinus type of initial guess for the laser field envelope is used")
            self.lf_init_guess = _LaserFields.laser_field_sqrsin
        elif conf_fitter.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.MAXWELL:
            print("Maxwell distribution-like type of initial guess for the laser field envelope is used")
            self.lf_init_guess = _LaserFields.laser_field_maxwell
        else:
            raise RuntimeError("Impossible case in the InitGuess class")

        if conf_fitter.init_guess_hf == TaskRootConfiguration.FitterConfiguration.InitGuessHf.EXP:
            if not conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.FILTERING and \
               not conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.SINGLE_POT:
                print("Exponential high-frequency part of initial guess for the laser field is used")
            self.lf_init_guess_hf = _LaserFieldsHighFrequencyPart.cexp
        elif conf_fitter.init_guess_hf == TaskRootConfiguration.FitterConfiguration.InitGuessHf.COS:
            print("Cos-like high-frequency part of initial guess for the laser field with frequency multiplier "
                  "'pcos' = %f is used" % conf_fitter.pcos)
            self.lf_init_guess_hf = _LaserFieldsHighFrequencyPart.cos
        elif conf_fitter.init_guess_hf == TaskRootConfiguration.FitterConfiguration.InitGuessHf.COS_SET:
            print("A sequence of cos-type terms with 'pcos' = %f as the high-frequency part of initial guess "
                  "for the laser field is used. The maximum frequency multiplier equal to floor(pcos) will be used"
                  % conf_fitter.pcos)
            if not conf_fitter.pcos > 1.0:
                raise ValueError("The maximum frequency multiplier in the high frequency part of the laser field, "
                                 "'pcos', has to be > 1")
            self.lf_init_guess_hf = _LaserFieldsHighFrequencyPart.cos_set
        else:
            raise RuntimeError("Impossible case in the InitGuessHf class")

        if conf_fitter.propagation.hamil_type == TaskRootConfiguration.FitterConfiguration.HamilType.NTRIV:
            print("Non-trivial type of the Hamiltonian is used")
            print("Number of 2-level basis vectors 'nb' = 1 is used")
            self.ntriv = 1
            if not conf_fitter.init_guess_hf == TaskRootConfiguration.FitterConfiguration.InitGuessHf.EXP:
                raise RuntimeError("For a non-trivial Hamiltonian an exponential high-frequency part of initial guess for the laser field has to be used!")
        elif conf_fitter.propagation.hamil_type == TaskRootConfiguration.FitterConfiguration.HamilType.BH_MODEL:
            print("Bose-Hubbard Hamiltonian is used")
            print("Number of %d-level basis vectors 'nb' = %d is used" % (conf_fitter.nb, conf_fitter.nb))
            self.ntriv = -1
        elif conf_fitter.propagation.hamil_type == TaskRootConfiguration.FitterConfiguration.HamilType.TWO_LEVELS:
            print("Simple trivial two-levels type of the Hamiltonian is used")
            print("Number of 2-level basis vectors 'nb' = 2 is used")
            self.ntriv = 0
            if not conf_fitter.nb == 2:
                raise RuntimeError("Number of basis vectors 'nb' for 'hamil_type' = 'two_levels' has to be equal to 2!")
        else:
            raise RuntimeError("Impossible case in the HamilType class")

        self.conf_fitter = conf_fitter
        self.init_dir = PropagationSolver.Direction.FORWARD


    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb):
        raise NotImplementedError()

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb):
        raise NotImplementedError()

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du, nu_L):
        raise NotImplementedError()

    def laser_field(self, E0, t, t0, sigma):
        return self.lf_init_guess(E0, t, t0, sigma)

    def laser_field_hf(self, nu_L, t, pcos):
        return self.lf_init_guess_hf(nu_L, t, pcos)


class HarmonicSingleStateTaskManager(TaskManager):
    def __init__(self, wf_type: TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: TaskRootConfiguration.FitterConfiguration):
        super().__init__(wf_type, conf_fitter)

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
        v.append((0.0, v_l))
        return v

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du, nu_L):
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
            nu_l    basic frequency of the laser field in Hz (dummy variable)

            OUTPUT
            v       real vector of length np describing the dimensionless potential V(X) """

        # Lower harmonic potential
        v = HarmonicSingleStateTaskManager._pot_level1(x, m, a)

        # Upper harmonic potential
        v_u = numpy.array([0.0] * np)
        v.append((0.0, v_u))

        return v

    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb) -> PsiBasis:
        psi_init_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_init_obj.psis[vec].f[0] = self.psi_init_impl(x, np, x0, p0, m, De, a, L)
            psi_init_obj.psis[vec].f[1] = _PsiFunctions.zero(np)
        return psi_init_obj

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb) -> PsiBasis:
        psi_goal_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_goal_obj.psis[vec].f[0] = _PsiFunctions.harmonic(x, np, x0, p0, m, De, a, L)
            psi_goal_obj.psis[vec].f[1] = _PsiFunctions.zero(np)
        return psi_goal_obj


class HarmonicMultipleStateTaskManager(HarmonicSingleStateTaskManager):
    def __init__(self, wf_type: TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: TaskRootConfiguration.FitterConfiguration):
        super().__init__(wf_type, conf_fitter)

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du, nu_L):
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
            nu_l    basic frequency of the laser field in Hz (dummy variable)

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

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb) -> PsiBasis:
        psi_goal_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_goal_obj.psis[vec].f[0] = _PsiFunctions.zero(np)
            psi_goal_obj.psis[vec].f[1] = _PsiFunctions.harmonic(x, np, x0p + x0, p0, m, De_e, a_e, L)
        return psi_goal_obj


class MorseSingleStateTaskManager(TaskManager):
    def __init__(self, wf_type: TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: TaskRootConfiguration.FitterConfiguration):
        super().__init__(wf_type, conf_fitter)

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
        v.append((0.0, v_l))

        return v

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du, nu_L):
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
            nu_l    basic frequency of the laser field in Hz (dummy variable)

            OUTPUT
            v       a list of real vectors of length np describing the potentials V_u(X) and V_l(X) """

        # Lower morse potential
        v = MorseSingleStateTaskManager._pot_level1(x, m, De, a)

        # Upper morse potential
        v_u = numpy.array([0.0] * np)
        v.append((0.0, v_u))

        return v

    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb) -> PsiBasis:
        psi_init_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_init_obj.psis[vec].f[0] = self.psi_init_impl(x, np, x0, p0, m, De, a, L)
            psi_init_obj.psis[vec].f[1] = _PsiFunctions.zero(np)
        return psi_init_obj

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb) -> PsiBasis:
        psi_goal_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_goal_obj.psis[vec].f[0] = _PsiFunctions.morse(x, np, x0, p0, m, De, a, L)
            psi_goal_obj.psis[vec].f[1] = _PsiFunctions.zero(np)
        return psi_goal_obj


class MorseMultipleStateTaskManager(MorseSingleStateTaskManager):
    def __init__(self, wf_type: TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: TaskRootConfiguration.FitterConfiguration):
        super().__init__(wf_type, conf_fitter)

    @staticmethod
    def _pot(x, np, m, De, a, x0p, De_e, a_e, Du, nu_L):
        # Lower morse potential
        v = MorseSingleStateTaskManager._pot_level1(x, m, De, a)

        # Upper morse potential
        v_u = numpy.array(
            [De_e * (1.0 - math.exp(-a_e * (xi - x0p))) * (1.0 - math.exp(-a_e * (xi - x0p))) + Du for xi in x])
        v.append((Du, v_u))

        return v

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du, nu_L):
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
            nu_l        basic frequency of the laser field in Hz (dummy variable)

            OUTPUT
            v       a list of real vectors of length np describing the potentials V_u(X) and V_l(X) """

        return self._pot(x, np, m, De, a, x0p, De_e, a_e, Du, nu_L)

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb) -> PsiBasis:
        psi_goal_obj = PsiBasis(nb)
        for vec in range(nb):
            psi_goal_obj.psis[vec].f[0] = _PsiFunctions.zero(np)
            psi_goal_obj.psis[vec].f[1] = _PsiFunctions.morse(x, np, x0p + x0, p0, m, De_e, a_e, L)
        return psi_goal_obj


class MultipleStateUnitTransformTaskManager(MorseMultipleStateTaskManager):
    def __init__(self, wf_type: TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: TaskRootConfiguration.FitterConfiguration):
        super().__init__(wf_type, conf_fitter)
        self.init_dir = PropagationSolver.Direction.BACKWARD

    @staticmethod
    def _quant_fourier_transform(nb):
        omega = cmath.exp(1j * 2.0 * math.pi / nb)
        F = numpy.zeros((nb, nb)).astype(complex)
        for v1 in range(nb):
            for v2 in range(nb):
                F.itemset((v1, v2), omega**(v1 * v2) / math.sqrt(nb))

        return F

    @staticmethod
    def _matrix_PsiBasis_mult(F: numpy.matrix, psi: PsiBasis, nb, np):
        phi: PsiBasis = PsiBasis(nb, nb)
        for bv in range(nb):
            for gl in range(nb):
                phi_gl = numpy.array([complex(0.0, 0.0)] * np)
                for il in range(nb):
                    F_psi_el_mult = F.item(gl, il) * psi.psis[bv].f[il]
                    phi_gl = numpy.add(phi_gl, F_psi_el_mult)
                phi.psis[bv].f[gl] = phi_gl

        return phi

    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb) -> PsiBasis:
        psi_init_obj = PsiBasis(nb, nb)

        for vect in range(nb):
            for vect1 in range(nb):
                if vect1 == vect:
                    psi_init_obj.psis[vect].f[vect1] = self.psi_init_impl(x, np, x0, p0, m, De, a, L)
                else:
                    psi_init_obj.psis[vect].f[vect1] = _PsiFunctions.zero(np)

        return psi_init_obj

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb) -> PsiBasis:
        F = self._quant_fourier_transform(nb)
        psi = self.psi_init(x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L, nb)
        phi = self._matrix_PsiBasis_mult(F, psi, nb, np)

        return phi

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du, nu_L):
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
            nu_l    basic frequency of the laser field in Hz

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

        elif self.ntriv == -1:
            #h_cm = phys_base.Red_Planck_h / phys_base.cm_to_erg  # s * cm^-1
            U = self.conf_fitter.propagation.U # U ~ 1 / cm   # * h_cm * h_cm  # U ~ cm / s^2
            Emax = self.conf_fitter.propagation.E0 * self.conf_fitter.Em # / phys_base.Hz_to_cm * h_cm
            l = (self.conf_fitter.nb - 1) / 2.0
            #delta_E = nu_L * phys_base.Hz_to_cm

            #U = delta_E / l

            # Maximum and minimum energies achieved during the calculation
            vmax = 2.0 * U * l**2 + 2.0 * Emax * l
            vmin = -2.0 * Emax * l

            # The lowest and the highest level energies
            #v_min = numpy.array([0.0] * np)
            #v_max = numpy.array([U * l**2] * np)
            #v.append((vmin, v_min))

            for n in range(self.conf_fitter.nb):
                #vn = -2.0 * Emax * (l - n)
                #v_n = numpy.array([U * (l - n)**2] * np)
                vmax_list = numpy.array([vmax] * np)
                v.append((vmin, vmax_list))
            #v.append((vmax, v_max))
#            for n in range(self.conf_fitter.nb):
#                print(v[n][1][0])
        else:
            raise RuntimeError("Unsupported type of potential!")

        return v


def create(conf_fitter: TaskRootConfiguration.FitterConfiguration):
    if conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.FILTERING or \
       conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.SINGLE_POT:

        task_manager_imp: TaskManager
        if conf_fitter.propagation.hamil_type != TaskRootConfiguration.FitterConfiguration.HamilType.NTRIV:
            raise RuntimeError("For 'task_type' = 'single_pot' and 'filtering' the Hamiltonian type 'hamil_type' = 'ntriv' should be specified!")
        if conf_fitter.nb != 1:
            raise RuntimeError("Number of basis vectors 'nb' for 'task_type' = 'single_pot' and 'filtering' should be equal to 1!")
        if conf_fitter.propagation.pot_type == TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.MORSE:
            print("Morse potentials are used")
            task_manager_imp = MorseSingleStateTaskManager(conf_fitter.propagation.wf_type, conf_fitter)
        elif conf_fitter.propagation.pot_type == TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.HARMONIC:
            print("Harmonic potentials are used")
            task_manager_imp = HarmonicSingleStateTaskManager(conf_fitter.propagation.wf_type, conf_fitter)
        else:
            raise RuntimeError("Impossible PotentialType")
    else:
        if conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
            task_manager_imp = MultipleStateUnitTransformTaskManager(conf_fitter.propagation.wf_type, conf_fitter)
            if conf_fitter.nb <= 1:
                raise RuntimeError(
                    "Number of basis vectors 'nb' for 'task_type' = 'optimal_control_unit_transform' should be more than 1!")
            if conf_fitter.propagation.pot_type == TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.NONE:
                print("No potentials is used")
            else:
                raise RuntimeError("Impossible PotentialType for the unitary transformation task")
        else:
            if conf_fitter.propagation.hamil_type != TaskRootConfiguration.FitterConfiguration.HamilType.NTRIV:
                raise RuntimeError(
                    "For 'task_type' = 'trans_wo_control', 'intuitive_control', 'local_control_population', "
                        "'local_control_projection', and 'optimal_control_krotov' the Hamiltonian type 'hamil_type' = 'ntriv' should be specified!")
            if conf_fitter.nb != 1:
                raise RuntimeError(
                        "Number of basis vectors 'nb' for 'task_type' = 'trans_wo_control', 'intuitive_control', 'local_control_population', "
                        "'local_control_projection', and 'optimal_control_krotov' should be equal to 1!")
            if conf_fitter.propagation.pot_type == TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.MORSE:
                print("Morse potentials are used")
                task_manager_imp = MorseMultipleStateTaskManager(conf_fitter.propagation.wf_type, conf_fitter)
            elif conf_fitter.propagation.pot_type == TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.HARMONIC:
                print("Harmonic potentials are used")
                task_manager_imp = HarmonicMultipleStateTaskManager(conf_fitter.propagation.wf_type, conf_fitter)
            else:
                raise RuntimeError("Impossible PotentialType")

    return task_manager_imp
