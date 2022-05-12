import cmath
import math
import phys_base
import numpy
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
        """ Calculates energy of external laser field impulse
            INPUT
            E0      amplitude value of the laser field energy envelope
            t0      initial time, when the laser field is switched on
            sigma   scaling parameter of the laser field envelope
            nu_L basic frequency of the laser field
            t       current time value
            OUTPUT
            E       complex value of current external laser field  """

        E = E0 * math.exp(-(t - t0) * (t - t0) / 2.0 / sigma / sigma)

        return E

    @staticmethod
    def laser_field_sqrsin(E0, t, t0, sigma):
        """ Calculates energy of external laser field impulse
            INPUT
            E0      amplitude value of the laser field energy envelope
            t0      initial time, when the laser field is switched on
            sigma   scaling parameter of the laser field envelope
            nu_L basic frequency of the laser field
            t       current time value
            OUTPUT
            E       complex value of current external laser field  """

        E = E0 * math.sin(2.0 * math.pi * (t - t0) / sigma) * math.sin(2.0 * math.pi * (t - t0) / sigma)

        return E

    @staticmethod
    def laser_field_maxwell(E0, t, t0, sigma):
        """ Calculates energy of external laser field impulse
            INPUT
            E0      amplitude value of the laser field energy envelope
            t0      initial time, when the laser field is switched on
            sigma   scaling parameter of the laser field envelope
            nu_L basic frequency of the laser field
            t       current time value
            OUTPUT
            E       complex value of current external laser field  """

        #E = E0 * t / sigma
        #E = E0 * (1.0 - t / sigma)
        E = E0 * t * t * math.exp(-(t - t0) * (t - t0) / 2.0 / sigma / sigma) / sigma / sigma

        return E


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

        self.conf_fitter = conf_fitter
        self.init_dir = PropagationSolver.Direction.FORWARD
        self.ntriv = 1

    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L):
        raise NotImplementedError()

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L):
        raise NotImplementedError()

    def ener_goal(self, psif: PsiBasis, v, akx2, np):
        n = len(psif)
        phif = []
        for i in range(n):
            phif.append([phys_base.hamil_cpu(psif.psis[i].f[0], v[0][1], akx2, np, self.ntriv),
                         phys_base.hamil_cpu(psif.psis[i].f[1], v[1][1], akx2, np, self.ntriv)])
        return phif

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du, nu_L):
        raise NotImplementedError()

    def laser_field(self, E0, t, t0, sigma):
        raise NotImplementedError()


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

    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L) -> PsiBasis:
        psi_init_obj = PsiBasis(1)
        psi_init_obj.psis[0].f[0] = self.psi_init_impl(x, np, x0, p0, m, De, a, L)
        psi_init_obj.psis[0].f[1] = _PsiFunctions.zero(np)
        return psi_init_obj

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L) -> PsiBasis:
        psi_goal_obj = PsiBasis(1)
        psi_goal_obj.psis[0].f[0] = _PsiFunctions.harmonic(x, np, x0, p0, m, De, a, L)
        psi_goal_obj.psis[0].f[1] = _PsiFunctions.zero(np)
        return psi_goal_obj

    def laser_field(self, E0, t, t0, sigma):
        return _LaserFields.zero(E0, t, t0, sigma)


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

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L) -> PsiBasis:
        psi_goal_obj = PsiBasis(1)
        psi_goal_obj.psis[0].f[0] = _PsiFunctions.zero(np)
        psi_goal_obj.psis[0].f[1] = _PsiFunctions.harmonic(x, np, x0p + x0, p0, m, De_e, a_e, L)
        return psi_goal_obj

    def laser_field(self, E0, t, t0, sigma):
        return _LaserFields.zero(E0, t, t0, sigma)


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

    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L) -> PsiBasis:
        psi_init_obj = PsiBasis(1)
        psi_init_obj.psis[0].f[0] = self.psi_init_impl(x, np, x0, p0, m, De, a, L)
        psi_init_obj.psis[0].f[1] = _PsiFunctions.zero(np)
        return psi_init_obj

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L) -> PsiBasis:
        psi_goal_obj = PsiBasis(1)
        psi_goal_obj.psis[0].f[0] = _PsiFunctions.morse(x, np, x0, p0, m, De, a, L)
        psi_goal_obj.psis[0].f[1] = _PsiFunctions.zero(np)
        return psi_goal_obj

    def laser_field(self, E0, t, t0, sigma):
        return _LaserFields.zero(E0, t, t0, sigma)


class MorseMultipleStateTaskManager(MorseSingleStateTaskManager):
    def __init__(self, wf_type: TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: TaskRootConfiguration.FitterConfiguration):
        super().__init__(wf_type, conf_fitter)

    @staticmethod
    def _pot(x, np, m, De, a, x0p, De_e, a_e, Du):
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
            nu_l    basic frequency of the laser field in Hz (dummy variable)

            OUTPUT
            v       a list of real vectors of length np describing the potentials V_u(X) and V_l(X) """

        return self._pot(x, np, m, De, a, x0p, De_e, a_e, Du)

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L) -> PsiBasis:
        psi_goal_obj = PsiBasis(1)
        psi_goal_obj.psis[0].f[0] = _PsiFunctions.zero(np)
        psi_goal_obj.psis[0].f[1] = _PsiFunctions.morse(x, np, x0p + x0, p0, m, De_e, a_e, L)
        return psi_goal_obj

    def laser_field(self, E0, t, t0, sigma):
        return _LaserFields.laser_field_gauss(E0, t, t0, sigma)


class MultipleStateUnitTransformTaskManager(MorseMultipleStateTaskManager):
    def __init__(self, wf_type: TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: TaskRootConfiguration.FitterConfiguration):
        super().__init__(wf_type, conf_fitter)
        self.init_dir = PropagationSolver.Direction.BACKWARD
        self.ntriv = 0

    def psi_init(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L) -> PsiBasis:
        psi_init_obj = PsiBasis(2)

        psi_init_obj.psis[0].f[0] = self.psi_init_impl(x, np, x0, p0, m, De, a, L) # self.psi_init_impl(x, np, x0, p0, m, De, a) #
        psi_init_obj.psis[0].f[1] = _PsiFunctions.zero(np)

        psi_init_obj.psis[1].f[0] = _PsiFunctions.zero(np)
        psi_init_obj.psis[1].f[1] = self.psi_init_impl(x, np, x0, p0, m, De, a, L) # self.psi_init_impl(x, np, x0p + x0, p0, m, De_e, a_e) #

        return psi_init_obj

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e, L) -> PsiBasis:
        psi_goal_obj = PsiBasis(2)

        psi_goal_obj.psis[0].f[0] = self.psi_init_impl(x, np, x0, p0, m, De, a, L) / math.sqrt(2.0) # self.psi_init_impl(x, np, x0, p0, m, De, a) / math.sqrt(2.0) #
        psi_goal_obj.psis[0].f[1] = self.psi_init_impl(x, np, x0, p0, m, De, a, L) / math.sqrt(2.0) # self.psi_init_impl(x, np, x0p + x0, p0, m, De_e, a_e) / math.sqrt(2.0) #

        psi_goal_obj.psis[1].f[0] = self.psi_init_impl(x, np, x0, p0, m, De, a, L) / math.sqrt(2.0) # self.psi_init_impl(x, np, x0, p0, m, De, a) / math.sqrt(2.0) #
        psi_goal_obj.psis[1].f[1] = -self.psi_init_impl(x, np, x0, p0, m, De, a, L) / math.sqrt(2.0) # -self.psi_init_impl(x, np, x0p + x0, p0, m, De_e, a_e) / math.sqrt(2.0) #

        return psi_goal_obj

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
        #Dl = 1521.6693
        Dl = 0.0
        # Lower morse potential
        v_l = numpy.array([Dl] * np)
        v.append((Dl, v_l))

        # Upper morse potential
        #D_u = nu_L * phys_base.Hz_to_cm
        D_u = Du + Dl
        v_u = numpy.array([D_u] * np)
        v.append((D_u, v_u))

        return v

    def laser_field(self, E0, t, t0, sigma):
        return _LaserFields.laser_field_maxwell(E0, t, t0, sigma) # laser_field_sqrsin


def create(conf_fitter: TaskRootConfiguration.FitterConfiguration):
    if conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.FILTERING or \
       conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.SINGLE_POT:

        task_manager_imp: TaskManager
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
            if conf_fitter.propagation.pot_type == TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.NONE:
                print("No potentials are used")
            else:
                raise RuntimeError("Impossible PotentialType for the unitary transformation task")
        elif conf_fitter.propagation.pot_type == TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.MORSE:
            print("Morse potentials are used")
            task_manager_imp = MorseMultipleStateTaskManager(conf_fitter.propagation.wf_type, conf_fitter)
        elif conf_fitter.propagation.pot_type == TaskRootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.HARMONIC:
            print("Harmonic potentials are used")
            task_manager_imp = HarmonicMultipleStateTaskManager(conf_fitter.propagation.wf_type, conf_fitter)
        else:
            raise RuntimeError("Impossible PotentialType")

    return task_manager_imp
