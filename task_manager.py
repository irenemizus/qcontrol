import cmath
import math
import phys_base
import numpy
from config import RootConfiguration

from phys_base import dalt_to_au, hart_to_cm

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
    def harmonic(x, np, x0, p0, m, De, a):
        """ Initial wave function generator
            INPUT
            x           vector of length np defining positions of grid points
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
    def morse(x, np, x0, p0, m, De, a):
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
    def __init__(self, wf_type: RootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: RootConfiguration.FitterConfiguration):
        if wf_type == RootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType.MORSE:
            print("Morse wavefunctions are used")
            self.psi_init_impl = _PsiFunctions.morse
        elif wf_type == RootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType.HARMONIC:
            print("Harmonic wavefunctions are used")
            self.psi_init_impl = _PsiFunctions.harmonic
        else:
            raise RuntimeError("Impossible case in the WaveFuncType class")

        self.conf_fitter = conf_fitter

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e):
        raise NotImplementedError()

    def ener_goal(self, psif, v, akx2, np):
        phif = []
        phif.append(phys_base.hamil(psif[0], v[0][1], akx2, np))
        phif.append(phys_base.hamil(psif[1], v[1][1], akx2, np))
        return phif

    def pot(self, x, np, m, De, a, x0p, De_e, a_e, Du):
        raise NotImplementedError()

    def psi_init(self, x, np, x0, p0, m, De, a):
        return [self.psi_init_impl(x, np, x0, p0, m, De, a), _PsiFunctions.zero(np)]


class HarmonicSingleStateTaskManager(TaskManager):
    def __init__(self, wf_type: RootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: RootConfiguration.FitterConfiguration):
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
        v_u = numpy.array([0.0] * np)
        v.append((0.0, v_u))

        return v

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e):
        return [_PsiFunctions.harmonic(x, np, x0, p0, m, De, a), _PsiFunctions.zero(np)]


class HarmonicMultipleStateTaskManager(HarmonicSingleStateTaskManager):
    def __init__(self, wf_type: RootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: RootConfiguration.FitterConfiguration):
        super().__init__(wf_type, conf_fitter)

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

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e):
        return [_PsiFunctions.zero(np), _PsiFunctions.harmonic(x, np, x0p + x0, p0, m, De_e, a_e)]


class MorseSingleStateTaskManager(TaskManager):
    def __init__(self, wf_type: RootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: RootConfiguration.FitterConfiguration):
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
        v_u = numpy.array([0.0] * np)
        v.append((0.0, v_u))

        return v


    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e):
        return [_PsiFunctions.morse(x, np, x0, p0, m, De, a), _PsiFunctions.zero(np)]


class MorseMultipleStateTaskManager(MorseSingleStateTaskManager):
    def __init__(self, wf_type: RootConfiguration.FitterConfiguration.PropagationConfiguration.WaveFuncType,
                 conf_fitter: RootConfiguration.FitterConfiguration):
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

    def psi_goal(self, x, np, x0, p0, x0p, m, De, De_e, Du, a, a_e):
        return [_PsiFunctions.zero(np), _PsiFunctions.morse(x, np, x0p + x0, p0, m, De_e, a_e)]


def create(conf_fitter: RootConfiguration.FitterConfiguration):
    if conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.FILTERING or \
        conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.SINGLE_POT:

        if conf_fitter.propagation.pot_type == RootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.MORSE:
            print("Morse potentials are used")
            task_manager_imp = MorseSingleStateTaskManager(conf_fitter.propagation.wf_type, conf_fitter)
        elif conf_fitter.propagation.pot_type == RootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.HARMONIC:
            print("Harmonic potentials are used")
            task_manager_imp = HarmonicSingleStateTaskManager(conf_fitter.propagation.wf_type, conf_fitter)
        else:
            raise RuntimeError("Impossible PotentialType")
    else:
        if conf_fitter.propagation.pot_type == RootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.MORSE:
            print("Morse potentials are used")
            task_manager_imp = MorseMultipleStateTaskManager(conf_fitter.propagation.wf_type, conf_fitter)
        elif conf_fitter.propagation.pot_type == RootConfiguration.FitterConfiguration.PropagationConfiguration.PotentialType.HARMONIC:
            print("Harmonic potentials are used")
            task_manager_imp = HarmonicMultipleStateTaskManager(conf_fitter.propagation.wf_type, conf_fitter)
        else:
            raise RuntimeError("Impossible PotentialType")

    return task_manager_imp