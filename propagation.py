import math
import cmath
import copy
from enum import Enum
import datetime
from typing import Callable, List, Optional

import numpy

import math_base
import phys_base
from psi_basis import Psi
from reporter import PropagationReporter


class PropagationSolver:
    # These values are calculated once and forever
    # They should NEVER change
    class StaticState:
        psi0: Psi
        psif: Psi

        def __init__(self, psi0: Psi, psif: Psi, moms0: phys_base.ExpectationValues,
                     cnorm0: List[complex],
                     cnormf: List[complex],
                     cener0: List[complex],
                     cenerf: List[complex],
                     overlp00: List[complex],
                     overlpf0: List[complex],
                     dt: float = 0.0, dx: float = 0.0,
                     x: numpy.ndarray = numpy.array(0), v=None, akx2=None):
            assert (psi0 is None and psif is None) or (psi0.f[0] is not psif.f[0] and psi0.f[1] is not psif.f[1]), \
                "A single array is passed twice (as psi0 and psif). Clone it!"

            self.psi0 = psi0
            self.psif = psif
            self.moms0 = moms0
            self.cnorm0 = cnorm0
            self.cnormf = cnormf
            self.cener0 = cener0
            self.cenerf = cenerf
            self.overlp00 = overlp00
            self.overlpf0 = overlpf0
            self.dt = dt
            self.dx = dx
            self.x = x
            self.v = v
            self.akx2 = akx2

    # These parameters are updated on each calculation step
    class DynamicState:
        psi: Psi
        psi_omega: Psi

        def __init__(self, l=0, t=0.0, psi: Psi = Psi(None), psi_omega: Psi = Psi(None),
                     E=0.0, freq_mult=1.0, dir=None):
            assert (psi is None and psi_omega is None) or \
                   (psi.f[0] is not psi_omega.f[0] and psi.f[1] is not psi_omega.f[1]), \
                "A single array is passed twice (as psi and psi_omega). Clone it!"

            self.l = l
            self.t = t
            self.psi = psi
            self.psi_omega = psi_omega
            self.E = E
            self.freq_mult = freq_mult
            self.dir = dir

    # These parameters are recalculated from scratch on each step,
    # and then follows an output of them to the user
    class InstrumentationOutputData:
        psigc_psie: complex

        def __init__(self, moms: phys_base.ExpectationValues, cnorm, psigc_psie: complex, psigc_dv_psie: complex,
                     cener: list[complex], E_full, overlp0, overlpf, emax, emin, t_sc, time_before, time_after):
            self.moms = moms
            self.cnorm = cnorm
            self.psigc_psie = psigc_psie
            self.psigc_dv_psie = psigc_dv_psie
            self.cener = cener
            self.E_full = E_full
            self.overlp0 = overlp0
            self.overlpf = overlpf
            self.emax = emax
            self.emin = emin
            self.t_sc = t_sc
            self.time_before = time_before
            self.time_after = time_after

    # The user's decision to correct/iterate or not to correct/iterate a step
    class StepReaction(Enum):
        OK = 0
        CORRECT = 1
        ITERATE = 2

    class Direction(Enum):
        FORWARD = 1
        BACKWARD = -1

    stat: Optional[StaticState]
    dyn: Optional[DynamicState]
    instr: Optional[InstrumentationOutputData]
    freq_multiplier: Optional[Callable[[DynamicState, StaticState], float]]

    def __init__(
            self,
            pot,
            _warning_collocation_points,
            _warning_time_steps,
            reporter: PropagationReporter,
            laser_field_envelope,
            freq_multiplier: Callable[[DynamicState, StaticState], float],
            dynamic_state_factory,
            mod_log,
            conf_prop):
        self.milliseconds_full = 0.0
        self.pot = pot
        self._warning_collocation_points = _warning_collocation_points
        self._warning_time_steps = _warning_time_steps
        self.reporter = reporter
        self.laser_field_envelope = laser_field_envelope
        self.freq_multiplier = freq_multiplier
        self.dynamic_state_factory = dynamic_state_factory
        self.mod_log = mod_log

        self.m = conf_prop.m
        self.L = conf_prop.L
        self.np = conf_prop.np
        self.nch = conf_prop.nch
        self.T = conf_prop.T
        self.nt = conf_prop.nt
        self.x0 = conf_prop.x0
        self.p0 = conf_prop.p0
        self.a = conf_prop.a
        self.De = conf_prop.De
        self.x0p = conf_prop.x0p
        self.a_e = conf_prop.a_e
        self.De_e = conf_prop.De_e
        self.Du = conf_prop.Du
        self.E0 = conf_prop.E0
        self.t0 = conf_prop.t0
        self.sigma = conf_prop.sigma
        self.nu_L = conf_prop.nu_L

        self.stat = None
        self.dyn = None
        self.instr = None

    @staticmethod
    def _norm_eval(psi: Psi, dx, np):
        cnorm = []
        cnorm.append(math_base.cprod(psi.f[0], psi.f[0], dx, np))
        cnorm.append(math_base.cprod(psi.f[1], psi.f[1], dx, np))
        return cnorm

    @staticmethod
    def _ener_eval(psi: Psi, v, akx2, dx, np):
        cener = []

        phi_l = phys_base.hamil_cpu(psi.f[0], v[0][1], akx2, np)
        cener.append(math_base.cprod(phi_l, psi.f[0], dx, np))

        phi_u = phys_base.hamil_cpu(psi.f[1], v[1][1], akx2, np)
        cener.append(math_base.cprod(phi_u, psi.f[1], dx, np))

        return cener

    @staticmethod
    def _pop_eval(psi_goal: Psi, psi: Psi, dx, np):
        overlp = []
        overlp.append(math_base.cprod(psi_goal.f[0], psi.f[0], dx, np))
        overlp.append(math_base.cprod(psi_goal.f[1], psi.f[1], dx, np))

        return overlp

    def report_static(self):
        # check if input data are correct in terms of the given problem
        # calculating the initial energy range of the Hamiltonian operator H
        emax0 = self.stat.v[0][1][0] + abs(self.stat.akx2[int(self.np / 2 - 1)]) + 2.0
        emin0 = self.stat.v[0][0]

        # calculating the initial minimum number of collocation points that is needed for convergence
        np_min0 = int(
            math.ceil(self.L * math.sqrt(
                      2.0 * self.m * (emax0 - emin0) * phys_base.dalt_to_au / phys_base.hart_to_cm) / math.pi
                      )
        )

        # calculating the initial minimum number of time steps that is needed for convergence
        nt_min0 = int(
            math.ceil((emax0 - emin0) * self.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h)
        )

        if self.np < np_min0:
            if self._warning_collocation_points:
                self._warning_collocation_points(self.np, np_min0)
        if self.nt < nt_min0:
            if self._warning_time_steps:
                self._warning_time_steps(self.nt, nt_min0)

        cener0_tot = self.stat.cener0[0] + self.stat.cener0[1]
        overlp0 =  self.stat.overlp00[0] + self.stat.overlp00[1]
        overlpf = self.stat.overlpf0[0] + self.stat.overlpf0[1]
        overlp0_abs = abs(overlp0) + abs(overlpf)
        max_ind_psi_l = numpy.argmax(self.stat.psi0.f[0])
        max_ind_psi_u = numpy.argmax(self.stat.psi0.f[1])

        fm_start = 1.0

        # plotting initial values
        self.reporter.print_time_point_prop(self.dyn.l, self.stat.psi0, self.dyn.t, self.stat.x, self.np, self.stat.moms0,
                                       self.stat.cener0[0].real, self.stat.cener0[1].real,
                                       overlp0, overlpf, overlp0_abs, cener0_tot.real,
                                       abs(self.stat.psi0.f[0][max_ind_psi_l]), self.stat.psi0.f[0][max_ind_psi_l].real,
                                       abs(self.stat.psi0.f[1][max_ind_psi_u]), self.stat.psi0.f[1][max_ind_psi_u].real,
                                       abs(self.dyn.E), fm_start)

        print("Initial emax = ", emax0)

        print(" Initial state features: ")
        print("Initial normalization (ground state): ", abs(self.stat.cnorm0[0]))
        print("Initial energy (ground state): ", abs(self.stat.cener0[0]))
        print("Initial normalization (excited state): ", abs(self.stat.cnorm0[1]))
        print("Initial energy (excited state): ", abs(self.stat.cener0[1]))

        print(" Final goal features: ")
        print("Final goal normalization (ground state): ", abs(self.stat.cnormf[0]))
        print("Final goal energy (ground state): ", abs(self.stat.cenerf[0]))
        print("Final goal normalization (excited state): ", abs(self.stat.cnormf[1]))
        print("Final goal energy (excited state): ", abs(self.stat.cenerf[1]))


    def report_dynamic(self):
        # calculating the minimum number of collocation points and time steps that are needed for convergence
        nt_min = int(math.ceil(
            (self.instr.emax - self.instr.emin) * self.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h))
        np_min = int(math.ceil(
            self.L * math.sqrt(2.0 * self.m *
                    (self.instr.emax - self.instr.emin) * phys_base.dalt_to_au / phys_base.hart_to_cm) / math.pi))

        cener_tot = self.instr.cener[0] + self.instr.cener[1]
        overlp0 = self.instr.overlp0[0] + self.instr.overlp0[1]
        overlpf = self.instr.overlpf[0] + self.instr.overlpf[1]
        overlp_abs = abs(overlp0) + abs(overlpf)

        time_span = self.instr.time_after - self.instr.time_before
        milliseconds_per_step = time_span.microseconds / 1000
        self.milliseconds_full += milliseconds_per_step

        max_ind_psi_l = numpy.argmax(abs(self.dyn.psi.f[0]))
        max_ind_psi_u = numpy.argmax(abs(self.dyn.psi.f[1]))

        self.reporter.print_time_point_prop(self.dyn.l, self.dyn.psi, self.dyn.t, self.stat.x, self.np,
                                            self.instr.moms, self.instr.cener[0].real, self.instr.cener[1].real,
                                            overlp0, overlpf, overlp_abs, cener_tot.real,
                                            abs(self.dyn.psi.f[0][max_ind_psi_l]), self.dyn.psi.f[0][max_ind_psi_l].real,
                                            abs(self.dyn.psi.f[1][max_ind_psi_u]), self.dyn.psi.f[1][max_ind_psi_u].real,
                                            abs(self.dyn.E), self.dyn.freq_mult)

        if self.dyn.l % self.mod_log == 0:
            if self.np < np_min:
                if self._warning_collocation_points:
                    self._warning_collocation_points(self.np, np_min)
            if self.nt < nt_min:
                if self._warning_time_steps:
                    self._warning_time_steps(self.nt, nt_min)

            print("l = ", self.dyn.l)
            print("t = ", self.dyn.t * 1e15, "fs")

            print("normalized scaled time interval = ", self.instr.t_sc)
            print("normalization on the ground state = ", abs(self.instr.cnorm[0]))
            print("energy on the ground state = ", self.instr.cener[0].real)
            print("overlap with initial wavefunction = ", abs(overlp0))
            print("overlap with final goal wavefunction = ", abs(overlpf))

            print(
                "milliseconds per step: " + str(milliseconds_per_step) + ", on average: " + str(
                    self.milliseconds_full / self.dyn.l))


    def start(self, dx, x, psi0, psif, dir: Direction):
        # evaluating of potential(s)
        v = self.pot(x, self.np, self.m, self.De, self.a, self.x0p, self.De_e, self.a_e, self.Du)

        # evaluating of k vector
        akx2 = math_base.initak(self.np, dx, 2)

        # evaluating of kinetic energy
        akx2 *= -phys_base.hart_to_cm / (2.0 * self.m * phys_base.dalt_to_au)

        # initial normalization check
        cnorm0 = self._norm_eval(psi0, dx, self.np)

        # calculating of initial ground/excited energies
        cener0 = self._ener_eval(psi0, v, akx2, dx, self.np)

        # final normalization check
        cnormf = self._norm_eval(psif, dx, self.np)

        # calculating of final excited/filtered energy
        cenerf = self._ener_eval(psif, v, akx2, dx, self.np)

        # time propagation
        dt = dir.value * self.T / self.nt
        psi = copy.deepcopy(psi0)

        # initial population
        overlp00 = self._pop_eval(psi0, psi, dx, self.np)
        overlpf0 = self._pop_eval(psif, psi, dx, self.np)

        # calculating of initial expectation values
        moms0 = phys_base.exp_vals_calc(psi, x, akx2, dx, self.np, self.m)

        self.stat = PropagationSolver.StaticState(psi0, psif, moms0, cnorm0, cnormf,
                     cener0, cenerf, overlp00, overlpf0, dt, dx, x, v, akx2)

        if dir == PropagationSolver.Direction.FORWARD:
            self.dyn = self.dynamic_state_factory(0, 0.0, psi, psi, 0.0, 1.0, dir)
        else:
            self.dyn = self.dynamic_state_factory(0, self.T, psi, psi, 0.0, 1.0, dir)

        # Calculating the initial field value
        self.dyn.E = self.laser_field_envelope(self, self.stat, self.dyn)

        self.report_static()

        self.dyn.l = 1


    def step(self, t_start):
        time_before = datetime.datetime.now()

        emax_list = []
        emin_list = []
        # calculating limits of energy ranges of the one-dimensional Hamiltonian operator H_l
        emax_list.append(self.stat.v[0][1][0] + abs(self.stat.akx2[int(self.np / 2 - 1)]) + 2.0)
        emin_list.append(self.stat.v[0][0])
        # calculating limits of energy ranges of the one-dimensional Hamiltonian operator H_u
        emax_list.append(self.stat.v[1][1][0] + abs(self.stat.akx2[int(self.np / 2 - 1)]) + 2.0)
        emin_list.append(self.stat.v[1][0])

        self.dyn.t = self.stat.dt * self.dyn.l + t_start

        self.dyn.freq_mult = self.freq_multiplier(self.dyn, self.stat)

        # Here we're transforming the problem to the one for psi_omega
        exp_L = cmath.exp(1j * math.pi * self.nu_L * self.dyn.freq_mult * self.dyn.t)
        psi_omega_l = self.dyn.psi.f[0] / exp_L
        self.dyn.psi_omega.f[0][:] = psi_omega_l[:]
        psi_omega_u = self.dyn.psi.f[1] * exp_L
        self.dyn.psi_omega.f[1][:] = psi_omega_u[:]

        # New energy ranges
        eL = self.nu_L * self.dyn.freq_mult * phys_base.Hz_to_cm / 2.0
        emax_omega = []
        emin_omega = []

        emax_omega.append(emax_list[0] + self.E0 + eL)
        emin_omega.append(emin_list[0] - self.E0 + eL)

        emax_omega.append(emax_list[1] + self.E0 - eL)
        emin_omega.append(emin_list[1] - self.E0 - eL)

        emax = max(emax_omega[0], emin_omega[0], emax_omega[1], emin_omega[1])
        emin = min(emax_omega[0], emin_omega[0], emax_omega[1], emin_omega[1])

        t_sc = self.stat.dt * (emax - emin) * phys_base.cm_to_erg / 4.0 / phys_base.Red_Planck_h

        self.dyn.E = self.laser_field_envelope(self, self.stat, self.dyn)
        E_full = self.dyn.E * exp_L * exp_L

        self.dyn.psi_omega = Psi(f=phys_base.prop_cpu(self.dyn.psi_omega.f, t_sc, self.nch, self.np, self.stat.v, self.stat.akx2, emin, emax, self.dyn.E, eL), lvls=self.dyn.psi_omega.lvls())

        cnorm = []
        cnorm.append(math_base.cprod(self.dyn.psi_omega.f[0], self.dyn.psi_omega.f[0], self.stat.dx, self.np))
        cnorm.append(math_base.cprod(self.dyn.psi_omega.f[1], self.dyn.psi_omega.f[1], self.stat.dx, self.np))
        cnorm_sum = cnorm[0] + cnorm[1]

        # renormalization
        if abs(cnorm_sum) > 0.0:
            self.dyn.psi_omega.f[0] /= math.sqrt(abs(cnorm_sum))
            self.dyn.psi_omega.f[1] /= math.sqrt(abs(cnorm_sum))

        psigc_psie = math_base.cprod(self.dyn.psi_omega.f[1], self.dyn.psi_omega.f[0], self.stat.dx, self.np)
        psigc_dv_psie = math_base.cprod3(self.dyn.psi_omega.f[1], self.stat.v[0][1] - self.stat.v[1][1], self.dyn.psi_omega.f[0], self.stat.dx, self.np)

        # converting back to psi
        self.dyn.psi.f[0] = self.dyn.psi_omega.f[0] * exp_L
        self.dyn.psi.f[1] = self.dyn.psi_omega.f[1] / exp_L

        # calculating of a current energy
        phi = phys_base.hamil2D_orig(self.dyn.psi.f, self.stat.v, self.stat.akx2, self.np, E_full)

        cener = []
        cener.append(math_base.cprod(phi[0], self.dyn.psi.f[0], self.stat.dx, self.np))
        cener.append(math_base.cprod(phi[1], self.dyn.psi.f[1], self.stat.dx, self.np))

        overlp0 = self._pop_eval(self.stat.psi0, self.dyn.psi, self.stat.dx, self.np)
        overlpf = self._pop_eval(self.stat.psif, self.dyn.psi, self.stat.dx, self.np)

        # calculating of expectation values
        moms = phys_base.exp_vals_calc(self.dyn.psi, self.stat.x, self.stat.akx2, self.stat.dx, self.np, self.m)

        time_after = datetime.datetime.now()

        self.instr = PropagationSolver.InstrumentationOutputData(moms, cnorm, psigc_psie, psigc_dv_psie,
                     cener, E_full, overlp0, overlpf, emax, emin, t_sc, time_before, time_after)

        self.report_dynamic()

        self.dyn.l += 1

        if self.dyn.l <= self.nt:
            return True
        else:
            return False


    @property
    def time_step(self):
        return abs(self.stat.dt)