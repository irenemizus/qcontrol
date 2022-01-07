import math
import cmath
import copy
from enum import Enum
import datetime

import math_base
import phys_base


class PropagationSolver:
    # These values are calculated once and forever
    # They should NEVER change
    class StaticState:
        def __init__(self, psi0=None, psif=None, moms0: phys_base.ExpectationValues=None,
                     cnorm0=None, cnormf=None,
                     cener0=None, cenerf=None,
                     E00=0.0, overlp00=None, overlpf0=None,
                     dt=0.0, dx=0.0, x=None, v=None, akx2=None):
            assert (psi0 is None and psif is None) or (psi0[0] is not psif[0] and psi0[1] is not psif[1]), \
                "A single array is passed twice (as psi0 and psif). Clone it!"

            self.psi0 = psi0
            self.psif = psif
            self.moms0 = moms0
            self.cnorm0 = cnorm0
            self.cnormf = cnormf
            self.cener0 = cener0
            self.cenerf = cenerf
            self.E00 = E00
            self.overlp00 = overlp00
            self.overlpf0 = overlpf0
            self.dt = dt
            self.dx = dx
            self.x = x
            self.v = v
            self.akx2 = akx2


    # These parameters are updated on each calculation step
    class DynamicState:
        def __init__(self, l=0, t=0.0, psi=None, psi_omega=None, E=0.0, freq_mult = 1.0):
            assert (psi is None and psi_omega is None) or (psi[0] is not psi_omega[0] and psi[1] is not psi_omega[1] ), "A single array is passed twice (as psi and psi_omega). Clone it!"

            self.l = l
            self.t = t
            self.psi = psi
            self.psi_omega = psi_omega
            self.E = E
            self.freq_mult = freq_mult


    # These parameters are recalculated from scratch on each step,
    # and then follows an output of them to the user
    class InstrumentationOutputData:
        def __init__(self, moms: phys_base.ExpectationValues, cnorm, psigc_psie, psigc_dv_psie,
                     cener, E_full, overlp0, overlpf, emax, emin, t_sc, time_before, time_after):
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

    def __init__(
            self,
            pot,
            report_static,
            report_dynamic,
            process_instrumentation,
            laser_field_envelope,
            freq_multiplier,
            dynamic_state_factory,
            conf_prop):
        self.pot = pot
        self.report_static = report_static
        self.report_dynamic = report_dynamic
        self.process_instrumentation = process_instrumentation
        self.laser_field_envelope = laser_field_envelope
        self.freq_multiplier = freq_multiplier
        self.dynamic_state_factory = dynamic_state_factory

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


    @staticmethod
    def _norm_eval(psi, dx, np):
        cnorm = []
        cnorm.append(math_base.cprod(psi[0], psi[0], dx, np))
        cnorm.append(math_base.cprod(psi[1], psi[1], dx, np))
        return cnorm


    @staticmethod
    def _ener_eval(psi, v, akx2, dx, np):
        cener = []

        phi_l = phys_base.hamil(psi[0], v[0][1], akx2, np)
        cener.append(math_base.cprod(phi_l, psi[0], dx, np))

        phi_u = phys_base.hamil(psi[1], v[1][1], akx2, np)
        cener.append(math_base.cprod(phi_u, psi[1], dx, np))

        return cener


    @staticmethod
    def _pop_eval(psi_goal, psi, dx, np):
        overlp = []
        overlp.append(math_base.cprod(psi_goal[0], psi[0], dx, np))
        overlp.append(math_base.cprod(psi_goal[1], psi[1], dx, np))

        return overlp


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
        dt = dir.value * self.T / (self.nt - 1)
        psi = copy.deepcopy(psi0)

        # initial laser field energy
        E00 = phys_base.laser_field(self.E0, 0.0, self.t0, self.sigma)

        # initial population
        overlp00 = self._pop_eval(psi0, psi, dx, self.np)
        overlpf0 = self._pop_eval(psif, psi, dx, self.np)

        # calculating of initial expectation values
        moms0 = phys_base.exp_vals_calc(psi, x, akx2, dx, self.np, self.m)

        self.stat = PropagationSolver.StaticState(psi0, psif, moms0, cnorm0, cnormf,
                     cener0, cenerf, E00, overlp00, overlpf0, dt, dx, x, v, akx2)
        self.report_static(self.stat)

        self.dyn = self.dynamic_state_factory(0, 0.0, psi, psi, 0.0, 1.0)
        self.report_dynamic(self.dyn)

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

        self.dyn.freq_mult = self.freq_multiplier(self.stat)

        # Here we're transforming the problem to the one for psi_omega
        exp_L = cmath.exp(1j * math.pi * self.nu_L * self.dyn.freq_mult * self.dyn.t)
        psi_omega_l = self.dyn.psi[0] / exp_L
        self.dyn.psi_omega[0][:] = psi_omega_l[:]
        psi_omega_u = self.dyn.psi[1] * exp_L
        self.dyn.psi_omega[1][:] = psi_omega_u[:]

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

        self.dyn.E = self.laser_field_envelope(self.stat, self.dyn)
        E_full = self.dyn.E * exp_L * exp_L

        self.dyn.psi_omega = phys_base.prop(self.dyn.psi_omega, t_sc, self.nch, self.np, self.stat.v, self.stat.akx2, emin, emax, self.dyn.E, eL)

        cnorm = []
        cnorm.append(math_base.cprod(self.dyn.psi_omega[0], self.dyn.psi_omega[0], self.stat.dx, self.np))
        cnorm.append(math_base.cprod(self.dyn.psi_omega[1], self.dyn.psi_omega[1], self.stat.dx, self.np))
        cnorm_sum = cnorm[0] + cnorm[1]

        # renormalization
        if cnorm_sum > 0.0:
            self.dyn.psi_omega[0] /= math.sqrt(abs(cnorm_sum))
            self.dyn.psi_omega[1] /= math.sqrt(abs(cnorm_sum))

        psigc_psie = math_base.cprod(self.dyn.psi_omega[1], self.dyn.psi_omega[0], self.stat.dx, self.np)
        psigc_dv_psie = math_base.cprod3(self.dyn.psi_omega[1], self.stat.v[0][1] - self.stat.v[1][1], self.dyn.psi_omega[0], self.stat.dx, self.np)

        # converting back to psi
        self.dyn.psi[0] = self.dyn.psi_omega[0] * exp_L
        self.dyn.psi[1] = self.dyn.psi_omega[1] / exp_L

        # calculating of a current energy
        phi = phys_base.hamil2D_orig(self.dyn.psi, self.stat.v, self.stat.akx2, self.np, E_full)

        cener = []
        cener.append(math_base.cprod(phi[0], self.dyn.psi[0], self.stat.dx, self.np))
        cener.append(math_base.cprod(phi[1], self.dyn.psi[1], self.stat.dx, self.np))

        overlp0 = self._pop_eval(self.stat.psi0, self.dyn.psi, self.stat.dx, self.np)
        overlpf = self._pop_eval(self.stat.psif, self.dyn.psi, self.stat.dx, self.np)

        # calculating of expectation values
        moms = phys_base.exp_vals_calc(self.dyn.psi, self.stat.x, self.stat.akx2, self.stat.dx, self.np, self.m)

        time_after = datetime.datetime.now()

        instr = PropagationSolver.InstrumentationOutputData(moms, cnorm, psigc_psie, psigc_dv_psie,
                     cener, E_full, overlp0, overlpf, emax, emin, t_sc, time_before, time_after)

        self.process_instrumentation(instr)
        self.report_dynamic(self.dyn)

        self.dyn.l += 1
        if self.dyn.l <= self.nt:
            return True
        else:
            return False