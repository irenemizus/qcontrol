import cmath
import math
import copy
from enum import Enum
import datetime
from typing import Callable, Optional

import numpy
from numpy.typing import NDArray

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

        def __init__(self, psi0: Psi, psif: Psi,
                     moms0: phys_base.ExpectationValues,
                     smoms0: phys_base.SigmaExpectationValues,
                     cnorm0: NDArray[numpy.complex128],
                     cnormf: NDArray[numpy.complex128],
                     cener0: NDArray[numpy.complex128],
                     cenerf: NDArray[numpy.complex128],
                     overlp00: NDArray[numpy.complex128],
                     overlpf0: NDArray[numpy.complex128],
                     dt: numpy.float64 = 0.0, dx: numpy.float64 = 0.0,
                     x: NDArray[numpy.float64] = numpy.empty(shape=0, dtype=numpy.float64), v=None, akx2=None):
            assert (psi0 is None and psif is None) or (psi0.f[0] is not psif.f[0] and psi0.f[1] is not psif.f[1]), \
                "A single array is passed twice (as psi0 and psif). Clone it!"

            self.psi0 = psi0
            self.psif = psif
            self.moms0 = moms0
            self.smoms0 = smoms0
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
                     E=0.0, freq_mult: numpy.float64 = 1.0, dir=None):
            assert (psi is None and psi_omega is None) or \
                   (psi.f[0] is not psi_omega.f[0] and psi.f[1] is not psi_omega.f[1]), \
                "A single array is passed twice (as psi and psi_omega). Clone it!"

            self.l = l
            self.t = t
            self.psi = psi
            self.psi_omega = psi_omega
            self.E = E
            self.freq_mult = numpy.float64(freq_mult)
            self.dir = dir

    # These parameters are recalculated from scratch on each step,
    # and then follows an output of them to the user
    class InstrumentationOutputData:
        psigc_psie: numpy.complex128

        def __init__(self, moms: phys_base.ExpectationValues, smoms: phys_base.SigmaExpectationValues,
                     cnorm, psigc_psie: numpy.complex128, psigc_dv_psie: numpy.complex128, cener: NDArray[numpy.complex128],
                     E_full, overlp0, overlpf, emax, emin, t_sc, time_before, time_after):
            self.moms = moms
            self.smoms = smoms
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
    freq_multiplier: Optional[Callable[[DynamicState, StaticState], numpy.float64]]

    def __init__(
            self,
            T,
            np,
            L,
            _warning_collocation_points,
            _warning_time_steps,
            reporter: PropagationReporter,
            hamil2D,
            laser_field_envelope,
            laser_field_hf,
            freq_multiplier: Callable[[DynamicState, StaticState], numpy.float64],
            dynamic_state_factory,
            pcos,
            w_list,
            mod_log,
            ntriv,
            hf_hide,
            conf_prop):
        self.milliseconds_full = 0.0
        self.T = T
        self.np = np
        self.L = L
        self._warning_collocation_points = _warning_collocation_points
        self._warning_time_steps = _warning_time_steps
        self.reporter = reporter
        self.hamil2D = hamil2D
        self.laser_field_envelope = laser_field_envelope
        self.laser_field_hf = laser_field_hf
        self.freq_multiplier = freq_multiplier
        self.dynamic_state_factory = dynamic_state_factory
        self.mod_log = mod_log
        self.ntriv = ntriv
        self.hf_hide = hf_hide
        self.pcos = pcos
        self.w_list = w_list
        self.m = conf_prop.m
        self.nch = conf_prop.nch
        self.nt = conf_prop.nt
        self.E0 = conf_prop.E0
        self.nu_L = conf_prop.nu_L

        self.stat = None
        self.dyn = None
        self.instr = None

    @staticmethod
    def _norm_eval(psi: Psi, dx, np):
        cnorm: NDArray[numpy.complex128] = numpy.zeros(len(psi.f), numpy.complex128)

        for n in range(len(psi.f)):
            cnorm[n] = math_base.cprod(psi.f[n], psi.f[n], dx, np)

        return cnorm

    def _ener_eval(self, psi: Psi, dx, np, E, eL, E_full, orig):
        cener: NDArray[numpy.complex128] = numpy.zeros(len(psi.f), numpy.complex128)

        phi = self.hamil2D(orig=orig, psi=psi, E=E, eL=eL, E_full=E_full)
        for n in range(len(psi.f)):
            cener[n] = math_base.cprod(psi.f[n], phi.f[n], dx, np)

        return cener

    @staticmethod
    def _pop_eval(psi_goal: Psi, psi: Psi, dx, np):
        overlp: NDArray[numpy.complex128] = numpy.zeros(len(psi.f), numpy.complex128)

        for n in range(len(psi.f)):
            overlp[n] = math_base.cprod(psi_goal.f[n], psi.f[n], dx, np)

        return overlp

    def report_static(self):
        # check if input data are correct in terms of the given problem
        # calculating the initial energy range of the Hamiltonian operator H
        extr = []

        for n in range(len(self.stat.psi0.f)):
            extr.append(self.stat.v[n][1][0] + abs(self.stat.akx2[int(self.np / 2 - 1)]))# + 2.0
            extr.append(self.stat.v[n][0])

        emax0 = max(extr)
        emin0 = min(extr)

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

        cener0_tot = 0.0
        overlp0 = 0.0
        overlpf = 0.0
        psi_max_abs = []
        psi_max_real = []
        for n in range(len(self.stat.psi0.f)):
            psi0n = self.stat.psi0.f[n]
            psi0n_abs = numpy.array([abs(el) for el in psi0n])
            psi0n_real = numpy.array([el.real for el in psi0n])
            psi0n_real_abs = numpy.array([abs(el.real) for el in psi0n])
            cener0_tot += self.stat.cener0[n]
            overlp0 += self.stat.overlp00[n]
            overlpf += self.stat.overlpf0[n]
            max_ind_psi_abs = numpy.argmax(psi0n_abs)
            max_ind_psi_real = numpy.argmax(psi0n_real_abs)
            psi_max_abs.append(psi0n_abs[max_ind_psi_abs])
            psi_max_real.append(psi0n_real[max_ind_psi_real])
        overlp0_tot = [overlp0, overlpf]

        fm_start = 1.0

        if self.ntriv == 1:
            E = abs(self.dyn.E)
        else:
            E = self.dyn.E.real

        # plotting initial values
        self.reporter.print_time_point_prop(self.dyn.l, self.stat.psi0, self.dyn.t, self.stat.x, self.np, self.nt,
                                            self.stat.moms0, self.stat.smoms0, self.stat.cener0, self.stat.cnorm0,
                                            self.stat.overlp00, self.stat.overlpf0, overlp0_tot, cener0_tot,
                                            psi_max_abs, psi_max_real, E, fm_start)

        print("Initial emax = ", emax0)
        print("Initial emin = ", emin0)

        print(" Initial state features: ")
        for n in range(len(self.stat.psi0.f)):
            print("Initial normalization (state #%d): %f" % (n, abs(self.stat.cnorm0[n])))
            print("Initial energy (state #%d): %f" % (n,  self.stat.cener0[n].real))

        print(" Final goal features: ")
        for n in range(len(self.stat.psi0.f)):
            print("Final goal normalization (state #%d): %f" % (n, abs(self.stat.cnormf[n])))
            print("Final goal energy (state #%d): %f" % (n, self.stat.cenerf[n].real))

    def report_dynamic(self):
        # calculating the minimum number of collocation points and time steps that are needed for convergence
        nt_min = int(math.ceil(
            (self.instr.emax - self.instr.emin) * self.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h))
        np_min = int(math.ceil(
            self.L * math.sqrt(2.0 * self.m *
                    (self.instr.emax - self.instr.emin) * phys_base.dalt_to_au / phys_base.hart_to_cm) / math.pi))

        cener_tot = 0.0
        overlp0 = 0.0
        overlpf = 0.0
        psi_max_abs = []
        psi_max_real = []
        for n in range(len(self.stat.psi0.f)):
            psin = self.dyn.psi.f[n]
            psin_abs = numpy.array([abs(el) for el in psin])
            psin_real = numpy.array([el.real for el in psin])
            psin_real_abs = numpy.array([abs(el.real) for el in psin])
            cener_tot += self.instr.cener[n]
            overlp0 += self.instr.overlp0[n]
            overlpf += self.instr.overlpf[n]
            max_ind_psi_abs = numpy.argmax(psin_abs)
            max_ind_psi_real = numpy.argmax(psin_real_abs)
            psi_max_abs.append(psin_abs[max_ind_psi_abs])
            psi_max_real.append(psin_real[max_ind_psi_real])
        overlp_tot = [overlp0, overlpf]

        time_span = self.instr.time_after - self.instr.time_before
        milliseconds_per_step = time_span.microseconds / 1000
        self.milliseconds_full += milliseconds_per_step

        if self.ntriv == 1:
            E = abs(self.dyn.E)
        else:
            E = self.dyn.E.real

        self.reporter.print_time_point_prop(self.dyn.l, self.dyn.psi, self.dyn.t, self.stat.x, self.np, self.nt,
                                            self.instr.moms, self.instr.smoms, self.instr.cener, self.instr.cnorm,
                                            self.instr.overlp0, self.instr.overlpf, overlp_tot, cener_tot,
                                            psi_max_abs, psi_max_real, E, self.dyn.freq_mult)

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
            for n in range(len(self.stat.psi0.f)):
                print("normalization on the state #%d = %f" % (n, abs(self.instr.cnorm[n])))
                print("energy on the state #%d = %f" % (n, self.instr.cener[n].real))
            print("overlap with initial wavefunction = ", abs(overlp0))
            print("overlap with final goal wavefunction = ", abs(overlpf))

            print(
                "milliseconds per step: " + str(milliseconds_per_step) + ", on average: " + str(
                    self.milliseconds_full / self.dyn.l))

    def start(self, v, akx2, dx, x, t_step, psi0, psif, dir: Direction):
        # initial normalization check
        cnorm0 = self._norm_eval(psi0, dx, self.np)

        # calculating of initial ground/excited energies
        eL = self.nu_L * phys_base.Hz_to_cm / 2.0
        cener0 = self._ener_eval(psi=psi0, dx=dx, np=self.np, E=numpy.float64(0.0),
                                 eL=eL, E_full=numpy.float64(0.0), orig=True)

        # final normalization check
        cnormf = self._norm_eval(psif, dx, self.np)

        # calculating of final excited/filtered energy
        cenerf = self._ener_eval(psi=psif, dx=dx, np=self.np, E=numpy.float64(0.0),
                                 eL=eL, E_full=numpy.float64(0.0), orig=True)

        # time propagation
        dt = dir.value * t_step
        psi = copy.deepcopy(psi0)

        # initial population
        overlp00 = self._pop_eval(psi0, psi, dx, self.np)
        overlpf0 = self._pop_eval(psif, psi, dx, self.np)

        # calculating of initial expectation values
        moms0 = phys_base.exp_vals_calc(psi, x, akx2, dx, self.np, self.m, self.ntriv)
        smoms0 = phys_base.exp_sigma_vals_calc(psi, dx, self.np, self.ntriv)

        self.stat = PropagationSolver.StaticState(psi0, psif, moms0, smoms0, cnorm0, cnormf,
                     cener0, cenerf, overlp00, overlpf0, dt, dx, x, v, akx2)

        if dir == PropagationSolver.Direction.FORWARD:
            self.dyn = self.dynamic_state_factory(0, numpy.float64(0.0), psi, psi, numpy.float64(0.0), numpy.float64(1.0), dir)
        else:
            self.dyn = self.dynamic_state_factory(0, self.T, psi, psi, numpy.float64(0.0), numpy.float64(1.0), dir)

        # Calculating the initial field value
        self.dyn.E = self.laser_field_envelope(self, self.stat, self.dyn)

        self.report_static()

        self.dyn.l = 1

    def step(self, t_start):
        time_before = datetime.datetime.now()

        nlvls = len(self.stat.psi0.f)

        extr = []
        # calculating limits of energy ranges of the one-dimensional Hamiltonian operator
        for n in range(nlvls):
            extr.append(self.stat.v[n][1][0] + abs(self.stat.akx2[int(self.np / 2 - 1)]))# + 2.0
            extr.append(self.stat.v[n][0])

        self.dyn.t = self.stat.dt * self.dyn.l + t_start

        eL = self.nu_L * self.dyn.freq_mult * phys_base.Hz_to_cm / 2.0
        exp_L = numpy.float64(1.0)

        # Here we're transforming the problem to the one for psi_omega -- if needed
        if self.hf_hide:
            self.dyn.freq_mult = self.freq_multiplier(self.dyn, self.stat)
            exp_L = numpy.complex128(cmath.sqrt(self.laser_field_hf(self.dyn.freq_mult, self.dyn.t, self.pcos, self.w_list)))

            psi_omega_l = self.dyn.psi.f[0] / exp_L
            self.dyn.psi_omega.f[0][:] = psi_omega_l[:]
            psi_omega_u = self.dyn.psi.f[1] * exp_L
            self.dyn.psi_omega.f[1][:] = psi_omega_u[:]

            # New energy ranges
            extr_omega = []

            extr_omega.append(extr[0] + self.E0 + eL)
            extr_omega.append(extr[1] - self.E0 + eL)

            extr_omega.append(extr[2] + self.E0 - eL)
            extr_omega.append(extr[3] - self.E0 - eL)

            emax = max(extr_omega)
            emin = min(extr_omega)
        else:
            emax = max(extr)
            emin = min(extr)

        t_sc = self.stat.dt * (emax - emin) * phys_base.cm_to_erg / 4.0 / phys_base.Red_Planck_h

        self.dyn.E = self.laser_field_envelope(self, self.stat, self.dyn)

        #print("l = %d" % self.dyn.l)
        #print("E = %f" % self.dyn.E)

        psigc_psie = numpy.float64(0.0)
        psigc_dv_psie = numpy.float64(0.0)

        cnorm = []
        if self.hf_hide:
            E_full = self.dyn.E * exp_L * exp_L
            self.dyn.psi_omega = phys_base.prop_cpu(psi=self.dyn.psi_omega, hamil2D=self.hamil2D, t_sc=t_sc,
                                                    nch=self.nch, np=self.np, emin=emin, emax=emax,
                                                    E=self.dyn.E, eL=eL, E_full=E_full, orig=False)

            cnorm_sum = 0.0
            for n in range(nlvls):
                cnormn = math_base.cprod(self.dyn.psi_omega.f[n], self.dyn.psi_omega.f[n], self.stat.dx, self.np)
                cnorm.append(cnormn)
                cnorm_sum += cnormn

            # renormalization
            if abs(cnorm_sum) > 0.0:
                for n in range(nlvls):
                    self.dyn.psi_omega.f[n] /= math.sqrt(abs(cnorm_sum))

            psigc_psie = math_base.cprod(self.dyn.psi_omega.f[1], self.dyn.psi_omega.f[0], self.stat.dx, self.np)
            psigc_dv_psie = math_base.cprod3(self.dyn.psi_omega.f[1], self.stat.v[0][1] - self.stat.v[1][1],
                                             self.dyn.psi_omega.f[0], self.stat.dx, self.np)

            # converting back to psi
            self.dyn.psi.f[0] = self.dyn.psi_omega.f[0] * exp_L
            self.dyn.psi.f[1] = self.dyn.psi_omega.f[1] / exp_L

        else:
            E_full = self.dyn.E
            self.dyn.psi = phys_base.prop_cpu(psi=self.dyn.psi, hamil2D=self.hamil2D, t_sc=t_sc, nch=self.nch, np=self.np,
                                              emin=emin, emax=emax, E=self.dyn.E, eL=eL, E_full=E_full, orig=False)

            #print(self.dyn.psi.f)

            cnorm_sum = 0.0
            for n in range(nlvls):
                cnormn = math_base.cprod(self.dyn.psi.f[n], self.dyn.psi.f[n], self.stat.dx, self.np)
                cnorm.append(cnormn)
                cnorm_sum += cnormn

            # renormalization
            if abs(cnorm_sum) > 0.0:
                for n in range(nlvls):
                    self.dyn.psi.f[n] /= math.sqrt(abs(cnorm_sum))

        # calculating of a current energy
        cener = self._ener_eval(psi=self.dyn.psi, dx=self.stat.dx, np=self.np, E=self.dyn.E, eL=eL,
                                E_full=E_full, orig=True)

        overlp0 = self._pop_eval(self.stat.psi0, self.dyn.psi, self.stat.dx, self.np)
        overlpf = self._pop_eval(self.stat.psif, self.dyn.psi, self.stat.dx, self.np)

        # calculating of expectation values
        moms = phys_base.exp_vals_calc(self.dyn.psi, self.stat.x, self.stat.akx2, self.stat.dx, self.np, self.m, self.ntriv)
        smoms = phys_base.exp_sigma_vals_calc(self.dyn.psi, self.stat.dx, self.np, self.ntriv)

        time_after = datetime.datetime.now()

        self.instr = PropagationSolver.InstrumentationOutputData(moms, smoms, cnorm, psigc_psie, psigc_dv_psie,
                     cener, E_full, overlp0, overlpf, emax, emin, t_sc, time_before, time_after)

        self.report_dynamic()

        self.dyn.l += 1

        if self.dyn.l <= self.nt:
            return True
        else:
            return False