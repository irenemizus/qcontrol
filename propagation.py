import math
import cmath
import sys
import copy
from enum import Enum

import datetime

import math_base
import phys_base
import harmonic


class PropagationSolver:
    def __init__(
            self,
            psi_init,
            pot,
            report_static,
            report_dynamic,
            process_instrumentation,
            m, L, np, nch, T, nt, x0, p0, a, De, x0p, E0,
            t0, sigma, nu_L, delay):

        self.pot = pot
        self.psi_init = psi_init
        self.report_static = report_static
        self.report_dynamic = report_dynamic
        self.process_instrumentation = process_instrumentation

        self.m = m
        self.L = L
        self.np = np
        self.nch = nch
        self.T = T
        self.nt = nt
        self.x0 = x0
        self.p0 = p0
        self.a = a
        self.De = De
        self.x0p = x0p
        self.E0 = E0
        self.t0 = t0
        self.sigma = sigma
        self.nu_L = nu_L
        self.delay = delay


    # These values are calculated once and forever
    # They should NEVER change
    class StaticState:
        def __init__(self, psi0 = None, psif = None, moms0: phys_base.ExpectationValues = None,
                     cnorm0 = None, cnormf = None,
                     cener0 = None, cener0_u = None, cenerf = None,
                     E00 = 0, overlp00 = None, overlpf0 = None,
                     dt = 0, dx = 0, x = None, v = None, akx2 = None):
            self.psi0 = psi0
            self.psif = psif
            self.moms0 = moms0
            self.cnorm0 = cnorm0
            self.cnormf = cnormf
            self.cener0 = cener0
            self.cener0_u = cener0_u
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
        def __init__(self, l = 0, psi = None):
            self.l = l
            self.psi = psi


    # These parameters are recalculated from scratch on each step,
    # and then follows an output of them to the user
    class InstrumentationOutputData:
        def __init__(self, moms: phys_base.ExpectationValues, cnorm_l, cnorm_u, psigc_psie,
                     cener_l, cener_u, E_full, overlp0, overlpf, emax, emin, t_sc, time_before, time_after):
            self.moms = moms
            self.cnorm_l = cnorm_l
            self.cnorm_u = cnorm_u
            self.psigc_psie = psigc_psie
            self.cener_l = cener_l
            self.cener_u = cener_u
            self.E_full = E_full
            self.overlp0 = overlp0
            self.overlpf = overlpf
            self.emax = emax
            self.emin = emin
            self.t_sc = t_sc
            self.time_before = time_before
            self.time_after = time_after


    # The user's decision to repeat or not to repeat
    class StepReaction(Enum):
        OK = 0
        REPEAT = 1


    def time_propagation(self):
        # calculating coordinate step of the problem
        dx = self.L / (self.np - 1)

        # setting the coordinate grid
        x = math_base.coord_grid(dx, self.np)

        # evaluating of potential(s)
        v = self.pot(x, self.np, self.m, self.De, self.a, self.x0p)

        # evaluating of k vector
        akx2 = math_base.initak(self.np, dx, 2)

        # evaluating of kinetic energy
        akx2 *= -phys_base.hart_to_cm / (2.0 * self.m * phys_base.dalt_to_au)

        # evaluating of initial wavefunction
        psi0 = self.psi_init(x, self.np, self.x0, self.p0, self.m, self.De, self.a)

        # initial normalization check
        cnorm0 = math_base.cprod(psi0[0], psi0[0], dx, self.np)

        # calculating of initial ground energy
        phi0 = phys_base.hamil(psi0[0], v[0][1], akx2, self.np)
        cener0 = math_base.cprod(phi0, psi0[0], dx, self.np)

        # initial excited energy
        phi0_u = phys_base.hamil(psi0[1], v[1][1], akx2, self.np)
        cener0_u = math_base.cprod(phi0_u, psi0[1], dx, self.np)

        # evaluating of the final goal -- upper state wavefunction
        psif = self.psi_init(x, self.np, self.x0p, self.p0, self.m, self.De / 2.0, self.a)

        # final normalization check
        cnormf = math_base.cprod(psif[0], psif[0], dx, self.np)

        # calculating of final excited energy
        phif = phys_base.hamil(psif[0], v[1][1], akx2, self.np)
        cenerf = math_base.cprod(phif, psif[0], dx, self.np)

        # time propagation
        dt = self.T / (self.nt - 1)
        psi = copy.deepcopy(psi0)

        # initial laser field energy
        E00 = phys_base.laser_field(self.E0, 0.0, self.t0, self.sigma)

        # initial population
        overlp00 = math_base.cprod(psi0[0], psi[0], dx, self.np)
        overlpf0 = math_base.cprod(psif[0], psi[1], dx, self.np)

        # calculating of initial expectation values
        moms0 = phys_base.exp_vals_calc(psi, x, akx2, dx, self.np, self.m)

        stat = PropagationSolver.StaticState(psi0, psif, moms0, cnorm0, cnormf,
                     cener0, cener0_u, cenerf, E00, overlp00, overlpf0, dt, dx, x, v, akx2)
        self.report_static(stat)

        dyn = PropagationSolver.DynamicState(0, psi)
        self.report_dynamic(dyn)


        # main propagation loop
        for dyn.l in range(1, self.nt + 1):
            self.step(stat, dyn)

    def step(self, stat: StaticState, dyn: DynamicState):
        dyn_bu = copy.deepcopy(dyn)
        take_back = True
        while take_back:
            time_before = datetime.datetime.now()

            # calculating limits of energy ranges of the one-dimensional Hamiltonian operator H_l
            emax_l = stat.v[0][1][0] + abs(stat.akx2[int(self.np / 2 - 1)]) + 2.0
            emin_l = stat.v[0][0]
            # calculating limits of energy ranges of the one-dimensional Hamiltonian operator H_u
            emax_u = stat.v[1][1][0] + abs(stat.akx2[int(self.np / 2 - 1)]) + 2.0
            emin_u = stat.v[1][0]

            t = stat.dt * dyn.l

            # Here we're transforming the problem to the one for psi_omega
            psi_omega = []
            exp_L = cmath.exp(1j * math.pi * self.nu_L * t)
            psi_omega_l = dyn.psi[0] / exp_L
            psi_omega.append(psi_omega_l)
            psi_omega_u = dyn.psi[1] * exp_L
            psi_omega.append(psi_omega_u)

            # New energy ranges
            eL = self.nu_L * phys_base.Hz_to_cm / 2.0
            emax_l_omega = emax_l + self.E0 + eL
            emin_l_omega = emin_l - self.E0 + eL

            emax_u_omega = emax_u + self.E0 - eL
            emin_u_omega = emin_u - self.E0 - eL

            emax = max(emax_l_omega, emin_l_omega, emax_u_omega, emin_u_omega)
            emin = min(emax_l_omega, emin_l_omega, emax_u_omega, emin_u_omega)

            t_sc = stat.dt * (emax - emin) * phys_base.cm_to_erg / 4.0 / phys_base.Red_Planck_h

            E = phys_base.laser_field(self.E0, t, self.t0, self.sigma)
            #E2 = phys_base.laser_field(self.E0, t, self.t0 + self.delay, self.sigma)
            #E = E1 + E2
            E_full = E * exp_L * exp_L
            psi_omega = phys_base.prop(psi_omega, t_sc, self.nch, self.np, stat.v, stat.akx2, emin, emax, E,
                                       eL)  # TODO move prop into this class

            cnorm_l = math_base.cprod(psi_omega[0], psi_omega[0], stat.dx, self.np)
            cnorm_u = math_base.cprod(psi_omega[1], psi_omega[1], stat.dx, self.np)
            cnorm = cnorm_l + cnorm_u

            # renormalization
            if cnorm > 0.0:
                psi_omega[0] /= math.sqrt(abs(cnorm))
                psi_omega[1] /= math.sqrt(abs(cnorm))

            # converting back to psi
            dyn.psi[0] = psi_omega[0] * exp_L
            dyn.psi[1] = psi_omega[1] / exp_L

            # calculating of a current energy
            phi = phys_base.hamil2D_orig(dyn.psi, stat.v, stat.akx2, self.np, E_full)

            cener_l = math_base.cprod(phi[0], dyn.psi[0], stat.dx, self.np)
            cener_u = math_base.cprod(phi[1], dyn.psi[1], stat.dx, self.np)

            overlp0 = math_base.cprod(stat.psi0[0], dyn.psi[0], stat.dx, self.np)
            overlpf = math_base.cprod(stat.psif[0], dyn.psi[1], stat.dx, self.np)

            psigc_psie = math_base.cprod(dyn.psi[1], dyn.psi[0], stat.dx, self.np)

            # calculating of expectation values
            moms = phys_base.exp_vals_calc(dyn.psi, stat.x, stat.akx2, stat.dx, self.np, self.m)

            time_after = datetime.datetime.now()

            instr = PropagationSolver.InstrumentationOutputData(moms, cnorm_l, cnorm_u, psigc_psie,
                         cener_l, cener_u, E_full, overlp0, overlpf, emax, emin, t_sc, time_before, time_after)

            res = self.process_instrumentation(instr)
            if res == PropagationSolver.StepReaction.OK:
                take_back = False
            elif res == PropagationSolver.StepReaction.REPEAT:
                take_back = True
            else:
                raise RuntimeError("Impossible case")

            if not take_back:
                self.report_dynamic(dyn)



    # def filtering(self):
    #     # filtering task for obtaining of an initial wavefunction in the given potential
    #     dt = self.T / (self.nt - 1)
    #     psi_init = harmonic.psi_init(self.x, self.np, self.x0, self.p0, self.m, self.De, self.a)
    #     psi_goal = self.psi_init(self.x, self.np, self.x0, self.p0, self.m, self.De, self.a)
    #     psi = copy.deepcopy(psi_init)
    #
    #     # plotting initial values
    #     self.plot(psi[0], 0.0, self.x, self.np)
    #
    #     # initial normalization check
    #     cnorm0 = math_base.cprod(psi[0], psi[0], self.dx, self.np)
    #
    #     # calculating of initial energy
    #     phi0 = phys_base.hamil(psi[0], self.v[0][1], self.akx2, self.np)
    #     cener0 = math_base.cprod(phi0, psi[0], self.dx, self.np)
    #
    #     print(" Initial state features: ")
    #     print("Initial normalization: ", abs(cnorm0))
    #     print("Initial energy: ", abs(cener0))
    #
    #     milliseconds_full = 0
    #
    #     # main propagation loop
    #     for l in range(1, self.nt + 1):
    #         time_before = datetime.datetime.now()
    #
    #         t = dt * l
    #         t_sc = dt * (self.emax0 - self.emin0) * phys_base.cm_to_erg / 4.0 / phys_base.Red_Planck_h
    #
    #         psi = phys_base.prop(psi, t_sc, self.nch, self.np, self.v, self.akx2, self.emin0, self.emax0, 0.0, 0.0)
    #
    #         cnorm = math_base.cprod(psi[0], psi[0], self.dx, self.np)
    #
    #         # renormalization
    #         if cnorm > 0.0:
    #             psi[0] /= math.sqrt(abs(cnorm))
    #
    #         phi = phys_base.hamil(psi[0], self.v[0][1], self.akx2, self.np)
    #
    #         cener = math_base.cprod(phi, psi[0], self.dx, self.np)
    #         overlp0 = math_base.cprod(psi_init[0], psi[0], self.dx, self.np)
    #         overlpg = math_base.cprod(psi_goal[0], psi[0], self.dx, self.np)
    #         moms0 = phys_base.ExpectationValues(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    #
    #         # plotting the result
    #         if l % self.mod_fileout == 0:
    #             if l >= self.lmin:
    #                 self.plot(psi[0], t, self.x, self.np)
    #
    #             if l >= self.lmin:
    #                 self.plot_mom(t, moms0, cener.real, 0.0, overlp0, cener.real)
    #
    #
    #         time_after = datetime.datetime.now()
    #         time_span = time_after - time_before
    #         milliseconds_per_step = time_span.microseconds / 1000
    #         milliseconds_full += milliseconds_per_step
    #
    #         if l % self.mod_stdout == 0:
    #             print("l = ", l)
    #             print("t = ", t * 1e15, "fs")
    #
    #             print("normalized scaled time interval = ", t_sc)
    #             print("normalization on the lower state = ", cnorm)
    #             print("overlap with initial wavefunction = ", abs(overlp0))
    #             print("overlap with the target wavefunction = ", abs(overlpg))
    #             print("energy on the lower state = ", cener.real)
    #
    #             print("milliseconds per step: " + str(milliseconds_per_step) + ", on average: " + str(milliseconds_full / l))












