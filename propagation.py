import math
import cmath
import sys
import copy

import datetime

import math_base
import phys_base
import harmonic


class PropagationSolver:
    def _warning_collocation_points(self, np_min):
        print("WARNING: The number of collocation points np = {} should be more than an estimated initial value {}. "
              "You've got a divergence!".format(self.np, np_min), file=sys.stderr)

    def _warning_time_steps(self, nt_min):
        print("WARNING: The number of time steps nt = {} should be more than an estimated value {}. "
              "You've got a divergence!".format(self.nt, nt_min), file=sys.stderr)

    def __init__(
            self,
            psi_init,
            pot,
            plot,
            plot_mom,
            plot_test,
            plot_up,
            plot_mom_up,
            on_after_step,
            m, L, np, nch, T, nt, x0, p0, a, De, x0p, E0,
            t0, sigma, nu_L, delay):

        self.pot = pot
        self.psi_init = psi_init
        self.plot = plot
        self.plot_mom = plot_mom
        self.plot_up = plot_up
        self.plot_mom_up = plot_mom_up
        self.plot_test = plot_test
        self.on_after_step = on_after_step

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


        # calculating coordinate step of the problem
        self.dx = self.L / (self.np - 1)

        # setting the coordinate grid
        self.x = math_base.coord_grid(self.dx, self.np)

        # evaluating of potential(s)
        self.v = pot(self.x, self.np, self.m, self.De, self.a, self.x0p)

        # evaluating of k vector
        self.akx2 = math_base.initak(self.np, self.dx, 2)

        # evaluating of kinetic energy
        self.coef_kin = -phys_base.hart_to_cm / (2.0 * self.m * phys_base.dalt_to_au)
        self.akx2 *= self.coef_kin

        # check if input data are correct in terms of the given problem
        # calculating the initial energy range of the Hamiltonian operator H
        self.emax0 = self.v[0][1][0] + abs(self.akx2[int(self.np / 2 - 1)]) + 2.0
        self.emin0 = self.v[0][0]

        print("Initial emax = ", self.emax0)

        # calculating the initial minimum number of collocation points that is needed for convergence
        self.np_min0 = int(
            math.ceil(self.L *
                      math.sqrt(2.0 * self.m * (self.emax0 - self.emin0) * phys_base.dalt_to_au / phys_base.hart_to_cm) /
                      math.pi
            )
        )

        # calculating the initial minimum number of time steps that is needed for convergence
        self.nt_min0 = int(
            math.ceil((self.emax0 - self.emin0) * self.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h
            )
        )

        if self.np < self.np_min0:
            self._warning_collocation_points(self.np_min0)
        if self.nt < self.nt_min0:
            self._warning_time_steps(self.nt_min0)

    class StepInit:
        def __init__(self, l, dt, psi, psi0, psif):
            self.l = l
            self.dt = dt
            self.psi = psi
            self.psi0 = psi0
            self.psif = psif


    class StepState:
        def __init__(self, moms: phys_base.ExpectationValues, cnorm_l, cnorm_u,
                     cener_l, cener_u, E_full, overlp0, overlpf, emax, emin, t_sc, time_before, time_after):
            self.moms = moms
            self.cnorm_l = cnorm_l
            self.cnorm_u = cnorm_u
            self.cener_l =  cener_l
            self.cener_u = cener_u
            self.E_full = E_full
            self.overlp0 = overlp0
            self.overlpf = overlpf
            self.emax = emax
            self.emin = emin
            self.t_sc = t_sc
            self.time_before = time_before
            self.time_after = time_after

    def step(self, init: StepInit):
        time_before = datetime.datetime.now()

        # calculating limits of energy ranges of the one-dimensional Hamiltonian operator H_l
        emax_l = self.v[0][1][0] + abs(self.akx2[int(self.np / 2 - 1)]) + 2.0
        emin_l = self.v[0][0]
        # calculating limits of energy ranges of the one-dimensional Hamiltonian operator H_u
        emax_u = self.v[1][1][0] + abs(self.akx2[int(self.np / 2 - 1)]) + 2.0
        emin_u = self.v[1][0]

        t = init.dt * init.l
        # Here we're transforming the problem to the one for psi_omega
        psi_omega = []
        exp_L = cmath.exp(1j * math.pi * self.nu_L * t)
        psi_omega_l = init.psi[0] / exp_L
        psi_omega.append(psi_omega_l)
        psi_omega_u = init.psi[1] * exp_L
        psi_omega.append(psi_omega_u)

        # New energy ranges
        eL = self.nu_L * phys_base.Hz_to_cm / 2.0
        emax_l_omega = emax_l + self.E0 + eL
        emin_l_omega = emin_l - self.E0 + eL

        emax_u_omega = emax_u + self.E0 - eL
        emin_u_omega = emin_u - self.E0 - eL

        emax = max(emax_l_omega, emin_l_omega, emax_u_omega, emin_u_omega)
        emin = min(emax_l_omega, emin_l_omega, emax_u_omega, emin_u_omega)

        t_sc = init.dt * (emax - emin) * phys_base.cm_to_erg / 4.0 / phys_base.Red_Planck_h

        E1 = phys_base.laser_field(self.E0, t, self.t0, self.sigma)
        E2 = phys_base.laser_field(self.E0, t, self.t0 + self.delay, self.sigma)
        E = E1 + E2
        E_full = E * exp_L * exp_L
        psi_omega = phys_base.prop(psi_omega, t_sc, self.nch, self.np, self.v, self.akx2, emin, emax, E,
                                   eL)  # TODO move prop into this class

        cnorm_l = math_base.cprod(psi_omega[0], psi_omega[0], self.dx, self.np)
        cnorm_u = math_base.cprod(psi_omega[1], psi_omega[1], self.dx, self.np)
        cnorm = cnorm_l + cnorm_u

        # renormalization
        if cnorm > 0.0:
            psi_omega[0] /= math.sqrt(abs(cnorm))
            psi_omega[1] /= math.sqrt(abs(cnorm))

        # converting back to psi
        init.psi[0] = psi_omega[0] * exp_L
        init.psi[1] = psi_omega[1] / exp_L

        # calculating of a current energy
        phi = phys_base.hamil2D_orig(init.psi, self.v, self.akx2, self.np, E_full)

        cener_l = math_base.cprod(phi[0], init.psi[0], self.dx, self.np)
        cener_u = math_base.cprod(phi[1], init.psi[1], self.dx, self.np)

        overlp0 = math_base.cprod(init.psi0[0], init.psi[0], self.dx, self.np)
        overlpf = math_base.cprod(init.psif[0], init.psi[1], self.dx, self.np)

        # calculating of expectation values
        moms = phys_base.exp_vals_calc(init.psi, self.x, self.akx2, self.dx, self.np, self.m)

        time_after = datetime.datetime.now()

        # plotting the result
        self.on_after_step(self, init, PropagationSolver.StepState(moms, cnorm_l, cnorm_u,
                                                             cener_l, cener_u, E_full, overlp0, overlpf,
                                                             emax, emin, t_sc, time_before, time_after))


    def time_propagation(self):
        # evaluating of initial wavefunction
        psi0 = self.psi_init(self.x, self.np, self.x0, self.p0, self.m, self.De, self.a)

        # initial normalization check
        cnorm0 = math_base.cprod(psi0[0], psi0[0], self.dx, self.np)

        # calculating of initial ground energy
        phi0 = phys_base.hamil(psi0[0], self.v[0][1], self.akx2, self.np)
        cener0 = math_base.cprod(phi0, psi0[0], self.dx, self.np)

        # initial excited energy
        phi0_u = phys_base.hamil(psi0[1], self.v[1][1], self.akx2, self.np)
        cener0_u = math_base.cprod(phi0_u, psi0[1], self.dx, self.np)
        cener0_tot = cener0 + cener0_u

        # evaluating of the final goal -- upper state wavefunction
        psif = self.psi_init(self.x, self.np, self.x0p, self.p0, self.m, self.De / 2.0, self.a)

        # final normalization check
        cnormf = math_base.cprod(psif[0], psif[0], self.dx, self.np)

        # calculating of final excited energy
        phif = phys_base.hamil(psif[0], self.v[1][1], self.akx2, self.np)
        cenerf = math_base.cprod(phif, psif[0], self.dx, self.np)

        print(" Initial state features: ")
        print("Initial normalization: ", abs(cnorm0))
        print("Initial energy: ", abs(cener0))

        print(" Final goal features: ")
        print("Final goal normalization: ", abs(cnormf))
        print("Final goal energy: ", abs(cenerf))

        # time propagation
        dt = self.T / (self.nt - 1)
        psi = copy.deepcopy(psi0)

        # plotting initial values
        self.plot(psi[0], 0.0, self.x, self.np)
        self.plot_up(psi[1], 0.0, self.x, self.np)

        # initial laser field energy
        E00 = phys_base.laser_field(self.E0, 0.0, self.t0, self.sigma)

        # initial population
        overlp00 = math_base.cprod(psi0[0], psi[0], self.dx, self.np)
        overlpf0 = math_base.cprod(psif[0], psi[1], self.dx, self.np)

        # calculating of initial expectation values
        moms0 = phys_base.exp_vals_calc(psi, self.x, self.akx2, self.dx, self.np, self.m)

        self.plot_mom(0.0, moms0, cener0.real, E00.real, overlp00, cener0_tot.real)
        self.plot_mom_up(0.0, moms0, cener0_u.real, E00.real, overlpf0, abs(overlp00) + abs(overlpf0))

        # main propagation loop
        for l in range(1, self.nt + 1):
            self.step(PropagationSolver.StepInit(l, dt, psi, psi0, psif))


    def filtering(self):
        # filtering task for obtaining of an initial wavefunction in the given potential
        dt = self.T / (self.nt - 1)
        psi_init = harmonic.psi_init(self.x, self.np, self.x0, self.p0, self.m, self.De, self.a)
        psi_goal = self.psi_init(self.x, self.np, self.x0, self.p0, self.m, self.De, self.a)
        psi = copy.deepcopy(psi_init)

        # plotting initial values
        self.plot(psi[0], 0.0, self.x, self.np)

        # initial normalization check
        cnorm0 = math_base.cprod(psi[0], psi[0], self.dx, self.np)

        # calculating of initial energy
        phi0 = phys_base.hamil(psi[0], self.v[0][1], self.akx2, self.np)
        cener0 = math_base.cprod(phi0, psi[0], self.dx, self.np)

        print(" Initial state features: ")
        print("Initial normalization: ", abs(cnorm0))
        print("Initial energy: ", abs(cener0))

        milliseconds_full = 0

        # main propagation loop
        for l in range(1, self.nt + 1):
            time_before = datetime.datetime.now()

            t = dt * l
            t_sc = dt * (self.emax0 - self.emin0) * phys_base.cm_to_erg / 4.0 / phys_base.Red_Planck_h

            psi = phys_base.prop(psi, t_sc, self.nch, self.np, self.v, self.akx2, self.emin0, self.emax0, 0.0, 0.0)

            cnorm = math_base.cprod(psi[0], psi[0], self.dx, self.np)

            # renormalization
            if cnorm > 0.0:
                psi[0] /= math.sqrt(abs(cnorm))

            phi = phys_base.hamil(psi[0], self.v[0][1], self.akx2, self.np)

            cener = math_base.cprod(phi, psi[0], self.dx, self.np)
            overlp0 = math_base.cprod(psi_init[0], psi[0], self.dx, self.np)
            overlpg = math_base.cprod(psi_goal[0], psi[0], self.dx, self.np)
            moms0 = phys_base.ExpectationValues(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            # plotting the result
            if l % self.mod_fileout == 0:
                if l >= self.lmin:
                    self.plot(psi[0], t, self.x, self.np)

                if l >= self.lmin:
                    self.plot_mom(t, moms0, cener.real, 0.0, overlp0, cener.real)


            time_after = datetime.datetime.now()
            time_span = time_after - time_before
            milliseconds_per_step = time_span.microseconds / 1000
            milliseconds_full += milliseconds_per_step

            if l % self.mod_stdout == 0:
                print("l = ", l)
                print("t = ", t * 1e15, "fs")

                print("normalized scaled time interval = ", t_sc)
                print("normalization on the lower state = ", cnorm)
                print("overlap with initial wavefunction = ", abs(overlp0))
                print("overlap with the target wavefunction = ", abs(overlpg))
                print("energy on the lower state = ", cener.real)

                print("milliseconds per step: " + str(milliseconds_per_step) + ", on average: " + str(milliseconds_full / l))












