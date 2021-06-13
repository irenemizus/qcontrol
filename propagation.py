import math
import sys
import copy

import math_base
import phys_base


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
            m=0.5,    # Dalton
            L=5.0,    # a_0   0.2 -- for a model harmonic oscillator with a = 1.0 # 4.0 a_0 -- for morse oscillator # 6.0 a_0 -- for dimensional harmonic oscillator
            np=1024,  #       128 -- for a model harmonic oscillator with a = 1.0 # 2048 -- for morse oscillator # 512 -- for dimensional harmonic oscillator
            nch=64,
            T=60e-15,  # s -- for morse oscillator
            nt=100000,
            x0=0,  # TODO: to fix x0 != 0
            p0=0,  # TODO: to fix p0 != 0
            a=1.0, # 1/a_0 -- for morse oscillator, a_0 -- for harmonic oscillator
            De=20000.0, # 1/cm
            E0=71.68, # 1/cm
            t0=25e-15,  # s
            sigma=10e-15, # s
            nu_L=0.599586e15, # Hz
            lmin=0):

        self.pot = pot
        self.psi_init = psi_init
        self.plot = plot
        self.plot_mom = plot_mom
        self.plot_up = plot_up
        self.plot_mom_up = plot_mom_up
        self.plot_test = plot_test

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
        self.E0 = E0
        self.t0 = t0
        self.sigma = sigma
        self.nu_L = nu_L
        self.lmin = lmin

        # analyze provided arguments
        if not math.log2(np).is_integer() or not math.log2(nch).is_integer():
            raise ValueError("The number of collocation points 'np' and of Chebyshev "
                  "interpolation points 'nch' must be positive integers and powers of 2")

        if lmin < 0:
            raise ValueError("The number 'lmin' of time iteration, from which the result"
                             "should be written to a file, should be positive or 0")

        if not L > 0.0 or not T > 0.0:
            raise ValueError("The value of spatial range 'L' and of time range 'T' of the problem"
                             "must be positive")

        if not m > 0.0 or not a > 0.0 or not De > 0.0:
            raise ValueError("The value of a reduced mass 'm/mass', of a scaling factor 'a'"
                             "and of a dissociation energy 'De' must be positive")

        if not E0 >= 0.0 or not sigma > 0.0 or not nu_L > 0.0:
            raise ValueError("The value of an amplitude value of the laser field energy envelope 'E0',"
                             "of a scaling parameter of the laser field envelope 'sigma'"
                             "and of a basic frequency of the laser field 'nu_L' must be positive")

        # calculating coordinate step of the problem
        self.dx = self.L / (self.np - 1)

        # setting the coordinate grid
        self.x = math_base.coord_grid(self.dx, self.np)

        # evaluating of potential(s)
        self.v = pot(self.x, self.np, self.m, self.De, self.a)
        #    print(v)

        # evaluating of initial wavefunction
        self.psi0 = psi_init(self.x, self.np, self.x0, self.p0, self.m, self.De, self.a)
        #    abs_psi0 = [abs(i) for i in psi0]
        #    print(abs_psi0)

        # initial normalization check
        self.cnorm0 = math_base.cprod(self.psi0[0], self.psi0[0], self.dx, self.np)
        print("Initial normalization: ", abs(self.cnorm0))

        #    cx1 = []
        #    for i in range(np):
        #        cx1.append(complex(1.0, 0.0))
        #    cnorm00 = cprod2(psi0, cx1, dx, np)
        #    print(abs(cnorm00))

        # evaluating of k vector
        self.akx2 = math_base.initak(self.np, self.dx, 2)
        #    print(akx2)

        # evaluating of kinetic energy
        self.coef_kin = -phys_base.hart_to_cm / (2.0 * self.m * phys_base.dalt_to_au)
        self.akx2 = [ak * self.coef_kin for ak in self.akx2]
        #   print(akx2)

        #    phi0_kin = diff(psi0, akx2, np)
        #    print(phi0_kin)

        self.E = phys_base.laser_field(0.0, 0.0, self.t0, self.sigma, self.nu_L)

        # calculating of initial energy
        self.phi0 = phys_base.hamil(self.psi0, self.v, self.akx2, self.np, self.E)

        self.cener0 = math_base.cprod(self.phi0[0], self.psi0[0], self.dx, self.np)
        print("Initial energy: ", abs(self.cener0))

        # check if input data are correct in terms of the given problem
        # calculating the initial energy range of the Hamiltonian operator H
        self.emax0 = self.v[0][1][0] + abs(self.akx2[int(self.np / 2 - 1)]) + 2.0
        print("Initial emax = ", self.emax0)
        self.emin0 = self.v[0][0]

        # calculating the initial minimum number of collocation points that is needed for convergence
        self.np_min0 = int(
            math.ceil(self.L *
                      math.sqrt(2.0 * self.m * (self.emax0 - self.emin0) * phys_base.dalt_to_au / phys_base.hart_to_cm) /
                      math.pi
            )
        )

        if self.np < self.np_min0:
            self._warning_collocation_points(self.np_min0)

    def time_propagation(self):
        # time propagation
        dt = self.T / (self.nt - 1)
        psi = copy.deepcopy(self.psi0)

        # main propagation loop
        for l in range(1, self.nt + 1):
            # calculating limits of energy ranges of the Hamiltonian operator H
            emax_l = self.v[0][1][0] + abs(self.akx2[int(self.np / 2 - 1)]) + 2.0
            emin_l = self.v[0][0]
            emax_u = self.v[1][1][0] + abs(self.akx2[int(self.np / 2 - 1)]) + 2.0
            emin_u = self.v[1][0]
            edges = [emax_l, emin_l, emax_u, emin_u]

            t_sc_l = dt * (emax_l - emin_l) * phys_base.cm_to_erg / 4.0 / phys_base.Red_Planck_h
            t_sc_u = dt * (emax_u - emin_u) * phys_base.cm_to_erg / 4.0 / phys_base.Red_Planck_h

            if l % 10 == 0:
                print("emax for the lower state = ", emax_l)
                print("emax for the upper state = ", emax_u)
                print("Normalized scaled time interval for the lower state = ", t_sc_l)
                print("Normalized scaled time interval for the upper state = ", t_sc_u)

            # calculating the minimum number of collocation points and time steps that are needed for convergence
            nt_min_l = int(math.ceil((emax_l - emin_l) * self.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h))
            nt_min_u = int(math.ceil((emax_u - emin_u) * self.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h))

            np_min_l = int(math.ceil(self.L * math.sqrt(2.0 * self.m * (emax_l - emin_l) * phys_base.dalt_to_au / phys_base.hart_to_cm) / math.pi))
            np_min_u = int(math.ceil(self.L * math.sqrt(2.0 * self.m * (emax_u - emin_u) * phys_base.dalt_to_au / phys_base.hart_to_cm) / math.pi))

            np_min = max(np_min_l, np_min_u)
            nt_min = max(nt_min_l, nt_min_u)

            if self.np < np_min and l % 10 == 0:
                self._warning_collocation_points(np_min)
            if self.nt < nt_min and l % 10 == 0:
                self._warning_time_steps(nt_min)

            t = dt * l
            E = phys_base.laser_field(self.E0, t, self.t0, self.sigma, self.nu_L)
            psi = phys_base.prop(psi, t_sc_l, t_sc_u, self.nch, self.np, self.v, self.akx2, edges, E) # TODO move prop into this class

            cnorm_l = math_base.cprod(psi[0], psi[0], self.dx, self.np)
            cnorm_u = math_base.cprod(psi[1], psi[1], self.dx, self.np)
            orthog_lu = math_base.cprod(psi[0], psi[1], self.dx, self.np)
            orthog_ul = math_base.cprod(psi[1], psi[0], self.dx, self.np)
            overlp = math_base.cprod(self.psi0[0], psi[0], self.dx, self.np)

            if l % 10 == 0:
                print("l = ", l)
                print("t = ", t * 1e15, "fs")
                print("normalization on the lower state = ", cnorm_l)
                print("normalization on the upper state = ", cnorm_u)
                print("orthogonality of the lower and upper wavefunctions (psi_0, psi_1^*) = ", orthog_lu)
                print("orthogonality of the upper and lower wavefunctions (psi_1, psi_0^*) = ", orthog_ul)
                print("overlap = ", overlp)

            # renormalization
            if cnorm_l > 0.0:
                psi[0] = [el / math.sqrt(abs(cnorm_l)) for el in psi[0]]
            if cnorm_u > 0.0:
                psi[1] = [el / math.sqrt(abs(cnorm_u)) for el in psi[1]]

            # calculating of a current energy
            phi = phys_base.hamil(psi, self.v, self.akx2, self.np, E)

            if l % 100 == 0:
                self.plot_test(l, phi[0], phi[1])

            cener_l = math_base.cprod(phi[0], psi[0], self.dx, self.np) - (E * orthog_ul)
            cener_u = math_base.cprod(phi[1], psi[1], self.dx, self.np) - (E.conjugate() * orthog_lu)
            if l % 10 == 0:
                print("energy on the lower state = ", cener_l.real)
                print("energy on the upper state = ", cener_u.real)

            # calculating of expectation values
            # for x
            momx_l = math_base.cprod2(psi[0], self.x, self.dx, self.np)
            momx_u = math_base.cprod2(psi[1], self.x, self.dx, self.np)

            # for x^2
            x2 = [el * el for el in self.x]

            momx2_l = math_base.cprod2(psi[0], x2, self.dx, self.np)
            momx2_u = math_base.cprod2(psi[1], x2, self.dx, self.np)

            # for p^2
            phi_kin_l = phys_base.diff(psi[0], self.akx2, self.np)
            phi_p2_l = [el * 2.0 * self.m for el in phi_kin_l]
            momp2_l = math_base.cprod(psi[0], phi_p2_l, self.dx, self.np)

            phi_kin_u = phys_base.diff(psi[1], self.akx2, self.np)
            phi_p2_u = [el * 2.0 * self.m for el in phi_kin_u]
            momp2_u = math_base.cprod(psi[1], phi_p2_u, self.dx, self.np)

            # for p
            akx = math_base.initak(self.np, self.dx, 1)
            akx = [el * phys_base.hart_to_cm / (-1j) / phys_base.dalt_to_au for el in akx]

            phip_l = phys_base.diff(psi[0], akx, self.np)
            momp_l = math_base.cprod(psi[0], phip_l, self.dx, self.np)

            phip_u = phys_base.diff(psi[1], akx, self.np)
            momp_u = math_base.cprod(psi[1], phip_u, self.dx, self.np)

            # plotting the result
            if l % 10 == 0:
                if l >= self.lmin:
                    self.plot(psi[0], t, self.x, self.np)
                    self.plot_up(psi[1], t, self.x, self.np)

                if l >= self.lmin:
                    self.plot_mom(t, momx_l, momx2_l, momp_l, momp2_l, cener_l.real, E.real)
                    self.plot_mom_up(t, momx_u, momx2_u, momp_u, momp2_u, cener_u.real, E.real)
