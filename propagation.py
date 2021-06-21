import math
import cmath
import sys
import copy
import numpy

import datetime

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
            m, L, np, nch, T, nt, x0, p0, a, De, E0,
            t0, sigma, nu_L, lmin):

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

        if not E0 >= 0.0 or not sigma > 0.0 or not nu_L >= 0.0:
            raise ValueError("The value of an amplitude value of the laser field energy envelope 'E0',"
                             "of a scaling parameter of the laser field envelope 'sigma'"
                             "and of a basic frequency of the laser field 'nu_L' must be positive")

        # calculating coordinate step of the problem
        self.dx = self.L / (self.np - 1)

        # setting the coordinate grid
        self.x = math_base.coord_grid(self.dx, self.np)

        # evaluating of potential(s)
        self.v = pot(self.x, self.np, self.m, self.De, self.a)

        # evaluating of initial wavefunction
        self.psi0 = psi_init(self.x, self.np, self.x0, self.p0, self.m, self.De, self.a)

        # initial normalization check
        self.cnorm0 = math_base.cprod(self.psi0[0], self.psi0[0], self.dx, self.np)

        # evaluating of the final goal -- upper state wavefunction
        self.psif = psi_init(self.x, self.np, self.x0, self.p0, self.m, self.De / 2.0, self.a)

        # final normalization check
        self.cnormf = math_base.cprod(self.psif[0], self.psif[0], self.dx, self.np)

        # evaluating of k vector
        self.akx2 = math_base.initak(self.np, self.dx, 2)

        # evaluating of kinetic energy
        self.coef_kin = -phys_base.hart_to_cm / (2.0 * self.m * phys_base.dalt_to_au)
        self.akx2 *= self.coef_kin

        # calculating of initial energy
        # ground state
        self.phi0 = phys_base.hamil(self.psi0[0], self.v[0][1], self.akx2, self.np)
        self.cener0 = math_base.cprod(self.phi0, self.psi0[0], self.dx, self.np)

        # excited state
        self.phi0_u = phys_base.hamil(self.psi0[1], self.v[1][1], self.akx2, self.np)
        self.cener0_u = math_base.cprod(self.phi0_u, self.psi0[1], self.dx, self.np)

        # check if input data are correct in terms of the given problem
        # calculating the initial energy range of the Hamiltonian operator H
        self.emax0 = self.v[0][1][0] + abs(self.akx2[int(self.np / 2 - 1)]) + 2.0
        self.emin0 = self.v[0][0]

        print(" Initial state features: ")
        print("Initial normalization: ", abs(self.cnorm0))
        print(" Final goal features: ")
        print("Final goal normalization: ", abs(self.cnormf))
        print(" Initial energy: ", abs(self.cener0))
        print(" Initial emax = ", self.emax0)

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

        # initial laser field energy
        E00 = phys_base.laser_field(self.E0, 0.0, self.t0, self.sigma)

        # initial population
        overlp00 = math_base.cprod(self.psi0[0], psi[0], self.dx, self.np)
        overlpf0 = math_base.cprod(self.psif[0], psi[1], self.dx, self.np)

        # calculating of initial expectation values
        # for x
        momx_l = math_base.cprod2(psi[0], self.x, self.dx, self.np)
        momx_u = math_base.cprod2(psi[1], self.x, self.dx, self.np)

        # for x^2
        x2 = numpy.multiply(self.x, self.x)
        momx2_l = math_base.cprod2(psi[0], x2, self.dx, self.np)
        momx2_u = math_base.cprod2(psi[1], x2, self.dx, self.np)

        # for p^2
        phi_kin_l = phys_base.diff(psi[0], self.akx2, self.np)
        phi_p2_l = phi_kin_l * (2.0 * self.m)
        momp2_l = math_base.cprod(psi[0], phi_p2_l, self.dx, self.np)

        phi_kin_u = phys_base.diff(psi[1], self.akx2, self.np)
        phi_p2_u = phi_kin_u * (2.0 * self.m)
        momp2_u = math_base.cprod(psi[1], phi_p2_u, self.dx, self.np)

        # for p
        akx = math_base.initak(self.np, self.dx, 1)
        akx_mul = phys_base.hart_to_cm / (-1j) / phys_base.dalt_to_au
        akx *= akx_mul

        phip_l = phys_base.diff(psi[0], akx, self.np)
        momp_l = math_base.cprod(psi[0], phip_l, self.dx, self.np)

        phip_u = phys_base.diff(psi[1], akx, self.np)
        momp_u = math_base.cprod(psi[1], phip_u, self.dx, self.np)

        # plotting initial values
        self.plot(psi[0], 0.0, self.x, self.np)
        self.plot_up(psi[1], 0.0, self.x, self.np)

        self.plot_mom(0.0, momx_l, momx2_l, momp_l, momp2_l, self.cener0.real, E00.real, overlp00)
        self.plot_mom_up(0.0, momx_u, momx2_u, momp_u, momp2_u, self.cener0_u.real, E00.real, overlpf0)


        milliseconds_full = 0

        # main propagation loop
        for l in range(1, self.nt + 1):
            time_before = datetime.datetime.now()

            # calculating limits of energy ranges of the one-dimensional Hamiltonian operator H_l
            emax_l = self.v[0][1][0] + abs(self.akx2[int(self.np / 2 - 1)]) + 2.0
            emin_l = self.v[0][0]
            # calculating limits of energy ranges of the one-dimensional Hamiltonian operator H_u
            emax_u = self.v[1][1][0] + abs(self.akx2[int(self.np / 2 - 1)]) + 2.0
            emin_u = self.v[1][0]

            t = dt * l
            # Here we're transforming the problem to the one for psi_omega
            psi_omega = []
            exp_L = cmath.exp(1j * math.pi * self.nu_L * t)
            psi_omega_l = psi[0] / exp_L
            psi_omega.append(psi_omega_l)
            psi_omega_u = psi[1] * exp_L
            psi_omega.append(psi_omega_u)

            # New energy ranges
            eL = self.nu_L * phys_base.Hz_to_cm / 2.0
            emax_l_omega = emax_l + self.E0 + eL
            emin_l_omega = emin_l - self.E0 + eL

            emax_u_omega = emax_u + self.E0 - eL
            emin_u_omega = emin_u - self.E0 - eL

            emax = max(emax_l_omega, emin_l_omega, emax_u_omega, emin_u_omega)
            emin = min(emax_l_omega, emin_l_omega, emax_u_omega, emin_u_omega)

            t_sc = dt * (emax - emin) * phys_base.cm_to_erg / 4.0 / phys_base.Red_Planck_h

            # calculating the minimum number of collocation points and time steps that are needed for convergence
            nt_min = int(math.ceil((emax - emin) * self.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h))
            np_min = int(math.ceil(self.L * math.sqrt(2.0 * self.m * (emax - emin) * phys_base.dalt_to_au / phys_base.hart_to_cm) / math.pi))

            E = phys_base.laser_field(self.E0, t, self.t0, self.sigma)
            E_full = E * exp_L * exp_L
            psi_omega = phys_base.prop(psi_omega, t_sc, self.nch, self.np, self.v, self.akx2, emin, emax, E, eL) # TODO move prop into this class

            cnorm_l = math_base.cprod(psi_omega[0], psi_omega[0], self.dx, self.np)
            cnorm_u = math_base.cprod(psi_omega[1], psi_omega[1], self.dx, self.np)
            orthog_lu = math_base.cprod(psi_omega[0], psi_omega[1], self.dx, self.np) * exp_L * exp_L
            orthog_ul = math_base.cprod(psi_omega[1], psi_omega[0], self.dx, self.np) / exp_L / exp_L

            # renormalization
            if cnorm_l > 0.0:
                psi_omega[0] /= math.sqrt(abs(cnorm_l))
            if cnorm_u > 0.0:
                psi_omega[1] /= math.sqrt(abs(cnorm_u))

            # calculating of a current energy
            phi_omega = phys_base.hamil2D(psi_omega, self.v, self.akx2, self.np, E, eL)

            cener_l = math_base.cprod(phi_omega[0], psi_omega[0], self.dx, self.np) - eL + E_full * orthog_ul
            cener_u = math_base.cprod(phi_omega[1], psi_omega[1], self.dx, self.np) + eL + E_full.conjugate() * orthog_lu

            # converting back to psi
            psi[0] = psi_omega[0] * exp_L
            psi[1] = psi_omega[1] / exp_L

#            if l % 100 == 0:
#                self.plot_test(l, phi[0], phi[1])

            overlp0 = math_base.cprod(self.psi0[0], psi[0], self.dx, self.np)
            overlpf = math_base.cprod(self.psif[0], psi[1], self.dx, self.np)

            # calculating of expectation values
            # for x
            momx_l = math_base.cprod2(psi[0], self.x, self.dx, self.np)
            momx_u = math_base.cprod2(psi[1], self.x, self.dx, self.np)

            # for x^2
            momx2_l = math_base.cprod2(psi[0], x2, self.dx, self.np)
            momx2_u = math_base.cprod2(psi[1], x2, self.dx, self.np)

            # for p^2
            phi_kin_l = phys_base.diff(psi[0], self.akx2, self.np)
            phi_p2_l = phi_kin_l * (2.0 * self.m)
            momp2_l = math_base.cprod(psi[0], phi_p2_l, self.dx, self.np)

            phi_kin_u = phys_base.diff(psi[1], self.akx2, self.np)
            phi_p2_u = phi_kin_u * (2.0 * self.m)
            momp2_u = math_base.cprod(psi[1], phi_p2_u, self.dx, self.np)

            # for p
            phip_l = phys_base.diff(psi[0], akx, self.np)
            momp_l = math_base.cprod(psi[0], phip_l, self.dx, self.np)

            phip_u = phys_base.diff(psi[1], akx, self.np)
            momp_u = math_base.cprod(psi[1], phip_u, self.dx, self.np)

            # plotting the result
            if l % 100 == 0:
                if l >= self.lmin:
                    self.plot(psi[0], t, self.x, self.np)
                    self.plot_up(psi[1], t, self.x, self.np)

                if l >= self.lmin:
                    self.plot_mom(t, momx_l, momx2_l, momp_l, momp2_l, cener_l.real, E_full.real, overlp0)
                    self.plot_mom_up(t, momx_u, momx2_u, momp_u, momp2_u, cener_u.real, E_full.real, overlpf)

            time_after = datetime.datetime.now()
            time_span = time_after - time_before
            milliseconds_per_step = time_span.microseconds / 1000
            milliseconds_full += milliseconds_per_step

            if l % 100 == 0:
                if self.np < np_min:
                    self._warning_collocation_points(np_min)
                if self.nt < nt_min:
                    self._warning_time_steps(nt_min)

                print("l = ", l)
                print("t = ", t * 1e15, "fs")

                print("emax = ", emax)
                print("emin = ", emin)
                print("normalized scaled time interval = ", t_sc)
                print("normalization on the lower state = ", cnorm_l)
                print("normalization on the upper state = ", cnorm_u)
                print("orthogonality of the lower and upper wavefunctions (psi_0, psi_1^*) = ", orthog_lu)
                print("orthogonality of the upper and lower wavefunctions (psi_1, psi_0^*) = ", orthog_ul)
                print("overlap with initial wavefunction = ", overlp0)
                print("overlap with final goal wavefunction = ", overlpf)
                print("energy on the lower state = ", cener_l.real)
                print("energy on the upper state = ", cener_u.real)

                print("milliseconds per step: " + str(milliseconds_per_step) + ", on average: " + str(milliseconds_full / l))
