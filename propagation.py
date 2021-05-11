import math
import sys

import math_base
import phys_base


class PropagationSolver:
    def _warning_collocation_points(self, np_min):
        print("WARNING: The number of collocation points np = {} should be more than an estimated initial value {}. "
              "You've got a divergence!".format(self.np, np_min), sys.stderr)

    def __init__(
            self,
            psi_init,
            pot,
            plot,
            plot_mom,
            m=0.5,
            L=6.0,    # 0.2 -- for a model harmonic oscillator with a = 1.0 # 4.0 a_0 -- for single morse oscillator # 6.0 a_0 -- for dimensional harmonic oscillator
            np=512,  # 8192
            nch=64,
            T=40e-15,  # s -- for single morse oscillator
            nt=130000,
            x0=0,  # TODO: to fix x0 != 0
            p0=0,  # TODO: to fix p0 != 0
            a=1.0,
            De=20000.0,
            lmin=0):

        self.pot = pot
        self.psi_init = psi_init
        self.plot = plot
        self.plot_mom = plot_mom

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

        # calculating coordinate step of the problem
        self.dx = self.L / (self.np - 1)

        # setting the coordinate grid
        self.x = math_base.coord_grid(self.dx, self.np)
        #    print(x)

        # evaluating of potential(s)
        self.v = pot(self.x, self.m, self.De, self.a)
        #    print(v)

        # evaluating of initial wavefunction
        self.psi0 = psi_init(self.x, self.x0, self.p0, self.m, self.De, self.a)
        #    abs_psi0 = [abs(i) for i in psi0]
        #    print(abs_psi0)

        # initial normalization check
        self.cnorm0 = math_base.cprod(self.psi0, self.psi0, self.dx, self.np)
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

        # calculating of initial energy
        self.phi0 = phys_base.hamil(self.psi0, self.v, self.akx2, self.np)

        self.cener0 = math_base.cprod(self.phi0, self.psi0, self.dx, self.np)
        print("Initial energy: ", abs(self.cener0))

        # check if input data are correct in terms of the given problem
        # calculating the initial energy range of the Hamiltonian operator H
        self.emax0 = self.v[0] + abs(self.akx2[int(self.np / 2 - 1)]) + 2.0
        print("Initial emax = ", self.emax0)

        # calculating the initial minimum number of collocation points that is needed for convergence
        self.np_min0 = int(
            math.ceil(self.L *
                      math.sqrt(2 * self.m * self.emax0 * phys_base.dalt_to_au / phys_base.hart_to_cm) /
                      math.pi
            )
        )

        if self.np < self.np_min0:
            self._warning_collocation_points(self.np_min0)

    def time_propagation(self):
        # time propagation
        dt = self.T / (self.nt - 1)
        psi = []
        psi[:] = self.psi0[:]

        # main propagation loop
        #with open(os.path.join(OUT_PATH, file_abs), 'w') as f_abs, \
        #        open(os.path.join(OUT_PATH, file_real), 'w') as f_real, \
        #        open(os.path.join(OUT_PATH, file_mom), 'w') as f_mom:

        for l in range(1, self.nt + 1):
            # calculating the energy range of the Hamiltonian operator H
            emax = self.v[0] + abs(self.akx2[int(self.np / 2 - 1)]) + 2.0
            t_sc = dt * emax * phys_base.cm_to_erg / 4.0 / phys_base.Red_Planck_h

            if l % 10 == 0:
                print("emax = ", emax)
                print("Normalized scaled time interval = ", t_sc)

            # calculating the minimum number of collocation points and time steps that are needed for convergence
            nt_min = int(math.ceil(emax * self.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h))
            np_min = int(math.ceil(self.L * math.sqrt(2 * self.m * emax * phys_base.dalt_to_au / phys_base.hart_to_cm) / math.pi))

            if self.np < np_min and l % 10 == 0:
                self._warning_collocation_points(np_min)
            if self.nt < nt_min and l % 10 == 0:
                print("The number of time steps nt = {} should be more than an estimated value {}. \
    You've got a divergence!".format(self.nt, nt_min))  # TODO make _warning_time_steps

            psi = phys_base.prop(psi, t_sc, self.nch, self.np, self.v, self.akx2, emax)     # TODO move prop into this class

            cnorm = math_base.cprod(psi, psi, self.dx, self.np)
            overlp = math_base.cprod(self.psi0, psi, self.dx, self.np)

            t = dt * l
            if l % 10 == 0:
                print("l = ", l)
                print("t = ", t * 1e15, "fs")
                print("normalization = ", cnorm)
                print("overlap = ", overlp)

            # renormalization
            psi = [el / math.sqrt(abs(cnorm)) for el in psi]

            # calculating of a current energy
            phi = phys_base.hamil(psi, self.v, self.akx2, self.np)
            cener = math_base.cprod(psi, phi, self.dx, self.np)
            if l % 10 == 0:
                print("energy = ", cener.real)

            # calculating of expectation values
            # for x
            momx = math_base.cprod2(psi, self.x, self.dx, self.np)
            # for x^2
            x2 = [el * el for el in self.x]
            momx2 = math_base.cprod2(psi, x2, self.dx, self.np)
            # for p^2
            phi_kin = phys_base.diff(psi, self.akx2, self.np)
            #            phi_p2 = [el / (-coef_kin) for el in phi_kin]
            phi_p2 = [el * 2.0 * self.m for el in phi_kin]
            momp2 = math_base.cprod(psi, phi_p2, self.dx, self.np)
            # for p
            akx = math_base.initak(self.np, self.dx, 1)
            akx = [el * phys_base.hart_to_cm / (-1j) / phys_base.dalt_to_au for el in akx]
            phip = phys_base.diff(psi, akx, self.np)
            momp = math_base.cprod(psi, phip, self.dx, self.np)

            # plotting the result
            if l >= self.lmin:
                self.plot(psi, t, self.x, self.np)  #, f_abs, f_real)
            if l >= self.lmin:
                self.plot_mom(t, momx, momx2, momp, momp2, cener.real) #, f_mom)
