import math

import propagation
import phys_base
import config


class FittingSolver:
    class FitterDynamicState(propagation.PropagationSolver.DynamicState):
        def __init__(self, l=0, psi=None, E=0.0, E_vel=0.0):
            super().__init__(l, psi, E)
            self.E_vel = E_vel

    def __init__(
            self,
            conf,
            psi_init,
            pot,
            _warning_collocation_points,
            _warning_time_steps,
            plot,
            plot_up,
            plot_mom,
            plot_mom_up
    ):
        self.conf=conf
        self.psi_init=psi_init
        self.pot=pot
        self._warning_collocation_points=_warning_collocation_points
        self._warning_time_steps=_warning_time_steps
        self.plot=plot
        self.plot_up=plot_up
        self.plot_mom=plot_mom
        self.plot_mom_up=plot_mom_up
        self.dt = 0
        self.stat_saved = propagation.PropagationSolver.StaticState()
        self.dyn_ref = FittingSolver.FitterDynamicState()
        self.milliseconds_full = 0
        self.res_saved = propagation.PropagationSolver.StepReaction.OK
        self.E_patched = 0.0
        self.dAdt_happy = 0.0

        self.solver = propagation.PropagationSolver(
            self.psi_init, self.pot,
            report_static=self.report_static,
            report_dynamic=self.report_dynamic,
            process_instrumentation=self.process_instrumentation,
            laser_field_envelope=self.LaserFieldEnvelope,
            dynamic_state_factory=self.fitter_dynamic_state_factory,
            m=conf.phys_syst_pars.m, L=conf.phys_calc_pars.L,
            np=conf.alg_calc_pars.np, nch=conf.alg_calc_pars.nch,
            T=conf.phys_calc_pars.T, nt=conf.alg_calc_pars.nt,
            x0=conf.init_conditions.x0, p0=conf.init_conditions.p0,
            a=conf.potential_pars.a, De=conf.potential_pars.De,
            x0p=conf.potential_pars.x0p, E0=conf.laser_field_pars.E0,
            t0=conf.laser_field_pars.t0, sigma=conf.laser_field_pars.sigma,
            nu_L=conf.laser_field_pars.nu_L, delay=conf.laser_field_pars.delay)


    def time_propagation(self):
        self.solver.time_propagation()


    def report_static(self, stat: propagation.PropagationSolver.StaticState):
        self.stat_saved = stat
        self.dt = stat.dt

        # check if input data are correct in terms of the given problem
        # calculating the initial energy range of the Hamiltonian operator H
        emax0 = stat.v[0][1][0] + abs(stat.akx2[int(self.conf.alg_calc_pars.np / 2 - 1)]) + 2.0
        emin0 = stat.v[0][0]

        # calculating the initial minimum number of collocation points that is needed for convergence
        np_min0 = int(
            math.ceil(self.conf.phys_calc_pars.L *
                      math.sqrt(
                          2.0 * self.conf.phys_syst_pars.m * (emax0 - emin0) * phys_base.dalt_to_au / phys_base.hart_to_cm) /
                      math.pi
                      )
        )

        # calculating the initial minimum number of time steps that is needed for convergence
        nt_min0 = int(
            math.ceil((emax0 - emin0) * self.conf.phys_calc_pars.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h
                      )
        )

        if self.conf.alg_calc_pars.np < np_min0:
            self._warning_collocation_points(self.conf.alg_calc_pars.np, np_min0)
        if self.conf.alg_calc_pars.nt < nt_min0:
            self._warning_time_steps(self.conf.alg_calc_pars.nt, nt_min0)

        cener0_tot = stat.cener0 + stat.cener0_u
        overlp0_abs = abs(stat.overlp00) + abs(stat.overlpf0)

        # plotting initial values
        self.plot(stat.psi0[0], 0.0, stat.x, self.conf.alg_calc_pars.np)
        self.plot_up(stat.psi0[1], 0.0, stat.x, self.conf.alg_calc_pars.np)

        self.plot_mom(0.0, stat.moms0, stat.cener0.real, stat.E00.real, stat.overlp00, cener0_tot.real,
                 abs(stat.psi0[0][520]), stat.psi0[0][520].real)
        self.plot_mom_up(0.0, stat.moms0, stat.cener0_u.real, stat.E00.real, stat.overlpf0, overlp0_abs,
                    abs(stat.psi0[1][520]), stat.psi0[1][520].real)

        print("Initial emax = ", emax0)

        print(" Initial state features: ")
        print("Initial normalization: ", abs(stat.cnorm0))
        print("Initial energy: ", abs(stat.cener0))

        print(" Final goal features: ")
        print("Final goal normalization: ", abs(stat.cnormf))
        print("Final goal energy: ", abs(stat.cenerf))


    def fitter_dynamic_state_factory(self, l, psi, E):
        return FittingSolver.FitterDynamicState(l, psi, E, 0.0)


    def report_dynamic(self, dyn: FitterDynamicState):
        self.dyn_ref = dyn


    def process_instrumentation(self, instr: propagation.PropagationSolver.InstrumentationOutputData):
        t = self.dt * self.dyn_ref.l

        # calculating the minimum number of collocation points and time steps that are needed for convergence
        nt_min = int(math.ceil(
            (instr.emax - instr.emin) * self.conf.phys_calc_pars.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h))
        np_min = int(math.ceil(
            self.conf.phys_calc_pars.L * math.sqrt(
                2.0 * self.conf.phys_syst_pars.m * (
                            instr.emax - instr.emin) * phys_base.dalt_to_au / phys_base.hart_to_cm) / math.pi))

        cener = instr.cener_l + instr.cener_u
        overlp_abs = abs(instr.overlp0) + abs(instr.overlpf)

        time_span = instr.time_after - instr.time_before
        milliseconds_per_step = time_span.microseconds / 1000
        self.milliseconds_full += milliseconds_per_step

        # local control algorithm
        coef = 2.0 * phys_base.cm_to_erg / phys_base.Red_Planck_h
        dAdt = self.dyn_ref.E * instr.psigc_psie.imag * coef

        if self.conf.phys_calc_pars.task_type != config.TaskType.LOCAL_CONTROL:
            res = propagation.PropagationSolver.StepReaction.OK
        else:
            if dAdt >= 0.0:
                res = propagation.PropagationSolver.StepReaction.OK
                self.dAdt_happy = dAdt
            else:
                if abs(instr.psigc_psie.imag) > self.conf.alg_calc_pars.epsilon:
                    self.E_patched = -self.dAdt_happy / (instr.psigc_psie.imag * coef)
                else:
                    print("Imaginary part in dA/dt is too small and has been replaces by epsilon")
                    self.E_patched = self.dAdt_happy / (self.conf.alg_calc_pars.epsilon * coef)
                res = propagation.PropagationSolver.StepReaction.REPEAT

        # plotting the result
        if self.dyn_ref.l % self.conf.print_pars.mod_fileout == 0: # and res == propagation.PropagationSolver.StepReaction.OK:
            if self.dyn_ref.l >= self.conf.print_pars.lmin:
                self.plot(self.dyn_ref.psi[0], t, self.stat_saved.x, self.conf.alg_calc_pars.np)
                self.plot_up(self.dyn_ref.psi[1], t, self.stat_saved.x, self.conf.alg_calc_pars.np)

            if self.dyn_ref.l >= self.conf.print_pars.lmin:
                self.plot_mom(t, instr.moms, instr.cener_l.real, self.dyn_ref.E, instr.overlp0, cener.real,
                         abs(self.dyn_ref.psi[0][520]), self.dyn_ref.psi[0][520].real)
                self.plot_mom_up(t, instr.moms, instr.cener_u.real, instr.E_full.real, instr.overlpf, overlp_abs,
                            abs(self.dyn_ref.psi[1][520]), self.dyn_ref.psi[1][520].real)

        if self.dyn_ref.l % self.conf.print_pars.mod_stdout == 0:
            if self.conf.alg_calc_pars.np < np_min:
                self._warning_collocation_points(self.conf.alg_calc_pars.np, np_min)
            if self.conf.alg_calc_pars.nt < nt_min:
                self._warning_time_steps(self.conf.alg_calc_pars.nt, nt_min)

            print("l = ", self.dyn_ref.l)
            print("t = ", t * 1e15, "fs")

            print("emax = ", instr.emax)
            print("emin = ", instr.emin)
            print("normalized scaled time interval = ", instr.t_sc)
            print("normalization on the lower state = ", abs(instr.cnorm_l))
            print("normalization on the upper state = ", abs(instr.cnorm_u))
            print("overlap with initial wavefunction = ", abs(instr.overlp0))
            print("overlap with final goal wavefunction = ", abs(instr.overlpf))
            print("energy on the lower state = ", instr.cener_l.real)
            print("energy on the upper state = ", instr.cener_u.real)
            print("Time derivation of the expectation value from the goal operator A = ", dAdt)

            print(
                "milliseconds per step: " + str(milliseconds_per_step) + ", on average: " + str(
                    self.milliseconds_full / self.dyn_ref.l))

            if res != propagation.PropagationSolver.StepReaction.OK:
                print("REPEATNG THE ITERATION")

        self.res_saved = res
        return res


    # calculating envelope of the laser field energy at the given time value
    def LaserFieldEnvelope(self, stat: propagation.PropagationSolver.StaticState,
                           dyn: propagation.PropagationSolver.DynamicState):
        if self.res_saved == propagation.PropagationSolver.StepReaction.OK:
            t = stat.dt * dyn.l
            self.E_patched = phys_base.laser_field(self.conf.laser_field_pars.E0, t, self.conf.laser_field_pars.t0, self.conf.laser_field_pars.sigma)
            # intuitive control algorithm
            if self.conf.phys_calc_pars.task_type == config.TaskType.INTUITIVE_CONTROL:
                for npul in range(1, self.conf.laser_field_pars.impulses_number):
                    self.E_patched += phys_base.laser_field(self.conf.laser_field_pars.E0, t,
                                               self.conf.laser_field_pars.t0 + (npul * self.conf.laser_field_pars.delay),
                                               self.conf.laser_field_pars.sigma)
        elif self.res_saved == propagation.PropagationSolver.StepReaction.REPEAT:
            pass
        else:
            raise RuntimeError("Impossible case")

        # Solving dynamic equation for E
        phi0 = 1.0e+29
        phi1 = 1.0e+35

        # Linear difference to the "wished"
        first = self.dyn_ref.E - self.E_patched

        if self.dyn_ref.E == 0:
            self.dyn_ref.E = self.E_patched

        if self.E_patched <= 0:
            raise RuntimeError("E_patched has to be positive")

        if self.dyn_ref.E <= 0:
            raise RuntimeError("E has to be positive")

        second = self.E_patched * math.log(self.dyn_ref.E / self.E_patched)

        BIG_NUMBER = 10
        if math.fabs(phi1 * second) > math.fabs(phi0 * first) * BIG_NUMBER:
            # Approx
            E0 = self.dyn_ref.E
            Ep = self.E_patched
            v0 = self.dyn_ref.E_vel

            if dyn.l % 200 == 0:
                print("approximating, current velocity is %f" % v0)

            A = phi0 * (E0 - Ep) + phi1 * Ep * math.log(E0 / Ep)
            B = phi1 * Ep / E0 + phi0
            arg0 = -math.atan(v0 * math.sqrt(B) / A)
            C0 = A / (B * math.cos(arg0))

            new_delta = C0 * math.cos(arg0 + math.sqrt(B) * stat.dt) - A / B

            E = E0 + new_delta
            self.dyn_ref.E_vel = -C0 * math.sqrt(B) * math.sin(arg0 + math.sqrt(B) * stat.dt)
        else:
            # Euler
            E_acc = -phi0 * first - phi1 * second
            self.dyn_ref.E_vel += E_acc * stat.dt
            E = self.dyn_ref.E + self.dyn_ref.E_vel * stat.dt

        #print("E_approx=%f, E_approx_vel=%f, E_euler=%f, E_euler_vel=%f" % (E_approx, E_vel_approx, E, self.dyn_ref.E_vel))
        return E
