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

        conf_prop = conf.fitter.propagation
        self.solver = propagation.PropagationSolver(
            self.psi_init, self.pot,
            report_static=self.report_static,
            report_dynamic=self.report_dynamic,
            process_instrumentation=self.process_instrumentation,
            laser_field_envelope=self.LaserFieldEnvelope,
            dynamic_state_factory=self.fitter_dynamic_state_factory,
            conf_prop=conf_prop)


    def time_propagation(self):
        self.solver.time_propagation()


    def report_static(self, stat: propagation.PropagationSolver.StaticState):
        self.stat_saved = stat
        self.dt = stat.dt

        # check if input data are correct in terms of the given problem
        # calculating the initial energy range of the Hamiltonian operator H
        emax0 = stat.v[0][1][0] + abs(stat.akx2[int(self.conf.fitter.propagation.np / 2 - 1)]) + 2.0
        emin0 = stat.v[0][0]

        # calculating the initial minimum number of collocation points that is needed for convergence
        np_min0 = int(
            math.ceil(self.conf.fitter.propagation.L *
                      math.sqrt(
                          2.0 * self.conf.fitter.propagation.m * (emax0 - emin0) * phys_base.dalt_to_au / phys_base.hart_to_cm) /
                      math.pi
                      )
        )

        # calculating the initial minimum number of time steps that is needed for convergence
        nt_min0 = int(
            math.ceil((emax0 - emin0) * self.conf.fitter.propagation.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h
                      )
        )

        if self.conf.fitter.propagation.np < np_min0:
            self._warning_collocation_points(self.conf.fitter.propagation.np, np_min0)
        if self.conf.fitter.propagation.nt < nt_min0:
            self._warning_time_steps(self.conf.fitter.propagation.nt, nt_min0)

        cener0_tot = stat.cener0 + stat.cener0_u
        overlp0_abs = abs(stat.overlp00) + abs(stat.overlpf0)

        # plotting initial values
        self.plot(stat.psi0[0], 0.0, stat.x, self.conf.fitter.propagation.np)
        self.plot_up(stat.psi0[1], 0.0, stat.x, self.conf.fitter.propagation.np)

        self.plot_mom(0.0, stat.moms0, stat.cener0.real, stat.E00.real, stat.overlp00, cener0_tot.real,
                 abs(stat.psi0[0][520]), stat.psi0[0][520].real) # TODO: replace 520 by expression
        self.plot_mom_up(0.0, stat.moms0, stat.cener0_u.real, stat.E00.real, stat.overlpf0, overlp0_abs,
                    abs(stat.psi0[1][520]), stat.psi0[1][520].real) # TODO: replace 520 by expression

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
            (instr.emax - instr.emin) * self.conf.fitter.propagation.T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h))
        np_min = int(math.ceil(
            self.conf.fitter.propagation.L * math.sqrt(
                2.0 * self.conf.fitter.propagation.m * (
                            instr.emax - instr.emin) * phys_base.dalt_to_au / phys_base.hart_to_cm) / math.pi))

        cener = instr.cener_l + instr.cener_u
        overlp_abs = abs(instr.overlp0) + abs(instr.overlpf)

        time_span = instr.time_after - instr.time_before
        milliseconds_per_step = time_span.microseconds / 1000
        self.milliseconds_full += milliseconds_per_step

        # local control algorithm
        coef = 2.0 * phys_base.cm_to_erg / phys_base.Red_Planck_h
        dAdt = self.dyn_ref.E * instr.psigc_psie.imag * coef

        if self.conf.fitter.task_type != config.RootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL:
            res = propagation.PropagationSolver.StepReaction.OK
        else:
            if dAdt >= 0.0:
                res = propagation.PropagationSolver.StepReaction.OK
                self.dAdt_happy = dAdt
            else:
                if abs(instr.psigc_psie.imag) > self.conf.fitter.epsilon:
                    self.E_patched = -self.dAdt_happy / (instr.psigc_psie.imag * coef)
                else:
                    print("Imaginary part in dA/dt is too small and has been replaces by epsilon")
                    self.E_patched = self.dAdt_happy / (self.conf.fitter.epsilon * coef)
                res = propagation.PropagationSolver.StepReaction.CORRECT

        # plotting the result
        if self.dyn_ref.l % self.conf.output.mod_fileout == 0:
            if self.dyn_ref.l >= self.conf.output.lmin:
                self.plot(self.dyn_ref.psi[0], t, self.stat_saved.x, self.conf.fitter.propagation.np)
                self.plot_up(self.dyn_ref.psi[1], t, self.stat_saved.x, self.conf.fitter.propagation.np)

            if self.dyn_ref.l >= self.conf.output.lmin:
                self.plot_mom(t, instr.moms, instr.cener_l.real, self.dyn_ref.E, instr.overlp0, cener.real,
                         abs(self.dyn_ref.psi[0][520]), self.dyn_ref.psi[0][520].real) # TODO: replace 520 by expression
                self.plot_mom_up(t, instr.moms, instr.cener_u.real, instr.E_full.real, instr.overlpf, overlp_abs,
                            abs(self.dyn_ref.psi[1][520]), self.dyn_ref.psi[1][520].real) # TODO: replace 520 by expression

        if self.dyn_ref.l % self.conf.output.mod_stdout == 0:
            if self.conf.fitter.propagation.np < np_min:
                self._warning_collocation_points(self.conf.fitter.propagation.np, np_min)
            if self.conf.fitter.propagation.nt < nt_min:
                self._warning_time_steps(self.conf.fitter.propagation.nt, nt_min)

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
                print("CORRECTING THE ITERATION")

        self.res_saved = res
        return res


    # calculating envelope of the laser field energy at the given time value
    def LaserFieldEnvelope(self, stat: propagation.PropagationSolver.StaticState,
                           dyn: propagation.PropagationSolver.DynamicState):
        t = stat.dt * dyn.l
        self.E_patched = phys_base.laser_field(self.conf.fitter.propagation.E0, t, self.conf.fitter.propagation.t0, self.conf.fitter.propagation.sigma)

        # transition without control
        if self.conf.fitter.task_type == config.RootConfiguration.FitterConfiguration.TaskType.TRANS_WO_CONTROL:
            E = self.E_patched
        # intuitive control algorithm
        elif self.conf.fitter.task_type == config.RootConfiguration.FitterConfiguration.TaskType.INTUITIVE_CONTROL:
            for npul in range(1, self.conf.fitter.impulses_number):
                self.E_patched += phys_base.laser_field(self.conf.fitter.propagation.E0, t,
                                            self.conf.fitter.propagation.t0 + (npul * self.conf.fitter.delay),
                                            self.conf.fitter.propagation.sigma)
            E = self.E_patched
        # local control algorithm (with A = Pe)
        elif self.conf.fitter.task_type == config.RootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL and \
                self.conf.fitter.task_subtype == config.RootConfiguration.FitterConfiguration.TaskSubType.GOAL_POPULATION:
            if self.dyn_ref.E == 0.0:
                self.dyn_ref.E = self.E_patched

            if self.E_patched <= 0:
                raise RuntimeError("E_patched has to be positive")

            if self.dyn_ref.E <= 0:
                raise RuntimeError("E has to be positive")

            # solving dynamic equation for E
            # linear difference to the "desired" value
            first = self.dyn_ref.E - self.E_patched
            # decay term
            second = self.dyn_ref.E_vel * math.pow(self.E_patched / self.dyn_ref.E, self.conf.fitter.pow)

            # Euler
            E_acc = -self.conf.fitter.k_E * first - self.conf.fitter.lamb * second
            self.dyn_ref.E_vel += E_acc * stat.dt
            E = self.dyn_ref.E + self.dyn_ref.E_vel * stat.dt
        else:
            E = self.E_patched

        return E
