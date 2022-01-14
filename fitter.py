import reporter
from config import RootConfiguration
from propagation import *


class FittingSolver:
    class FitterDynamicState():
        def __init__(self, E_vel=0.0, freq_mult_vel=0.0, iter_step=0):
            self.E_vel = E_vel
            self.freq_mult_vel = freq_mult_vel
            self.iter_step = iter_step
            self.propagation_dyn_ref = None

    def __init__(
            self,
            conf_fitter,
            psi_init,
            psi_goal,
            pot,
            reporter: reporter.FitterReporter,
            _warning_collocation_points,
            _warning_time_steps
    ):
        self.conf_fitter = conf_fitter
        self.psi_init = psi_init
        self.psi_goal = psi_goal
        self.pot = pot
        self.reporter = reporter
        self.dyn = None
        self._warning_collocation_points = _warning_collocation_points
        self._warning_time_steps = _warning_time_steps
        self.E_patched = 0.0
        self.freq_mult_patched = 1.0
        self.dAdt_happy = 0.0
        self.chi_tlist = []
        self.goal_close = 0.0
        self.res = PropagationSolver.StepReaction.OK

        self.E_int = 0.0
        self.E_tlist = []

        conf_prop = conf_fitter.propagation
        self.propagation_reporter = self.reporter.create_propagation_reporter("one_and_only")
        self.propagation_reporter.open()
        self.solver = PropagationSolver(
            pot=self.pot,
            _warning_collocation_points=self._warning_collocation_points,
            _warning_time_steps=self._warning_time_steps,
            reporter=self.propagation_reporter,
            laser_field_envelope=self.LaserFieldEnvelope,
            freq_multiplier=self.FreqMultiplier,
            dynamic_state_factory=self.propagation_dynamic_state_factory,
            mod_log=conf_fitter.mod_log,
            conf_prop=conf_prop)


    def time_propagation(self, dx, x):
        self.dyn = FittingSolver.FitterDynamicState(0.0, 0.0, 0)

        #with reporter.PlotReporter(RootConfiguration.OutputPlotConfiguration("0f")) as reporter_imp:
        if self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV or \
                self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_GRADIENT:
            print("Iteration = ", self.dyn.iter_step, ", Forward direction begins...")
    #    self.reporter = reporter_imp
        self.solver.laser_field_envelope = self.LaserFieldEnvelope
        self.solver.start(dx, x, self.psi_init, self.psi_goal, PropagationSolver.Direction.FORWARD)
        self.dyn.propagation_dyn_ref = self.solver.dyn
        self.E_tlist.append(0.0)

        # main propagation loop
        while self.solver.step(0.0):
            self.do_the_thing(self.solver.instr)

        if self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV or \
           self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_GRADIENT:

            assert self.solver.dyn.l - 1 == self.conf_fitter.propagation.nt
            self.goal_close = math_base.cprod(self.solver.stat.psif[1], self.solver.dyn.psi[1], self.solver.stat.dx,
                                                  self.conf_fitter.propagation.np)

            if abs(self.goal_close - 1.0) > self.conf_fitter.epsilon:
                print("Iteration = ", self.dyn.iter_step, ", Backward direction begins...")
                self.res = PropagationSolver.StepReaction.ITERATE

                chiT = []
                chiT.append(numpy.array([0.0] * self.conf_fitter.propagation.np).astype(complex))
                chiT.append(self.goal_close * self.solver.stat.psif[1])
                self.chi_tlist.append(chiT)

                ##with reporter.PlotReporter(RootConfiguration.OutputPlotConfiguration("0b")) as reporter_imp:
                #    self.reporter = reporter_imp
                self.solver.laser_field_envelope = self.LaserFieldEnvelopeBackward
                self.solver.start(dx, x, chiT, self.psi_init, PropagationSolver.Direction.BACKWARD)
                self.dyn.propagation_dyn_ref = self.solver.dyn
                # propagation loop for the 0-th back iteration
                while self.solver.step(self.conf_fitter.propagation.T):
                    self.chi_tlist.append(self.solver.dyn.psi_omega)
                    self.do_the_thing(self.solver.instr)
            else:
                print("The goal has been reached on the very first iteration. You don't need the control!")
                self.res = PropagationSolver.StepReaction.OK

            # iterative procedure
            while abs(self.goal_close - 1.0) > self.conf_fitter.epsilon or \
                    self.dyn.iter_step <= self.conf_fitter.iter_max:
                self.dyn.iter_step += 1
                print("Iteration = ", self.dyn.iter_step, ", Forward direction begins...")
                self.res = PropagationSolver.StepReaction.ITERATE

                #with reporter.PlotReporter(RootConfiguration.OutputPlotConfiguration(f"{self.dyn.iter_step}f")) as reporter_imp:
                #    self.reporter = reporter_imp
                self.solver.laser_field_envelope = self.LaserFieldEnvelope
                self.solver.start(dx, x, self.psi_init, self.psi_goal, PropagationSolver.Direction.FORWARD)
                self.dyn.propagation_dyn_ref = self.solver.dyn

                # propagation loop for the next forward iterations
                self.E_int = 0.0
                self.E_tlist = []
                self.E_tlist.append(0.0)
                while self.solver.step(0.0):
                    self.do_the_thing(self.solver.instr)

                print("Iteration = ", self.dyn.iter_step, ", Backward direction begins...")

                chiT = []
                assert self.solver.dyn.l - 1 == self.conf_fitter.propagation.nt
                self.goal_close = math_base.cprod(self.solver.stat.psif[1], self.solver.dyn.psi[1],
                                                      self.solver.stat.dx, self.conf_fitter.propagation.np)
                chiT.append(numpy.array([0.0] * self.conf_fitter.propagation.np).astype(complex))
                chiT.append(self.goal_close * self.solver.stat.psif[1])

                #with reporter.PlotReporter(RootConfiguration.OutputPlotConfiguration(f"{self.dyn.iter_step}b")) as reporter_imp:
                #    self.reporter = reporter_imp
                self.solver.laser_field_envelope = self.LaserFieldEnvelopeBackward
                self.solver.start(dx, x, chiT, self.psi_init, PropagationSolver.Direction.BACKWARD)
                self.dyn.propagation_dyn_ref = self.solver.dyn
                # propagation loop for the next backward iterations
                chi_tlist_new = []
                chi_tlist_new.append(chiT)
                while self.solver.step(self.conf_fitter.propagation.T):
                    chi_tlist_new.append(self.solver.dyn.psi_omega)
                    self.do_the_thing(self.solver.instr)

                self.chi_tlist[:] = chi_tlist_new[:]
            self.res = PropagationSolver.StepReaction.OK
        self.propagation_reporter.close()


    def propagation_dynamic_state_factory(self, l, t, psi, psi_omega, E, freq_mult):
        psi_omega_copy = copy.deepcopy(psi_omega)
        return PropagationSolver.DynamicState(l, t, psi, psi_omega_copy, E, freq_mult)


    def do_the_thing(self, instr: PropagationSolver.InstrumentationOutputData):
        # algorithm without control
        if self.conf_fitter.task_type != RootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION and \
           self.conf_fitter.task_type != RootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
            self.res = PropagationSolver.StepReaction.OK
            dAdt = 0.0
        # local control algorithm with goal population
        elif self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION:
            coef = 2.0 * phys_base.cm_to_erg / phys_base.Red_Planck_h
            dAdt = self.dyn.propagation_dyn_ref.E * instr.psigc_psie.imag * coef
            if dAdt >= 0.0:
                self.res = PropagationSolver.StepReaction.OK
                self.dAdt_happy = dAdt
            else:
                if abs(instr.psigc_psie.imag) > self.conf_fitter.epsilon:
                    self.E_patched = -self.dAdt_happy / (instr.psigc_psie.imag * coef)
                else:
                    print("Imaginary part in dA/dt is too small and has been replaces by epsilon")
                    self.E_patched = self.dAdt_happy / (self.conf_fitter.epsilon * coef)
                self.res = PropagationSolver.StepReaction.CORRECT
        # local control algorithm with goal projection
        else:
            coef2 = -4.0 * phys_base.cm_to_erg / phys_base.Red_Planck_h
            Sge2 = instr.psigc_psie * instr.psigc_psie
            Sdvge = instr.psigc_psie * instr.psigc_dv_psie
            freq_cm = phys_base.Hz_to_cm * self.conf_fitter.propagation.nu_L
            body = Sdvge + freq_cm * self.dyn.propagation_dyn_ref.freq_mult * Sge2
            dAdt = body.imag * coef2
            if dAdt >= 0.0:
                self.res = PropagationSolver.StepReaction.OK
                self.dAdt_happy = dAdt
            else:
                if Sge2.imag > self.conf_fitter.epsilon:
                    self.freq_mult_patched = (self.dAdt_happy - coef2 * Sdvge.imag) / (Sge2.imag * freq_cm * coef2)
                elif Sge2.imag > 0.0:
                    print("Imaginary part in dA/dt is positive but too small and has been replaces by epsilon")
                    self.freq_mult_patched = (self.dAdt_happy - coef2 * Sdvge.imag) / (self.conf_fitter.epsilon * freq_cm * coef2)
                elif Sge2.imag < -self.conf_fitter.epsilon:
                    self.freq_mult_patched = - Sdvge.imag / (Sge2.imag * freq_cm)
                else:
                    print("Imaginary part in dA/dt is negative but too small and has been replaces by -epsilon")
                    self.freq_mult_patched = Sdvge.imag / (self.conf_fitter.epsilon * freq_cm)

                if self.freq_mult_patched < 0.0:
                    self.freq_mult_patched = 0.0

                self.res = PropagationSolver.StepReaction.CORRECT

        # plotting the result
        self.reporter.print_time_point_fitter(self.dyn.propagation_dyn_ref.l, self.dyn.propagation_dyn_ref.t,
                                              self.dyn.propagation_dyn_ref.E, self.dyn.propagation_dyn_ref.freq_mult)

        if self.dyn.propagation_dyn_ref.l % self.conf_fitter.mod_log == 0:
            if self.conf_fitter.task_type != RootConfiguration.FitterConfiguration.TaskType.FILTERING:
                print("emax = ", instr.emax)
                print("emin = ", instr.emin)
            if self.conf_fitter.task_type != RootConfiguration.FitterConfiguration.TaskType.FILTERING and \
               self.conf_fitter.task_type != RootConfiguration.FitterConfiguration.TaskType.SINGLE_POT:
                print("normalization on the excited state = ", abs(instr.cnorm[1]))
            if self.conf_fitter.task_type != RootConfiguration.FitterConfiguration.TaskType.FILTERING and \
               self.conf_fitter.task_type != RootConfiguration.FitterConfiguration.TaskType.SINGLE_POT:
                print("energy on the excited state = ", instr.cener[1].real)
            if self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION or \
               self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
                print("Time derivation of the expectation value from the goal operator A = ", dAdt)

            if self.res == PropagationSolver.StepReaction.CORRECT:
                print("CORRECTING THE ITERATION")
            elif self.res == PropagationSolver.StepReaction.ITERATE:
                print("THE OPTIMAL CONTROL ITERATIVE ALGORITHM PROCEEDS \n"
                      "Current iteration = ", self.dyn.propagation_dyn_ref.iter_step)
            else:
                pass


    # calculating envelope of the laser field energy at the given time value
    def LaserFieldEnvelope(self, stat: PropagationSolver.StaticState,
                           dyn: PropagationSolver.DynamicState):
        self.E_patched = phys_base.laser_field(self.conf_fitter.propagation.E0, dyn.t, self.conf_fitter.propagation.t0, self.conf_fitter.propagation.sigma)

        # transition without control
        if self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.TRANS_WO_CONTROL:
            E = self.E_patched
        # intuitive control algorithm
        elif self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.INTUITIVE_CONTROL:
            for npul in range(1, self.conf_fitter.impulses_number):
                self.E_patched += phys_base.laser_field(self.conf_fitter.propagation.E0, dyn.t,
                                            self.conf_fitter.propagation.t0 + (npul * self.conf_fitter.delay),
                                            self.conf_fitter.propagation.sigma)
            E = self.E_patched
        # local control algorithm (with A = Pe)
        elif self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION:
            if self.dyn.propagation_dyn_ref.E == 0.0:
                self.dyn.propagation_dyn_ref.E = self.E_patched

            if self.E_patched <= 0:
                raise RuntimeError("E_patched has to be positive")

            if self.dyn.propagation_dyn_ref.E <= 0:
                raise RuntimeError("E has to be positive")

            # solving dynamic equation for E
            # linear difference to the "desired" value
            first = self.dyn.propagation_dyn_ref.E - self.E_patched
            # decay term
            second = self.dyn.E_vel * math.pow(self.E_patched / self.dyn.propagation_dyn_ref.E, self.conf_fitter.pow)

            # Euler
            E_acc = -self.conf_fitter.k_E * first - self.conf_fitter.lamb * second
            self.dyn.E_vel += E_acc * stat.dt
            E = self.dyn.propagation_dyn_ref.E + self.dyn.E_vel * stat.dt
        # optimal control algorithm
        elif self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV or \
             self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_GRADIENT:
            if self.dyn.iter_step == 0:
                E = self.E_patched
                self.E_tlist.append(E)
            else:
                conf_prop = self.conf_fitter.propagation
                Enorm0 = math.sqrt(math.pi) * conf_prop.E0 * conf_prop.E0 * conf_prop.sigma * \
                         (math.erf((conf_prop.T - conf_prop.t0) / conf_prop.sigma) +
                          math.erf(conf_prop.t0 / conf_prop.sigma)) / 2.0

                chie_psig = math_base.cprod(self.chi_tlist[conf_prop.nt - self.dyn.propagation_dyn_ref.l][1],
                                            self.dyn.propagation_dyn_ref.psi_omega[0], stat.dx, conf_prop.np)
                psie_chig = math_base.cprod(self.dyn.propagation_dyn_ref.psi_omega[1],
                                            self.chi_tlist[conf_prop.nt - self.dyn.propagation_dyn_ref.l][0],stat.dx, conf_prop.np)

                Ediff = chie_psig - psie_chig
                self.E_int += Ediff * Ediff.conjugate() * self.solver.stat.dt
                E = 1j * math.sqrt(Enorm0) * Ediff / math.sqrt(self.E_int)
                self.E_tlist.append(E)
        else:
            E = self.E_patched

        return E


    # calculating envelope of the laser field energy at the given time value for back propagation
    def LaserFieldEnvelopeBackward(self, stat: PropagationSolver.StaticState,
                                    dyn: PropagationSolver.DynamicState):
        self.E_patched = self.E_tlist[self.conf_fitter.propagation.nt - dyn.l]
        E = self.E_patched
        return E


    # calculating a frequency multiplier value at the given time value
    def FreqMultiplier(self, stat: PropagationSolver.StaticState):
        # local control algorithm (with A = Pg + Pe)
        if self.conf_fitter.task_type == RootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
            if self.freq_mult_patched < 0:
                raise RuntimeError("freq_mult_patched has to be positive or zero")

            if self.dyn.propagation_dyn_ref.freq_mult < 0:
                raise RuntimeError("Frequency multiplicator has to be positive or zero")

            # solving dynamic equation for frequency multiplicator
            # linear difference to the "desired" value
            first = self.dyn.propagation_dyn_ref.freq_mult - self.freq_mult_patched
            # decay term
            second = self.dyn.freq_mult_vel * math.pow(self.freq_mult_patched / self.dyn.propagation_dyn_ref.freq_mult, self.conf_fitter.pow)

            # Euler
            freq_mult_acc = -self.conf_fitter.k_E * first - self.conf_fitter.lamb * second
            self.dyn.freq_mult_vel += freq_mult_acc * stat.dt
            freq_mult = self.dyn.propagation_dyn_ref.freq_mult + self.dyn.freq_mult_vel * stat.dt
#            freq_mult = self.freq_mult_patched
        else:
            freq_mult = 1.0

        return freq_mult
