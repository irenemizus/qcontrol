import collections.abc
#import os.path
import numpy

import reporter
import json
from config import TaskRootConfiguration
from propagation import *


class FittingSolver:
    class FitterDynamicState():
        def __init__(self, E_vel=0.0, freq_mult_vel=0.0, iter_step=0, dir = PropagationSolver.Direction.FORWARD):
            self.E_vel = E_vel
            self.freq_mult_vel = freq_mult_vel
            self.iter_step = iter_step
            self.dir = dir

            self.E_patched = 0.0
            self.freq_mult_patched = 1.0
            self.dAdt_happy = 0.0

            self.chi_tlist = []
            self.psi_omega_tlist = []

            self.goal_close = 0.0
            #self.E_int = 0.0
            #self.  = []
            self.res = PropagationSolver.StepReaction.OK

            self.propagation_dyn_ref = None

        @staticmethod
        def from_json(json_str: str):
            def object_hook(dct: dict):
                new_chi_tlist = []
                for time_point in dct['chi_tlist']:
                    new_time_point = []
                    for level in time_point:
                        new_level = numpy.array(level).astype(complex)
                        new_time_point.append(new_level)
                    new_chi_tlist.append(new_time_point)
                dct['chi_tlist'] = new_chi_tlist

                new_psi_omega_tlist = []
                for time_point in dct['psi_omega_tlist']:
                    new_time_point = []
                    for level in time_point:
                        new_level = numpy.array(level).astype(complex)
                        new_time_point.append(new_level)
                    new_psi_omega_tlist.append(new_time_point)
                dct['psi_omega_tlist'] = new_psi_omega_tlist

                #new_E_tlist = numpy.array(dct['E_tlist']).astype(complex)
                #dct['E_tlist'] = new_E_tlist

                dct['goal_close'] = complex(dct['goal_close'])

                new_dir = PropagationSolver.Direction[dct['dir']]
                dct['dir'] = new_dir

                new_res = PropagationSolver.StepReaction[dct['res']]
                dct['res'] = new_res

                return dct

            res = json.loads(json_str, object_hook=object_hook)
            return res

        def to_json_with_bins(self):
            array_index = 0
            arrays = []

            def default(obj):
                nonlocal array_index, arrays

                if isinstance(obj, complex):
                    return str(obj)
                elif isinstance(obj, numpy.ndarray):
                    res = str(array_index)
                    arrays.append(obj)
                    array_index += 1
                    return res
                elif isinstance(obj, PropagationSolver.StepReaction):
                    return str(obj._name_)
                elif isinstance(obj, PropagationSolver.Direction):
                    return str(obj._name_)
                elif isinstance(obj, collections.abc.Mapping):
                    return None
                else:
                    return obj.__dict__

            return json.dumps(self, default=default,
                              sort_keys=True, indent=4), arrays

    def __initialize_propagation(self, prop_id: str, laser_field_envelope):
        self.propagation_reporter = self.reporter.create_propagation_reporter(prop_id)
        self.propagation_reporter.open()
        self.solver = PropagationSolver(
            pot=self.pot_func,
            _warning_collocation_points=self._warning_collocation_points,
            _warning_time_steps=self._warning_time_steps,
            reporter=self.propagation_reporter,
            laser_field_envelope=laser_field_envelope,
            freq_multiplier=self.FreqMultiplier,
            dynamic_state_factory=self.propagation_dynamic_state_factory,
            mod_log=self.conf_fitter.mod_log,
            conf_prop=self.conf_fitter.propagation)

    def __finalize_propagation(self):
        self.dyn.propagation_dyn_ref = None
        self.propagation_reporter.close()
        self.propagation_reporter = None

    def __init__(
            self,
            conf_fitter,
            psi_init,
            psi_goal,
            pot_func,
            laser_field,
            reporter: reporter.FitterReporter,
            _warning_collocation_points,
            _warning_time_steps
    ):
        self._warning_collocation_points = _warning_collocation_points
        self._warning_time_steps = _warning_time_steps
        self.pot_func = pot_func
        self.laser_field = laser_field

        self.reporter = reporter

        self.conf_fitter = conf_fitter
        self.psi_init = psi_init
        self.psi_goal = psi_goal

        self.dyn = None

    # single propagation to the given direction; returns new chiT
    def __single_propagation(self, dx, x, direct: PropagationSolver.Direction, chiT, goal_close_abs):
        self.dyn.dir = direct

        if direct == PropagationSolver.Direction.FORWARD:
            ind_dir = "f"
            laser_field = self.LaserFieldEnvelope

            if self.dyn.iter_step > 0:
                print("Iteration = ", self.dyn.iter_step, ", Forward direction begins...")
                self.dyn.res = PropagationSolver.StepReaction.ITERATE
                #self.dyn.E_tlist = []
                #if self.dyn.iter_step == 1:
                #    self.dyn.E_int = self.__E_int(dx)

            psi_init_omega_copy = [
                self.psi_init[0].copy(),
                self.psi_init[1].copy()
            ]

            self.dyn.psi_omega_tlist = [psi_init_omega_copy]
            init_psi = self.psi_init
            fin_psi = self.psi_goal
            t_init = 0.0
            #self.dyn.E_tlist.append(self.laser_field(self.conf_fitter.propagation.E0, 0.0, self.conf_fitter.propagation.t0, self.conf_fitter.propagation.sigma))
        else:
            print("Iteration = ", self.dyn.iter_step, ", Backward direction begins...")
            ind_dir = "b"
            laser_field = self.LaserFieldEnvelopeBackward
            self.dyn.res = PropagationSolver.StepReaction.ITERATE
            init_psi = chiT
            fin_psi = self.psi_init
            t_init = self.conf_fitter.propagation.T

        self.__initialize_propagation("iter_" + str(self.dyn.iter_step) + ind_dir, laser_field)
        self.solver.start(dx, x, init_psi, fin_psi, self.dyn.dir)
        self.dyn.propagation_dyn_ref = self.solver.dyn

        # propagation loop
        while self.solver.step(t_init):
            psi_omega_copy = [
                self.solver.dyn.psi_omega[0].copy(),
                self.solver.dyn.psi_omega[1].copy()
            ]
            if direct == PropagationSolver.Direction.BACKWARD:
                self.dyn.chi_tlist.append(psi_omega_copy)
            else:
                self.dyn.psi_omega_tlist.append(psi_omega_copy)

            self.do_the_thing(self.solver.instr)

        if direct == PropagationSolver.Direction.FORWARD:
            chiT = []
            assert self.solver.dyn.l - 1 == self.conf_fitter.propagation.nt
            self.dyn.goal_close = math_base.cprod(self.solver.stat.psif[1], self.solver.dyn.psi[1],
                                                  self.solver.stat.dx, self.conf_fitter.propagation.np)
            goal_close_abs = abs(self.dyn.goal_close)
            self.reporter.print_iter_point_fitter(self.dyn.iter_step, goal_close_abs)
            chiT.append(numpy.array([0.0] * self.conf_fitter.propagation.np).astype(complex))
            chiT.append(self.dyn.goal_close * self.solver.stat.psif[1])

            # renormalization
            cnorm = math_base.cprod(chiT[1], chiT[1], dx, self.conf_fitter.propagation.np)
            if abs(cnorm) > 0.0:
                for el in chiT[1]:
                    el /= math.sqrt(abs(cnorm))

            chiT_omega = [
                chiT[0].copy(),
                chiT[1].copy()
            ]
            for el in chiT_omega[1]:
                el *= cmath.exp(1j * math.pi * self.conf_fitter.propagation.nu_L * self.dyn.propagation_dyn_ref.freq_mult * self.conf_fitter.propagation.T)
            self.dyn.chi_tlist = [chiT_omega]

        self.__finalize_propagation()
        return chiT, goal_close_abs

    # single simple forward propagation
    def __single_iteration_simple(self, dx, x):
        chiT, goal_close_abs = self.__single_propagation(dx, x, PropagationSolver.Direction.FORWARD, [], 0.0)
        return 0.0

    # single iteration for an optimal control task
    def __single_iteration_optimal(self, dx, x):
        assert (dx > 0.0)
        chiT = []
        goal_close_abs = 0.0
        print("Iteration = ", self.dyn.iter_step, ", Forward direction begins...")


        for direct in PropagationSolver.Direction:
            #if direct == PropagationSolver.Direction.FORWARD:
            #    ind_dir = 'f'
            #else:
            #    ind_dir = 'b'

            # calculating chiT
            chiT, goal_close_abs = self.__single_propagation(dx, x, direct, chiT, goal_close_abs)

            #saved_json, arrays = self.dyn.to_json_with_bins()

            #if not os.path.exists("savings"):
            #    os.mkdir("savings")

            #path = os.path.join("savings", "iter_" + str(self.dyn.iter_step) + ind_dir)
            #if not os.path.exists(path):
            #    os.mkdir(path)

            #with open(os.path.join(path, "fitter_state.json"), 'w') as f:
            #    f.write(saved_json)

            #np.savez_compressed(os.path.join(path, "fitter_state_bins.npz"), arrays)

            #print(f"abs(goal_close_abs - 1.0) = {abs(goal_close_abs - 1.0)}")
            if abs(goal_close_abs - 1.0) <= self.conf_fitter.epsilon and self.dyn.iter_step == 0:
                print("The goal has been reached on the very first iteration. You don't need the control!")
                self.dyn.res = PropagationSolver.StepReaction.OK
                break

        return goal_close_abs


    def time_propagation(self, dx, x):
        self.dyn = FittingSolver.FitterDynamicState(0.0, 0.0, 0, PropagationSolver.Direction.FORWARD)

        if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV or \
           self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_GRADIENT:
            # 0-th iteration
            goal_close_abs = self.__single_iteration_optimal(dx, x)
                #self.dyn = FittingSolver.FitterDynamicState.from_json(saved_json)
                #print(self.dyn)

            # iterative procedure
            while abs(goal_close_abs - 1.0) > self.conf_fitter.epsilon and \
                    self.dyn.iter_step < self.conf_fitter.iter_max:
                self.dyn.iter_step += 1
                goal_close_abs = self.__single_iteration_optimal(dx, x)

            if abs(goal_close_abs - 1.0) <= self.conf_fitter.epsilon:
                print("The goal has been successfully reached on the " + str(self.dyn.iter_step) + " iteration.")
            else:
                print("The goal has not been reached during the calculation.")

            self.dyn.res = PropagationSolver.StepReaction.OK
        else:
            self.__single_iteration_simple(dx, x)


    def propagation_dynamic_state_factory(self, l, t, psi, psi_omega, E, freq_mult, dir):
        psi_omega_copy = copy.deepcopy(psi_omega)
        return PropagationSolver.DynamicState(l, t, psi, psi_omega_copy, E, freq_mult, dir)


    def do_the_thing(self, instr: PropagationSolver.InstrumentationOutputData):
        # algorithm without control
        if self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION and \
           self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
            dAdt = 0.0
        # local control algorithm with goal population
        elif self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION:
            coef = 2.0 * phys_base.cm_to_erg / phys_base.Red_Planck_h
            dAdt = self.dyn.propagation_dyn_ref.E * instr.psigc_psie.imag * coef
            if dAdt >= 0.0:
                self.dyn.res = PropagationSolver.StepReaction.OK
                self.dyn.dAdt_happy = dAdt
            else:
                if abs(instr.psigc_psie.imag) > self.conf_fitter.epsilon:
                    self.dyn.E_patched = -self.dyn.dAdt_happy / (instr.psigc_psie.imag * coef)
                else:
                    print("Imaginary part in dA/dt is too small and has been replaces by epsilon")
                    self.dyn.E_patched = self.dyn.dAdt_happy / (self.conf_fitter.epsilon * coef)
                self.dyn.res = PropagationSolver.StepReaction.CORRECT
        # local control algorithm with goal projection
        else:
            coef2 = -4.0 * phys_base.cm_to_erg / phys_base.Red_Planck_h
            Sge2 = instr.psigc_psie * instr.psigc_psie
            Sdvge = instr.psigc_psie * instr.psigc_dv_psie
            freq_cm = phys_base.Hz_to_cm * self.conf_fitter.propagation.nu_L
            body = Sdvge + freq_cm * self.dyn.propagation_dyn_ref.freq_mult * Sge2
            dAdt = body.imag * coef2
            if dAdt >= 0.0:
                self.dyn.res = PropagationSolver.StepReaction.OK
                self.dyn.dAdt_happy = dAdt
            else:
                if Sge2.imag > self.conf_fitter.epsilon:
                    self.dyn.freq_mult_patched = (self.dyn.dAdt_happy - coef2 * Sdvge.imag) / (Sge2.imag * freq_cm * coef2)
                elif Sge2.imag > 0.0:
                    print("Imaginary part in dA/dt is positive but too small and has been replaces by epsilon")
                    self.dyn.freq_mult_patched = (self.dyn.dAdt_happy - coef2 * Sdvge.imag) / (self.conf_fitter.epsilon * freq_cm * coef2)
                elif Sge2.imag < -self.conf_fitter.epsilon:
                    self.dyn.freq_mult_patched = - Sdvge.imag / (Sge2.imag * freq_cm)
                else:
                    print("Imaginary part in dA/dt is negative but too small and has been replaces by -epsilon")
                    self.dyn.freq_mult_patched = Sdvge.imag / (self.conf_fitter.epsilon * freq_cm)

                if self.dyn.freq_mult_patched < 0.0:
                    self.dyn.freq_mult_patched = 0.0

                self.dyn.res = PropagationSolver.StepReaction.CORRECT

        if self.dyn.propagation_dyn_ref.l % self.conf_fitter.mod_log == 0:
            if self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.FILTERING:
                print("emax = ", instr.emax)
                print("emin = ", instr.emin)
            if self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.FILTERING and \
               self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.SINGLE_POT:
                print("normalization on the excited state = ", abs(instr.cnorm[1]))
            if self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.FILTERING and \
               self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.SINGLE_POT:
                print("energy on the excited state = ", instr.cener[1].real)
            if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION or \
               self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
                print("Time derivation of the expectation value from the goal operator A = ", dAdt)

            if self.dyn.res == PropagationSolver.StepReaction.CORRECT:
                print("CORRECTING THE ITERATION")
            elif self.dyn.res == PropagationSolver.StepReaction.ITERATE:
                print("THE OPTIMAL CONTROL ITERATIVE ALGORITHM PROCEEDS \n"
                      "Current iteration = ", self.dyn.iter_step)
            else:
                pass


    #def __E_int(self, dx):
    #    conf_prop = self.conf_fitter.propagation

    #    E_int = 0.0
    #    for l in range(conf_prop.nt):
    #        chie_old_psig_old = math_base.cprod(self.dyn.chi_tlist[conf_prop.nt - l - 1][1],
    #                                            self.dyn.psi_omega_tlist[l][0], dx,
    #                                            conf_prop.np)
    #        psie_old_chig_old = math_base.cprod(self.dyn.psi_omega_tlist[l][1],
    #                                            self.dyn.chi_tlist[conf_prop.nt - l - 1][0],
    #                                            dx, conf_prop.np)
    #        Ediff_old_old = chie_old_psig_old - psie_old_chig_old

    #        E_int += (Ediff_old_old * Ediff_old_old.conjugate()).real * self.solver.time_step

    #    print(f"E_int = {self.dyn.E_int}")
    #    return E_int


    # calculating envelope of the laser field energy at the given time value
    def LaserFieldEnvelope(self, stat: PropagationSolver.StaticState,
                           dyn: PropagationSolver.DynamicState):
        self.dyn.E_patched = self.laser_field(self.conf_fitter.propagation.E0, dyn.t, self.conf_fitter.propagation.t0, self.conf_fitter.propagation.sigma)

        # transition without control
        if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.TRANS_WO_CONTROL:
            E = self.dyn.E_patched

        # intuitive control algorithm
        elif self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.INTUITIVE_CONTROL:
            for npul in range(1, self.conf_fitter.impulses_number):
                self.dyn.E_patched += self.laser_field(self.conf_fitter.propagation.E0, dyn.t,
                                            self.conf_fitter.propagation.t0 + (npul * self.conf_fitter.delay),
                                            self.conf_fitter.propagation.sigma)
            E = self.dyn.E_patched

        # local control algorithm (with A = Pe)
        elif self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION:
            if self.solver.dyn.E == 0.0:
                self.solver.dyn.E = self.dyn.E_patched

            if self.dyn.E_patched <= 0:
                raise RuntimeError("E_patched has to be positive")

            if self.solver.dyn.E <= 0:
                raise RuntimeError("E has to be positive")

            # solving dynamic equation for E
            # linear difference to the "desired" value
            first = self.solver.dyn.E - self.dyn.E_patched
            # decay term
            second = self.dyn.E_vel * math.pow(self.dyn.E_patched / self.solver.dyn.E, self.conf_fitter.pow)

            # Euler
            E_acc = -self.conf_fitter.k_E * first - self.conf_fitter.lamb * second
            self.dyn.E_vel += E_acc * stat.dt
            E = self.solver.dyn.E + self.dyn.E_vel * stat.dt

        # optimal control algorithm
        elif (self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV or
              self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_GRADIENT) and \
              self.dyn.dir == PropagationSolver.Direction.FORWARD:
            if self.dyn.iter_step == 0:
                E = self.dyn.E_patched
                #self.dyn.E_tlist.append(E)
            else:
                conf_prop = self.conf_fitter.propagation
                #Enorm0 = math.sqrt(math.pi) * conf_prop.E0 * conf_prop.E0 * conf_prop.sigma * \
                #         (math.erf((conf_prop.T - conf_prop.t0) / conf_prop.sigma) +
                #          math.erf(conf_prop.t0 / conf_prop.sigma)) / 2.0

                #print(f"Enorm0 = {Enorm0}")

                chie_old_psig_new = math_base.cprod(self.dyn.chi_tlist[conf_prop.nt - self.solver.dyn.l - 1][1],
                                                    dyn.psi_omega[0],
                                                    stat.dx, conf_prop.np)
                psie_new_chig_old = math_base.cprod(dyn.psi_omega[1],
                                                    self.dyn.chi_tlist[conf_prop.nt - self.solver.dyn.l - 1][0],
                                                    stat.dx, conf_prop.np)
                #Ediff_new_old = chie_old_psig_new - psie_new_chig_old

                #E = 1j * math.sqrt(Enorm0) * Ediff_new_old / math.sqrt(self.dyn.E_int)
                h_lambda = 0.0066
                #E = 1j * Ediff_new_old / h_lambda
                E = -2.0 * chie_old_psig_new.imag / h_lambda
                #self.dyn.E_tlist.append(E)
        else:
            E = self.dyn.E_patched

        return E


    # calculating envelope of the laser field energy at the given time value for back propagation
    def LaserFieldEnvelopeBackward(self, stat: PropagationSolver.StaticState,
                                    dyn: PropagationSolver.DynamicState):
        #self.dyn.E_patched = self.dyn.E_tlist[self.conf_fitter.propagation.nt - dyn.l]
        #E = self.dyn.E_patched
        conf_prop = self.conf_fitter.propagation
        chie_new_psig_old = math_base.cprod(dyn.psi_omega[1],
                                            self.dyn.psi_omega_tlist[conf_prop.nt - self.solver.dyn.l - 1][0],
                                            stat.dx, conf_prop.np)
        psie_old_chig_new = math_base.cprod(self.dyn.psi_omega_tlist[conf_prop.nt - self.solver.dyn.l - 1][1],
                                            dyn.psi_omega[0],
                                            stat.dx, conf_prop.np)
        #Ediff_new_old = chie_new_psig_old - psie_old_chig_new

        # E = 1j * math.sqrt(Enorm0) * Ediff_new_old / math.sqrt(self.dyn.E_int)
        h_lambda = 0.0066
        #E = 1j * Ediff_new_old / h_lambda
        E = -2.0 * chie_new_psig_old.imag / h_lambda

        return E


    # calculating a frequency multiplier value at the given time value
    def FreqMultiplier(self, stat: PropagationSolver.StaticState):
        # local control algorithm (with A = Pg + Pe)
        if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
            if self.dyn.freq_mult_patched < 0:
                raise RuntimeError("freq_mult_patched has to be positive or zero")

            if self.dyn.propagation_dyn_ref.freq_mult < 0:
                raise RuntimeError("Frequency multiplicator has to be positive or zero")

            # solving dynamic equation for frequency multiplicator
            # linear difference to the "desired" value
            first = self.dyn.propagation_dyn_ref.freq_mult - self.dyn.freq_mult_patched
            # decay term
            second = self.dyn.freq_mult_vel * math.pow(self.dyn.freq_mult_patched / self.dyn.propagation_dyn_ref.freq_mult, self.conf_fitter.pow)

            # Euler
            freq_mult_acc = -self.conf_fitter.k_E * first - self.conf_fitter.lamb * second
            self.dyn.freq_mult_vel += freq_mult_acc * stat.dt
            freq_mult = self.dyn.propagation_dyn_ref.freq_mult + self.dyn.freq_mult_vel * stat.dt
        else:
            freq_mult = 1.0

        return freq_mult
