import collections.abc
import os.path
import json

import reporter
from config import TaskRootConfiguration
from propagation import *
from psi_basis import PsiBasis, Psi


class FittingSolver:
    class FitterDynamicState():
        chi_tlist: list[PsiBasis]
        psi_tlist: list[PsiBasis]
        E_tlist: list[complex]
        chi_cur: PsiBasis
        psi_cur: PsiBasis
        goal_close: List[complex]

        def __init__(self, basis_length, E_vel=0.0, freq_mult_vel=0.0, iter_step=0, dir = PropagationSolver.Direction.FORWARD):
            self.E_vel = E_vel
            self.freq_mult_vel = freq_mult_vel
            self.iter_step = iter_step
            self.dir = dir

            self.E_patched = 0.0
            self.freq_mult_patched = 1.0
            self.dAdt_happy = 0.0

            self.chi_tlist = []
            self.psi_tlist = []

            self.chi_cur = PsiBasis(basis_length)
            self.psi_cur = PsiBasis(basis_length)

            self.goal_close = [complex(0.0)] * basis_length
            self.E_tlist = []

            self.res = PropagationSolver.StepReaction.OK

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

                new_psi_tlist = []
                for time_point in dct['psi_tlist']:
                    new_time_point = []
                    for level in time_point:
                        new_level = numpy.array(level).astype(complex)
                        new_time_point.append(new_level)
                    new_psi_tlist.append(new_time_point)
                dct['psi_tlist'] = new_psi_tlist

                new_E_tlist = numpy.array(dct['E_tlist']).astype(complex)
                dct['E_tlist'] = new_E_tlist

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

    solvers: list[PropagationSolver]
    propagation_reporters: list[PropagationReporter]

    def __initialize_propagation(self, prop_id: str, laser_field_envelope, ntriv):

        self.solvers = []
        self.propagation_reporters = [None] * self.basis_length
        for vect in range(self.basis_length):
            propagation_reporter = self.reporter.create_propagation_reporter(os.path.join(prop_id, "basis_" + str(vect)))
            propagation_reporter.open()
            self.propagation_reporters[vect] = propagation_reporter

            self.solvers.append(PropagationSolver(
                pot=self.pot_func,
                _warning_collocation_points=self._warning_collocation_points,
                _warning_time_steps=self._warning_time_steps,
                reporter=propagation_reporter,
                laser_field_envelope=laser_field_envelope,
                freq_multiplier=self.FreqMultiplier,
                dynamic_state_factory=self.propagation_dynamic_state_factory,
                mod_log=self.conf_fitter.mod_log,
                ntriv = ntriv,
                conf_prop=self.conf_fitter.propagation))

    def __finalize_propagation(self):
        for vect in range(self.basis_length):
            self.propagation_reporters[vect].close()
            self.propagation_reporters[vect] = None
        self.propagation_reporters = None

    psi_init_basis: PsiBasis
    psi_goal_basis: PsiBasis
    dyn: FitterDynamicState

    def __init__(
            self,
            conf_fitter,
            init_dir,
            ntriv,
            psi_init_basis: PsiBasis,
            psi_goal_basis: PsiBasis,
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
        self.init_dir = init_dir
        self.ntriv = ntriv
        self.psi_init_basis = psi_init_basis
        self.psi_goal_basis = psi_goal_basis

        self.basis_length = len(psi_init_basis)
        self.a0 = complex(0.0, 0.0)

        self.dyn = None

        #self.TMP_delta_E = []

    # single propagation to the given direction; returns new chiT
    def __single_propagation(self, dx, x, t_step, direct: PropagationSolver.Direction, chiT: PsiBasis, goal_close_abs):
        self.dyn.dir = direct
        init_psi_basis: PsiBasis
        fin_psi_basis: PsiBasis
        chiT_omega = PsiBasis(self.basis_length)
        self.dyn.chi_cur = None
        self.dyn.psi_cur = None
        if direct == PropagationSolver.Direction.FORWARD:
            ind_dir = "f"
            laser_field = self.LaserFieldEnvelope

            if self.dyn.iter_step > 0:
                self.dyn.res = PropagationSolver.StepReaction.ITERATE

            psi_init_copy = copy.deepcopy(self.psi_init_basis)

            self.dyn.psi_tlist = [ psi_init_copy ]
            self.dyn.psi_cur = self.dyn.psi_tlist[0]
            init_psi_basis = self.psi_init_basis
            fin_psi_basis = self.psi_goal_basis
            t_init = 0.0
        else:
            ind_dir = "b"
            laser_field = self.LaserFieldEnvelopeBackward

            if self.dyn.iter_step > 0:
                self.dyn.res = PropagationSolver.StepReaction.ITERATE

            if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                chiT = copy.deepcopy(self.psi_goal_basis)
#                chiT_omega = copy.deepcopy(self.psi_goal_basis)
#                for vect in range(self.basis_length):
#                    chiT_omega.psis[vect].f[1] *= cmath.exp(1j * math.pi * self.conf_fitter.propagation.nu_L * self.conf_fitter.propagation.T)
#                    chiT_omega.psis[vect].f[0] *= cmath.exp(-1j * math.pi * self.conf_fitter.propagation.nu_L * self.conf_fitter.propagation.T)

                self.dyn.chi_tlist = [ chiT ]

            self.dyn.chi_cur = self.dyn.chi_tlist[0]
            init_psi_basis = chiT
            fin_psi_basis = self.psi_init_basis
            t_init = self.conf_fitter.propagation.T

        self.__initialize_propagation("iter_" + str(self.dyn.iter_step) + ind_dir, laser_field, self.ntriv)

        # Propagation loop
        finished: set[int] = set()

        # Starting solvers
        for vect in range(self.basis_length):
            solver = self.solvers[vect]
            solver.start(dx, x, t_step, init_psi_basis.psis[vect], fin_psi_basis.psis[vect], self.dyn.dir)

        # Paranoidly checking E EVERYWHERE
        E_checked = self.solvers[0].dyn.E
        for vect in range(self.basis_length):
            if abs(E_checked - self.solvers[vect].dyn.E) > 0.001:
                raise AssertionError("Different energies in different solvers")

        E_tlist_new: list[complex] = []
        # Working with solvers
        if direct == PropagationSolver.Direction.FORWARD:
            # with open("test_chi_" + str(self.dyn.iter_step) + ".txt", "w") as f:
            #     f.write("chi_tlist = [\n")
            #     for l in self.dyn.chi_tlist:
            #         f.write("Vectors = [\n")
            #         f.write("    " + str(l.psis[0].f[0]) + ",\n")
            #         f.write("    " + str(l.psis[0].f[1]) + ",\n")
            #         f.write("    " + str(l.psis[1].f[0]) + ",\n")
            #         f.write("    " + str(l.psis[1].f[1]) + ",\n")
            #         f.write("]\n\n")
            #     f.write("]\n\n")
            E_tlist_new = [ E_checked ]

        #self.TMP_delta_E = []
        while True:
            chi_new: PsiBasis = PsiBasis(self.basis_length)
            psi_new: PsiBasis = PsiBasis(self.basis_length)

            for vect in range(self.basis_length):
                solver = self.solvers[vect]

                if vect in finished:
                    raise AssertionError(f"The solver #{vect} has finished, but asked to proceed. It's a pity.")

                if (solver.dyn.l - 1) % self.conf_fitter.mod_log == 0 and \
                   self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                    print(f"The solver for the basis vector #{vect} is running...")

                if not solver.step(t_init):
                    finished.add(vect)

                if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                    psi_copy = copy.deepcopy(solver.dyn.psi)
                else:
                    psi_copy = copy.deepcopy(solver.dyn.psi_omega)

                #print("|psi6| = ", abs(psi_copy.f[0]) + abs(psi_copy.f[1]))

                if direct == PropagationSolver.Direction.BACKWARD:
                    chi_new.psis[vect] = psi_copy
                else:
                    psi_new.psis[vect] = psi_copy

                self.do_the_thing(solver.dyn, solver.instr)

                #print("|psi7| = ", abs(psi_copy.f[0]) + abs(psi_copy.f[1]))

            # Checking sanity
            t_checked = self.solvers[0].dyn.t
            E_checked = self.solvers[0].dyn.E
            for vect in range(self.basis_length):
                if abs(t_checked - self.solvers[vect].dyn.t) > t_step / 1000.0:
                    raise AssertionError("Different times in different solvers")
                if abs(E_checked - self.solvers[vect].dyn.E) > 0.001:
                    raise AssertionError("Different laser field energies in different solvers")

            if direct == PropagationSolver.Direction.BACKWARD:
                self.dyn.chi_cur = chi_new
                self.dyn.chi_tlist.append(chi_new)
            else:
                E_tlist_new.append(E_checked)
                self.dyn.psi_cur = psi_new
                if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV:
                    self.dyn.psi_tlist.append(psi_new)

            do_continue = len(finished) < self.basis_length
            if not do_continue: break

        if direct == PropagationSolver.Direction.FORWARD:
            self.dyn.E_tlist = E_tlist_new
            #with open("test_E_" + str(self.dyn.iter_step) + ".txt", "w") as f:
            #    f.write("TMP_delta_E = [\n")
            #    for l in self.TMP_delta_E:
            #        f.write("    " + str(l) + ",\n")
            #     f.write("]\n\n")

            # with open("test_E_" + str(self.dyn.iter_step) + ".txt", "w") as f:
            #     f.write("E_tlist = [\n")
            #     for l in self.dyn.E_tlist:
            #         f.write("    " + str(l) + ",\n")
            #     f.write("]\n\n")

        self.dyn.goal_close = [complex(0.0, 0.0)] * self.basis_length
        for vect in range(self.basis_length):
            solver = self.solvers[vect]

            if direct == PropagationSolver.Direction.FORWARD:
                chiT_part = Psi()
                assert solver.dyn.l - 1 == self.conf_fitter.propagation.nt
                print("phase(psi_g(T)) = ", cmath.phase(solver.dyn.psi.f[0][0]))
                print("phase(psif_g) = ", cmath.phase(solver.stat.psif.f[0][0]))
                print("phase(psi0_g) = ", cmath.phase(solver.stat.psi0.f[0][0]))
                print("phase(psi_e(T)) = ", cmath.phase(solver.dyn.psi.f[1][0]))
                print("phase(psif_e) = ", cmath.phase(solver.stat.psif.f[1][0]))
                print("phase(psi0_e) = ", cmath.phase(solver.stat.psi0.f[1][0]))

                print("|psi_g(T)| = ", abs(solver.dyn.psi.f[0][0]))
                print("|psif_g| = ", abs(solver.stat.psif.f[0][0]))
                print("|psi0_g| = ", abs(solver.stat.psi0.f[0][0]))
                print("|psi_e(T)| = ", abs(solver.dyn.psi.f[1][0]))
                print("|psif_e| = ", abs(solver.stat.psif.f[1][0]))
                print("|psi0_e| = ", abs(solver.stat.psi0.f[1][0]))

                print("psi_g(T) = ", solver.dyn.psi.f[0][0])
                print("psi_e(T) = ", solver.dyn.psi.f[1][0])

                self.dyn.goal_close[vect] = math_base.cprod(solver.stat.psif.f[1], solver.dyn.psi.f[1],
                                                      solver.stat.dx, self.conf_fitter.propagation.np) + \
                                            math_base.cprod(solver.stat.psif.f[0], solver.dyn.psi.f[0],
                                                      solver.stat.dx, self.conf_fitter.propagation.np)

                if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV:
                    chiT_part.f[0] = numpy.array([0.0] * self.conf_fitter.propagation.np).astype(complex)
                    chiT_part.f[1] = self.dyn.goal_close[vect] * solver.stat.psif.f[1]

                    # renormalization
                    cnorm = math_base.cprod(chiT_part.f[1], chiT_part.f[1], dx, self.conf_fitter.propagation.np)
                    if abs(cnorm) > 0.0:
                        chiT_part.f[1] /= math.sqrt(abs(cnorm))

                    chiTp_omega = copy.deepcopy(chiT_part)
                    chiTp_omega.f[1] *= cmath.exp(1j * math.pi * self.conf_fitter.propagation.nu_L * solver.dyn.freq_mult * self.conf_fitter.propagation.T)
                    chiT_omega.psis[vect] = chiTp_omega
                    chiT.psis[vect] = copy.deepcopy(chiT_part)


                elif self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                    print("goal_close value for the basis vector ", vect, " = ", self.dyn.goal_close[vect])
                else:
                    pass

            #goal_close_abs += abs(self.dyn.goal_close[vect])
            goal_close_abs += self.dyn.goal_close[vect]

        if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV and \
                direct == PropagationSolver.Direction.FORWARD:
            self.dyn.chi_tlist = [ chiT_omega ]

        goal_close_abs = abs(goal_close_abs)
        self.__finalize_propagation()
        return chiT, goal_close_abs


    # single simple forward propagation
    def __single_iteration_simple(self, dx, x, t_step, t_list):
        chiT, goal_close_abs = self.__single_propagation(dx, x, t_step, PropagationSolver.Direction.FORWARD, self.psi_goal_basis, 0.0)
        self.reporter.print_iter_point_fitter(self.dyn.iter_step, goal_close_abs, self.dyn.E_tlist, t_list,
                                              self.conf_fitter.propagation.nt)

        return 0.0


    # single iteration for an optimal control task
    def __single_iteration_optimal(self, dx, x, t_step, t_list):
        assert (dx > 0.0)

        chiT = PsiBasis(self.basis_length)
        goal_close_abs = 0.0

        direct = self.init_dir
        for run in range(2):
            print(f"Iteration = {self.dyn.iter_step}, {direct.name} direction begins...")
            # calculating chiT
            chiT, goal_close_abs = self.__single_propagation(dx, x, t_step, direct, chiT, goal_close_abs)

            #if direct == PropagationSolver.Direction.FORWARD:
            #    ind_dir = 'f'
            #else:
            #    ind_dir = 'b'

            #saved_json, arrays = self.dyn.to_json_with_bins()

            #if not os.path.exists("savings"):
            #    os.mkdir("savings")

            #path = os.path.join("savings", "iter_" + str(self.dyn.iter_step) + ind_dir)
            #if not os.path.exists(path):
            #    os.mkdir(path)

            #with open(os.path.join(path, "fitter_state.json"), 'w') as f:
            #    f.write(saved_json)

            #np.savez_compressed(os.path.join(path, "fitter_state_bins.npz"), arrays)

            if self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                if abs(goal_close_abs - self.basis_length) <= self.conf_fitter.epsilon and self.dyn.iter_step == 0:
                    print("The goal has been reached on the very first iteration. You don't need the control!")
                    self.dyn.res = PropagationSolver.StepReaction.OK
                    break

            if direct == PropagationSolver.Direction.FORWARD:
                self.reporter.print_iter_point_fitter(self.dyn.iter_step, goal_close_abs, self.dyn.E_tlist, t_list,
                                                  self.conf_fitter.propagation.nt)

            direct = PropagationSolver.Direction(-direct.value)

        return goal_close_abs


    def time_propagation(self, dx, x, t_step, t_list):
        self.dyn = FittingSolver.FitterDynamicState(self.basis_length, E_vel=0.0, freq_mult_vel=0.0,
                                                    iter_step=0, dir = self.init_dir)
        if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:

            E_tlist_init = []
            goal_close_abs_init = abs(math_base.cprod(self.psi_goal_basis.psis[0].f[0], self.psi_init_basis.psis[0].f[0],
                                                  dx, self.conf_fitter.propagation.np) + \
                                  math_base.cprod(self.psi_goal_basis.psis[0].f[1], self.psi_init_basis.psis[0].f[1],
                                                  dx, self.conf_fitter.propagation.np) + \
                                  math_base.cprod(self.psi_goal_basis.psis[1].f[0], self.psi_init_basis.psis[1].f[0],
                                                  dx, self.conf_fitter.propagation.np) + \
                                  math_base.cprod(self.psi_goal_basis.psis[1].f[1], self.psi_init_basis.psis[1].f[1],
                                                  dx, self.conf_fitter.propagation.np))

            for t in t_list:
                #cos = cmath.exp(1j * 2.0 * math.pi * self.conf_fitter.propagation.nu_L * t)
                cos = math.cos(2.0 * math.pi * self.conf_fitter.propagation.nu_L * t)
                E_tlist_init.append(self.laser_field(self.conf_fitter.propagation.E0, t, self.conf_fitter.propagation.t0, self.conf_fitter.propagation.sigma) * cos)

            self.reporter.print_iter_point_fitter(-1, goal_close_abs_init, E_tlist_init, t_list,
                                              self.conf_fitter.propagation.nt)

        if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV or \
           self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
            # 0-th iteration
            goal_close_abs = self.__single_iteration_optimal(dx, x, t_step, t_list)
                #self.dyn = FittingSolver.FitterDynamicState.from_json(saved_json)
                #print(self.dyn)

            #self.dyn.freq_mult_patched += -0.0025  # Adding 30 percent to frequency

            # iterative procedure
            #with open("test_nu_L.txt", "w") as f:
            while abs(goal_close_abs - self.basis_length) > self.conf_fitter.epsilon and \
                    (self.dyn.iter_step < self.conf_fitter.iter_max or self.conf_fitter.iter_max == -1):
                self.dyn.iter_step += 1

                #self.dyn.freq_mult_patched += 0.000025  # Adding 0.005 percent to frequency

                goal_close_abs = self.__single_iteration_optimal(dx, x, t_step, t_list)

                    #f.write("{:2d} {:.6e}\n".format(self.dyn.iter_step, self.solvers[0].nu_L * self.dyn.freq_mult_patched))
                    #f.flush()

            if abs(goal_close_abs - self.basis_length) <= self.conf_fitter.epsilon:
                print("The goal has been successfully reached on the " + str(self.dyn.iter_step) + " iteration.")
            else:
                print("The goal has not been reached during the calculation.")

            self.dyn.res = PropagationSolver.StepReaction.OK
        else:
            self.__single_iteration_simple(dx, x, t_step, t_list)


    def propagation_dynamic_state_factory(self, l, t, psi: Psi, psi_omega: Psi, E, freq_mult, dir):
        psi_omega_copy = copy.deepcopy(psi_omega)
        psi_copy = copy.deepcopy(psi)
        return PropagationSolver.DynamicState(l, t, psi_copy, psi_omega_copy, E, freq_mult, dir)


    def do_the_thing(self, dyn: PropagationSolver.DynamicState, instr: PropagationSolver.InstrumentationOutputData):
        # algorithm without control
        if self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION and \
           self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
            dAdt = 0.0
        # local control algorithm with goal population
        elif self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION:
            coef = 2.0 * phys_base.cm_to_erg / phys_base.Red_Planck_h
            dAdt = dyn.E * instr.psigc_psie.imag * coef
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
            body = Sdvge + freq_cm * dyn.freq_mult * Sge2
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

        if dyn.l % self.conf_fitter.mod_log == 0:
            if self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.FILTERING:
                print("emax = ", instr.emax)
                print("emin = ", instr.emin)
            if self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.FILTERING and \
               self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.SINGLE_POT and \
               self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                print("normalization on the excited state = ", abs(instr.cnorm[1]))
            if self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.FILTERING and \
               self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.SINGLE_POT and \
               self.conf_fitter.task_type != TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
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


    # calculating envelope of the laser field energy at the given time value
    def LaserFieldEnvelope(self, prop: PropagationSolver, stat: PropagationSolver.StaticState, dyn: PropagationSolver.DynamicState):
        assert self.dyn.dir == PropagationSolver.Direction.FORWARD
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
            if prop.dyn.E == 0.0:
                prop.dyn.E = self.dyn.E_patched

            if self.dyn.E_patched <= 0:
                raise RuntimeError("E_patched has to be positive")

            if prop.dyn.E <= 0:
                raise RuntimeError("E has to be positive")

            # solving dynamic equation for E
            # linear difference to the "desired" value
            first = prop.dyn.E - self.dyn.E_patched
            # decay term
            second = self.dyn.E_vel * math.pow(self.dyn.E_patched / prop.dyn.E, self.conf_fitter.pow)

            # Euler
            E_acc = -self.conf_fitter.k_E * first - self.conf_fitter.lamb * second
            self.dyn.E_vel += E_acc * abs(stat.dt)
            E = prop.dyn.E + self.dyn.E_vel * abs(stat.dt)

        # optimal control algorithm
        elif (self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV) and \
              self.dyn.dir == PropagationSolver.Direction.FORWARD:
            if self.dyn.iter_step == 0:
                E = self.dyn.E_patched
            else:
                conf_prop = self.conf_fitter.propagation

                chie_old_psig_new = math_base.cprod(self.dyn.chi_tlist[conf_prop.nt - prop.dyn.l].psis[0].f[1],
                                                    dyn.psi_omega.f[0],
                                                    stat.dx, conf_prop.np)

                E = -2.0 * chie_old_psig_new.imag / self.conf_fitter.h_lambda

        # optimal control unitary transformation algorithm
        elif (self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM) and \
              self.dyn.dir == PropagationSolver.Direction.FORWARD:
            conf_prop = self.conf_fitter.propagation
            if prop.dyn.l == 0:
                E = self.dyn.E_patched
                chi_basis_0 = self.dyn.chi_tlist[-1]

                # psi/chi ~ forward/backward propagation wf, init ~ t = 0.0, {0;1} ~ # of basis vector, {g;e} ~ ground/excited state
                chi0_init_g = chi_basis_0.psis[0].f[0]
                chi1_init_e = chi_basis_0.psis[1].f[1]
                chi0_init_e = chi_basis_0.psis[0].f[1]
                chi1_init_g = chi_basis_0.psis[1].f[0]

                psi0_init_g = self.psi_init_basis.psis[0].f[0]
                psi1_init_e = self.psi_init_basis.psis[1].f[1]
                psi0_init_e = self.psi_init_basis.psis[0].f[1]
                psi1_init_g = self.psi_init_basis.psis[1].f[0]

                self.a0 = math_base.cprod(psi0_init_g, chi0_init_g, stat.dx, conf_prop.np) + \
                          math_base.cprod(psi0_init_e, chi0_init_e, stat.dx, conf_prop.np) + \
                          math_base.cprod(psi1_init_g, chi1_init_g, stat.dx, conf_prop.np) + \
                          math_base.cprod(psi1_init_e, chi1_init_e, stat.dx, conf_prop.np)
            else:
                chi_basis = self.dyn.chi_tlist[-prop.dyn.l]

                # psi/chi ~ forward/backward propagation wf, {0;1} ~ # of basis vector, {g;e} ~ ground/excited state
                psi0_g = self.dyn.psi_cur.psis[0].f[0]
                psi1_e = self.dyn.psi_cur.psis[1].f[1]
                psi0_e = self.dyn.psi_cur.psis[0].f[1]
                psi1_g = self.dyn.psi_cur.psis[1].f[0]

                chi0_g = chi_basis.psis[0].f[0]
                chi1_e = chi_basis.psis[1].f[1]
                chi0_e = chi_basis.psis[0].f[1]
                chi1_g = chi_basis.psis[1].f[0]

                sum = math_base.cprod(chi0_g, psi0_g, stat.dx, conf_prop.np) + \
                      math_base.cprod(chi0_e, psi0_e, stat.dx, conf_prop.np) + \
                      math_base.cprod(chi1_g, psi1_g, stat.dx, conf_prop.np) + \
                      math_base.cprod(chi1_e, psi1_e, stat.dx, conf_prop.np)

                s = self.laser_field(conf_prop.E0, dyn.t - (abs(stat.dt) / 2.0), conf_prop.t0, conf_prop.sigma) / conf_prop.E0
                #E_init = s * conf_prop.E0 * cmath.exp(1j * 2.0 * math.pi * conf_prop.nu_L * (dyn.t - (abs(stat.dt) / 2.0)))
                E_init = s * conf_prop.E0 * math.cos(2.0 * math.pi * conf_prop.nu_L * (dyn.t - (abs(stat.dt) / 2.0)))
                delta_E = - s * (self.a0 * sum).imag / self.conf_fitter.h_lambda #/ phys_base.Red_Planck_h * phys_base.cm_to_erg
                #delta_E *= cmath.exp(-2.0 * 1j * math.pi * conf_prop.nu_L * (dyn.t - (abs(stat.dt) / 2.0)))

                #print(f"===== Got {delta_E}")
                #if abs(self.TMP_delta_E) > abs(delta_E):
                #    print("===== Got max")
                #self.TMP_delta_E.append(delta_E)
                #print(f"delta_E[{len(self.TMP_delta_E)}] = {delta_E}")

                if self.dyn.iter_step == 0:
                    E = E_init + delta_E
                else:
                    E = self.dyn.E_tlist[prop.dyn.l] + delta_E
        else:
            E = self.dyn.E_patched

        return E


    # calculating envelope of the laser field energy at the given time value for back propagation
    def LaserFieldEnvelopeBackward(self, prop: PropagationSolver, stat: PropagationSolver.StaticState,
                                    dyn: PropagationSolver.DynamicState):
        conf_prop = self.conf_fitter.propagation
        assert self.dyn.dir == PropagationSolver.Direction.BACKWARD

        # optimal control algorithm
        if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV:
            chie_new_psig_old = math_base.cprod(dyn.psi_omega.f[1],
                                                self.dyn.psi_tlist[conf_prop.nt - prop.dyn.l].psis[0].f[0],
                                                stat.dx, conf_prop.np)

            E = -2.0 * chie_new_psig_old.imag / self.conf_fitter.h_lambda

        # optimal control unitary transformation algorithm
        elif self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
            if self.dyn.iter_step == 0:
                #cos = cmath.exp(1j * 2.0 * math.pi * conf_prop.nu_L * dyn.t)
                cos = math.cos(2.0 * math.pi * conf_prop.nu_L * dyn.t)
                E = self.laser_field(conf_prop.E0, dyn.t, conf_prop.t0, conf_prop.sigma) * cos
            else:
                if prop.dyn.l == 0:
                    E = self.dyn.E_tlist[-1]
                else:
                    E = self.dyn.E_tlist[-prop.dyn.l]
        else:
            E = self.laser_field(conf_prop.E0, dyn.t, conf_prop.t0, conf_prop.sigma)

        return E


    # calculating a frequency multiplier value at the given time value
    def FreqMultiplier(self, dyn: PropagationSolver.DynamicState, stat: PropagationSolver.StaticState):
        # local control algorithm (with A = Pg + Pe)
        if self.conf_fitter.task_type == TaskRootConfiguration.FitterConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
            if self.dyn.freq_mult_patched < 0:
                raise RuntimeError("freq_mult_patched has to be positive or zero")

            if dyn.freq_mult < 0:
                raise RuntimeError("Frequency multiplicator has to be positive or zero")

            # solving dynamic equation for frequency multiplicator
            # linear difference to the "desired" value
            first = dyn.freq_mult - self.dyn.freq_mult_patched
            # decay term
            second = self.dyn.freq_mult_vel * math.pow(self.dyn.freq_mult_patched / dyn.freq_mult, self.conf_fitter.pow)

            # Euler
            freq_mult_acc = -self.conf_fitter.k_E * first - self.conf_fitter.lamb * second
            self.dyn.freq_mult_vel += freq_mult_acc * abs(stat.dt)
            freq_mult = dyn.freq_mult + self.dyn.freq_mult_vel * abs(stat.dt)
        else:
            freq_mult = self.dyn.freq_mult_patched

        return freq_mult
