import collections.abc
import os.path
import json

from numpy.typing import NDArray

import reporter
from config import TaskRootConfiguration
from propagation import *
from psi_basis import PsiBasis, Psi
from typing import List

class FittingSolver:
    class FitterDynamicState:
        chi_tlist: List[PsiBasis]
        psi_tlist: List[PsiBasis]
        E_tlist: List[numpy.complex128]
        chi_cur: PsiBasis
        psi_cur: PsiBasis
        goal_close_vec: NDArray[numpy.complex128]

        def __init__(self, basis_length, levels_number, E_vel=numpy.float64(0.0), freq_mult_vel=numpy.float64(0.0),
                     iter_step=0, dir=PropagationSolver.Direction.FORWARD):
            self.E_vel = E_vel
            self.freq_mult_vel = freq_mult_vel
            self.iter_step = iter_step
            self.dir = dir

            self.E_patched = numpy.float64(0.0)
            self.freq_mult_patched = numpy.float64(1.0)
            self.dAdt_happy = numpy.float64(0.0)
            self.Fsm = numpy.complex128(0)
            self.E_int = numpy.float64(0.0)
            self.J = numpy.float64(0.0)
            self.h_lambda = numpy.float64(0.0)
            self.F_sm_mid_1 = numpy.float64(0.0)
            self.F_sm_mid_2 = numpy.float64(0.0)

            self.chi_tlist = []
            self.psi_tlist = []

            self.chi_cur = PsiBasis(basis_length, levels_number)
            self.psi_cur = PsiBasis(basis_length, levels_number)

            self.goal_close_abs = numpy.float64(0.0)
            self.goal_close_scal = numpy.float64(0.0)
            self.goal_close_vec = numpy.zeros(basis_length, numpy.complex128)
            self.E_tlist = []

            self.res = PropagationSolver.StepReaction.OK

        @staticmethod
        def from_json(json_str: str):
            def object_hook(dct: dict):
                new_chi_tlist = []
                for time_point in dct['chi_tlist']:
                    new_time_point = []
                    for level in time_point:
                        new_level = numpy.array(level).astype(numpy.complex128)
                        new_time_point.append(new_level)
                    new_chi_tlist.append(new_time_point)
                dct['chi_tlist'] = new_chi_tlist

                new_psi_tlist = []
                for time_point in dct['psi_tlist']:
                    new_time_point = []
                    for level in time_point:
                        new_level = numpy.array(level).astype(numpy.complex128)
                        new_time_point.append(new_level)
                    new_psi_tlist.append(new_time_point)
                dct['psi_tlist'] = new_psi_tlist

                new_E_tlist = numpy.array(dct['E_tlist']).astype(numpy.complex128)
                dct['E_tlist'] = new_E_tlist

                dct['goal_close'] = (dct['goal_close']).astype(numpy.complex128)

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

                if isinstance(obj, numpy.complex128):
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

    solvers: List[PropagationSolver]
    propagation_reporters: List[PropagationReporter]

    def __initialize_propagation(self, prop_id: str, laser_field_envelope, laser_field_hf, hamil2D, ntriv):

        self.solvers = []
        self.propagation_reporters = [None] * self.basis_length
        for vect in range(self.basis_length):
            propagation_reporter = self.reporter.create_propagation_reporter(os.path.join(prop_id, "basis_" + str(vect)), self.levels_number)
            propagation_reporter.open()
            self.propagation_reporters[vect] = propagation_reporter

            self.solvers.append(PropagationSolver(
                pot=self.pot_func,
                T=self.T,
                _warning_collocation_points=self._warning_collocation_points,
                _warning_time_steps=self._warning_time_steps,
                reporter=propagation_reporter,
                hamil2D=hamil2D,
                laser_field_envelope=laser_field_envelope,
                laser_field_hf=laser_field_hf,
                freq_multiplier=self.FreqMultiplier,
                dynamic_state_factory=self.propagation_dynamic_state_factory,
                pcos=self.conf_fitter.pcos,
                w_list=self.conf_fitter.w_list,
                mod_log=self.conf_fitter.mod_log,
                ntriv=ntriv,
                hf_hide=self.conf_fitter.hf_hide,
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
            task_type,
            T,
            init_dir,
            ntriv,
            psi_init_basis: PsiBasis,
            psi_goal_basis: PsiBasis,
            pot_func,
            Fgoal,
            laser_field,
            laser_field_hf,
            F_type,
            aF_type,
            hamil2D,
            reporter: reporter.FitterReporter,
            _warning_collocation_points,
            _warning_time_steps
    ):
        self._warning_collocation_points = _warning_collocation_points
        self._warning_time_steps = _warning_time_steps
        self.pot_func = pot_func
        self.laser_field = laser_field
        self.laser_field_hf = laser_field_hf
        self.F_type = F_type
        self.aF_type = aF_type
        self.hamil2D = hamil2D

        self.reporter = reporter

        self.task_type = task_type
        self.T = T

        self.conf_fitter = conf_fitter
        self.init_dir = init_dir
        self.ntriv = ntriv
        self.Fgoal = Fgoal
        self.psi_init_basis = psi_init_basis
        self.psi_goal_basis = psi_goal_basis

        self.basis_length = len(psi_init_basis)
        self.levels_number = len(psi_init_basis.psis[0].f)
        self.a0 = numpy.complex128(0)

        self.dyn = None

#        self.TMP_delta_E = []

    # single propagation to the given direction; returns new chiT
    def __single_propagation(self, dx, x, t_step, direct: PropagationSolver.Direction, chiT: PsiBasis):
        self.dyn.dir = direct
        init_psi_basis: PsiBasis
        fin_psi_basis: PsiBasis
        chiT_omega = PsiBasis(self.basis_length, self.levels_number)
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
            t_init = numpy.float64(0.0)
        else:
            ind_dir = "b"
            laser_field = self.LaserFieldEnvelopeBackward

            if self.dyn.iter_step > 0:
                self.dyn.res = PropagationSolver.StepReaction.ITERATE

            if self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                chiT = copy.deepcopy(self.psi_goal_basis)

                self.dyn.chi_tlist = [ chiT ]

            self.dyn.chi_cur = self.dyn.chi_tlist[0]
            init_psi_basis = chiT
            fin_psi_basis = self.psi_init_basis
            t_init = self.T

        self.__initialize_propagation("iter_" + str(self.dyn.iter_step) + ind_dir, laser_field, self.laser_field_hf, self.hamil2D, self.ntriv)

        # Propagation loop
        finished: set[int] = set()

        # Starting solvers
        for vect in range(self.basis_length):
            solver = self.solvers[vect]
            solver.start(dx, x, t_step, init_psi_basis.psis[vect], fin_psi_basis.psis[vect], self.dyn.dir)

        # Paranoid checking E EVERYWHERE
        E_checked = self.solvers[0].dyn.E
        for vect in range(self.basis_length):
            if abs(E_checked - self.solvers[vect].dyn.E) > 0.001:
                raise AssertionError("Different energies in different solvers")

        E_tlist_new: List[numpy.complex128] = []
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

#        self.TMP_delta_E = []
        while True:
            chi_new: PsiBasis = PsiBasis(self.basis_length, self.levels_number)
            psi_new: PsiBasis = PsiBasis(self.basis_length, self.levels_number)

            for vect in range(self.basis_length):
                solver = self.solvers[vect]

                if vect in finished:
                    raise AssertionError(f"The solver #{vect} has finished, but asked to proceed. It's a pity.")

                if (solver.dyn.l - 1) % self.conf_fitter.mod_log == 0 and \
                   self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                    print(f"The solver for the basis vector #{vect} is running...")

                if not solver.step(t_init):
                    finished.add(vect)

                if not self.conf_fitter.hf_hide:
                    psi_copy = copy.deepcopy(solver.dyn.psi)
                else:
                    psi_copy = copy.deepcopy(solver.dyn.psi_omega)

                if direct == PropagationSolver.Direction.BACKWARD:
                    chi_new.psis[vect] = psi_copy
                else:
                    psi_new.psis[vect] = psi_copy

                self.do_the_thing(solver.dyn, solver.instr)

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
                if self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV:
                    self.dyn.psi_tlist.append(psi_new)

            do_continue = len(finished) < self.basis_length
            if not do_continue:
                break

        if direct == PropagationSolver.Direction.FORWARD:
            self.dyn.E_tlist = E_tlist_new

            self.dyn.E_int = numpy.float64(0.0)
            for el in range(len(self.dyn.E_tlist) - 1):
                E_cur = self.dyn.E_tlist[el + 1]
                self.dyn.E_int += E_cur * E_cur.conjugate() * (t_step * 1e15)

#            with open("test_E_" + str(self.dyn.iter_step) + ".txt", "w") as f:
#                f.write("TMP_delta_E = [\n")
#                for l in self.TMP_delta_E:
#                    f.write("    " + str(l) + ",\n")
#                f.write("]\n\n")

#            with open("test_E_" + str(self.dyn.iter_step) + ".txt", "w") as f:
#                f.write("E_tlist = [\n")
#                for l in self.dyn.E_tlist:
#                    f.write("    " + str(l) + ",\n")
#                f.write("]\n\n")

        self.dyn.goal_close_vec = numpy.zeros(self.basis_length, dtype=numpy.complex128)
        for vect in range(self.basis_length):
            solver = self.solvers[vect]

            if direct == PropagationSolver.Direction.FORWARD:
                chiT_part = Psi()
                assert solver.dyn.l - 1 == self.conf_fitter.propagation.nt
                for n in range(self.levels_number):
                    print("phase(psi_%d(T)) = %f" % (n, cmath.phase(solver.dyn.psi.f[n][0])))
                    print("phase(psif_%d) = %f" % (n, cmath.phase(solver.stat.psif.f[n][0])))
                    print("phase(psi0_%d) = %f" % (n, cmath.phase(solver.stat.psi0.f[n][0])))

                    print("|psi_%d(T)| = %f" % (n, abs(solver.dyn.psi.f[n][0])))
                    print("|psif_%d| = %f" % (n, abs(solver.stat.psif.f[n][0])))
                    print("|psi0_%d| = %f" % (n, abs(solver.stat.psi0.f[n][0])))

                    print("psi_%d(T)" % n + " = ", solver.dyn.psi.f[n][0])

                    self.dyn.goal_close_vec[vect] += math_base.cprod(solver.stat.psif.f[n], solver.dyn.psi.f[n],
                                                                     solver.stat.dx, self.conf_fitter.propagation.np)

                if self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV:
                    chiT_part.f[0] = numpy.zeros(self.conf_fitter.propagation.np, dtype=numpy.complex128)
                    chiT_part.f[1] = self.dyn.goal_close_vec[vect] * solver.stat.psif.f[1]

                    # renormalization
                    cnorm = math_base.cprod(chiT_part.f[1], chiT_part.f[1], dx, self.conf_fitter.propagation.np)
                    if abs(cnorm) > 0.0:
                        chiT_part.f[1] /= math.sqrt(abs(cnorm))

                    chiTp_omega = copy.deepcopy(chiT_part)
                    hf_part = self.laser_field_hf(solver.dyn.freq_mult,
                                                  self.T,
                                                  self.conf_fitter.pcos,
                                                  self.conf_fitter.w_list)
                    chiTp_omega.f[1] *= cmath.sqrt(hf_part)
                    chiT_omega.psis[vect] = chiTp_omega
                    chiT.psis[vect] = copy.deepcopy(chiT_part)

                elif self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                    print("goal_close value for the basis vector ", vect, " = ", self.dyn.goal_close_vec[vect])
                else:
                    pass

            self.dyn.goal_close_scal += self.dyn.goal_close_vec[vect] # tau

        if self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV and \
                direct == PropagationSolver.Direction.FORWARD:
            self.dyn.chi_tlist = [ chiT_omega ]

        if direct == PropagationSolver.Direction.FORWARD:
            self.dyn.goal_close_abs = abs(self.dyn.goal_close_scal)

        self.dyn.Fsm = self.F_type(self.dyn.goal_close_vec, self.basis_length)

        print("Fsm = ", self.dyn.Fsm.real)

        if self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV:
            h_lambda = self.conf_fitter.h_lambda
        elif self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
            h_lambda = self.dyn.h_lambda
        else:
            h_lambda = numpy.float64(0.0)

        self.dyn.J = self.dyn.Fsm.real - h_lambda * h_lambda * self.dyn.E_int.real

        self.__finalize_propagation()
        return chiT

    # single simple forward propagation
    def __single_iteration_simple(self, dx, x, t_step, t_list):
        chiT = self.__single_propagation(dx, x, t_step, PropagationSolver.Direction.FORWARD, self.psi_goal_basis)

        E_list = []
        for E in self.dyn.E_tlist:
            if E.imag:
                E_list.append(abs(E))
            else:
                E_list.append(E.real)

        self.reporter.print_iter_point_fitter(self.dyn.iter_step, self.dyn.goal_close_abs, E_list, t_list, self.dyn.Fsm,
                                              self.dyn.E_int, self.dyn.J, self.conf_fitter.propagation.nt)

        return 0.0

    # single iteration for an optimal control task
    def __single_iteration_optimal(self, dx, x, t_step, t_list):
        assert (dx > 0.0)

        chiT = PsiBasis(self.basis_length, self.levels_number)
        self.dyn.goal_close_scal = numpy.float64(0.0)

        direct = self.init_dir
        for run in range(2):
            print(f"Iteration = {self.dyn.iter_step}, {direct.name} direction begins...")
            # calculating chiT
            chiT = self.__single_propagation(dx, x, t_step, direct, chiT)

#            if direct == PropagationSolver.Direction.FORWARD:
#                ind_dir = 'f'
#            else:
#                ind_dir = 'b'

#            saved_json, arrays = self.dyn.to_json_with_bins()

#            if not os.path.exists("savings"):
#                os.mkdir("savings")

#            path = os.path.join("savings", "iter_" + str(self.dyn.iter_step) + ind_dir)
#            if not os.path.exists(path):
#                os.mkdir(path)

#            with open(os.path.join(path, "fitter_state.json"), 'w') as f:
#                f.write(saved_json)

#            np.savez_compressed(os.path.join(path, "fitter_state_bins.npz"), arrays)

            if self.task_type != TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                if abs(self.dyn.Fsm.real - self.Fgoal) <= self.conf_fitter.epsilon and self.dyn.iter_step == 0:
                    print("The goal has been reached on the very first iteration. You don't need the control!")
                    self.dyn.res = PropagationSolver.StepReaction.OK
                    break

            if direct == PropagationSolver.Direction.FORWARD:
                E_list = []
                for E in self.dyn.E_tlist:
                    if E.imag:
                        E_list.append(abs(E))
                    else:
                        E_list.append(E.real)

                self.reporter.print_iter_point_fitter(self.dyn.iter_step, self.dyn.goal_close_abs, E_list, t_list, self.dyn.Fsm,
                                                      self.dyn.E_int, self.dyn.J, self.conf_fitter.propagation.nt)

            direct = PropagationSolver.Direction(-direct.value)

    def time_propagation(self, dx, x, t_step, t_list):
        self.dyn = FittingSolver.FitterDynamicState(self.basis_length, self.levels_number, E_vel=numpy.float64(0.0),
                                                    freq_mult_vel=numpy.float64(0.0), iter_step=0, dir=self.init_dir)
        if self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
            E_tlist_init = []
            goal_close_init = numpy.zeros(self.basis_length, dtype=numpy.complex128)
            goal_close_scal_init = numpy.float64(0.0)
            for vect in range(self.basis_length):
                for n in range(self.levels_number):
                    goal_close_init[vect] += math_base.cprod(self.psi_goal_basis.psis[vect].f[n], self.psi_init_basis.psis[vect].f[n],
                                                           dx, self.conf_fitter.propagation.np)
                goal_close_scal_init += goal_close_init[vect]

            goal_close_abs_init = abs(goal_close_scal_init)

#            Fsm_init = numpy.complex128(0)
#            for vect in range(self.basis_length):
#                for vect1 in range(self.basis_length):
#                    Fsm_init -= goal_close_init[vect] * goal_close_init[vect1].conjugate()

            Fsm_init = self.F_type(goal_close_init, self.basis_length)
            for t in t_list:
                hf_part = self.laser_field_hf(numpy.float64(1.0), t, self.conf_fitter.pcos, self.conf_fitter.w_list)
                E = self.laser_field(self.conf_fitter.propagation.E0, t, self.conf_fitter.propagation.t0,
                                     self.conf_fitter.propagation.sigma) * hf_part
                E_tlist_init.append(E.real)

            E_int_init = numpy.float64(0.0)
            for el in range(len(E_tlist_init) - 1):
                E_cur_init = E_tlist_init[el + 1]
                E_int_init += E_cur_init * E_cur_init.conjugate() * (t_step * 1e15)

            if self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV or \
               self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                if self.ntriv == -1:
                    h_lambda_init = self.conf_fitter.h_lambda * phys_base.Red_Planck_h / phys_base.cm_to_erg
                else:
                    h_lambda_init = self.conf_fitter.h_lambda
            else:
                h_lambda_init = numpy.float64(0.0)

            J_init = Fsm_init.real - h_lambda_init * h_lambda_init * E_int_init

            self.reporter.print_iter_point_fitter(-1, goal_close_abs_init, E_tlist_init, t_list, Fsm_init,
                                                  E_int_init, J_init, self.conf_fitter.propagation.nt)

        if self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV or \
           self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
            # 0-th iteration
            self.__single_iteration_optimal(dx, x, t_step, t_list)
#                self.dyn = FittingSolver.FitterDynamicState.from_json(saved_json)
#                print(self.dyn)

            # iterative procedure
            while abs(self.dyn.Fsm.real - self.Fgoal) > self.conf_fitter.epsilon and \
                    (self.dyn.iter_step < self.conf_fitter.iter_max or self.conf_fitter.iter_max == -1):
                self.dyn.iter_step += 1
                self.__single_iteration_optimal(dx, x, t_step, t_list)

                if self.conf_fitter.q:
                    if self.dyn.iter_step == self.conf_fitter.iter_mid_1:
                        self.dyn.F_sm_mid_1 = self.dyn.Fsm.real
                    elif self.dyn.iter_step == self.conf_fitter.iter_mid_2:
                        self.dyn.F_sm_mid_2 = self.dyn.Fsm.real
                        if (self.dyn.F_sm_mid_1 > -self.basis_length * self.basis_length * self.conf_fitter.q or
                            self.dyn.F_sm_mid_2 > -self.basis_length * self.basis_length * self.conf_fitter.q or
                            self.dyn.F_sm_mid_1 < self.dyn.F_sm_mid_2):
                            print(f"Fsm(iter_mid_1) = {str(self.dyn.F_sm_mid_1)}, Fsm(iter_mid_2) = {str(self.dyn.F_sm_mid_2)}. "
                                  f"Is the calculation diverging?..")
                            raise ValueError("The calculation is probably going to diverge.  Stopping the run...")
                    else:
                        pass

            if abs(self.dyn.Fsm.real - self.Fgoal) <= self.conf_fitter.epsilon:
                print("The goal has been successfully reached on the " + str(self.dyn.iter_step) + " iteration.")
            else:
                print("The goal has not been reached during the calculation.")

            self.dyn.res = PropagationSolver.StepReaction.OK
        else:
            self.__single_iteration_simple(dx, x, t_step, t_list)

    def propagation_dynamic_state_factory(self, l, t, psi: Psi, psi_omega: Psi, E, freq_mult: numpy.float64, dir):
        psi_omega_copy = copy.deepcopy(psi_omega)
        psi_copy = copy.deepcopy(psi)
        return PropagationSolver.DynamicState(l, t, psi_copy, psi_omega_copy, E, freq_mult, dir)

    def do_the_thing(self, dyn: PropagationSolver.DynamicState, instr: PropagationSolver.InstrumentationOutputData):
        # algorithm without control
        if self.task_type != TaskRootConfiguration.TaskType.LOCAL_CONTROL_POPULATION and \
           self.task_type != TaskRootConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
            dAdt = numpy.float64(0.0)
        # local control algorithm with goal population
        elif self.task_type == TaskRootConfiguration.TaskType.LOCAL_CONTROL_POPULATION:
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
                    self.dyn.freq_mult_patched = numpy.float64(0.0)

                self.dyn.res = PropagationSolver.StepReaction.CORRECT

        if dyn.l % self.conf_fitter.mod_log == 0:
            if self.task_type != TaskRootConfiguration.TaskType.FILTERING:
                print("emax = ", instr.emax)
                print("emin = ", instr.emin)
            if self.task_type != TaskRootConfiguration.TaskType.FILTERING and \
               self.task_type != TaskRootConfiguration.TaskType.SINGLE_POT and \
               self.task_type != TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                print("normalization on the state #1 = ", abs(instr.cnorm[1]))
            if self.task_type != TaskRootConfiguration.TaskType.FILTERING and \
               self.task_type != TaskRootConfiguration.TaskType.SINGLE_POT and \
               self.task_type != TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                print("energy on the state #1 = ", instr.cener[1].real)
            if self.task_type == TaskRootConfiguration.TaskType.LOCAL_CONTROL_POPULATION or \
               self.task_type == TaskRootConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
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
        if self.conf_fitter.hf_hide:
            self.dyn.E_patched = self.laser_field(self.conf_fitter.propagation.E0, dyn.t, self.conf_fitter.propagation.t0, self.conf_fitter.propagation.sigma)
        else:
            env = self.laser_field(self.conf_fitter.propagation.E0, dyn.t,
                                                  self.conf_fitter.propagation.t0, self.conf_fitter.propagation.sigma)
            hf_part = self.laser_field_hf(dyn.freq_mult, dyn.t, self.conf_fitter.pcos, self.conf_fitter.w_list)
            self.dyn.E_patched = env * hf_part

        # transition without control
        if self.task_type == TaskRootConfiguration.TaskType.TRANS_WO_CONTROL:
            E = self.dyn.E_patched

        # intuitive control algorithm
        elif self.task_type == TaskRootConfiguration.TaskType.INTUITIVE_CONTROL:
            for npul in range(1, self.conf_fitter.impulses_number):
                self.dyn.E_patched += self.laser_field(self.conf_fitter.propagation.E0, dyn.t,
                                                       self.conf_fitter.propagation.t0 + (npul * self.conf_fitter.delay),
                                                       self.conf_fitter.propagation.sigma)
            E = self.dyn.E_patched

        # local control algorithm (with A = Pe)
        elif self.task_type == TaskRootConfiguration.TaskType.LOCAL_CONTROL_POPULATION:
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
        elif (self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV) and \
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
        elif (self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM) and \
              self.dyn.dir == PropagationSolver.Direction.FORWARD:
            conf_prop = self.conf_fitter.propagation
            if prop.dyn.l == 0:
                E = self.dyn.E_patched
                chi_init = self.dyn.chi_tlist[-1]
                psi_init = self.psi_init_basis

                #numpy.angle(chi_init)
                print("Current goal_close_abs:\t\t"   f"{self.dyn.goal_close_abs}\n")

                if self.ntriv == -1:
                    h_lambda_0 = self.conf_fitter.h_lambda * phys_base.Red_Planck_h / phys_base.cm_to_erg
                else:
                    h_lambda_0 = self.conf_fitter.h_lambda

                if self.conf_fitter.h_lambda_mode == TaskRootConfiguration.FitterConfiguration.HlambdaModeType.DYNAMICAL:
                    if self.dyn.goal_close_abs:
                        self.dyn.h_lambda = h_lambda_0 * self.basis_length / math.sqrt(self.dyn.goal_close_abs)
                       #self.conf_fitter.h_lambda *= pow(5, 1.0 / 200)
                    else:
                        self.dyn.h_lambda = h_lambda_0
                elif self.conf_fitter.h_lambda_mode == TaskRootConfiguration.FitterConfiguration.HlambdaModeType.CONST:
                    self.dyn.h_lambda = h_lambda_0
                else:
                    raise RuntimeError("Impossible case in the HlambdaModeType class")

                self.a0 = self.aF_type(psi_init, chi_init, stat.dx, self.basis_length, self.levels_number, conf_prop.np)
                pass
            else:
                chi_basis = self.dyn.chi_tlist[-prop.dyn.l]
                psi_basis = self.dyn.psi_cur
                sum = numpy.float64(0.0)
#                sum1 = 0.0

                m1 = numpy.float64(0.0)
                for vect in range(self.basis_length):
                    for n in range(self.levels_number):
                        sum += self.a0[vect] * math_base.cprod(chi_basis.psis[vect].f[n], psi_basis.psis[vect].f[n], stat.dx, conf_prop.np)
                        abs_chi = math_base.cprod(chi_basis.psis[vect].f[n], chi_basis.psis[vect].f[n], stat.dx, conf_prop.np)
                        abs_psi = math_base.cprod(psi_basis.psis[vect].f[n], psi_basis.psis[vect].f[n], stat.dx,
                                                  conf_prop.np)
                        m1 += (abs_chi - abs_psi).real
#                        sum1 += math_base.cprod(chi_basis.psis[vect].f[n], psi_basis.psis[vect].f[n], stat.dx, conf_prop.np)
                hf_part = self.laser_field_hf(dyn.freq_mult, dyn.t - (abs(stat.dt) / 2.0), self.conf_fitter.pcos, self.conf_fitter.w_list)
                s = self.laser_field(conf_prop.E0, dyn.t - (abs(stat.dt) / 2.0), conf_prop.t0, conf_prop.sigma) / conf_prop.E0
                E_init = s * conf_prop.E0 * hf_part

                if prop.dyn.l <= 5:
                    print("sum:\t\t"   f"{sum}\n")
#                    print("sum1 * a0[0]:\t\t"   f"{sum1 * self.a0[0]}\n")

                delta_E = -s * sum.imag / self.dyn.h_lambda

                d = pow(self.basis_length - self.dyn.goal_close_abs, 0.25)
                m1 = numpy.float64(m1 / abs(m1)) if abs(m1) != 0.0 else numpy.float64(0.0)
                #delta_E = -s * (sum.imag * d + m1 * d*d) / self.dyn.h_lambda

#                delta_E = - s * (self.a0[0] * sum1).imag / self.dyn.h_lambda

#                print(f"===== Got {delta_E}")
#                if abs(self.TMP_delta_E) > abs(delta_E):
#                    print("===== Got max")
#                self.TMP_delta_E.append(delta_E)
#                print(f"delta_E[{len(self.TMP_delta_E)}] = {delta_E}")

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
        if self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV:
            chie_new_psig_old = math_base.cprod(dyn.psi_omega.f[1],
                                                self.dyn.psi_tlist[conf_prop.nt - prop.dyn.l].psis[0].f[0],
                                                stat.dx, conf_prop.np)

            E = -2.0 * chie_new_psig_old.imag / self.conf_fitter.h_lambda

        # optimal control unitary transformation algorithm
        elif self.task_type == TaskRootConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
            if self.dyn.iter_step == 0:
                hf_part = self.laser_field_hf(dyn.freq_mult, dyn.t, self.conf_fitter.pcos, self.conf_fitter.w_list)
                E = self.laser_field(conf_prop.E0, dyn.t, conf_prop.t0, conf_prop.sigma) * hf_part
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
        if self.task_type == TaskRootConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
            if self.dyn.freq_mult_patched < 0:
                raise RuntimeError("freq_mult_patched has to be positive or zero")

            if dyn.freq_mult < 0:
                raise RuntimeError("Frequency multiplier has to be positive or zero")

            # solving dynamic equation for frequency multiplier
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

        return numpy.float64(freq_mult)
