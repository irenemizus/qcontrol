import unittest

import fitter
import grid_setup
import hamil_2d
import math_base
import phys_base
import task_manager
import test_data
from config import TaskRootConfiguration
from propagation import PropagationSolver
from psi_basis import PsiBasis
from test_tools import *

PATH_REP = os.path.join("iter_0f", "basis_0")


class fitter_Tests(unittest.TestCase):
    @staticmethod
    def _test_setup(user_conf):
        conf = TaskRootConfiguration()
        conf.load(user_conf)
        print(conf)

        task_manager_imp = task_manager.create(conf)

        return conf, task_manager_imp

    @staticmethod
    def _pi_pulse_test_setup(user_conf):
        conf = TaskRootConfiguration()
        conf.load(user_conf)
        print(conf)
        conf_prop = conf.fitter.propagation

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf)
        dx, x = grid.grid_setup()

        # setup of the time grid
        forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_task=conf)
        t_step, t_list = forw_time_grid.grid_setup()

        psi0 = PsiBasis(1)
        # evaluating of initial wavefunction
        psi0.psis[0].f[0] = task_manager._PsiFunctions.one(x, conf.np, conf.x0, conf.p0, conf_prop.m,
                                             conf.De, conf.a, conf.L)
        psi0.psis[0].f[1] = task_manager._PsiFunctions.zero(conf.np)

        psif = PsiBasis(1)
        # evaluating of the final goal
        psif.psis[0].f[0] = task_manager._PsiFunctions.zero(conf.np)
        psif.psis[0].f[1] = task_manager._PsiFunctions.one(x, conf.np, conf.x0p + conf.x0, conf.p0,
                                              conf_prop.m, conf.De_e, conf.a_e, conf.L)

        # evaluating of potential(s)
        v = task_manager.MultipleStateUnitTransformTaskManager._pot(x, conf.np, conf_prop.m,
                                                            conf.De, conf.a, conf.x0p, conf.De_e,
                                                            conf.a_e, conf.Du, -2, conf)

        # evaluating of k vector
        akx2 = math_base.initak(conf.np, dx, 2, -2)

        # evaluating of kinetic energy
        akx2 *= -phys_base.hart_to_cm / (2.0 * conf_prop.m * phys_base.dalt_to_au)

        # Hamiltonian for the current task
        hamil2D = hamil_2d.Hamil2DQubit(v, akx2, conf.np, conf.U, conf.W, conf.delta, -2)

        # number of levels
        nlevs = len(psi0.psis[0].f)

        return conf, task_manager_imp, dx, x, t_step, t_list, nlevs, psi0, psif, v, akx2, hamil2D

    def test_single_harm(self):
        user_conf = {
            "task_type": "single_pot",
            "pot_type": "harmonic",
            "wf_type": "harmonic",
            "hamil_type": "ntriv",
            "init_guess": "zero",
            "nb": 1,
            "nlevs": 2,
            "T": 280e-15,
            "L": 10.0,
            "np": 512,
            "a": 1.0,
            "x0p": 0.0,
            "a_e": 0.0,
            "De_e": 0.0,
            "Du": 0.0,
            "x0": 1.0,
            "p0": 0.0,
            "fitter": {
                "impulses_number": 0,
                "propagation": {
                  "m": 0.5,
                  "nch": 64,
                  "nt": 200000,
                  "E0": 0.0,
                  "nu_L": 0.0
                },
                "mod_log": 500
            }
        }
        mod_fileout = 1000
        lmin = 0
        imod_fileout = 1
        imin = 0

        conf, task_manager_imp = self._test_setup(user_conf)
        # number of levels
        nlevs = len(task_manager_imp.psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              task_manager_imp.init_dir, task_manager_imp.ntriv,
                                              task_manager_imp.psi0, task_manager_imp.psif,
                                              task_manager_imp.v, task_manager_imp.akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              task_manager_imp.hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(task_manager_imp.dx, task_manager_imp.x, task_manager_imp.t_step, task_manager_imp.t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_single_harm_.py")
        #prop_reporter.print_all("../test_data/prop_single_harm_.py", "../test_data/fitter_single_harm_.py")

        psi_prop_comparer = TableComparer(epsilon=(np.complex128(0.0001 + 0.0001j), 0.0001, 0.0001), delta=np.float64(1.e-12))  # psi, t, x
        tvals_prop_comparer = TableComparer(epsilon=(0.0001, 0.02, 0.001, 0.0001, 0.00001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                             0.000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                             np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                             0.0001, 0.0001), delta=np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.0001, 0.0001, 0.0001, 0.000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E

        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_single_harm.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_single_harm.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_single_harm.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_single_harm.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_single_harm.iter_tab_E))

    def test_single_morse(self):
        user_conf = {
            "task_type": "single_pot",
            "pot_type": "morse",
            "wf_type": "morse",
            "hamil_type": "ntriv",
            "init_guess": "zero",
            "nb": 1,
            "nlevs": 2,
            "T": 280e-16,
            "L": 4.0,
            "np": 2048,
            "a": 1.0,
            "De": 20000.0,
            "x0p": 0.0,
            "a_e": 0.0,
            "De_e": 0.0,
            "Du": 0.0,
            "x0": 0.0,
            "p0": 0.0,
            "fitter": {
                "impulses_number": 0,
                "propagation": {
                  "m": 0.5,
                  "nch": 64,
                  "nt": 20000,
                  "E0": 0.0,
                  "nu_L": 0.0
                },
                "mod_log": 500
            }
        }
        mod_fileout = 1000
        lmin = 0
        imod_fileout = 1
        imin = 0

        conf, task_manager_imp = self._test_setup(user_conf)
        # number of levels
        nlevs = len(task_manager_imp.psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              task_manager_imp.init_dir, task_manager_imp.ntriv,
                                              task_manager_imp.psi0, task_manager_imp.psif,
                                              task_manager_imp.v, task_manager_imp.akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              task_manager_imp.hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(task_manager_imp.dx, task_manager_imp.x, task_manager_imp.t_step, task_manager_imp.t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_single_morse_.py")
        #prop_reporter.print_all("../test_data/prop_single_morse_.py", "../test_data/fitter_single_morse_.py")

        psi_prop_comparer = TableComparer(epsilon=(np.complex128(0.0001 + 0.0001j), 0.0001, 0.0001), delta=np.float64(1.e-12))  # psi, t, x
        tvals_prop_comparer = TableComparer(epsilon=(0.0001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                            0.0000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                            0.0001, 0.0001), delta=np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.0001, 0.0001, 0.0001, 0.0000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E

        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_single_morse.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_single_morse.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_single_morse.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_single_morse.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_single_morse.iter_tab_E))

    def test_filter(self):
        user_conf = {
            "task_type": "filtering",
            "pot_type": "morse",
            "wf_type": "harmonic",
            "hamil_type": "ntriv",
            "init_guess": "zero",
            "nb": 1,
            "nlevs": 2,
            "T": 1980e-15,
            "L": 5.0,
            "np": 2048,
            "a": 1.0,
            "De": 20000.0,
            "x0p": 0.0,
            "a_e": 0.0,
            "De_e": 0.0,
            "Du": 0.0,
            "fitter": {
                "impulses_number": 0,
                "propagation": {
                    "m": 0.5,
                    "nch": 64,
                    "nt": 800000,
                    "E0": 0.0,
                    "nu_L": 0.0
                },
                "mod_log": 1000
            }
        }
        mod_fileout = 20000
        lmin = 0
        imod_fileout = 1
        imin = 0

        conf, task_manager_imp = self._test_setup(user_conf)
        # number of levels
        nlevs = len(task_manager_imp.psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              task_manager_imp.init_dir, task_manager_imp.ntriv,
                                              task_manager_imp.psi0, task_manager_imp.psif,
                                              task_manager_imp.v, task_manager_imp.akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              task_manager_imp.hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(task_manager_imp.dx, task_manager_imp.x, task_manager_imp.t_step, task_manager_imp.t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_filter_.py")
        #prop_reporter.print_all("../test_data/prop_filter_.py", "../test_data/fitter_filter_.py")

        psi_prop_comparer = TableComparer((np.complex128(0.0001 + 0.0001j), 0.000001, 0.0001), np.float64(1.e-12)) # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.0001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                      0.0000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                      np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                      0.0001, 0.0001), np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.000001, 0.0001, 0.0001, 0.0000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E


        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_filter.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_filter.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_filter.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_filter.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_filter.iter_tab_E))

    def test_trans_woc(self):
        user_conf = {
            "task_type": "trans_wo_control",
            "pot_type": "morse",
            "wf_type": "morse",
            "hamil_type": "ntriv",
            "init_guess": "gauss",
            "init_guess_hf": "exp",
            "nb": 1,
            "nlevs": 2,
            "T": 330e-15,
            "L": 5.0,
            "np": 1024,
            "a": 1.0,
            "De": 20000.0,
            "x0p": -0.17,
            "a_e": 1.0,
            "De_e": 10000.0,
            "Du": 20000.0,
            "fitter": {
                "impulses_number": 1,
                "propagation": {
                    "m": 0.5,
                    "nch": 64,
                    "nt": 230000,
                    "E0": 71.54,
                    "t0": 200e-15,
                    "sigma": 50e-15,
                    "nu_L": 0.29297e15
                },
                "mod_log": 500
            }
        }
        mod_fileout = 10000
        lmin = 0
        imod_fileout = 1
        imin = 0

        conf, task_manager_imp = self._test_setup(user_conf)
        # number of levels
        nlevs = len(task_manager_imp.psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              task_manager_imp.init_dir, task_manager_imp.ntriv,
                                              task_manager_imp.psi0, task_manager_imp.psif,
                                              task_manager_imp.v, task_manager_imp.akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              task_manager_imp.hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(task_manager_imp.dx, task_manager_imp.x, task_manager_imp.t_step, task_manager_imp.t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_trans_woc_.py")
        #prop_reporter.print_all("../test_data/prop_trans_woc_.py", "../test_data/fitter_trans_woc_.py")

        psi_prop_comparer = TableComparer((np.complex128(0.0001 + 0.0001j), 0.000001, 0.0001), np.float64(1.e-12)) # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                      0.0000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                      np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                      0.0001, 0.0001), np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E

        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_trans_woc.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_trans_woc.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_trans_woc.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_trans_woc.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_trans_woc.iter_tab_E))

    def test_int_ctrl(self):
        user_conf = {
            "task_type": "intuitive_control",
            "pot_type": "morse",
            "wf_type": "morse",
            "hamil_type": "ntriv",
            "init_guess": "gauss",
            "init_guess_hf": "exp",
            "nb": 1,
            "nlevs": 2,
            "T": 700e-15,
            "L": 5.0,
            "np": 1024,
            "a": 1.0,
            "De": 20000.0,
            "x0p": -0.17,
            "a_e": 1.0,
            "De_e": 10000.0,
            "Du": 20000.0,
            "fitter": {
                "impulses_number": 2,
                "delay": 300e-15,
                "propagation": {
                    "m": 0.5,
                    "nch": 64,
                    "nt": 490000,
                    "E0": 71.54,
                    "t0": 200e-15,
                    "sigma": 50e-15,
                    "nu_L": 0.29297e15
                },
                "mod_log": 500
            }
        }
        mod_fileout = 20000
        lmin = 0
        imod_fileout = 1
        imin = 0

        conf, task_manager_imp = self._test_setup(user_conf)
        # number of levels
        nlevs = len(task_manager_imp.psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              task_manager_imp.init_dir, task_manager_imp.ntriv,
                                              task_manager_imp.psi0, task_manager_imp.psif,
                                              task_manager_imp.v, task_manager_imp.akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              task_manager_imp.hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(task_manager_imp.dx, task_manager_imp.x, task_manager_imp.t_step, task_manager_imp.t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_int_ctrl_.py")
        #prop_reporter.print_all("../test_data/prop_int_ctrl_.py", "../test_data/fitter_int_ctrl_.py")

        psi_prop_comparer = TableComparer((np.complex128(0.0001 + 0.0001j), 0.000001, 0.0001), np.float64(1.e-12)) # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                      0.0000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                      np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                      0.0001, 0.0001), np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E

        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_int_ctrl.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_int_ctrl.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_int_ctrl.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_int_ctrl.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_int_ctrl.iter_tab_E))

    def test_loc_ctrl_pop(self):
        user_conf = {
            "task_type": "local_control_population",
            "pot_type": "morse",
            "wf_type": "morse",
            "hamil_type": "ntriv",
            "init_guess": "gauss",
            "init_guess_hf": "exp",
            "nb": 1,
            "nlevs": 2,
            "T": 400e-15,
            "L": 5.0,
            "np": 1024,
            "a": 1.0,
            "De": 20000.0,
            "x0p": -0.17,
            "a_e": 1.0,
            "De_e": 10000.0,
            "Du": 20000.0,
            "fitter": {
                "k_E": 1e29,
                "lamb": 4e14,
                "pow": 0.8,
                "epsilon": 1e-15,
                "impulses_number": 1,
                "propagation": {
                    "m": 0.5,
                    "nch": 64,
                    "nt": 280000,
                    "E0": 71.54,
                    "t0": 200e-15,
                    "sigma": 50e-15,
                    "nu_L": 0.29297e15
                },
                "mod_log": 500
            }
        }
        mod_fileout = 10000
        lmin = 0
        imod_fileout = 1
        imin = 0

        conf, task_manager_imp = self._test_setup(user_conf)
        # number of levels
        nlevs = len(task_manager_imp.psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              task_manager_imp.init_dir, task_manager_imp.ntriv,
                                              task_manager_imp.psi0, task_manager_imp.psif,
                                              task_manager_imp.v, task_manager_imp.akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              task_manager_imp.hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(task_manager_imp.dx, task_manager_imp.x, task_manager_imp.t_step, task_manager_imp.t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_loc_ctrl_pop_.py")
        #prop_reporter.print_all("../test_data/prop_loc_ctrl_pop_.py", "../test_data/fitter_loc_ctrl_pop_.py")

        psi_prop_comparer = TableComparer((np.complex128(0.0001 + 0.0001j), 0.000001, 0.0001), np.float64(1.e-12)) # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                      0.0000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                      np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                      0.0001, 0.0001), np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E

        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_loc_ctrl_pop.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_loc_ctrl_pop.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_loc_ctrl_pop.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_loc_ctrl_pop.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_loc_ctrl_pop.iter_tab_E))

    def test_loc_ctrl_proj(self):
        user_conf = {
            "task_type": "local_control_projection",
            "pot_type": "morse",
            "wf_type": "morse",
            "hamil_type": "ntriv",
            "init_guess": "gauss",
            "init_guess_hf": "exp",
            "nb": 1,
            "nlevs": 2,
            "T": 450e-15,
            "L": 5.0,
            "np": 1024,
            "a": 1.0,
            "De": 20000.0,
            "x0p": -0.17,
            "a_e": 1.0,
            "De_e": 10000.0,
            "Du": 20000.0,
            "fitter": {
                "k_E": 1e29,
                "lamb": 8e14,
                "pow": 0.65,
                "epsilon": 1e-15,
                "impulses_number": 1,
                "propagation": {
                    "m": 0.5,
                    "nch": 64,
                    "nt": 315000,
                    "E0": 71.54,
                    "t0": 200e-15,
                    "sigma": 50e-15,
                    "nu_L": 0.29297e15
                },
                "mod_log": 500
            }
        }
        mod_fileout = 10000
        lmin = 0
        imod_fileout = 1
        imin = 0

        conf, task_manager_imp = self._test_setup(user_conf)
        # number of levels
        nlevs = len(task_manager_imp.psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              task_manager_imp.init_dir, task_manager_imp.ntriv,
                                              task_manager_imp.psi0, task_manager_imp.psif,
                                              task_manager_imp.v, task_manager_imp.akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              task_manager_imp.hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(task_manager_imp.dx, task_manager_imp.x, task_manager_imp.t_step, task_manager_imp.t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_loc_ctrl_proj_.py")
        #prop_reporter.print_all("../test_data/prop_loc_ctrl_proj_.py", "../test_data/fitter_loc_ctrl_proj_.py")

        psi_prop_comparer = TableComparer((np.complex128(0.0001 + 0.0001j), 0.000001, 0.0001), np.float64(1.e-12)) # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                      0.0000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                      np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                      0.0001, 0.0001), np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E

        for n in range(nlevs):
            self.assertTrue(
                psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_loc_ctrl_proj.psi_tabs[n]))
            self.assertTrue(
                tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_loc_ctrl_proj.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_loc_ctrl_proj.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_loc_ctrl_proj.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_loc_ctrl_proj.iter_tab_E))

    def test_opt_ctrl_krot(self):
        user_conf = {
            "task_type": "optimal_control_krotov",
            "pot_type": "morse",
            "wf_type": "morse",
            "hamil_type": "ntriv",
            "init_guess": "gauss",
            "init_guess_hf": "exp",
            "nb": 1,
            "nlevs": 2,
            "T": 350e-15,
            "L": 5.0,
            "np": 1024,
            "a": 1.0,
            "De": 20000.0,
            "x0p": -0.17,
            "a_e": 1.0,
            "De_e": 10000.0,
            "Du": 20000.0,
            "fitter": {
                "epsilon": 1e-8,
                "impulses_number": 1,
                "iter_max": 1,
                "h_lambda": 0.0066,
                "propagation": {
                    "m": 0.5,
                    "nch": 64,
                    "nt": 230000,
                    "E0": 71.54,
                    "t0": 200e-15,
                    "sigma": 50e-15,
                    "nu_L": 0.29297e15
                },
                "mod_log": 500
            }
        }
        mod_fileout = 10000
        lmin = 0
        imod_fileout = 1
        imin = 0

        conf, task_manager_imp = self._test_setup(user_conf)
        # number of levels
        nlevs = len(task_manager_imp.psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              task_manager_imp.init_dir, task_manager_imp.ntriv,
                                              task_manager_imp.psi0, task_manager_imp.psif,
                                              task_manager_imp.v, task_manager_imp.akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              task_manager_imp.hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(task_manager_imp.dx, task_manager_imp.x, task_manager_imp.t_step, task_manager_imp.t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_opt_ctrl_krot_.py")
        #prop_reporter.print_all("../test_data/prop_opt_ctrl_krot_.py", "../test_data/fitter_opt_ctrl_krot_.py")

        psi_prop_comparer = TableComparer((np.complex128(0.0001 + 0.0001j), 0.000001, 0.0001), np.float64(1.e-12)) # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                      0.0000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                      np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                      0.0001, 0.0001), np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E

        for n in range(nlevs):
            self.assertTrue(
                psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_opt_ctrl_krot.psi_tabs[n]))
            self.assertTrue(
                tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_opt_ctrl_krot.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_opt_ctrl_krot.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_opt_ctrl_krot.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_opt_ctrl_krot.iter_tab_E))

    def test_opt_ctrl_ut_HB_2lvls_Jz(self):
        user_conf = {
            "task_type": "optimal_control_unit_transform",
            "pot_type": "none",
            "wf_type": "const",
            "hamil_type": "BH_model",
            "lf_aug_type": "z",
            "init_guess": "sqrsin",
            "init_guess_hf": "cos_set",
            "nb": 2,
            "nlevs": 2,
            "T": 3.306555E-13,
            "np": 1,
            "L": 1.0,
            "Du": 1.0,
            "U": 5.0,
            "delta": 25.0,
            "fitter": {
                "epsilon": 1e-8,
                "impulses_number": 1,
                "iter_max": 50,
                "h_lambda": 0.005,
                "hf_hide": False,
                "w_list": [
                    0.88, 0.34, 0.92, 0.27, 0.39, 0.82, 0.68
                ],
                "pcos": 4.0,
                "Em": 5.0,
                "propagation": {
                    "nch": 8,
                    "t0": 0.0,
                    "E0": 40.0,
                    "nt": 3500,
                    "sigma": 6.671270E-13,
                    "nu_L": 0.299793E14
                },
                "mod_log": 500
            }
        }
        mod_fileout = 100
        lmin = 0
        imod_fileout = 1
        imin = -1

        conf, task_manager_imp = self._test_setup(user_conf)
        # number of levels
        nlevs = len(task_manager_imp.psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              task_manager_imp.init_dir, task_manager_imp.ntriv,
                                              task_manager_imp.psi0, task_manager_imp.psif,
                                              task_manager_imp.v, task_manager_imp.akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              task_manager_imp.hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(task_manager_imp.dx, task_manager_imp.x, task_manager_imp.t_step, task_manager_imp.t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_opt_ctrl_ut_HB_2lvls_Jz_.py")
        #prop_reporter.print_all("../test_data/prop_opt_ctrl_ut_HB_2lvls_Jz_.py", "../test_data/fitter_opt_ctrl_ut_HB_2lvls_Jz_.py")

        psi_prop_comparer = TableComparer((np.complex128(0.0001 + 0.0001j), 0.000001, 0.0001), np.float64(1.e-12)) # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                      0.0000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                      np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                      0.0001, 0.0001), np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E

        for n in range(nlevs):
            self.assertTrue(
                psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_opt_ctrl_ut_HB_2lvls_Jz.psi_tabs[n]))
            self.assertTrue(
                tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_opt_ctrl_ut_HB_2lvls_Jz.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_opt_ctrl_ut_HB_2lvls_Jz.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_opt_ctrl_ut_HB_2lvls_Jz.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_opt_ctrl_ut_HB_2lvls_Jz.iter_tab_E))

    def test_opt_ctrl_ut_HB_2lvls_Jx(self):
        user_conf = {
            "task_type": "optimal_control_unit_transform",
            "pot_type": "none",
            "wf_type": "const",
            "hamil_type": "BH_model",
            "lf_aug_type": "x",
            "init_guess": "sqrsin",
            "init_guess_hf": "sin_set",
            "nb": 2,
            "T": 5.4E-13,
            "np": 1,
            "L": 1.0,
            "Du": 1.0,
            "U": 30.0,
            "W": 30.0,
            "delta": 15.0,
            "sigma_auto": True,
            "nu_L_auto": True,
            "fitter": {
                "epsilon": 1e-5,
                "impulses_number": 1,
                "iter_max": 3,
                "iter_mid_1": 250,
                "iter_mid_2": 300,
                "q": 0.75,
                "h_lambda": 0.00005,
                "h_lambda_mode": "dynamical",
                "pcos": 2.0,
                "hf_hide": False,
                "w_list": [1.5, -1.0, 0.7999999999999998],
                "propagation": {
                    "nch": 8,
                    "t0": 0.0,
                    "E0": 40.0,
                    "nt": 3500
                },
                "mod_log": 500
            }
        }
        mod_fileout = 100
        lmin = 0
        imod_fileout = 1
        imin = -1

        conf, task_manager_imp = self._test_setup(user_conf)
        # number of levels
        nlevs = len(task_manager_imp.psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              task_manager_imp.init_dir, task_manager_imp.ntriv,
                                              task_manager_imp.psi0, task_manager_imp.psif,
                                              task_manager_imp.v, task_manager_imp.akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              task_manager_imp.hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(task_manager_imp.dx, task_manager_imp.x, task_manager_imp.t_step, task_manager_imp.t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_opt_ctrl_ut_HB_2lvls_Jx_.py")
        #prop_reporter.print_all("../test_data/prop_opt_ctrl_ut_HB_2lvls_Jx_.py", "../test_data/fitter_opt_ctrl_ut_HB_2lvls_Jx_.py")

        psi_prop_comparer = TableComparer((np.complex128(0.0001 + 0.0001j), 0.000001, 0.0001), np.float64(1.e-12)) # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                             0.0000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                             np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                             0.0001, 0.0001), np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E

        for n in range(nlevs):
            self.assertTrue(
                psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_opt_ctrl_ut_HB_2lvls_Jx.psi_tabs[n]))
            self.assertTrue(
                tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_opt_ctrl_ut_HB_2lvls_Jx.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_opt_ctrl_ut_HB_2lvls_Jx.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_opt_ctrl_ut_HB_2lvls_Jx.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_opt_ctrl_ut_HB_2lvls_Jx.iter_tab_E))

    def test_opt_ctrl_ut_HB_s2s_Jx(self):
        user_conf = {
            "task_type": "optimal_control_unit_transform",
            "pot_type": "none",
            "wf_type": "const",
            "hamil_type": "BH_model",
            "lf_aug_type": "x",
            "init_guess": "gauss",
            "init_guess_hf": "exp",
            "nb": 1,
            "nlevs": 2,
            "T": 2.779700E-13,
            "np": 1,
            "L": 1.0,
            "Du": 1.0,
            "U": 30.0,
            "W": 30.0,
            "delta": 15.0,
            "sigma_auto": True,
            "nu_L_auto": True,
            "t0_auto": True,
            "fitter": {
                "epsilon": 1e-6,
                "impulses_number": 1,
                "iter_max": 50,
                "iter_mid_1": 30,
                "iter_mid_2": 50,
                "q": 0.75,
                "h_lambda": 0.00005,
                "h_lambda_mode": "dynamical",
                "F_type": "sm",
                "hf_hide": False,
                "pcos": 1.0,
                "propagation": {
                    "nch": 8,
                    "E0": 40.0,
                    "nt": 3500
                },
                "mod_log": 500
            }
        }
        mod_fileout = 100
        lmin = 0
        imod_fileout = 1
        imin = -1

        conf, task_manager_imp = self._test_setup(user_conf)
        # number of levels
        nlevs = len(task_manager_imp.psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              task_manager_imp.init_dir, task_manager_imp.ntriv,
                                              task_manager_imp.psi0, task_manager_imp.psif,
                                              task_manager_imp.v, task_manager_imp.akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              task_manager_imp.hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(task_manager_imp.dx, task_manager_imp.x, task_manager_imp.t_step, task_manager_imp.t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_opt_ctrl_ut_HB_s2s_Jx_.py")
        #prop_reporter.print_all("../test_data/prop_opt_ctrl_ut_HB_s2s_Jx_.py", "../test_data/fitter_opt_ctrl_ut_HB_s2s_Jx_.py")

        psi_prop_comparer = TableComparer((np.complex128(0.0001 + 0.0001j), 0.000001, 0.0001), np.float64(1.e-12)) # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                             0.0000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                             np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                             0.0001, 0.0001), np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E

        for n in range(nlevs):
            self.assertTrue(
                psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_opt_ctrl_ut_HB_s2s_Jx.psi_tabs[n]))
            self.assertTrue(
                tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_opt_ctrl_ut_HB_s2s_Jx.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_opt_ctrl_ut_HB_s2s_Jx.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_opt_ctrl_ut_HB_s2s_Jx.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_opt_ctrl_ut_HB_s2s_Jx.iter_tab_E))

    def test_pi_pulse(self):
        user_conf = {
            "task_type": "trans_wo_control",
            "pot_type": "none",
            "wf_type": "const",
            "hamil_type": "BH_model",
            "lf_aug_type": "x",
            "init_guess": "const",
            "init_guess_hf": "exp",
            "nb": 1,
            "nlevs": 2,
            "T": 555.9416E-15,
            "np": 1,
            "L": 1.0,
            "Du": 1.0,
            "U": 30.0,
            "W": 0.0,
            "delta": 15.0,
            "fitter": {
                "epsilon": 1e-5,
                "impulses_number": 1,
                "iter_max": 300,
                "iter_mid_1": 250,
                "iter_mid_2": 300,
                "q": 0.75,
                "h_lambda": 0.00005,
                "h_lambda_mode": "const",
                "F_type": "sm",
                "pcos": 1.0,
                "hf_hide": False,
                "propagation": {
                    "nch": 8,
                    "t0": 0.0,
                    "E0": 1.0,
                    "nt": 512,
                    "nu_L": 0.00179875E15
                },
                "mod_log": 500
            }
        }
        mod_fileout = 100
        lmin = 0
        imod_fileout = 1
        imin = -1
        ntriv = -2
        init_dir = PropagationSolver.Direction.FORWARD

        conf, task_manager_imp, dx, x, t_step, t_list, nlevs, psi0, psif, v, akx2, hamil2D = self._pi_pulse_test_setup(user_conf)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf.fitter, conf.task_type, conf.T, conf.np, conf.L,
                                              init_dir, ntriv, psi0, psif, v, akx2,
                                              task_manager_imp.F_goal,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              task_manager_imp.F_type,
                                              task_manager_imp.aF_type,
                                              hamil2D, fit_reporter_imp,
                                              None, None)

        fitting_solver.time_propagation(dx, x, t_step, t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters[PATH_REP]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("../test_data/fit_iter_pi_pulse_.py")
        #prop_reporter.print_all("../test_data/prop_pi_pulse_.py", "../test_data/fitter_pi_pulse_.py")

        psi_prop_comparer = TableComparer((np.complex128(0.0001 + 0.0001j), 0.000001, 0.0001), np.float64(1.e-12)) # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                             0.0000001, np.complex128(0.001 + 0.001j), # ener, norm,
                                             np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp0, overlpf,
                                             0.0001, 0.0001), np.float64(1.e-12)) # psi_max_abs, psi_max_real

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001, # t, E, freq_mult, ener_tot,
                                            np.complex128(0.001 + 0.001j), np.complex128(0.001 + 0.001j), # overlp_tot[0], overlp_tot[1],
                                            0.001, 0.001, 0.001), np.float64(1.e-12)) # smoms.x, smoms.y, smoms.z
        iter_fit_comparer = TableComparer((0, 0.0001, 0.00001, # iter, goal_close, Fsm,
                                           0.0001, 0.00001), np.float64(1.e-12)) # E_int, J
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), np.float64(1.e-12)) # iter, t, E

        for n in range(nlevs):
            self.assertTrue(
                psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_pi_pulse.psi_tabs[n]))
            self.assertTrue(
                tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_pi_pulse.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_pi_pulse.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_pi_pulse.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_pi_pulse.iter_tab_E))


if __name__ == '__main__':
    unittest.main()
