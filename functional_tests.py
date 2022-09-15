import unittest

import fitter
import grid_setup
import task_manager
import test_data
from config import TaskRootConfiguration
from test_tools import *


class fitter_Tests(unittest.TestCase):
    def test_single_harm(self):
        conf = TaskRootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "single_pot",
            "impulses_number": 0,
            "init_guess": "zero",
            "nb": 1,
            "propagation": {
              "m": 0.5,
              "pot_type": "harmonic",
              "hamil_type": "ntriv",
              "a": 1.0,
              "x0p": 0.0,
              "a_e": 0.0,
              "De_e": 0.0,
              "Du": 0.0,
              "wf_type": "harmonic",
              "x0": 1.0,
              "p0": 0.0,
              "L": 10.0,
              "T": 280e-15,
              "np": 512,
              "nch": 64,
              "nt": 200000,
              "E0": 0.0,
              "nu_L": 0.0
            },
            "mod_log": 500
        }

        conf.load(user_conf)
        print(conf)

        mod_fileout = 1000
        lmin = 0
        imod_fileout = 1
        imin = 0

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # setup of the time grid
        forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_prop=conf.propagation)
        t_step, t_list = forw_time_grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # initial propagation direction
        init_dir = task_manager_imp.init_dir
        # checking of triviality of the system
        ntriv = task_manager_imp.ntriv
        # number of levels
        nlevs = len(psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf, init_dir, ntriv, psi0, psif,
                                              task_manager_imp.pot,
                                              task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf,
                                              fit_reporter_imp,
                                              None, None)
        #fitting_solver.time_propagation(dx, x, t_step, t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters["iter_0f/basis_0"]

        # Uncomment in case of emergency :)
        #fit_reporter_imp.print_all("test_data/fit_iter_single_harm_.py")
        #prop_reporter.print_all("test_data/prop_single_harm_.py", "test_data/fitter_single_harm_.py")

        psi_prop_comparer = TableComparer(epsilon=(complex(0.0001, 0.0001), 0.0001, 0.0001), delta=1.e-21)
        tvals_prop_comparer = TableComparer(epsilon=(0.0001, 0.02, 0.001, 0.0001, 0.00001, 0.000001,
                                             complex(0.001, 0.001), complex(0.001, 0.001),
                                             0.0001, 0.0001), delta=1.e-21)

        tvals_fit_comparer = TableComparer((0.0001, 0.0001, 0.0001, 0.000001, complex(0.001, 0.001), complex(0.001, 0.001)), 1.e-21)
        iter_fit_comparer = TableComparer((0, 0.0001, complex(0.00001, 0.00001)), 1.e-21)
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), 1.e-21)

        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_single_harm.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_single_harm.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_single_harm.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_single_harm.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_single_harm.iter_tab_E))

    def test_single_morse(self):
        conf = TaskRootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "single_pot",
            "impulses_number": 0,
            "init_guess": "zero",
            "nb": 1,
            "propagation": {
              "m": 0.5,
              "pot_type": "morse",
              "hamil_type": "ntriv",
              "a": 1.0,
              "De": 20000,
              "x0p": 0.0,
              "a_e": 0.0,
              "De_e": 0.0,
              "Du": 0.0,
              "wf_type": "morse",
              "x0": 0.0,
              "p0": 0.0,
              "L": 4.0,
              "T": 280e-16,
              "np": 2048,
              "nch": 64,
              "nt": 20000,
              "E0": 0.0,
              "nu_L": 0.0
            },
            "mod_log": 500
        }

        conf.load(user_conf)
        print(conf)

        mod_fileout = 1000
        lmin = 0
        imod_fileout = 1
        imin = 0

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # setup of the time grid
        forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_prop=conf.propagation)
        t_step, t_list = forw_time_grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # initial propagation direction
        init_dir = task_manager_imp.init_dir
        # checking of triviality of the system
        ntriv = task_manager_imp.ntriv
        # number of levels
        nlevs = len(psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf, init_dir, ntriv, psi0, psif,
                                              task_manager_imp.pot, task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf, fit_reporter_imp,
                                              None, None)
        fitting_solver.time_propagation(dx, x, t_step, t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters["iter_0f/basis_0"]

        # Uncomment in case of emergency :)
        fit_reporter_imp.print_all("test_data/fit_iter_single_morse_.py")
        prop_reporter.print_all("test_data/prop_single_morse_.py", "test_data/fitter_single_morse_.py")

        psi_prop_comparer = TableComparer(epsilon=(complex(0.0001, 0.0001), 0.0001, 0.0001), delta=1.e-21)
        tvals_prop_comparer = TableComparer(epsilon=(0.0001, 0.001, 0.001, 0.001, 0.000001, 0.0000001,
                                             complex(0.001, 0.001), complex(0.001, 0.001),
                                             0.0001, 0.0001), delta=1.e-21)

        tvals_fit_comparer = TableComparer((0.0001, 0.0001, 0.0001, 0.0000001,
                                            complex(0.001, 0.001), complex(0.001, 0.001)), 1.e-21)
        iter_fit_comparer = TableComparer((0, 0.0001, complex(0.00001, 0.00001)), 1.e-21)
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), 1.e-21)

        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_single_morse.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_single_morse.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_single_morse.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_single_morse.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_single_morse.iter_tab_E))

    def test_filter(self):
        conf = TaskRootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "filtering",
            "impulses_number": 0,
            "init_guess": "zero",
            "nb": 1,
            "propagation": {
                "m": 0.5,
                "pot_type": "morse",
                "hamil_type": "ntriv",
                "a": 1.0,
                "De": 20000,
                "x0p": 0.0,
                "a_e": 0.0,
                "De_e": 0.0,
                "Du": 0.0,
                "wf_type": "harmonic",
                "L": 5.0,
                "T": 1980e-15,
                "np": 2048,
                "nch": 64,
                "nt": 800000,
                "E0": 0.0,
                "nu_L": 0.0
            },
            "mod_log": 1000
        }
        conf.load(user_conf)
        print(conf)

        mod_fileout = 20000
        lmin = 0
        imod_fileout = 1
        imin = 0

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # setup of the time grid
        forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_prop=conf.propagation)
        t_step, t_list = forw_time_grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # initial propagation direction
        init_dir = task_manager_imp.init_dir
        # checking of triviality of the system
        ntriv = task_manager_imp.ntriv
        # number of levels
        nlevs = len(psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf, init_dir, ntriv, psi0, psif,
                                              task_manager_imp.pot, task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf, fit_reporter_imp,
                                              None, None)
        fitting_solver.time_propagation(dx, x, t_step, t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters["iter_0f/basis_0"]

        # Uncomment in case of emergency :)
        fit_reporter_imp.print_all("test_data/fit_iter_filter_.py")
        prop_reporter.print_all("test_data/prop_filter_.py", "test_data/fitter_filter_.py")

        psi_prop_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-21)
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.0001, 0.000001,
                                      0.0000001, complex(0.001, 0.001), complex(0.001, 0.001),
                                      0.0001, 0.0001), 1.e-21)

        tvals_fit_comparer = TableComparer((0.000001, 0.0001, 0.0001, 0.0000001,
                                            complex(0.001, 0.001), complex(0.001, 0.001)), 1.e-21)
        iter_fit_comparer = TableComparer((0, 0.0001, complex(0.00001, 0.00001)), 1.e-21)
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), 1.e-21)


        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_filter.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_filter.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_filter.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_filter.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_filter.iter_tab_E))

    def test_trans_woc(self):
        conf = TaskRootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "trans_wo_control",
            "impulses_number": 1,
            "init_guess": "gauss",
            "init_guess_hf": "exp",
            "nb": 1,
            "propagation": {
                "m": 0.5,
                "pot_type": "morse",
                "hamil_type": "ntriv",
                "a": 1.0,
                "De": 20000,
                "x0p": -0.17,
                "a_e": 1.0,
                "De_e": 10000,
                "Du": 20000,
                "wf_type": "morse",
                "L": 5.0,
                "T": 330e-15,
                "np": 1024,
                "nch": 64,
                "nt": 230000,
                "E0": 71.54,
                "t0": 200e-15,
                "sigma": 50e-15,
                "nu_L": 0.29297e15
            },
            "mod_log": 500
        }

        conf.load(user_conf)
        print(conf)

        mod_fileout = 10000
        lmin = 0
        imod_fileout = 1
        imin = 0

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # setup of the time grid
        forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_prop=conf.propagation)
        t_step, t_list = forw_time_grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # initial propagation direction
        init_dir = task_manager_imp.init_dir
        # checking of triviality of the system
        ntriv = task_manager_imp.ntriv
        # number of levels
        nlevs = len(psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf, init_dir, ntriv, psi0, psif,
                                              task_manager_imp.pot, task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf, fit_reporter_imp,
                                              None, None)
        fitting_solver.time_propagation(dx, x, t_step, t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters["iter_0f/basis_0"]

        # Uncomment in case of emergency :)
        fit_reporter_imp.print_all("test_data/fit_iter_trans_woc_.py")
        prop_reporter.print_all("test_data/prop_trans_woc_.py", "test_data/fitter_trans_woc_.py")

        psi_prop_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-21)
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, complex(0.001, 0.001), complex(0.001, 0.001),
                                      0.0001, 0.0001), 1.e-21)

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001,
                                            complex(0.001, 0.001), complex(0.001, 0.001)), 1.e-21)
        iter_fit_comparer = TableComparer((0, 0.0001, complex(0.00001, 0.00001)), 1.e-21)
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), 1.e-21)

        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_trans_woc.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_trans_woc.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_trans_woc.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_trans_woc.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_trans_woc.iter_tab_E))

    def test_int_ctrl(self):
        conf = TaskRootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "intuitive_control",
            "impulses_number": 2,
            "delay": 300e-15,
            "init_guess": "gauss",
            "init_guess_hf": "exp",
            "nb": 1,
            "propagation": {
                "m": 0.5,
                "pot_type": "morse",
                "hamil_type": "ntriv",
                "a": 1.0,
                "De": 20000,
                "x0p": -0.17,
                "a_e": 1.0,
                "De_e": 10000,
                "Du": 20000,
                "wf_type": "morse",
                "L": 5.0,
                "T": 700e-15,
                "np": 1024,
                "nch": 64,
                "nt": 490000,
                "E0": 71.54,
                "t0": 200e-15,
                "sigma": 50e-15,
                "nu_L": 0.29297e15
            },
            "mod_log": 500
        }

        conf.load(user_conf)
        print(conf)

        mod_fileout = 20000
        lmin = 0
        imod_fileout = 1
        imin = 0

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # setup of the time grid
        forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_prop=conf.propagation)
        t_step, t_list = forw_time_grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # initial propagation direction
        init_dir = task_manager_imp.init_dir
        # checking of triviality of the system
        ntriv = task_manager_imp.ntriv
        # number of levels
        nlevs = len(psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf, init_dir, ntriv, psi0, psif,
                                              task_manager_imp.pot, task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf, fit_reporter_imp,
                                              None, None)
        fitting_solver.time_propagation(dx, x, t_step, t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters["iter_0f/basis_0"]

        # Uncomment in case of emergency :)
        fit_reporter_imp.print_all("test_data/fit_iter_int_ctrl_.py")
        prop_reporter.print_all("test_data/prop_int_ctrl_.py", "test_data/fitter_int_ctrl_.py")

        psi_prop_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-21)
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, complex(0.001, 0.001), complex(0.001, 0.001),
                                      0.0001, 0.0001), 1.e-21)

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001,
                                            complex(0.001, 0.001), complex(0.001, 0.001)), 1.e-21)
        iter_fit_comparer = TableComparer((0, 0.0001, complex(0.00001, 0.00001)), 1.e-21)
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), 1.e-21)

        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_int_ctrl.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_int_ctrl.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_int_ctrl.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_int_ctrl.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_int_ctrl.iter_tab_E))

    def test_loc_ctrl_pop(self):
        conf = TaskRootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "local_control_population",
            "k_E": 1e29,
            "lamb": 4e14,
            "pow": 0.8,
            "epsilon": 1e-15,
            "impulses_number": 1,
            "init_guess": "gauss",
            "init_guess_hf": "exp",
            "nb": 1,
            "propagation": {
                "m": 0.5,
                "pot_type": "morse",
                "hamil_type": "ntriv",
                "a": 1.0,
                "De": 20000,
                "x0p": -0.17,
                "a_e": 1.0,
                "De_e": 10000,
                "Du": 20000,
                "wf_type": "morse",
                "L": 5.0,
                "T": 400e-15,
                "np": 1024,
                "nch": 64,
                "nt": 280000,
                "E0": 71.54,
                "t0": 200e-15,
                "sigma": 50e-15,
                "nu_L": 0.29297e15
            },
            "mod_log": 500
        }

        conf.load(user_conf)
        print(conf)

        mod_fileout = 10000
        lmin = 0
        imod_fileout = 1
        imin = 0

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # setup of the time grid
        forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_prop=conf.propagation)
        t_step, t_list = forw_time_grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # initial propagation direction
        init_dir = task_manager_imp.init_dir
        # checking of triviality of the system
        ntriv = task_manager_imp.ntriv
        # number of levels
        nlevs = len(psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf, init_dir, ntriv, psi0, psif,
                                              task_manager_imp.pot, task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf, fit_reporter_imp,
                                              None, None)
        fitting_solver.time_propagation(dx, x, t_step, t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters["iter_0f/basis_0"]

        # Uncomment in case of emergency :)
        fit_reporter_imp.print_all("test_data/fit_iter_loc_ctrl_pop_.py")
        prop_reporter.print_all("test_data/prop_loc_ctrl_pop_.py", "test_data/fitter_loc_ctrl_pop_.py")

        psi_prop_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-21)
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, complex(0.001, 0.001), complex(0.001, 0.001),
                                      0.0001, 0.0001), 1.e-21)

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001,
                                            complex(0.001, 0.001), complex(0.001, 0.001)), 1.e-21)
        iter_fit_comparer = TableComparer((0, 0.0001, complex(0.00001, 0.00001)), 1.e-21)
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), 1.e-21)

        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_loc_ctrl_pop.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_loc_ctrl_pop.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_loc_ctrl_pop.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_loc_ctrl_pop.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_loc_ctrl_pop.iter_tab_E))

    def test_loc_ctrl_proj(self):
        conf = TaskRootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "local_control_projection",
            "k_E": 1e29,
            "lamb": 8e14,
            "pow": 0.65,
            "epsilon": 1e-15,
            "impulses_number": 1,
            "init_guess": "gauss",
            "init_guess_hf": "exp",
            "nb": 1,
            "propagation": {
                "m": 0.5,
                "pot_type": "morse",
                "hamil_type": "ntriv",
                "a": 1.0,
                "De": 20000,
                "x0p": -0.17,
                "a_e": 1.0,
                "De_e": 10000,
                "Du": 20000,
                "wf_type": "morse",
                "L": 5.0,
                "T": 450e-15,
                "np": 1024,
                "nch": 64,
                "nt": 315000,
                "E0": 71.54,
                "t0": 200e-15,
                "sigma": 50e-15,
                "nu_L": 0.29297e15
            },
            "mod_log": 500
        }

        conf.load(user_conf)
        print(conf)

        mod_fileout = 10000
        lmin = 0
        imod_fileout = 1
        imin = 0

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # setup of the time grid
        forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_prop=conf.propagation)
        t_step, t_list = forw_time_grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # initial propagation direction
        init_dir = task_manager_imp.init_dir
        # checking of triviality of the system
        ntriv = task_manager_imp.ntriv
        # number of levels
        nlevs = len(psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf, init_dir, ntriv, psi0, psif,
                                              task_manager_imp.pot, task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf, fit_reporter_imp,
                                              None, None)
        fitting_solver.time_propagation(dx, x, t_step, t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters["iter_0f/basis_0"]

        # Uncomment in case of emergency :)
        fit_reporter_imp.print_all("test_data/fit_iter_loc_ctrl_proj_.py")
        prop_reporter.print_all("test_data/prop_loc_ctrl_proj_.py", "test_data/fitter_loc_ctrl_proj_.py")

        psi_prop_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-21)
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, complex(0.001, 0.001), complex(0.001, 0.001),
                                      0.0001, 0.0001), 1.e-21)

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001,
                                            complex(0.001, 0.001), complex(0.001, 0.001)), 1.e-21)
        iter_fit_comparer = TableComparer((0, 0.0001, complex(0.00001, 0.00001)), 1.e-21)
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), 1.e-21)

        for n in range(nlevs):
            self.assertTrue(
                psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_loc_ctrl_proj.psi_tabs[n]))
            self.assertTrue(
                tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_loc_ctrl_proj.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_loc_ctrl_proj.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_loc_ctrl_proj.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_loc_ctrl_proj.iter_tab_E))

    def test_opt_ctrl_krot(self):
        conf = TaskRootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "optimal_control_krotov",
            "epsilon": 1e-8,
            "impulses_number": 1,
            "iter_max": 1,
            "h_lambda": 0.0066,
            "init_guess": "gauss",
            "init_guess_hf": "exp",
            "nb": 1,
            "propagation": {
                "m": 0.5,
                "pot_type": "morse",
                "hamil_type": "ntriv",
                "a": 1.0,
                "De": 20000,
                "x0p": -0.17,
                "a_e": 1.0,
                "De_e": 10000,
                "Du": 20000,
                "wf_type": "morse",
                "L": 5.0,
                "T": 350e-15,
                "np": 1024,
                "nch": 64,
                "nt": 230000,
                "E0": 71.54,
                "t0": 200e-15,
                "sigma": 50e-15,
                "nu_L": 0.29297e15
            },
            "mod_log": 500
        }

        conf.load(user_conf)
        print(conf)

        mod_fileout = 10000
        lmin = 0
        imod_fileout = 1
        imin = 0

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # setup of the time grid
        forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_prop=conf.propagation)
        t_step, t_list = forw_time_grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # initial propagation direction
        init_dir = task_manager_imp.init_dir
        # checking of triviality of the system
        ntriv = task_manager_imp.ntriv
        # number of levels
        nlevs = len(psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf, init_dir, ntriv, psi0, psif,
                                              task_manager_imp.pot, task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf, fit_reporter_imp,
                                              None, None)
        fitting_solver.time_propagation(dx, x, t_step, t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters["iter_0f/basis_0"]

        # Uncomment in case of emergency :)
        fit_reporter_imp.print_all("test_data/fit_iter_opt_ctrl_krot_.py")
        prop_reporter.print_all("test_data/prop_opt_ctrl_krot_.py", "test_data/fitter_opt_ctrl_krot_.py")

        psi_prop_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-21)
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, complex(0.001, 0.001), complex(0.001, 0.001),
                                      0.0001, 0.0001), 1.e-21)

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001,
                                            complex(0.001, 0.001), complex(0.001, 0.001)), 1.e-21)
        iter_fit_comparer = TableComparer((0, 0.0001, complex(0.00001, 0.00001)), 1.e-21)
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), 1.e-21)

        for n in range(nlevs):
            self.assertTrue(
                psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_opt_ctrl_krot.psi_tabs[n]))
            self.assertTrue(
                tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_opt_ctrl_krot.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_opt_ctrl_krot.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_opt_ctrl_krot.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_opt_ctrl_krot.iter_tab_E))

    def test_opt_ctrl_ut_H1(self):
        conf = TaskRootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "optimal_control_unit_transform",
            "epsilon": 1e-8,
            "impulses_number": 1,
            "iter_max": 7,
            "h_lambda": 0.5,
            "init_guess": "sqrsin",
            "init_guess_hf": "cos",
            "pcos": 0.58,
            "nb": 2,
            "propagation": {
                "pot_type": "none",
                "wf_type": "const",
                "hamil_type": "two_levels",
                "np": 1,
                "L": 1.0,
                "nch": 64,
                "Du": 100,
                "t0": 0.0,
                "E0": 40.0,
                "nt": 1000,
                "T": 33.35635E-13,
                "sigma": 66.7127E-13,
                "nu_L": 0.299792458e13
            },
            "mod_log": 500
        }

        conf.load(user_conf)
        print(conf)

        mod_fileout = 100
        lmin = 0
        imod_fileout = 1
        imin = -1

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # setup of the time grid
        forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_prop=conf.propagation)
        t_step, t_list = forw_time_grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e,
                                         conf.propagation.L, conf.nb)

        # initial propagation direction
        init_dir = task_manager_imp.init_dir
        # checking of triviality of the system
        ntriv = task_manager_imp.ntriv
        # number of levels
        nlevs = len(psi0.psis[0].f)

        fit_reporter_imp = TestFitterReporter(mod_fileout, lmin, imod_fileout, imin)
        fit_reporter_imp.open()

        fitting_solver = fitter.FittingSolver(conf, init_dir, ntriv, psi0, psif,
                                              task_manager_imp.pot, task_manager_imp.laser_field,
                                              task_manager_imp.laser_field_hf, fit_reporter_imp,
                                              None, None)
        fitting_solver.time_propagation(dx, x, t_step, t_list)
        fit_reporter_imp.close()

        prop_reporter = fit_reporter_imp.prop_reporters["iter_0f/basis_0"]

        # Uncomment in case of emergency :)
        fit_reporter_imp.print_all("test_data/fit_iter_opt_ctrl_ut_H1_.py")
        prop_reporter.print_all("test_data/prop_opt_ctrl_ut_H1_.py", "test_data/fitter_opt_ctrl_ut_H1_.py")

        psi_prop_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-21)
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, complex(0.001, 0.001), complex(0.001, 0.001),
                                      0.0001, 0.0001), 1.e-21)

        tvals_fit_comparer = TableComparer((0.000001, 0.00001, 0.0001, 0.0000001,
                                            complex(0.001, 0.001), complex(0.001, 0.001)), 1.e-21)
        iter_fit_comparer = TableComparer((0, 0.0001, complex(0.00001, 0.00001)), 1.e-21)
        iter_fit_E_comparer = TableComparer((0, 0.0001, 0.0001), 1.e-21)

        for n in range(nlevs):
            self.assertTrue(
                psi_prop_comparer.compare(prop_reporter.psi_tab[n], test_data.prop_opt_ctrl_ut_H1.psi_tabs[n]))
            self.assertTrue(
                tvals_prop_comparer.compare(prop_reporter.prop_tab[n], test_data.prop_opt_ctrl_ut_H1.prop_tabs[n]))

        self.assertTrue(tvals_fit_comparer.compare(prop_reporter.fit_tab, test_data.fitter_opt_ctrl_ut_H1.tvals_tab))
        self.assertTrue(iter_fit_comparer.compare(fit_reporter_imp.iter_tab, test_data.fit_iter_opt_ctrl_ut_H1.iter_tab))
        self.assertTrue(iter_fit_E_comparer.compare(fit_reporter_imp.iter_tab_E, test_data.fit_iter_opt_ctrl_ut_H1.iter_tab_E))


if __name__ == '__main__':
    unittest.main()
