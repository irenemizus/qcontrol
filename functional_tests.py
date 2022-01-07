import unittest

import fitter
import grid_setup
import task_manager
import test_data
from config import RootConfiguration
from test_tools import *

class test_facilities_Tests(unittest.TestCase):
    def test_table_comparer(self):
        cmp = TableComparer((3.5, 0.2, np.complex(5.0, 1.0)))
        tab1 = [
            (3.01, 5.0, np.array([np.complex(1.0, 1.0), np.complex(2.01, 2.001)]))
        ]
        tab2 = [
            (3.0, 5.0, np.array([np.complex(1.0, 1.0), np.complex(2.0, 2.0)]))
        ]

        self.assertTrue(cmp.compare(tab1, tab2))


class fitter_Tests(unittest.TestCase):
    def test_single_harm(self):
        conf = RootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "single_pot",
            "k_E": 1e29,
            "lamb": 4e14,
            "pow": 0.8,
            "epsilon": 1e-15,
            "impulses_number": 0,
            "delay": 600e-15,
            "propagation": {
                "m": 0.5,
                "pot_type": "harmonic",
                "a": 1.0,
                "De": 20000,
                "x0p": 0.0,
                "a_e": 0.0,
                "De_e": 0.0,
                "Du": 0.0,
                "wf_type": "harmonic",
                "x0": 1.0,
                "p0": 0.0,
                "L": 10.0,
                "T": 280e-16,
                "np": 512,
                "nch": 64,
                "nt": 20000,
                "E0": 0.0,
                "t0": 300e-15,
                "sigma": 50e-15,
                "nu_L": 0.0
            },
            "mod_log": 500
        }

        conf.load(user_conf)
        print(conf)

        mod_fileout = 1000
        lmin = 0

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.m,
                                         conf.propagation.De, conf.propagation.a)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e)

        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, psi0, psif, task_manager_imp,
                                                  reporter_impl, None, None)
            fitting_solver.time_propagation(dx, x)

        # Uncomment in case of emergency :)
        #reporter_impl.print_all("test_data/fitter_single_harm.py")

        psi_comparer = TableComparer((complex(0.0001, 0.0001), 0.0001, 0.0001))
        mom_comparer = TableComparer((0.0001, 0.02, 0.001, 0.0001, 0.00001,
                                      0.000001, 0.0001, 0.0001, complex(0.001, 0.001), 0.000001,
                                      0.0001, 0.0001))
        mom_up_comparer = TableComparer((0.0001, 0.02, 0.001, 0.0001, 0.00001,
                                      0.000001, 0.0001, 0.0001, complex(0.001, 0.001), 0.001,
                                      0.0001, 0.0001))

        self.assertTrue(psi_comparer.compare(reporter_impl.psi_tab, test_data.fitter_single_harm.psi_tab))
        self.assertTrue(psi_comparer.compare(reporter_impl.psi_up_tab, test_data.fitter_single_harm.psi_up_tab))

        self.assertTrue(mom_comparer.compare(reporter_impl.mom_tab, test_data.fitter_single_harm.mom_tab))
        self.assertTrue(mom_up_comparer.compare(reporter_impl.mom_up_tab, test_data.fitter_single_harm.mom_up_tab))


    def test_single_morse(self):
        conf = RootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "single_pot",
            "k_E": 1e29,
            "lamb": 4e14,
            "pow": 0.8,
            "epsilon": 1e-15,
            "impulses_number": 0,
            "delay": 600e-15,
            "propagation": {
                "m": 0.5,
                "pot_type": "morse",
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
                "t0": 300e-15,
                "sigma": 50e-15,
                "nu_L": 0.0
            },
            "mod_log": 500
        }

        conf.load(user_conf)
        print(conf)

        mod_fileout = 1000
        lmin = 0

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.m,
                                         conf.propagation.De, conf.propagation.a)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e)

        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, psi0, psif, task_manager_imp,
                                                  reporter_impl, None, None)
            fitting_solver.time_propagation(dx, x)

        # Uncomment in case of emergency :)
        # reporter_impl.print_all("test_data/fitter_single_morse.py")

        psi_comparer = TableComparer((complex(0.0001, 0.0001), 0.0001, 0.0001))
        mom_comparer = TableComparer((0.0001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, 0.0001, 0.0001, complex(0.001, 0.001), 0.0000001,
                                      0.0001, 0.0001))
        mom_up_comparer = TableComparer((0.0001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, 0.0001, 0.0001, complex(0.001, 0.001), 0.001,
                                      0.0001, 0.0001))

        self.assertTrue(psi_comparer.compare(reporter_impl.psi_tab, test_data.fitter_single_morse.psi_tab))
        self.assertTrue(psi_comparer.compare(reporter_impl.psi_up_tab, test_data.fitter_single_morse.psi_up_tab))

        self.assertTrue(mom_comparer.compare(reporter_impl.mom_tab, test_data.fitter_single_morse.mom_tab))
        self.assertTrue(mom_up_comparer.compare(reporter_impl.mom_up_tab, test_data.fitter_single_morse.mom_up_tab))


    def test_filter(self):
        conf = RootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "filtering",
            "k_E": 1e29,
            "lamb": 8e14,
            "pow": 0.65,
            "epsilon": 1e-15,
            "impulses_number": 0,
            "delay": 600e-15,
            "propagation": {
              "m": 0.5,
              "pot_type": "morse",
              "a": 1.0,
              "De": 20000,
              "x0p": 0.0,
              "a_e": 0.0,
              "De_e": 0.0,
              "Du": 0.0,
              "wf_type": "harmonic",
              "x0": 0.0,
              "p0": 0.0,
              "L": 5.0,
              "T": 1980e-15,
              "np": 2048,
              "nch": 64,
              "nt": 800000,
              "E0": 0.0,
              "t0": 300e-15,
              "sigma": 50e-15,
              "nu_L": 0.0
            },
            "mod_log": 1000
        }

        conf.load(user_conf)
        print(conf)

        mod_fileout = 20000
        lmin = 0

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.m,
                                         conf.propagation.De, conf.propagation.a)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e)

        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, psi0, psif, task_manager_imp,
                                                  reporter_impl, None, None)
            fitting_solver.time_propagation(dx, x)

        # Uncomment in case of emergency :)
        # reporter_impl.print_all("test_data/fitter_filter.py")

        psi_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001))
        mom_comparer = TableComparer((0.000001, 0.001, 0.001, 0.0001, 0.000001,
                                      0.0000001, 0.0001, 0.0001, complex(0.001, 0.001), 0.0000001,
                                      0.0001, 0.0001))
        mom_up_comparer = TableComparer((0.000001, 0.001, 0.001, 0.0001, 0.000001,
                                      0.0000001, 0.0001, 0.0001, complex(0.001, 0.001), 0.001,
                                      0.0001, 0.0001))

        self.assertTrue(psi_comparer.compare(reporter_impl.psi_tab, test_data.fitter_filter.psi_tab))
        self.assertTrue(psi_comparer.compare(reporter_impl.psi_up_tab, test_data.fitter_filter.psi_up_tab))

        self.assertTrue(mom_comparer.compare(reporter_impl.mom_tab, test_data.fitter_filter.mom_tab))
        self.assertTrue(mom_up_comparer.compare(reporter_impl.mom_up_tab, test_data.fitter_filter.mom_up_tab))


    def test_trans_woc(self):
        conf = RootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "trans_wo_control",
            "k_E": 1e29,
            "lamb": 4e14,
            "pow": 0.8,
            "epsilon": 1e-15,
            "impulses_number": 1,
            "delay": 600e-15,
            "propagation": {
                "m": 0.5,
                "pot_type": "morse",
                "a": 1.0,
                "De": 20000,
                "x0p": -0.17,
                "a_e": 1.0,
                "De_e": 10000,
                "Du": 20000,
                "wf_type": "morse",
                "x0": 0.0,
                "p0": 0.0,
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

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.m,
                                         conf.propagation.De, conf.propagation.a)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e)

        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, psi0, psif, task_manager_imp,
                                                  reporter_impl, None, None)
            fitting_solver.time_propagation(dx, x)

        # Uncomment in case of emergency :)
        # reporter_impl.print_all("test_data/fitter_trans_woc.py")

        psi_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001))
        mom_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, 0.00001, 0.0001, complex(0.001, 0.001), 0.0000001,
                                      0.0001, 0.0001))
        mom_up_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, 0.00001, 0.0001, complex(0.001, 0.001), 0.001,
                                      0.0001, 0.0001))

        self.assertTrue(psi_comparer.compare(reporter_impl.psi_tab, test_data.fitter_trans_woc.psi_tab))
        self.assertTrue(psi_comparer.compare(reporter_impl.psi_up_tab, test_data.fitter_trans_woc.psi_up_tab))

        self.assertTrue(mom_comparer.compare(reporter_impl.mom_tab, test_data.fitter_trans_woc.mom_tab))
        self.assertTrue(mom_up_comparer.compare(reporter_impl.mom_up_tab, test_data.fitter_trans_woc.mom_up_tab))


    def test_int_ctrl(self):
        conf = RootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "intuitive_control",
            "k_E": 1e29,
            "lamb": 4e14,
            "pow": 0.8,
            "epsilon": 1e-15,
            "impulses_number": 2,
            "delay": 300e-15,
            "propagation": {
                "m": 0.5,
                "pot_type": "morse",
                "a": 1.0,
                "De": 20000,
                "x0p": -0.17,
                "a_e": 1.0,
                "De_e": 10000,
                "Du": 20000,
                "wf_type": "morse",
                "x0": 0.0,
                "p0": 0.0,
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

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.m,
                                         conf.propagation.De, conf.propagation.a)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e)

        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, psi0, psif, task_manager_imp,
                                                  reporter_impl, None, None)
            fitting_solver.time_propagation(dx, x)

        # Uncomment in case of emergency :)
        # reporter_impl.print_all("test_data/fitter_int_ctrl.py")

        psi_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001))
        mom_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, 0.00001, 0.0001, complex(0.001, 0.001), 0.0000001,
                                      0.0001, 0.0001))
        mom_up_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, 0.00001, 0.0001, complex(0.001, 0.001), 0.001,
                                      0.0001, 0.0001))

        self.assertTrue(psi_comparer.compare(reporter_impl.psi_tab, test_data.fitter_int_ctrl.psi_tab))
        self.assertTrue(psi_comparer.compare(reporter_impl.psi_up_tab, test_data.fitter_int_ctrl.psi_up_tab))

        self.assertTrue(mom_comparer.compare(reporter_impl.mom_tab, test_data.fitter_int_ctrl.mom_tab))
        self.assertTrue(mom_up_comparer.compare(reporter_impl.mom_up_tab, test_data.fitter_int_ctrl.mom_up_tab))


    def test_loc_ctrl_pop(self):
        conf = RootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "local_control_population",
            "k_E": 1e29,
            "lamb": 4e14,
            "pow": 0.8,
            "epsilon": 1e-15,
            "impulses_number": 1,
            "delay": 600e-15,
            "propagation": {
                "m": 0.5,
                "pot_type": "morse",
                "a": 1.0,
                "De": 20000,
                "x0p": -0.17,
                "a_e": 1.0,
                "De_e": 10000,
                "Du": 20000,
                "wf_type": "morse",
                "x0": 0.0,
                "p0": 0.0,
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

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.m,
                                         conf.propagation.De, conf.propagation.a)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e)

        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, psi0, psif, task_manager_imp,
                                                  reporter_impl, None, None)
            fitting_solver.time_propagation(dx, x)

        # Uncomment in case of emergency :)
        # reporter_impl.print_all("test_data/fitter_loc_ctrl_pop.py")

        psi_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001))
        mom_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, 0.00001, 0.0001, complex(0.001, 0.001), 0.0000001,
                                      0.0001, 0.0001))
        mom_up_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, 0.00001, 0.0001, complex(0.001, 0.001), 0.001,
                                      0.0001, 0.0001))

        self.assertTrue(psi_comparer.compare(reporter_impl.psi_tab, test_data.fitter_loc_ctrl_pop.psi_tab))
        self.assertTrue(psi_comparer.compare(reporter_impl.psi_up_tab, test_data.fitter_loc_ctrl_pop.psi_up_tab))

        self.assertTrue(mom_comparer.compare(reporter_impl.mom_tab, test_data.fitter_loc_ctrl_pop.mom_tab))
        self.assertTrue(mom_up_comparer.compare(reporter_impl.mom_up_tab, test_data.fitter_loc_ctrl_pop.mom_up_tab))


    def test_loc_ctrl_proj(self):
        conf = RootConfiguration.FitterConfiguration()

        user_conf = {
            "task_type": "local_control_projection",
            "k_E": 1e29,
            "lamb": 8e14,
            "pow": 0.65,
            "epsilon": 1e-15,
            "impulses_number": 1,
            "delay": 600e-15,
            "propagation": {
                "m": 0.5,
                "pot_type": "morse",
                "a": 1.0,
                "De": 20000,
                "x0p": -0.17,
                "a_e": 1.0,
                "De_e": 10000,
                "Du": 20000,
                "wf_type": "morse",
                "x0": 0.0,
                "p0": 0.0,
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

        task_manager_imp = task_manager.create(conf)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = task_manager_imp.psi_init(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.m,
                                         conf.propagation.De, conf.propagation.a)

        # evaluating of the final goal
        psif = task_manager_imp.psi_goal(x, conf.propagation.np, conf.propagation.x0,
                                         conf.propagation.p0, conf.propagation.x0p,
                                         conf.propagation.m, conf.propagation.De,
                                         conf.propagation.De_e, conf.propagation.Du,
                                         conf.propagation.a, conf.propagation.a_e)

        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, psi0, psif, task_manager_imp,
                                                  reporter_impl, None, None)
            fitting_solver.time_propagation(dx, x)

        # Uncomment in case of emergency :)
        # reporter_impl.print_all("test_data/fitter_loc_ctrl_proj.py")

        psi_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001))
        mom_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, 0.00001, 0.0001, complex(0.001, 0.001), 0.0000001,
                                      0.0001, 0.0001))
        mom_up_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, 0.00001, 0.0001, complex(0.001, 0.001), 0.001,
                                      0.0001, 0.0001))

        self.assertTrue(psi_comparer.compare(reporter_impl.psi_tab, test_data.fitter_loc_ctrl_proj.psi_tab))
        self.assertTrue(psi_comparer.compare(reporter_impl.psi_up_tab, test_data.fitter_loc_ctrl_proj.psi_up_tab))

        self.assertTrue(mom_comparer.compare(reporter_impl.mom_tab, test_data.fitter_loc_ctrl_proj.mom_tab))
        self.assertTrue(mom_up_comparer.compare(reporter_impl.mom_up_tab, test_data.fitter_loc_ctrl_proj.mom_up_tab))


if __name__ == '__main__':
    unittest.main()
