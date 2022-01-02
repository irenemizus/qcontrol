import sys
import unittest

import fitter
import task_manager
import test_data
from config import RootConfiguration
from reporter import Reporter
import numpy as np


class TableComparer:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def compare(self, tab1, tab2):
        # Comparing tables length
        if len(tab1) != len(tab2):
            return False

        # For each line
        for i in range(len(tab1)):
            # Comparing the length of the tab1 line with the one of tab2 line
            if len(tab1[i]) != len(tab2[i]):
                return False

            if len(tab1[i]) != len(self.epsilon):
                return False

            for k in range(len(tab1[i])):
                el1 = tab1[i][k]
                el2 = tab2[i][k]
                eps = self.epsilon[k]

                if isinstance(el1, float) and isinstance(el2, float) and isinstance(eps, float):
                    # Trivially comparing two floats
                    if el2 == 0.0 and el1 == 0.0:
                        return True
                    elif el2 == 0.0 and el1 != 0.0:
                        return False
                    elif abs(el1 - el2) / abs(el2) * 100.0 >= abs(eps):
                        return False
                elif isinstance(el1, np.ndarray) and \
                     isinstance(el2, np.ndarray) and \
                     isinstance(eps, np.complex):
                    # Comparing each element of two complex arrays
                    if len(el1) != len(el2):
                        raise RuntimeError("Complex arrays have different lengths")

                    for l in range(len(el1)):
                        if el2[l].real == 0.0 and el1[l].real == 0.0 and \
                           el2[l].imag == 0.0 and el1[l].imag == 0.0:
                            return True
                        elif el2[l].real == 0.0 and el1[l].real != 0.0:
                            return False
                        elif el2[l].imag == 0.0 and el1[l].imag != 0.0:
                            return False
                        elif abs(el1[l].real - el2[l].real) / abs(el2[l].real) * 100.0 >= eps.real or \
                           abs(el1[l].imag - el2[l].imag) / abs(el2[l].imag) * 100.0 >= eps.imag:
                            return False
        return True

class TestReporter(Reporter):
    def __init__(self, mod_fileout, lmin):
        self.mod_fileout = mod_fileout
        self.lmin = lmin

        self.psi_tab = []
        self.mom_tab = []
        self.psi_up_tab = []
        self.mom_up_tab = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def plot(self, psi, t, x, np):
        self.psi_tab.append((
            psi[0], t, x
        ))

    def plot_mom(self, t, moms, ener, E, freq_mult, overlp, ener_tot, abs_psi_max, real_psi_max):
        self.mom_tab.append((
            t,
            moms.x_l.real, moms.x2_l.real, moms.p_l.real, moms.p2_l.real,
            ener, E, freq_mult, overlp, ener_tot, abs_psi_max, real_psi_max
        ))

    def plot_up(self, psi, t, x, np):
        self.psi_up_tab.append((
            psi[1], t, x
        ))

    def plot_mom_up(self, t, moms, ener, E, freq_mult, overlp, overlp_tot, abs_psi_max, real_psi_max):
        self.mom_up_tab.append((
            t,
            moms.x_u.real, moms.x2_u.real, moms.p_u.real, moms.p2_u.real,
            ener, E, freq_mult, overlp, overlp_tot, abs_psi_max, real_psi_max
        ))

    def print_time_point(self, l, psi, t, x, np, moms, ener, ener_u, E, freq_mult, overlp, overlp_u, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u):
        if l % self.mod_fileout == 0 and l >= self.lmin:
            self.plot(psi, t, x, np)
            self.plot_up(psi, t, x, np)
            self.plot_mom(t, moms, ener, E, freq_mult, overlp, ener_tot, abs_psi_max, real_psi_max)
            self.plot_mom_up(t, moms, ener_u, E, freq_mult, overlp_u, overlp_tot, abs_psi_max_u, real_psi_max_u)

    def print_all(self, filename):
        # To print the whole arrays without truncation ('...')
        np.set_printoptions(threshold=sys.maxsize)

        with open(filename, "w") as f:
            f.write("from numpy import array\n\n")
            f.write("psi_tab = [\n")
            for l in self.psi_tab:
                f.write("    " + str(l) + ",\n")
            f.write("]\n")

            f.write("mom_tab = [\n")
            for l in self.mom_tab:
                f.write("    " + str(l) + ",\n")
            f.write("]\n\n")

            f.write("psi_up_tab = [\n")
            for l in self.psi_up_tab:
                f.write("    " + str(l) + ",\n")
            f.write("]\n\n")

            f.write("mom_up_tab = [\n")
            for l in self.mom_up_tab:
                f.write("    " + str(l) + ",\n")
            f.write("]\n\n")

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
        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, task_manager_imp.psi_init, task_manager_imp,
                                                  task_manager_imp.pot, reporter_impl,
                                                  None, None)
            fitting_solver.time_propagation()

        # Uncomment in case of emergency :)
        # reporter_impl.print_all("test_data/fitter_single_harm.py")

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
        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, task_manager_imp.psi_init, task_manager_imp,
                                                  task_manager_imp.pot, reporter_impl,
                                                  None, None)
            fitting_solver.time_propagation()

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
        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, task_manager_imp.psi_init, task_manager_imp,
                                                  task_manager_imp.pot, reporter_impl,
                                                  None, None)
            fitting_solver.time_propagation()

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
        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, task_manager_imp.psi_init, task_manager_imp,
                                                  task_manager_imp.pot, reporter_impl,
                                                  None, None)
            fitting_solver.time_propagation()

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
        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, task_manager_imp.psi_init, task_manager_imp,
                                                  task_manager_imp.pot, reporter_impl,
                                                  None, None)
            fitting_solver.time_propagation()

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
        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, task_manager_imp.psi_init, task_manager_imp,
                                                  task_manager_imp.pot, reporter_impl,
                                                  None, None)
            fitting_solver.time_propagation()

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
        with TestReporter(mod_fileout, lmin) as reporter_impl:
            fitting_solver = fitter.FittingSolver(conf, task_manager_imp.psi_init, task_manager_imp,
                                                  task_manager_imp.pot, reporter_impl,
                                                  None, None)
            fitting_solver.time_propagation()

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
