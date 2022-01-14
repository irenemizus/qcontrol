import unittest

import test_data
from propagation import *
import grid_setup
import task_manager
from config import RootConfiguration
from test_tools import *


class test_facilities_Tests(unittest.TestCase):
    def test_table_comparer_trivial(self):
        cmp = TableComparer((3.5, 0.2, np.complex(5.0, 1.0)), 1.e-20)
        tab1 = [
            (3.01, 5.0, np.array([np.complex(1.0, 1.0), np.complex(2.01, 2.001)]))
        ]
        tab2 = [
            (3.0, 5.0, np.array([np.complex(1.0, 1.0), np.complex(2.0, 2.0)]))
        ]

        self.assertTrue(cmp.compare(tab1, tab2))

    def test_table_comparer(self):
        psi_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-51)
        tvals_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                        0.0000001, complex(0.001, 0.001), 0.0000001,
                                        0.0001, 0.0001), 1.e-51)
        tvals_up_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                           0.0000001, complex(0.001, 0.001), 0.001,
                                           0.0001, 0.0001), 1.e-51)

        self.assertTrue(psi_comparer.compare(test_data.prop_trans_woc.psi_tab, test_data.prop_trans_woc.psi_tab))
        self.assertTrue(psi_comparer.compare(test_data.prop_trans_woc.psi_up_tab, test_data.prop_trans_woc.psi_up_tab))

        self.assertTrue(tvals_comparer.compare(test_data.prop_trans_woc.tvals_tab, test_data.prop_trans_woc.tvals_tab))
        self.assertTrue(tvals_up_comparer.compare(test_data.prop_trans_woc.tvals_up_tab, test_data.prop_trans_woc.tvals_up_tab))


class propagation_Tests(unittest.TestCase):
    @staticmethod
    def _test_setup():
        conf = RootConfiguration.FitterConfiguration()

        conf_fitter = {
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

        conf.load(conf_fitter)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # evaluating of initial wavefunction
        psi0 = [
            task_manager._PsiFunctions.morse(x,
                                             conf.propagation.np,
                                             conf.propagation.x0,
                                             conf.propagation.p0,
                                             conf.propagation.m,
                                             conf.propagation.De,
                                             conf.propagation.a),
            task_manager._PsiFunctions.zero(conf.propagation.np)
        ]

        # evaluating of the final goal
        psif = [
            task_manager._PsiFunctions.zero(conf.propagation.np),
            task_manager._PsiFunctions.morse(x,
                                              conf.propagation.np,
                                              conf.propagation.x0p + conf.propagation.x0,
                                              conf.propagation.p0,
                                              conf.propagation.m,
                                              conf.propagation.De_e,
                                              conf.propagation.a_e)
        ]

        return conf, dx, x, psi0, psif


    def test_prop_forward(self):
        conf, dx, x, psi0, psif = self._test_setup()

        def _warning_collocation_points(np, np_min):
            pass

        def _warning_time_steps(nt, nt_min):
            pass

        def process_instrumentation(instr: PropagationSolver.InstrumentationOutputData):
            pass

        def freq_multiplier(stat: PropagationSolver.StaticState):
            return 1.0

        def laser_field_envelope(stat: PropagationSolver.StaticState,
                               dyn: PropagationSolver.DynamicState):
            return phys_base.laser_field(conf.propagation.E0, dyn.t,
                                         conf.propagation.t0, conf.propagation.sigma)

        def dynamic_state_factory(l, t, psi, psi_omega, E, freq_mult):
            psi_omega_copy = copy.deepcopy(psi_omega)
            return PropagationSolver.DynamicState(l, t, psi, psi_omega_copy, E, freq_mult)


        mod_fileout = 10000
        lmin = 0

        reporter_impl = TestPropagationReporter(mod_fileout, lmin)
        reporter_impl.open()

        solver = PropagationSolver(
            pot=task_manager.MorseMultipleStateTaskManager._pot,
            _warning_collocation_points=_warning_collocation_points,
            _warning_time_steps=_warning_time_steps,
            reporter=reporter_impl,
            laser_field_envelope=laser_field_envelope,
            freq_multiplier=freq_multiplier,
            dynamic_state_factory=dynamic_state_factory,
            mod_log=conf.mod_log,
            conf_prop=conf.propagation)

        solver.start(dx, x, psi0, psif, PropagationSolver.Direction.FORWARD)

        # main propagation loop
        while solver.step(0.0):
            pass

        reporter_impl.close()

        # Uncomment in case of emergency :)
        #reporter_impl.print_all("test_data/prop_trans_woc_forw.py")

        psi_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-51)
        tvals_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, complex(0.001, 0.001), 0.0000001,
                                      0.0001, 0.0001), 1.e-51)
        tvals_up_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, complex(0.001, 0.001), 0.001,
                                      0.0001, 0.0001), 1.e-51)

        self.assertTrue(psi_comparer.compare(reporter_impl.psi_tab, test_data.prop_trans_woc_forw.psi_tab))
        self.assertTrue(psi_comparer.compare(reporter_impl.psi_up_tab, test_data.prop_trans_woc_forw.psi_up_tab))

        self.assertTrue(tvals_comparer.compare(reporter_impl.tvals_tab, test_data.prop_trans_woc_forw.tvals_tab))
        self.assertTrue(tvals_up_comparer.compare(reporter_impl.tvals_up_tab, test_data.prop_trans_woc_forw.tvals_up_tab))


    def test_prop_backward(self):
        conf, dx, x, psi0, psif = self._test_setup()

        def _warning_collocation_points(np, np_min):
            pass

        def _warning_time_steps(nt, nt_min):
            pass

        def process_instrumentation(instr: PropagationSolver.InstrumentationOutputData):
            pass

        def freq_multiplier(stat: PropagationSolver.StaticState):
            return 1.0

        def laser_field_envelope(stat: PropagationSolver.StaticState,
                               dyn: PropagationSolver.DynamicState):
            return phys_base.laser_field(conf.propagation.E0, dyn.t,
                                         conf.propagation.t0, conf.propagation.sigma)

        def dynamic_state_factory(l, t, psi, psi_omega, E, freq_mult):
            psi_omega_copy = copy.deepcopy(psi_omega)
            return PropagationSolver.DynamicState(l, t, psi, psi_omega_copy, E, freq_mult)


        mod_fileout = 10000
        lmin = 0

        reporter_impl = TestPropagationReporter(mod_fileout, lmin)
        reporter_impl.open()

        solver = PropagationSolver(
            pot=task_manager.MorseMultipleStateTaskManager._pot,
            _warning_collocation_points=_warning_collocation_points,
            _warning_time_steps=_warning_time_steps,
            reporter=reporter_impl,
            laser_field_envelope=laser_field_envelope,
            freq_multiplier=freq_multiplier,
            dynamic_state_factory=dynamic_state_factory,
            mod_log=conf.mod_log,
            conf_prop=conf.propagation)

        solver.start(dx, x, psif, psi0, PropagationSolver.Direction.BACKWARD)

        # main propagation loop
        while solver.step(conf.propagation.T):
            pass

        reporter_impl.close()

        # Uncomment in case of emergency :)
        #reporter_impl.print_all("test_data/prop_trans_woc_back.py")

        psi_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-51)
        tvals_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, complex(0.001, 0.001), 0.0000001,
                                      0.0001, 0.0001), 1.e-51)
        tvals_up_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,
                                      0.0000001, complex(0.001, 0.001), 0.001,
                                      0.0001, 0.0001), 1.e-51)

        self.assertTrue(psi_comparer.compare(reporter_impl.psi_tab, test_data.prop_trans_woc_back.psi_tab))
        self.assertTrue(psi_comparer.compare(reporter_impl.psi_up_tab, test_data.prop_trans_woc_back.psi_up_tab))

        self.assertTrue(tvals_comparer.compare(reporter_impl.tvals_tab, test_data.prop_trans_woc_back.tvals_tab))
        self.assertTrue(tvals_up_comparer.compare(reporter_impl.tvals_up_tab, test_data.prop_trans_woc_back.tvals_up_tab))
