import unittest

import test_data
import test_tools
from propagation import *
import grid_setup
import task_manager
from config import RootConfiguration
from test_tools import *


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
        print(conf)

        conf_prop = conf.propagation

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

        return conf_prop, dx, x, psi0, psif


    def test_prop_forward(self):
        conf_prop, dx, x, psi0, psif = self._test_setup()

        def report_static(stat: PropagationSolver.StaticState):
            pass

        def report_dynamic(dyn: PropagationSolver.DynamicState):
            pass

        def process_instrumentation(instr: PropagationSolver.InstrumentationOutputData):
            pass

        def freq_multiplier(stat: PropagationSolver.StaticState):
            return 1.0

        def laser_field_envelope(stat: PropagationSolver.StaticState,
                               dyn: PropagationSolver.DynamicState):
            return phys_base.laser_field(conf_prop.E0, dyn.t,
                                         conf_prop.t0, conf_prop.sigma)

        def dynamic_state_factory(l, t, psi, psi_omega, E, freq_mult):
            psi_omega_copy = copy.deepcopy(psi_omega)
            return PropagationSolver.DynamicState(l, t, psi, psi_omega_copy, E, freq_mult)

        solver = PropagationSolver(pot=task_manager.MorseMultipleStateTaskManager._pot,
                                report_static=report_static,
                                report_dynamic=report_dynamic,
                                process_instrumentation=process_instrumentation,
                                laser_field_envelope=laser_field_envelope,
                                freq_multiplier=freq_multiplier,
                                dynamic_state_factory=dynamic_state_factory,
                                conf_prop=conf_prop)

        mod_fileout = 10000
        lmin = 0

        with TestReporter(mod_fileout, lmin) as reporter_impl:
            solver.start(dx, x, psi0, psif, PropagationSolver.Direction.FORWARD)

            # main propagation loop
            while solver.step(0.0):
                pass

        # Uncomment in case of emergency :)
        reporter_impl.print_all("test_data/prop_trans_woc.py")

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

