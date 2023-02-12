import unittest

import test_data
from propagation import *
import grid_setup
import task_manager
from config import TaskRootConfiguration
from psi_basis import PsiBasis
from tests.test_tools import *


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
        psi_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-21) # psi, t, x
        tvals_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                        0.0000001, complex(0.001, 0.001), # ener, norm,
                                        complex(0.001, 0.001), complex(0.001, 0.001), # overlp0, overlpf,
                                        0.0001, 0.0001), 1.e-21) # psi_max_abs, psi_max_real

        nlevs = 2
        for n in range(nlevs):
            self.assertTrue(psi_comparer.compare(test_data.prop_trans_woc.psi_tabs[n], test_data.prop_trans_woc.psi_tabs[n]))
            self.assertTrue(tvals_comparer.compare(test_data.prop_trans_woc.prop_tabs[n], test_data.prop_trans_woc.prop_tabs[n]))


class propagation_Tests(unittest.TestCase):
    @staticmethod
    def _test_setup():
        conf = TaskRootConfiguration.FitterConfiguration()

        conf_fitter = {
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

        conf.load(conf_fitter)

        # setup of the grid
        grid = grid_setup.GridConstructor(conf.propagation)
        dx, x = grid.grid_setup()

        # setup of the time grid
        forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_prop=conf.propagation)
        t_step, t_list = forw_time_grid.grid_setup()

        psi0 = PsiBasis(1)
        # evaluating of initial wavefunction
        psi0.psis[0].f[0] = task_manager._PsiFunctions.morse(x,
                                             conf.propagation.np,
                                             conf.propagation.x0,
                                             conf.propagation.p0,
                                             conf.propagation.m,
                                             conf.propagation.De,
                                             conf.propagation.a,
                                             conf.propagation.L)
        psi0.psis[0].f[1] = task_manager._PsiFunctions.zero(conf.propagation.np)

        psif = PsiBasis(1)
        # evaluating of the final goal
        psif.psis[0].f[0] = task_manager._PsiFunctions.zero(conf.propagation.np)
        psif.psis[0].f[1] = task_manager._PsiFunctions.morse(x,
                                              conf.propagation.np,
                                              conf.propagation.x0p + conf.propagation.x0,
                                              conf.propagation.p0,
                                              conf.propagation.m,
                                              conf.propagation.De_e,
                                              conf.propagation.a_e,
                                              conf.propagation.L)

        return conf, dx, x, psi0, psif, t_step, t_list


    def test_prop_forward(self):
        conf, dx, x, psi0, psif, t_step, t_list = self._test_setup()

        def _warning_collocation_points(np, np_min):
            pass

        def _warning_time_steps(nt, nt_min):
            pass

        def process_instrumentation(instr: PropagationSolver.InstrumentationOutputData):
            pass

        def freq_multiplier(dyn: PropagationSolver.DynamicState, stat: PropagationSolver.StaticState):
            return 1.0

        def laser_field_envelope(prop: PropagationSolver, stat: PropagationSolver.StaticState,
                               dyn: PropagationSolver.DynamicState):
            return task_manager._LaserFields.laser_field_gauss(conf.propagation.E0, dyn.t,
                                         conf.propagation.t0, conf.propagation.sigma)

        def laser_field_hf(freq_mult, t, pcos, w_list):
            return task_manager._LaserFieldsHighFrequencyPart.cexp(conf.propagation.nu_L, 1.0, t, pcos, w_list)

        def dynamic_state_factory(l, t, psi, psi_omega, E, freq_mult, dir):
            assert dir == PropagationSolver.Direction.FORWARD
            psi_omega_copy = copy.deepcopy(psi_omega)
            return PropagationSolver.DynamicState(l, t, psi, psi_omega_copy, E, freq_mult, dir)


        mod_fileout = 10000
        lmin = 0
        ntriv = 1
        nlevs = 2

        reporter_impl = TestPropagationReporter(mod_fileout, lmin, nlevs)
        reporter_impl.open()

        solver = PropagationSolver(
            pot=task_manager.MorseMultipleStateTaskManager._pot,
            _warning_collocation_points=_warning_collocation_points,
            _warning_time_steps=_warning_time_steps,
            reporter=reporter_impl,
            laser_field_envelope=laser_field_envelope,
            laser_field_hf=laser_field_hf,
            freq_multiplier=freq_multiplier,
            dynamic_state_factory=dynamic_state_factory,
            pcos=conf.pcos,
            w_list=conf.w_list,
            mod_log=conf.mod_log,
            ntriv=ntriv,
            conf_prop=conf.propagation)

        solver.start(dx, x, t_step, psi0.psis[0], psif.psis[0], PropagationSolver.Direction.FORWARD)

        # main propagation loop
        while solver.step(0.0):
            pass

        reporter_impl.close()

        # Uncomment in case of emergency :)
        reporter_impl.print_all("../test_data/prop_trans_woc_forw_.py", None)

        psi_prop_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-21) # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001, # t, moms.x, moms.x2, moms.p, moms.p2,
                                      0.0000001, complex(0.001, 0.001), # ener, norm,
                                      complex(0.001, 0.001), complex(0.001, 0.001), # overlp0, overlpf,
                                      0.0001, 0.0001), 1.e-21) # psi_max_abs, psi_max_real

        for n in range(nlevs):
            self.assertTrue(psi_prop_comparer.compare(reporter_impl.psi_tab[n], test_data.prop_trans_woc_forw.psi_tabs[n]))
            self.assertTrue(tvals_prop_comparer.compare(reporter_impl.prop_tab[n], test_data.prop_trans_woc_forw.prop_tabs[n]))


    def test_prop_backward(self):
        conf, dx, x, psi0, psif, t_step, t_list = self._test_setup()

        def _warning_collocation_points(np, np_min):
            pass

        def _warning_time_steps(nt, nt_min):
            pass

        def process_instrumentation(instr: PropagationSolver.InstrumentationOutputData):
            pass

        def freq_multiplier(dyn: PropagationSolver.DynamicState, stat: PropagationSolver.StaticState):
            return 1.0

        def laser_field_envelope(prop: PropagationSolver, stat: PropagationSolver.StaticState,
                               dyn: PropagationSolver.DynamicState):
            return task_manager._LaserFields.laser_field_gauss(conf.propagation.E0, dyn.t,
                                         conf.propagation.t0, conf.propagation.sigma)

        def laser_field_hf(freq_mult, t, pcos, w_list):
            return task_manager._LaserFieldsHighFrequencyPart.cexp(conf.propagation.nu_L, 1.0, t, pcos, w_list)

        def dynamic_state_factory(l, t, psi, psi_omega, E, freq_mult, dir):
            assert dir == PropagationSolver.Direction.BACKWARD
            psi_omega_copy = copy.deepcopy(psi_omega)
            return PropagationSolver.DynamicState(l, t, psi, psi_omega_copy, E, freq_mult, dir)


        mod_fileout = 10000
        lmin = 0
        ntriv = 1
        nlevs = 2

        reporter_impl = TestPropagationReporter(mod_fileout, lmin, nlevs)
        reporter_impl.open()

        solver = PropagationSolver(
            pot=task_manager.MorseMultipleStateTaskManager._pot,
            _warning_collocation_points=_warning_collocation_points,
            _warning_time_steps=_warning_time_steps,
            reporter=reporter_impl,
            laser_field_envelope=laser_field_envelope,
            laser_field_hf=laser_field_hf,
            freq_multiplier=freq_multiplier,
            dynamic_state_factory=dynamic_state_factory,
            pcos=conf.pcos,
            w_list=conf.w_list,
            mod_log=conf.mod_log,
            ntriv=ntriv,
            conf_prop=conf.propagation)

        solver.start(dx, x, t_step, psif.psis[0], psi0.psis[0], PropagationSolver.Direction.BACKWARD)

        # main propagation loop
        while solver.step(conf.propagation.T):
            pass

        reporter_impl.close()

        # Uncomment in case of emergency :)
        reporter_impl.print_all("../test_data/prop_trans_woc_back_.py", None)

        psi_prop_comparer = TableComparer((complex(0.0001, 0.0001), 0.000001, 0.0001), 1.e-21)  # psi, t, x
        tvals_prop_comparer = TableComparer((0.000001, 0.001, 0.001, 0.001, 0.000001,  # t, moms.x, moms.x2, moms.p, moms.p2,
                                      0.0000001, complex(0.001, 0.001), # ener, norm,
                                      complex(0.001, 0.001), complex(0.001, 0.001), # overlp0, overlpf,
                                      0.0001, 0.0001), 1.e-21) # psi_max_abs, psi_max_real

        for n in range(nlevs):
            self.assertTrue(
                psi_prop_comparer.compare(reporter_impl.psi_tab[n], test_data.prop_trans_woc_back.psi_tabs[n]))
            self.assertTrue(
                tvals_prop_comparer.compare(reporter_impl.prop_tab[n], test_data.prop_trans_woc_back.prop_tabs[n]))
