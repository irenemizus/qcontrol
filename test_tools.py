#from __future__ import print_function
import sys
import numpy as np
from reporter import *


class TableComparer:
    def __init__(self, epsilon, delta: float):
        self.epsilon = epsilon
        self.delta = delta

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
                    if abs(el2) < self.delta and abs(el1) < self.delta:
                        return True
                    elif abs(el2) < self.delta and abs(el1) >= self.delta:
                        return False
                    elif abs(el1 - el2) / abs(el2) >= abs(eps):
                        return False
                elif isinstance(el1, np.ndarray) and \
                     isinstance(el2, np.ndarray) and \
                     isinstance(eps, np.complex):
                    # Comparing each element of two complex arrays
                    if len(el1) != len(el2):
                        raise RuntimeError("Complex arrays have different lengths")

                    for l in range(len(el1)):
                        if abs(el2[l].real) < self.delta and abs(el1[l].real) < self.delta and \
                           abs(el2[l].imag) < self.delta and abs(el1[l].imag) < self.delta:
                            return True
                        elif abs(el2[l].real) < self.delta and abs(el1[l].real) >= self.delta:
                            return False
                        elif abs(el2[l].imag) < self.delta and abs(el1[l].imag) >= self.delta:
                            return False
                        elif abs(el1[l].real - el2[l].real) / abs(el2[l].real) >= eps.real or \
                           abs(el1[l].imag - el2[l].imag) / abs(el2[l].imag) >= eps.imag:
                            return False
        return True


class TestPropagationReporter(PropagationReporter):
    def __init__(self, mod_fileout, lmin):
        super().__init__("")
        self.mod_fileout = mod_fileout
        self.lmin = lmin

        self.psi_tab = []
        self.tvals_tab = []
        self.psi_up_tab = []
        self.tvals_up_tab = []
        self.tvals_tab_fit = []

    def open(self):
        return self

    def close(self):
        pass

    def plot(self, psi: Psi, t, x, np):
        self.psi_tab.append((
            psi.f[0], t, x
        ))

    def plot_tvals(self, t, moms, ener, overlp, ener_tot, abs_psi_max, real_psi_max):
        self.tvals_tab.append((
            t,
            moms.x_l.real, moms.x2_l.real, moms.p_l.real, moms.p2_l.real,
            ener, overlp, ener_tot, abs_psi_max, real_psi_max
        ))

    def plot_up(self, psi: Psi, t, x, np):
        self.psi_up_tab.append((
            psi.f[1], t, x
        ))

    def plot_tvals_up(self, t, moms, ener, overlp, overlp_tot, abs_psi_max, real_psi_max):
        self.tvals_up_tab.append((
            t,
            moms.x_u.real, moms.x2_u.real, moms.p_u.real, moms.p2_u.real,
            ener, overlp, overlp_tot, abs_psi_max, real_psi_max
        ))

    def plot_tvals_fit(self, t, E, freq_mult):
        self.tvals_tab_fit.append((
            t, E, freq_mult
        ))

    def print_time_point_prop(self, l, psi: Psi, t, x, np, moms, ener, ener_u, overlp, overlp_u, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u, E, freq_mult):
        if l % self.mod_fileout == 0 and l >= self.lmin:
            self.plot(psi, t, x, np)
            self.plot_up(psi, t, x, np)
            self.plot_tvals(t, moms, ener, overlp, ener_tot, abs_psi_max, real_psi_max)
            self.plot_tvals_up(t, moms, ener_u, overlp_u, overlp_tot, abs_psi_max_u, real_psi_max_u)
            self.plot_tvals_fit(t, E, freq_mult)


    def print_all(self, filename, filename_fit):
        # To print the whole arrays without truncation ('...')
        np.set_printoptions(threshold=sys.maxsize)

        with open(filename, "w") as f:
            f.write("from numpy import array\n\n")
            f.write("psi_tab = [\n")
            for l in self.psi_tab:
                f.write("    " + str(l) + ",\n")
            f.write("]\n")

            f.write("tvals_tab = [\n")
            for l in self.tvals_tab:
                f.write("    " + str(l) + ",\n")
            f.write("]\n\n")

            f.write("psi_up_tab = [\n")
            for l in self.psi_up_tab:
                f.write("    " + str(l) + ",\n")
            f.write("]\n\n")

            f.write("tvals_up_tab = [\n")
            for l in self.tvals_up_tab:
                f.write("    " + str(l) + ",\n")
            f.write("]\n\n")

        if filename_fit:
            with open(filename_fit, "w") as f_fit:
                f_fit.write("from numpy import array\n\n")
                f_fit.write("tvals_tab = [\n")
                for l in self.tvals_tab_fit:
                    f_fit.write("    " + str(l) + ",\n")
                f_fit.write("]\n\n")


class TestFitterReporter(FitterReporter):
    def __init__(self, mod_fileout, lmin, imod_fileout, imin):
        super().__init__()
        self.mod_fileout = mod_fileout
        self.lmin = lmin

        self.imod_fileout = imod_fileout
        self.imin = imin

        self.iter_tab = []
        self.iter_tab_E = []

        self.prop_reporters = {}

    def open(self):
        return self

    def close(self):
        pass

    def plot_iter(self, iter, goal_close):
        self.iter_tab.append((
            iter, goal_close
        ))

    def plot_i_E(self, E_tlist, iter, t_list, nt):
        self.iter_tab_E.append((
            iter, t_list, E_tlist
        ))

    def print_iter_point_fitter(self, iter, goal_close, E_tlist, t_list, nt):
        if iter % self.imod_fileout == 0 and iter >= self.imin:
            self.plot_iter(iter, goal_close)
            self.plot_i_E(E_tlist, iter, t_list, nt)

    def print_all(self, filename):
        # To print the whole arrays without truncation ('...')
        np.set_printoptions(threshold=sys.maxsize)

        with open(filename, "w") as f:
            f.write("from numpy import array\n\n")
            f.write("iter_tab = [\n")
            for l in self.iter_tab:
                f.write("    " + str(l) + ",\n")
            f.write("]\n\n")

            f.write("iter_tab_E = [\n")
            for l in self.iter_tab_E:
                f.write("    " + str(l) + ",\n")
            f.write("]\n\n")


    def create_propagation_reporter(self, prop_id: str):
        new_prop_rep = TestPropagationReporter(self.mod_fileout, self.lmin)
        self.prop_reporters[prop_id] = new_prop_rep
        return new_prop_rep
