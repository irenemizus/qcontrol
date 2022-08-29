import sys

import numpy
import numpy as np
from reporter import *


class TableComparer:
    def __init__(self, epsilon, delta: float):
        self.epsilon = epsilon
        self.delta = delta

    @staticmethod
    def is_array(el):
        return isinstance(el, np.ndarray) or isinstance(el, list) or isinstance(el, tuple)

    @staticmethod
    def is_complex(el):
        return isinstance(el, np.complex128) or isinstance(el, complex)

    def compare_el(self, el1, el2, eps):
        if isinstance(el1, int) and isinstance(el2, int):
            # Trivially comparing two ints
            if el1 != el2:
                return False
        elif isinstance(el1, float) and isinstance(el2, float) and isinstance(eps, float):
            # Trivially comparing two floats
            if abs(el2) < self.delta and abs(el1) < self.delta:
                pass  # Going on
            elif abs(el2) < self.delta and abs(el1) >= self.delta:
                return False
            elif abs(el1 - el2) / abs(el2) >= abs(eps):
                return False
        elif isinstance(el1, complex) and isinstance(el2, complex) and isinstance(eps, complex):
            # Trivially comparing two complexes
            if abs(el2.real) < self.delta and abs(el1.real) < self.delta and \
                    abs(el2.imag) < self.delta and abs(el1.imag) < self.delta:
                pass  # Going on
            elif abs(el2.real) < self.delta and abs(el1.real) >= self.delta:
                return False
            elif abs(el2.imag) < self.delta and abs(el1.imag) >= self.delta:
                return False
            elif abs(el2.imag) < self.delta and abs(el1.imag) < self.delta and \
                 abs(el2.real) >= self.delta and abs(el1.real) >= self.delta:
                if abs(el1.real - el2.real) / abs(el2.real) >= abs(eps.real):
                    return False
                else:
                    pass
            elif abs(el1.real - el2.real) / abs(el2.real) >= eps.real or \
                    abs(el1.imag - el2.imag) / abs(el2.imag) >= eps.imag:
                return False
        elif isinstance(el1, np.ndarray) and isinstance(el2, np.ndarray) and \
                self.is_complex(eps):
            # Comparing each element of two complex arrays
            if len(el1) != len(el2):
                raise RuntimeError("Complex arrays have different lengths")

            for l in range(len(el1)):
                if abs(el2[l].real) < self.delta and abs(el1[l].real) < self.delta and \
                        abs(el2[l].imag) < self.delta and abs(el1[l].imag) < self.delta:
                    pass  # Going on
                elif abs(el2[l].real) < self.delta and abs(el1[l].real) >= self.delta:
                    return False
                elif abs(el2[l].imag) < self.delta and abs(el1[l].imag) >= self.delta:
                    return False
                elif abs(el1[l].real - el2[l].real) / abs(el2[l].real) >= eps.real or \
                        abs(el1[l].imag - el2[l].imag) / abs(el2[l].imag) >= eps.imag:
                    return False
        elif self.is_array(el1) and self.is_array(el2) and isinstance(eps, float):
            # Comparing each element of two real arrays
            if len(el1) != len(el2):
                raise RuntimeError("Real arrays have different lengths")

            for l in range(len(el1)):
                if abs(el2[l]) < self.delta and abs(el1[l]) < self.delta:
                    pass  # Going on
                elif abs(el2[l]) < self.delta and abs(el1[l]) >= self.delta:
                    return False
                elif abs(el1[l] - el2[l]) / abs(el2[l]) >= eps:
                    return False
        else:
            raise ValueError("Invalid types to compare")

        return True

    def compare(self, tab1, tab2):
        # Comparing tables length
        if len(tab1) != len(tab2):
            return False

        # For each line
        for i in range(len(tab1)):
            # Comparing the length of the tab1 line with the one of tab2 line
            if not self.is_array(tab1[i]) and not self.is_array(tab2[i]) and not self.is_array(self.epsilon):
                el1 = tab1[i]
                el2 = tab2[i]
                eps = self.epsilon
                if not self.compare_el(el1, el2, eps):
                    return False
            else:
                if len(tab1[i]) != len(tab2[i]):
                    return False

                if len(tab1[i]) != len(self.epsilon):
                    return False

                for k in range(len(tab1[i])):
                    el1 = tab1[i][k]
                    el2 = tab2[i][k]
                    eps = self.epsilon[k]
                    if not self.compare_el(el1, el2, eps):
                        return False

        return True


class TestPropagationReporter(PropagationReporter):
    def __init__(self, mod_fileout, lmin, nlevs):
        super().__init__("", nlevs)
        self.mod_fileout = mod_fileout
        self.lmin = lmin
        self.nlevs = nlevs

        self.psi_tab = [None] * nlevs
        self.prop_tab = [None] * nlevs
        self.fit_tab = []

    def open(self):
        return self

    def close(self):
        pass

    def plot(self, psi: Psi, t, x, np):
        for n in range(self.nlevs):
            self.psi_tab[n] = (
                psi.f[n], t, x
            )

    def plot_prop(self, t, moms, ener, overlp0, overlpf, psi_max_abs, psi_max_real):
        for n in range(self.nlevs):
            self.prop_tab[n] = (
                t,
                moms.x[n].real, moms.x2[n].real, moms.p[n].real, moms.p2[n].real,
                ener[n], overlp0[n], overlpf[n], psi_max_abs[n], psi_max_real[n]
        )

    def plot_fitter(self, t, E, freq_mult, ener_tot, overlp_tot):
        self.fit_tab.append((
            t, E, freq_mult, ener_tot, overlp_tot[0], overlp_tot[1]
        ))

    def print_time_point_prop(self, l, psi: Psi, t, x, np, moms, ener, overlp0, overlpf, overlp_tot, ener_tot,
                              psi_max_abs, psi_max_real, E, freq_mult):
        if l % self.mod_fileout == 0 and l >= self.lmin:
            self.plot(psi, t, x, np)
            self.plot_prop(t, moms, ener, overlp0, overlpf, psi_max_abs, psi_max_real)
            self.plot_fitter(t, E, freq_mult, ener_tot, overlp_tot)


    def print_all(self, filename, filename_fit):
        # To print the whole arrays without truncation ('...')
        np.set_printoptions(threshold=sys.maxsize)

        with open(filename, "w") as f:
            f.write("from numpy import array\n\n")
            for n in range(self.nlevs):
                f.write(f"psi_tabs_{n} = [\n")
                psin = numpy.array(self.psi_tab[n]).astype(complex)
                for l in psin:
                    f.write("    " + str(l) + ",\n")
                f.write("]\n")

            for n in range(self.nlevs):
                f.write(f"prop_tabs_{n} = [\n")
                propn = numpy.array(self.prop_tab[n]).astype(complex)
                for l in propn:
                    f.write("    " + str(l) + ",\n")
                f.write("]\n")

        if filename_fit:
            with open(filename_fit, "w") as f_fit:
                f_fit.write("from numpy import array\n\n")
                f_fit.write("tvals_tab = [\n")
                for l in self.fit_tab:
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

    def plot_iter(self, iter, goal_close, Fsm):
        self.iter_tab.append((
            iter, goal_close, Fsm
        ))

    def plot_i_E(self, E_tlist, iter, t_list, nt):
        self.iter_tab_E.append((
            iter, t_list, E_tlist
        ))

    def print_iter_point_fitter(self, iter, goal_close, E_tlist, t_list, Fsm, nt):
        if iter % self.imod_fileout == 0 and iter >= self.imin:
            self.plot_iter(iter, goal_close, Fsm)
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


    def create_propagation_reporter(self, prop_id: str, nlevs):
        new_prop_rep = TestPropagationReporter(self.mod_fileout, self.lmin, nlevs)
        self.prop_reporters[prop_id] = new_prop_rep
        return new_prop_rep
