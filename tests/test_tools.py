import sys

import numpy as np
from reporter import *


class TableComparer:
    def __init__(self, epsilon, delta: np.float64):
        self.epsilon = epsilon
        self.delta = delta

    @staticmethod
    def is_array(el):
        return isinstance(el, np.ndarray) or isinstance(el, list) or isinstance(el, tuple)

    @staticmethod
    def is_complex(el):
        return isinstance(el, np.complex128) or isinstance(el, complex)

    @staticmethod
    def is_float(el):
        return isinstance(el, np.float64) or isinstance(el, float)

    def compare_floats(self, el1, el2, eps):
        # Trivially comparing two floats
        if abs(el2) < self.delta and abs(el1) < self.delta:
            pass  # Both are smalls. Treated as equals
        elif abs(el2) < self.delta < abs(el1):
            return False  # el2 is small, el1 is not. Not equal
        elif abs(el1 - el2) / abs(el2) > abs(eps):
            return False  # The relative difference is larger than eps. Not equal

        return True

    def compare_complex(self, el1, el2, eps):
        r1 = el1.real
        r2 = el2.real
        i1 = el1.imag
        i2 = el2.imag

        if abs(i1) < self.delta and abs(i2) < self.delta:
            return self.compare_floats(r1, r2, eps.real)
        elif abs(r1) < self.delta and abs(r2) < self.delta:
            return self.compare_floats(i1, i2, eps.imag)
        elif abs(r1) > self.delta > abs(r2):
            return False
        elif abs(r2) > self.delta > abs(r1):
            return False
        elif abs(i1) > self.delta > abs(i2):
            return False
        elif abs(i2) > self.delta > abs(i1):
            return False
        elif not self.compare_floats(r1, r2, eps.real) or \
                not self.compare_floats(i1, i2, eps.imag):
            return False

        return True

    def compare_el(self, el1, el2, eps):
        if isinstance(el1, int) and isinstance(el2, int):
            # Trivially comparing two ints
            if el1 != el2:
                return False
        elif self.is_float(el1) and self.is_float(el2) and self.is_float(eps):
            # Trivially comparing two floats
            if not self.compare_floats(el1, el2, eps):
                return False
        elif self.is_complex(el1) and self.is_complex(el2) and self.is_complex(eps):
            # Trivially comparing two complexes
            if not self.compare_complex(el1, el2, eps):
                return False
        elif self.is_array(el1) and self.is_array(el2) and self.is_complex(eps):
            # Comparing each element of two complex arrays
            if len(el1) != len(el2):
                raise RuntimeError("Complex arrays have different lengths")

            for l in range(len(el1)):
                if not self.compare_complex(el1[l], el2[l], eps):
                    return False
        elif self.is_array(el1) and self.is_array(el2) and self.is_float(eps):
            # Comparing each element of two real arrays
            if len(el1) != len(el2):
                raise RuntimeError("Real arrays have different lengths")

            for l in range(len(el1)):
                if not self.compare_floats(el1[l], el2[l], eps):
                    return False
        else:
            raise ValueError(f"Invalid types to compare: {str(type(el1))} and {str(type(el2))}")

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

        self.psi_tab = []
        for n in range(self.nlevs):
            self.psi_tab.append([])

        self.prop_tab = [] * nlevs
        for n in range(self.nlevs):
            self.prop_tab.append([])


        self.fit_tab = []

    def open(self):
        return self

    def close(self):
        pass

    def plot(self, psi: Psi, t, x, np):
        for n in range(self.nlevs):
            self.psi_tab[n].append(
                (psi.f[n], t, x)
            )
        pass

    def plot_prop(self, t, moms, ener, norm, overlp0, overlpf, psi_max_abs, psi_max_real):
        for n in range(self.nlevs):
            self.prop_tab[n].append(
                (t,
                moms.x[n].real, moms.x2[n].real, moms.p[n].real, moms.p2[n].real,
                ener[n].real, norm[n], overlp0[n], overlpf[n], psi_max_abs[n], psi_max_real[n])
        )

    def plot_fitter(self, t, E, freq_mult, ener_tot, overlp_tot, smoms):
        self.fit_tab.append((
            t, E, freq_mult, ener_tot.real, overlp_tot[0], overlp_tot[1], smoms.x.real, smoms.y.real, smoms.z.real
        ))

    def print_time_point_prop(self, l, psi: Psi, t, x, np, nt, moms, smoms, ener, norm, overlp0, overlpf, overlp_tot, ener_tot,
                              psi_max_abs, psi_max_real, E, freq_mult):
        if l % self.mod_fileout == 0 and l >= self.lmin:
            self.plot(psi, t, x, np)
            self.plot_prop(t, moms, ener, norm, overlp0, overlpf, psi_max_abs, psi_max_real)
            self.plot_fitter(t, E, freq_mult, ener_tot, overlp_tot, smoms)

    @staticmethod
    def __print_any_array(f, a, prefix = "    "):
        for m in range(len(a) - 1):
            f.write(prefix + str(a[m]) + ", ")
            if (m+1) % 4 == 0: f.write('\n')
        f.write(prefix + str(a[-1]) + "\n")

    def print_all(self, filename, filename_fit):
        # To print the whole arrays without truncation ('...')
        np.set_printoptions(threshold=sys.maxsize)

        with open(filename, "w") as f:
            f.write("from numpy import array\n\n")
            f.write(f"psi_tabs = [\n    # level #0\n")
            for n in range(self.nlevs):
                f.write("[")
                for q in range(len(self.psi_tab[n])):
                    fun, t, x = self.psi_tab[n][q]

                    f.write("    (")
                    f.write("array([\n")
                    TestPropagationReporter.__print_any_array(f, fun, "        ")
                    f.write("    ]),\n")

                    f.write("    " + str(t) + ",\n")

                    f.write("    array([\n")
                    TestPropagationReporter.__print_any_array(f, x, "        ")
                    f.write("    ])")
                    f.write("),")

                f.write("]")
                if n != self.nlevs - 1:
                    f.write(f",\n    # level #{n + 1}\n")
            f.write("]\n\n")

            f.write(f"prop_tabs = [\n    # level #0\n")
            for n in range(self.nlevs):
                f.write("[\n")
                for l in self.prop_tab[n]:
                    f.write("    " + str(l) + ",\n")

                f.write("]")
                if n != self.nlevs - 1:
                    f.write(f",\n    # level #{n + 1}\n")
            f.write("]\n\n")

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

    def plot_iter(self, iter, goal_close, Fsm, E_int, J):
        self.iter_tab.append((
            iter, goal_close, Fsm.real, E_int.real, J
        ))

    def plot_i_E(self, E_tlist, iter, t_list, nt):
        self.iter_tab_E.append((
            iter, t_list, E_tlist
        ))

    def print_iter_point_fitter(self, iter, goal_close, E_tlist, t_list, Fsm, E_int, J, nt):
        if iter % self.imod_fileout == 0 and iter >= self.imin:
            self.plot_iter(iter, goal_close, Fsm, E_int, J)
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
