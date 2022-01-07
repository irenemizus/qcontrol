#from __future__ import print_function
import sys
import numpy as np
from reporter import Reporter


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
        super().__init__()
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
