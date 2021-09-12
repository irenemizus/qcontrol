import os.path

import phys_base

class Reporter:
    def __init__(self, conf_output, out_path):
        self.conf_output = conf_output
        self.out_path = out_path
        self.f_abs = None
        self.f_real = None
        self.f_mom = None
        self.f_abs_up = None
        self.f_real_up = None
        self.f_mom_up = None

    def __enter__(self):
        self.f_abs = open(os.path.join(self.out_path, self.conf_output.file_abs), 'w')
        self.f_real = open(os.path.join(self.out_path, self.conf_output.file_real), 'w')
        self.f_mom = open(os.path.join(self.out_path, self.conf_output.file_mom), 'w')
        self.f_abs_up = open(os.path.join(self.out_path, self.conf_output.file_abs) + "_exc", 'w')
        self.f_real_up = open(os.path.join(self.out_path, self.conf_output.file_real + "_exc"), 'w')
        self.f_mom_up = open(os.path.join(self.out_path, self.conf_output.file_mom + "_exc"), 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f_abs.close()
        self.f_abs = None
        self.f_real.close()
        self.f_real = None
        self.f_mom.close()
        self.f_mom = None
        self.f_abs_up.close()
        self.f_abs_up = None
        self.f_real_up.close()
        self.f_real_up = None
        self.f_mom_up.close()
        self.f_mom_up = None

    @staticmethod
    def __plot_file(psi, t, x, np, f_abs, f_real):
        """ Plots absolute and real values of the current wavefunction """
        for i in range(np):
            f_abs.write("{:.6f} {:.6f} {:.6e}\n".format(t * 1e+15, x[i], abs(psi[i])))
            f_real.write("{:.6f} {:.6f} {:.6e}\n".format(t * 1e+15, x[i], psi[i].real))
            f_abs.flush()
            f_real.flush()

    @staticmethod
    def __plot_mom_file(t, momx, momx2, momp, momp2, ener, E, overlp, tot, abs_psi_max, real_psi_max, file_mom):
        """ Plots expectation values of the current x, x*x, p and p*p """
        file_mom.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
            t * 1e+15, momx.real, momx2.real, momp.real, momp2.real, ener, E, abs(overlp), tot, abs_psi_max,
            real_psi_max))
        file_mom.flush()

    def plot(self, psi, t, x, np):
        self.__plot_file(psi, t, x, np, self.f_abs, self.f_real)

    def plot_mom(self, t, moms: phys_base.ExpectationValues, ener, E, overlp, ener_tot, abs_psi_max, real_psi_max):
        self.__plot_mom_file(t, moms.x_l, moms.x2_l, moms.p_l, moms.p2_l, ener, E, overlp, ener_tot,
                                              abs_psi_max, real_psi_max, self.f_mom)

    def plot_up(self, psi, t, x, np):
        self.__plot_file(psi, t, x, np, self.f_abs_up, self.f_real_up)

    def plot_mom_up(self, t, moms, ener, E, overlp, overlp_tot, abs_psi_max, real_psi_max):
        self.__plot_mom_file(t, moms.x_u, moms.x2_u, moms.p_u, moms.p2_u, ener, E, overlp, overlp_tot,
                                              abs_psi_max, real_psi_max, self.f_mom_up)

    @staticmethod
    def __plot_test_file(l, phi_l, phi_u, f):
        f.write("Step number: {0}\n".format(l))
        f.write("Lower state wavefunction:")
        for i in range(len(phi_l)):
            f.write("{0}\n".format(phi_l[i]))
        f.write("Upper state wavefunction:")
        for i in range(len(phi_u)):
            f.write("{0}\n".format(phi_u[i]))

