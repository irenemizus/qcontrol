import plotly.graph_objects as go
import os.path

from tools import print_err

class Reporter:
    def __init__(self):
        pass

    def plot(self, psi, t, x, np):
        raise NotImplementedError()

    def plot_mom(self, t, moms, ener, E, overlp, ener_tot, abs_psi_max, real_psi_max):
        raise NotImplementedError()

    def plot_up(self, psi, t, x, np):
        raise NotImplementedError()

    def plot_mom_up(self, t, moms, ener, E, overlp, overlp_tot, abs_psi_max, real_psi_max):
        raise NotImplementedError()

    def print_time_point(self, l, psi, t, x, np, moms, ener, ener_u, E, overlp, overlp_u, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u):
        raise NotImplementedError()


class TableReporter(Reporter):
    def __init__(self, conf_output):
        self.conf_output = conf_output
        self.f_abs = None
        self.f_real = None
        self.f_mom = None
        self.f_abs_up = None
        self.f_real_up = None
        self.f_mom_up = None

    def __enter__(self):
        self.f_abs = open(os.path.join(self.conf_output.out_path, self.conf_output.tab_abs), 'w')
        self.f_real = open(os.path.join(self.conf_output.out_path, self.conf_output.tab_real), 'w')
        self.f_mom = open(os.path.join(self.conf_output.out_path, self.conf_output.tab_mom), 'w')
        self.f_abs_up = open(os.path.join(self.conf_output.out_path, self.conf_output.tab_abs + "_exc"), 'w')
        self.f_real_up = open(os.path.join(self.conf_output.out_path, self.conf_output.tab_real + "_exc"), 'w')
        self.f_mom_up = open(os.path.join(self.conf_output.out_path, self.conf_output.tab_mom + "_exc"), 'w')
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
        self.__plot_file(psi[0], t, x, np, self.f_abs, self.f_real)

    def plot_mom(self, t, moms, ener, E, overlp, ener_tot, abs_psi_max, real_psi_max):
        self.__plot_mom_file(t, moms.x_l, moms.x2_l, moms.p_l, moms.p2_l, ener, E, overlp, ener_tot,
                                              abs_psi_max, real_psi_max, self.f_mom)

    def plot_up(self, psi, t, x, np):
        self.__plot_file(psi[1], t, x, np, self.f_abs_up, self.f_real_up)

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


    def print_time_point(self, l, psi, t, x, np, moms, ener, ener_u, E, overlp, overlp_u, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u):
        if l % self.conf_output.mod_fileout == 0 and l >= self.conf_output.lmin:
            self.plot(psi, t, x, np)
            self.plot_up(psi, t, x, np)
            self.plot_mom(t, moms, ener, E, overlp, ener_tot, abs_psi_max, real_psi_max)
            self.plot_mom_up(t, moms, ener_u, E, overlp_u, overlp_tot, abs_psi_max_u, real_psi_max_u)


class PlotReporter(Reporter):
    def __init__(self, conf_output):
        self.conf_output = conf_output


    def __enter__(self):
        # Time
        self.t_list = []
        self.t_u_list = []

        # Coordinate
        self.x_list = []

        # X = Time
        self.x_l_list = []
        self.x2_l_list = []
        self.p_l_list = []
        self.p2_l_list = []
        self.x_u_list = []
        self.x2_u_list = []
        self.p_u_list = []
        self.p2_u_list = []
        self.ener_list = []
        self.ener_u_list = []
        self.E_list = []
        self.overlp_list = []
        self.overlp_u_list = []
        self.ener_tot_list = []
        self.overlp_tot_list = []
        self.abs_psi_max_list = []
        self.real_psi_max_list = []
        self.x_u_list = []
        self.x2_u_list = []
        self.p_u_list = []
        self.p2_u_list = []
        self.ener_u_list = []
        self.E_list = []
        self.overlp_list = []
        self.overlp_u_list = []
        self.overlp_tot = []
        self.ener_tot_list = []
        self.abs_psi_max_list = []
        self.real_psi_max_list = []
        self.abs_psi_max_u_list = []
        self.real_psi_max_u_list = []

        # X = Coordinate
        self.psi_abs = {}  # key: t, value: {'x': [], 'y': []}
        self.psi_real = {}  # key: t, value: {'x': [], 'y': []}
        self.psi_abs_u = {}
        self.psi_real_u = {}

        self.i = 0

        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


    @staticmethod
    def __plot_update_graph(psi, numb_plotout, title_plot, title_y, plot_name):
        fig = go.Figure()

        # Filtering the curves
        psi_filt = {}

        len_abs = len(psi)
        mod_plotoutput = len_abs / (numb_plotout - 1) # we have the initial plots, as well
        if mod_plotoutput == 0: mod_plotoutput = 1
        k_filt = -1
        K_all = 0
        for el in psi:
            new_k = round(K_all / mod_plotoutput)
            if new_k > k_filt:
                psi_filt[el] = psi[el]
            k_filt = new_k
            K_all += 1

        for t in psi_filt:
            sc = go.Scatter(x=psi_filt[t]['x'], y=psi_filt[t]['y'], name = str(t), mode="lines")
            fig.add_trace(sc)  # , row=1, col=1

            fig.update_layout(
                title={
                    'text': title_plot,
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title={
                    'text': 'x'
                },
                yaxis_title={
                    'text': title_y
                }
            )

        fig.write_image(plot_name)


    @staticmethod
    def __plot_moms_update_graph(t_list, moms_list, namem, title_plot, title_y, plot_name):
        fig_mom = go.Figure()

        sc_x = go.Scatter(x=t_list, y=moms_list[0], name=namem[0], mode="lines")
        sc_x2 = go.Scatter(x=t_list, y=moms_list[1], name=namem[1], mode="lines")
        sc_p = go.Scatter(x=t_list, y=moms_list[2], name=namem[2], mode="lines")

        fig_mom.add_trace(sc_x)
        fig_mom.add_trace(sc_x2)
        fig_mom.add_trace(sc_p)

        if len(moms_list) == 4:
            sc_p2 = go.Scatter(x=t_list, y=moms_list[3], name=namem[3], mode="lines")
            fig_mom.add_trace(sc_p2)

        fig_mom.update_layout(
            title={
                'text': title_plot,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'time'
            },
            yaxis_title={
                'text': title_y
            }
        )

        fig_mom.write_image(plot_name)

    @staticmethod
    def __plot_tvals_update_graph(t_list, val_list, title_plot, title_y, plot_name):
        fig = go.Figure()

        sc = go.Scatter(x=t_list, y=val_list, mode="lines")
        fig.add_trace(sc)

        fig.update_layout(
            title={
                'text': title_plot,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'time'
            },
            yaxis_title={
                'text': title_y
            }
        )

        fig.write_image(plot_name)


    def plot(self, psi, t, x, np):
        psi0_abs = []
        psi0_real = []
        for i in range(np):
            psi0_abs.append(abs(psi[0][i]))
            psi0_real.append(psi[0][i].real)

        self.psi_abs[t] = {'x': x, 'y': psi0_abs}
        self.psi_real[t] = {'x': x, 'y': psi0_real}

        if self.i % self.conf_output.mod_update == 0:
            # Updating the graph for psi_abs
            self.__plot_update_graph(self.psi_abs, self.conf_output.number_plotout,
                                     "Absolute value of the wave function on the ground state",
                                     "abs(Ψ)", os.path.join(self.conf_output.out_path, self.conf_output.gr_abs_grd))

            # Updating the graph for psi_real
            self.__plot_update_graph(self.psi_real, self.conf_output.number_plotout,
                                     "Real value of the wave function on the ground state",
                                     "Re(Ψ)", os.path.join(self.conf_output.out_path, self.conf_output.gr_real_grd))


    def plot_up(self, psi, t, x, np):
        psi1_abs = []
        psi1_real = []
        for i in range(np):
            psi1_abs.append(abs(psi[1][i]))
            psi1_real.append(psi[1][i].real)

        self.psi_abs_u[t] = {'x': x, 'y': psi1_abs}
        self.psi_real_u[t] = {'x': x, 'y': psi1_real}

        if self.i % self.conf_output.mod_update == 0:
            # Updating the graph for psi_abs
            self.__plot_update_graph(self.psi_abs_u, self.conf_output.number_plotout,
                                     "Absolute value of the wave function on the excited state",
                                     "abs(Ψ)", os.path.join(self.conf_output.out_path, self.conf_output.gr_abs_exc))

            # Updating the graph for psi_real
            self.__plot_update_graph(self.psi_real_u, self.conf_output.number_plotout,
                                     "Real value of the wave function on the excited state",
                                     "Re(Ψ)", os.path.join(self.conf_output.out_path, self.conf_output.gr_real_exc))


    def plot_mom(self, t, moms, ener, E, overlp, ener_tot, abs_psi_max, real_psi_max):
        self.t_list.append(t)
        self.x_l_list.append(moms.x_l.real)
        self.x2_l_list.append(moms.x2_l.real)
        self.p_l_list.append(moms.p_l.real)
        self.p2_l_list.append(moms.p2_l.real)
        self.ener_list.append(ener)
        self.E_list.append(E)
        self.overlp_list.append(abs(overlp))
        self.ener_tot_list.append(ener_tot)
        self.abs_psi_max_list.append(abs_psi_max)
        self.real_psi_max_list.append(real_psi_max)

        namem = ["<x>", "<x^2>", "<p>", "<p^2>"]
        moms_list = [self.x_l_list, self.x2_l_list, self.p_l_list]

        if self.i % self.conf_output.mod_update == 0:
            # Updating the graph for moms without <p^2>
            self.__plot_moms_update_graph(self.t_list, moms_list, namem,
                                          "Expectation values for the ground state", "",
                                          os.path.join(self.conf_output.out_path, self.conf_output.gr_moms_low_grd))

            moms_list.append(self.p2_l_list)
            # Updating the graph for moms
            self.__plot_moms_update_graph(self.t_list, moms_list, namem,
                                          "Expectation values for the ground state", "",
                                          os.path.join(self.conf_output.out_path, self.conf_output.gr_moms_grd))

            # Updating the graph for ener
            self.__plot_tvals_update_graph(self.t_list, self.ener_list,
                                           "Energy on the ground state", "Energy",
                                           os.path.join(self.conf_output.out_path, self.conf_output.gr_ener_grd))

            # Updating the graph for laser field energy
            self.__plot_tvals_update_graph(self.t_list, self.E_list,
                                           "Laser field energy envelope", "E",
                                           os.path.join(self.conf_output.out_path, self.conf_output.gr_lf_en))

            # Updating the graph for lower state population
            self.__plot_tvals_update_graph(self.t_list, self.overlp_list,
                                           "Ground state population", "abs((psi0, psi))",
                                           os.path.join(self.conf_output.out_path, self.conf_output.gr_overlp_grd))

            # Updating the graph for total energy
            self.__plot_tvals_update_graph(self.t_list, self.ener_tot_list,
                                           "Total energy", "Total energy",
                                           os.path.join(self.conf_output.out_path, self.conf_output.gr_ener_tot))

            # Updating the graph for maximum absolute value of ground state wavefunction
            self.__plot_tvals_update_graph(self.t_list, self.abs_psi_max_list,
                                           "Time dependence of max|Ψ| for the ground state wavefunction", "max|Ψ|",
                                           os.path.join(self.conf_output.out_path, self.conf_output.gr_abs_max_grd))

            # Updating the graph for maximum real value of ground state wavefunction
            self.__plot_tvals_update_graph(self.t_list, self.real_psi_max_list,
                                           "Time dependence of maximum value of real(Ψ) for the ground state wavefunction",
                                           "real(Ψ)", os.path.join(self.conf_output.out_path, self.conf_output.gr_real_max_grd))


    def plot_mom_up(self, t, moms, ener, E, overlp, overlp_tot, abs_psi_max, real_psi_max):
        self.t_u_list.append(t)
        self.x_u_list.append(moms.x_u.real)
        self.x2_u_list.append(moms.x2_u.real)
        self.p_u_list.append(moms.p_u.real)
        self.p2_u_list.append(moms.p2_u.real)
        self.ener_u_list.append(ener)
        self.overlp_u_list.append(abs(overlp))
        self.overlp_tot_list.append(overlp_tot)
        self.abs_psi_max_u_list.append(abs_psi_max)
        self.real_psi_max_u_list.append(real_psi_max)

        namem = ["<x>", "<x^2>", "<p>", "<p^2>"]
        moms_list = [self.x_u_list, self.x2_u_list, self.p_u_list]

        if self.i % self.conf_output.mod_update == 0:
            # Updating the graph for moms
            self.__plot_moms_update_graph(self.t_u_list, moms_list, namem,
                                          "Expectation values for the excited state", "",
                                          os.path.join(self.conf_output.out_path, self.conf_output.gr_moms_low_exc))

            moms_list.append(self.p2_u_list)
            # Updating the graph for moms without <p^2>
            self.__plot_moms_update_graph(self.t_u_list, moms_list, namem,
                                          "Expectation values for the excited state", "",
                                          os.path.join(self.conf_output.out_path, self.conf_output.gr_moms_exc))

            # Updating the graph for ener
            self.__plot_tvals_update_graph(self.t_u_list, self.ener_u_list,
                                           "Energy on the excited state", "Energy",
                                           os.path.join(self.conf_output.out_path, self.conf_output.gr_ener_exc))

            # Updating the graph for excited state population
            self.__plot_tvals_update_graph(self.t_u_list, self.overlp_u_list,
                                           "Excited state population", "abs((psi1, psi))",
                                           os.path.join(self.conf_output.out_path, self.conf_output.gr_overlp_exc))

            # Updating the graph for total population
            self.__plot_tvals_update_graph(self.t_u_list, self.overlp_tot_list,
                                           "Total population", "Total population",
                                           os.path.join(self.conf_output.out_path, self.conf_output.gr_overlp_tot))

            # Updating the graph for maximum absolute value of excited state wavefunction
            self.__plot_tvals_update_graph(self.t_u_list, self.abs_psi_max_u_list,
                                           "Time dependence of max|Ψ| for the excited state wavefunction", "max|Ψ|",
                                           os.path.join(self.conf_output.out_path, self.conf_output.gr_abs_max_exc))

            # Updating the graph for maximum real value of excited state wavefunction
            self.__plot_tvals_update_graph(self.t_u_list, self.real_psi_max_u_list,
                                           "Time dependence of maximum value of real(Ψ) for the excited state wavefunction",
                                           "real(Ψ)", os.path.join(self.conf_output.out_path, self.conf_output.gr_real_max_exc))


    def print_time_point(self, l, psi, t, x, np, moms, ener, ener_u, E, overlp, overlp_u, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u):
        try:
            if l % self.conf_output.mod_plotout == 0 and l >= self.conf_output.lmin:
                self.plot(psi, t, x, np)
                self.plot_up(psi, t, x, np)
                self.plot_mom(t, moms, ener, E, overlp, ener_tot, abs_psi_max, real_psi_max)
                self.plot_mom_up(t, moms, ener_u, E, overlp_u, overlp_tot, abs_psi_max_u, real_psi_max_u)
                self.i += 1
        except ValueError as err:
            print_err("A nasty error has occurred during the reporting: ", err)
            print_err("Hopefully that doesn't affect the calculations, so the application is going on...")


class MultipleReporter(Reporter):
    def __init__(self, conf_output):
        self.reps = []
        if not conf_output.plot.is_empty():
            self.reps.append(PlotReporter(conf_output.plot))
        if not conf_output.table.is_empty():
            self.reps.append(TableReporter(conf_output.table))

    def __enter__(self):
        for rep in self.reps:
            rep.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def plot(self, psi, t, x, np):
        for rep in self.reps:
            rep.plot(self, psi, t, x, np)


    def plot_mom(self, t, moms, ener, E, overlp, ener_tot, abs_psi_max, real_psi_max):
        for rep in self.reps:
            rep.plot_mom(self, t, moms, ener, E, overlp, ener_tot, abs_psi_max, real_psi_max)


    def plot_up(self, psi, t, x, np):
        for rep in self.reps:
            rep.plot_up(self, psi, t, x, np)


    def plot_mom_up(self, t, moms, ener, E, overlp, overlp_tot, abs_psi_max, real_psi_max):
        for rep in self.reps:
            rep.plot_mom(self, t, moms, ener, E, overlp, overlp_tot, abs_psi_max, real_psi_max)


    def print_time_point(self, l, psi, t, x, np, moms, ener, ener_u, E, overlp, overlp_u, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u):
        for rep in self.reps:
            rep.print_time_point(l, psi, t, x, np, moms, ener, ener_u, E, overlp, overlp_u, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u)

