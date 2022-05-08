import copy

# Disable the orca response timeout.
from typing import List

import plotly.io._orca
import retrying

from psi_basis import Psi

unwrapped = plotly.io._orca.request_image_with_retrying.__wrapped__
wrapped = retrying.retry(wait_random_min=1000)(unwrapped)
plotly.io._orca.request_image_with_retrying = wrapped

import plotly.graph_objects as go
import os.path

from tools import print_err

import config


class PropagationReporter:
    def __init__(self, out_path: str):
        self._out_path = out_path

    def open(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def print_time_point_prop(self, l, psi: Psi, t, x, np, moms, ener, ener_u, overlp0, overlpf, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u, E, freq_mult):
        raise NotImplementedError()


class TablePropagationReporter(PropagationReporter):
    def __init__(self, out_path: str,
                 conf: config.ReportRootConfiguration.ReportFitterConfiguration.ReportTablePropagationConfiguration):
        super().__init__(out_path=out_path)
        self.conf = conf
        self.f_abs = None
        self.f_real = None
        self.f_prop = None
        self.f_abs_up = None
        self.f_real_up = None
        self.f_prop_up = None
        self.f_fit = None

    @staticmethod
    def name_template(name: str, level: int):
        return name.replace("{level}", str(level))

    def open(self):
        if not os.path.exists(self._out_path):
            os.mkdir(self._out_path)

        self.f_abs = open(os.path.join(self._out_path, self.name_template(self.conf.tab_abs, 0)), 'w')
        self.f_real = open(os.path.join(self._out_path, self.name_template(self.conf.tab_real, 0)), 'w')
        self.f_prop = open(os.path.join(self._out_path, self.name_template(self.conf.tab_tvals, 0)), 'w')
        self.f_abs_up = open(os.path.join(self._out_path, self.name_template(self.conf.tab_abs, 1)), 'w')
        self.f_real_up = open(os.path.join(self._out_path, self.name_template(self.conf.tab_real, 1)), 'w')
        self.f_prop_up = open(os.path.join(self._out_path, self.name_template(self.conf.tab_tvals, 1)), 'w')
        self.f_fit = open(os.path.join(self._out_path, self.conf.tab_tvals_fit), 'w')
        return self


    def close(self):
        self.f_abs.close()
        self.f_abs = None
        self.f_real.close()
        self.f_real = None
        self.f_prop.close()
        self.f_prop = None
        self.f_abs_up.close()
        self.f_abs_up = None
        self.f_real_up.close()
        self.f_real_up = None
        self.f_prop_up.close()
        self.f_prop_up = None
        self.f_fit.close()
        self.f_fit = None


    @staticmethod
    def __plot_file(psi, t, x, np, f_abs, f_real):
        """ Plots absolute and real values of the current wavefunction """
        for i in range(np):
            f_abs.write("{:.6f} {:.6f} {:.6e}\n".format(t * 1e+15, x[i], abs(psi[i])))
            f_real.write("{:.6f} {:.6f} {:.6e}\n".format(t * 1e+15, x[i], psi[i].real))
            f_abs.flush()
            f_real.flush()


    @staticmethod
    def __plot_t_file_prop(t, momx, momx2, momp, momp2, ener, overlp0, overlpf, abs_psi_max, real_psi_max, file_prop):
        """ Plots expectation values of the current x, x*x, p and p*p, and other values,
        which are modified by propagation, as a function of time """
        file_prop.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
            t * 1e+15, momx.real, momx2.real, momp.real, momp2.real, ener, abs(overlp0), abs(overlpf), abs_psi_max,
            real_psi_max))
        file_prop.flush()


    @staticmethod
    def __plot_t_file_fitter(t, E, freq_mult, ener_tot, overlp_tot, file_fit):
        """ Plots the values, which are modified by fitter, as a function of time """
        file_fit.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(t * 1e+15, E, freq_mult, ener_tot, overlp_tot[0], overlp_tot[1]))
        file_fit.flush()


    def plot(self, psi: Psi, t, x, np):
        self.__plot_file(psi.f[0], t, x, np, self.f_abs, self.f_real)

    def plot_prop(self, t, moms, ener, overlp0, overlpf, abs_psi_max, real_psi_max):
        self.__plot_t_file_prop(t, moms.x_l, moms.x2_l, moms.p_l, moms.p2_l, ener, overlp0, overlpf,
                             abs_psi_max, real_psi_max, self.f_prop)

    def plot_up(self, psi: Psi, t, x, np):
        self.__plot_file(psi.f[1], t, x, np, self.f_abs_up, self.f_real_up)

    def plot_prop_up(self, t, moms, ener, overlp0, overlpf, abs_psi_max, real_psi_max):
        self.__plot_t_file_prop(t, moms.x_u, moms.x2_u, moms.p_u, moms.p2_u, ener, overlp0, overlpf,
                             abs_psi_max, real_psi_max, self.f_prop_up)

    def plot_fitter(self, t, E, freq_mult, ener_tot, overlp_tot):
        self.__plot_t_file_fitter(t, E, freq_mult, ener_tot, overlp_tot, self.f_fit)


    @staticmethod
    def __plot_test_file(l, phi_l, phi_u, f):
        f.write("Step number: {0}\n".format(l))
        f.write("Lower state wavefunction:")
        for i in range(len(phi_l)):
            f.write("{0}\n".format(phi_l[i]))
        f.write("Upper state wavefunction:")
        for i in range(len(phi_u)):
            f.write("{0}\n".format(phi_u[i]))


    def print_time_point_prop(self, l, psi: Psi, t, x, np, moms, ener, ener_u, overlp0, overlpf, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u, E, freq_mult):
        if l % self.conf.mod_fileout == 0 and l >= self.conf.lmin:
            self.plot(psi, t, x, np)
            self.plot_up(psi, t, x, np)
            self.plot_prop(t, moms, ener, overlp0[0], overlpf[0], abs_psi_max, real_psi_max)
            self.plot_prop_up(t, moms, ener_u, overlp0[1], overlpf[1], abs_psi_max_u, real_psi_max_u)
            self.plot_fitter(t, E, freq_mult, ener_tot, overlp_tot)


class PlotPropagationReporter(PropagationReporter):
    def __init__(self, out_path: str,
                 conf: config.ReportRootConfiguration.ReportFitterConfiguration.ReportPlotPropagationConfiguration):
        super().__init__(out_path=out_path)
        self.conf = conf

    def open(self):
        if not os.path.exists(self._out_path):
            os.mkdir(self._out_path)

        # Time
        self.t_list = []
        self.t_u_list = []
        self.t_fit_list = []

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
        self.overlp0_list = []
        self.overlp0_u_list = []
        self.overlpf_list = []
        self.overlpf_u_list = []
        self.ener_tot_list = []
        self.overlp0_tot_list = []
        self.overlpf_tot_list = []
        self.abs_psi_max_list = []
        self.real_psi_max_list = []
        self.overlp_tot = []
        self.abs_psi_max_u_list = []
        self.real_psi_max_u_list = []

        self.E_list = []
        self.freq_mult_list = []

        # X = Coordinate
        self.psi_abs = {}  # key: t, value: {'x': [], 'y': []}
        self.psi_real = {}  # key: t, value: {'x': [], 'y': []}
        self.psi_abs_u = {}
        self.psi_real_u = {}

        self.i = 0

        return self

    def close(self):
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


    def plot(self, psi:Psi, t, x, np):
        psi0_abs = []
        psi0_real = []
        for i in range(np):
            psi0_abs.append(abs(psi.f[0][i]))
            psi0_real.append(psi.f[0][i].real)

        self.psi_abs[t] = {'x': x, 'y': psi0_abs}
        self.psi_real[t] = {'x': x, 'y': psi0_real}

        if self.i % self.conf.mod_update == 0:
            # Updating the graph for psi_abs
            self.__plot_update_graph(self.psi_abs, self.conf.number_plotout,
                                     "Absolute value of the wave function on the ground state",
                                     "abs(Ψ)", os.path.join(self._out_path, self.conf.gr_abs_grd))

            # Updating the graph for psi_real
            self.__plot_update_graph(self.psi_real, self.conf.number_plotout,
                                     "Real value of the wave function on the ground state",
                                     "Re(Ψ)", os.path.join(self._out_path, self.conf.gr_real_grd))


    def plot_up(self, psi: Psi, t, x, np):
        psi1_abs = []
        psi1_real = []
        for i in range(np):
            psi1_abs.append(abs(psi.f[1][i]))
            psi1_real.append(psi.f[1][i].real)

        self.psi_abs_u[t] = {'x': x, 'y': psi1_abs}
        self.psi_real_u[t] = {'x': x, 'y': psi1_real}

        if self.i % self.conf.mod_update == 0:
            # Updating the graph for psi_abs
            self.__plot_update_graph(self.psi_abs_u, self.conf.number_plotout,
                                     "Absolute value of the wave function on the excited state",
                                     "abs(Ψ)", os.path.join(self._out_path, self.conf.gr_abs_exc))

            # Updating the graph for psi_real
            self.__plot_update_graph(self.psi_real_u, self.conf.number_plotout,
                                     "Real value of the wave function on the excited state",
                                     "Re(Ψ)", os.path.join(self._out_path, self.conf.gr_real_exc))


    def plot_prop(self, t, moms, ener, overlp0, overlpf, abs_psi_max, real_psi_max):
        self.t_list.append(t)
        self.x_l_list.append(moms.x_l.real)
        self.x2_l_list.append(moms.x2_l.real)
        self.p_l_list.append(moms.p_l.real)
        self.p2_l_list.append(moms.p2_l.real)
        self.ener_list.append(ener)
        self.overlp0_list.append(abs(overlp0))
        self.overlpf_list.append(abs(overlpf))
        self.abs_psi_max_list.append(abs_psi_max)
        self.real_psi_max_list.append(real_psi_max)

        namem = ["<x>", "<x^2>", "<p>", "<p^2>"]
        moms_list = [self.x_l_list, self.x2_l_list, self.p_l_list]

        if self.i % self.conf.mod_update == 0:
            # Updating the graph for moms without <p^2>
            self.__plot_moms_update_graph(self.t_list, moms_list, namem,
                                          "Expectation values for the ground state", "",
                                          os.path.join(self._out_path, self.conf.gr_moms_low_grd))

            moms_list.append(self.p2_l_list)
            # Updating the graph for moms
            self.__plot_moms_update_graph(self.t_list, moms_list, namem,
                                          "Expectation values for the ground state", "",
                                          os.path.join(self._out_path, self.conf.gr_moms_grd))

            # Updating the graph for ener
            self.__plot_tvals_update_graph(self.t_list, self.ener_list,
                                           "Energy on the ground state", "Energy",
                                           os.path.join(self._out_path, self.conf.gr_ener_grd))

            # Updating the graph for lower state population
            self.__plot_tvals_update_graph(self.t_list, self.overlp0_list,
                                           "Ground state population", "abs((psi0, psi))",
                                           os.path.join(self._out_path, self.conf.gr_overlp0_grd))

            self.__plot_tvals_update_graph(self.t_list, self.overlpf_list,
                                           "Ground state population", "abs((psif, psi))",
                                           os.path.join(self._out_path, self.conf.gr_overlpf_grd))

            # Updating the graph for maximum absolute value of ground state wavefunction
            self.__plot_tvals_update_graph(self.t_list, self.abs_psi_max_list,
                                           "Time dependence of max|Ψ| for the ground state wavefunction", "max|Ψ|",
                                           os.path.join(self._out_path, self.conf.gr_abs_max_grd))

            # Updating the graph for maximum real value of ground state wavefunction
            self.__plot_tvals_update_graph(self.t_list, self.real_psi_max_list,
                                           "Time dependence of maximum value of real(Ψ) for the ground state wavefunction",
                                           "real(Ψ)", os.path.join(self._out_path, self.conf.gr_real_max_grd))


    def plot_prop_up(self, t, moms, ener, overlp0, overlpf, abs_psi_max, real_psi_max):
        self.t_u_list.append(t)
        self.x_u_list.append(moms.x_u.real)
        self.x2_u_list.append(moms.x2_u.real)
        self.p_u_list.append(moms.p_u.real)
        self.p2_u_list.append(moms.p2_u.real)
        self.ener_u_list.append(ener)
        self.overlp0_u_list.append(abs(overlp0))
        self.overlpf_u_list.append(abs(overlpf))
        self.abs_psi_max_u_list.append(abs_psi_max)
        self.real_psi_max_u_list.append(real_psi_max)

        namem = ["<x>", "<x^2>", "<p>", "<p^2>"]
        moms_list = [self.x_u_list, self.x2_u_list, self.p_u_list]

        if self.i % self.conf.mod_update == 0:
            # Updating the graph for moms
            self.__plot_moms_update_graph(self.t_u_list, moms_list, namem,
                                          "Expectation values for the excited state", "",
                                          os.path.join(self._out_path, self.conf.gr_moms_low_exc))

            moms_list.append(self.p2_u_list)
            # Updating the graph for moms without <p^2>
            self.__plot_moms_update_graph(self.t_u_list, moms_list, namem,
                                          "Expectation values for the excited state", "",
                                          os.path.join(self._out_path, self.conf.gr_moms_exc))

            # Updating the graph for ener
            self.__plot_tvals_update_graph(self.t_u_list, self.ener_u_list,
                                           "Energy on the excited state", "Energy",
                                           os.path.join(self._out_path, self.conf.gr_ener_exc))

            # Updating the graph for excited state population
            self.__plot_tvals_update_graph(self.t_u_list, self.overlp0_u_list,
                                           "Excited state population", "abs((psi0, psi))",
                                           os.path.join(self._out_path, self.conf.gr_overlp0_exc))

            self.__plot_tvals_update_graph(self.t_u_list, self.overlpf_u_list,
                                           "Excited state population", "abs((psif, psi))",
                                           os.path.join(self._out_path, self.conf.gr_overlpf_exc))

            # Updating the graph for maximum absolute value of excited state wavefunction
            self.__plot_tvals_update_graph(self.t_u_list, self.abs_psi_max_u_list,
                                           "Time dependence of max|Ψ| for the excited state wavefunction", "max|Ψ|",
                                           os.path.join(self._out_path, self.conf.gr_abs_max_exc))

            # Updating the graph for maximum real value of excited state wavefunction
            self.__plot_tvals_update_graph(self.t_u_list, self.real_psi_max_u_list,
                                           "Time dependence of maximum value of real(Ψ) for the excited state wavefunction",
                                           "real(Ψ)", os.path.join(self._out_path, self.conf.gr_real_max_exc))


    def plot_fitter(self, t, E, freq_mult, ener_tot, overlp_tot):
        self.t_fit_list.append(t)
        self.E_list.append(E)
        self.freq_mult_list.append(freq_mult)
        self.ener_tot_list.append(ener_tot)
        self.overlp0_tot_list.append(overlp_tot[0])
        self.overlpf_tot_list.append(overlp_tot[1])

        if self.i % self.conf.mod_update == 0:
            # Updating the graph for laser field energy
            self.__plot_tvals_update_graph(self.t_fit_list, self.E_list,
                                           "Laser field energy envelope", "E",
                                           os.path.join(self._out_path, self.conf.gr_lf_en))

            # Updating the graph for laser field frequency multiplier
            self.__plot_tvals_update_graph(self.t_fit_list, self.freq_mult_list,
                                           "Laser field frequency multiplier", "f",
                                           os.path.join(self._out_path, self.conf.gr_lf_fr))

            # Updating the graph for total energy
            self.__plot_tvals_update_graph(self.t_fit_list, self.ener_tot_list,
                                           "Total energy", "Total energy",
                                           os.path.join(self._out_path, self.conf.gr_ener_tot))

            # Updating the graph for total population
            self.__plot_tvals_update_graph(self.t_fit_list, self.overlp0_tot_list,
                                           "Closeness to the initial state", "(Ψ, Ψ_init)",
                                           os.path.join(self._out_path, self.conf.gr_overlp0_tot))

            self.__plot_tvals_update_graph(self.t_fit_list, self.overlpf_tot_list,
                                           "Closeness to the goal state", "(Ψ, Ψ_goal)",
                                           os.path.join(self._out_path, self.conf.gr_overlpf_tot))


    def print_time_point_prop(self, l, psi: Psi, t, x, np, moms, ener, ener_u, overlp0, overlpf, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u, E, freq_mult):
        try:
            if l % self.conf.mod_plotout == 0 and l >= self.conf.lmin:
                self.plot(psi, t, x, np)
                self.plot_up(psi, t, x, np)
                self.plot_prop(t, moms, ener, overlp0[0], overlpf[0], abs_psi_max, real_psi_max)
                self.plot_prop_up(t, moms, ener_u, overlp0[1], overlpf[1], abs_psi_max_u, real_psi_max_u)
                self.plot_fitter(t, E, freq_mult, ener_tot, overlp_tot)
                self.i += 1
        except ValueError as err:
            print_err("A nasty error has occurred during the reporting: ", err)
            print_err("Hopefully that doesn't affect the calculations, so the application is going on...")


class MultiplePropagationReporter(PropagationReporter):
    reps: List[PropagationReporter]

    def __init__(self, out_path: str, conf_rep_table, conf_rep_plot):
        super(MultiplePropagationReporter, self).__init__(out_path=out_path)
        self.reps = []
        if not conf_rep_plot.is_empty():
            self.reps.append(PlotPropagationReporter(conf=conf_rep_plot, out_path=os.path.join(self._out_path, "plots")))
        if not conf_rep_table.is_empty():
            self.reps.append(TablePropagationReporter(conf=conf_rep_table, out_path=os.path.join(self._out_path, "tables")))

    def open(self):
        for rep in self.reps:
            rep.open()
        return self

    def close(self):
        pass

    def print_time_point_prop(self, l, psi: Psi, t, x, np, moms, ener, ener_u, overlp0, overlpf, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u, E, freq_mult):
        for rep in self.reps:
            rep.print_time_point_prop(l, psi, t, x, np, moms, ener, ener_u, overlp0, overlpf, overlp_tot, ener_tot,
                         abs_psi_max, real_psi_max, abs_psi_max_u, real_psi_max_u, E, freq_mult)


# FitterReporter

class FitterReporter:
    def __init__(self):
        pass

    def open(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def print_iter_point_fitter(self, iter, goal_close, E_tlist, t_list, nt):
        raise NotImplementedError()

    def create_propagation_reporter(self, prop_id: str):
        raise NotImplementedError()


class TableFitterReporter(FitterReporter):
    def __init__(self, conf: config.ReportRootConfiguration.ReportTableFitterConfiguration):
        super().__init__()
        self.conf = conf
        self.f_ifit = None
        self.f_ifit_E = None

    def create_propagation_reporter(self, prop_id: str):
        prop_conf_output = copy.deepcopy(self.conf)
        return TablePropagationReporter(out_path=os.path.join(prop_conf_output.out_path, prop_id),
                                        conf=prop_conf_output.propagation)

    def open(self):
        if not os.path.exists(self.conf.out_path):
            os.mkdir(self.conf.out_path)

        self.f_ifit = open(os.path.join(self.conf.out_path, self.conf.tab_iter), 'w')
        self.f_ifit_E = open(os.path.join(self.conf.out_path, self.conf.tab_iter_E), 'w')
        return self

    def close(self):
        self.f_ifit.close()
        self.f_ifit = None
        self.f_ifit_E.close()
        self.f_ifit_E = None
        pass


    @staticmethod
    def __plot_i_file_fitter(iter, goal_close, file_ifit):
        """ Plots the values, which are modified by fitter, as a function of iteration """
        file_ifit.write("{:2d} {:.6f} \n".format(int(iter), goal_close))
        file_ifit.flush()

    @staticmethod
    def __plot_i_file_E(E_tlist, iter, t_list, nt, f_ifit_E):
        """ Plots laser field energy envelope on the current iteration """
        for i in range(nt + 1):
            f_ifit_E.write("{:2d} {:.6f} {:.6e}\n".format(iter, t_list[i] * 1e+15, abs(E_tlist[i])))
            f_ifit_E.flush()


    def plot_i_E(self, E_tlist, iter, t_list, nt):
        self.__plot_i_file_E(E_tlist, iter, t_list, nt, self.f_ifit_E)

    def plot_fitter(self, iter, goal_close):
        self.__plot_i_file_fitter(iter, goal_close, self.f_ifit)

    def print_iter_point_fitter(self, iter, goal_close, E_tlist, t_list, nt):
        if iter % self.conf.imod_fileout == 0 and iter >= self.conf.imin:
            self.plot_fitter(iter, goal_close)
            self.plot_i_E(E_tlist, iter, t_list, nt)


class PlotFitterReporter(FitterReporter):
    def __init__(self, conf: config.ReportRootConfiguration.ReportPlotFitterConfiguration):
        super().__init__()
        self.conf = conf

    def create_propagation_reporter(self, prop_id: str):
        prop_conf_output = copy.deepcopy(self.conf)
        return PlotPropagationReporter(out_path=os.path.join(prop_conf_output.out_path, prop_id),
                                       conf=prop_conf_output.propagation)

    def open(self):
        # Iterations
        self.i_list = []
        self.gc_list = []

        # t = Coordinate
        self.E_abs = {}  # key: iter, value: {'t': [], 'y': []}
        return self

    def close(self):
        pass


    @staticmethod
    def __plot_iter_time_update_graph(E_tlist, numb_plotout, title_plot, title_y, plot_name):
        fig = go.Figure()

        # Filtering the curves
        E_filt = {}

        len_E = len(E_tlist)
        mod_plotoutput = len_E / (numb_plotout - 1) # we have the initial plots, as well
        if mod_plotoutput == 0: mod_plotoutput = 1
        k_filt = -1
        K_all = 0
        for el in E_tlist:
            new_k = round(K_all / mod_plotoutput)
            if new_k > k_filt:
                E_filt[el] = E_tlist[el]
            k_filt = new_k
            K_all += 1

        for i in E_filt:
            sc = go.Scatter(x=E_filt[i]['t'], y=E_filt[i]['y'], name = str(i), mode="lines")
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
                    'text': 'time'
                },
                yaxis_title={
                    'text': title_y
                }
            )

        fig.write_image(plot_name)

    @staticmethod
    def __plot_iter_update_graph(i_list, val_list, title_plot, title_y, plot_name):
        fig = go.Figure()

        sc = go.Scatter(x=i_list, y=val_list, mode="lines")
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
                'text': 'Iteration'
            },
            yaxis_title={
                'text': title_y
            }
        )

        fig.write_image(plot_name)


    def plot_fitter(self, iter, goal_close):
        self.i_list.append(iter)
        self.gc_list.append(goal_close)

        # Updating the graph for closeness of the result to the goal
        self.__plot_iter_update_graph(self.i_list, self.gc_list,
                                       "Closeness of the current result to the goal", "(Ψ, Ψ_goal)",
                                       os.path.join(self.conf.out_path, self.conf.gr_iter))

    def plot_E(self, E, iter, t, nt):
        E_abs = []
        for i in range(nt + 1):
            E_abs.append(abs(E[i]))

        self.E_abs[iter] = {'t': t, 'y': E_abs}

        # Updating the graph for E_abs
        self.__plot_iter_time_update_graph(self.E_abs, self.conf.inumber_plotout,
                                 "Absolute value of the laser field envelope",
                                 "abs(E)", os.path.join(self.conf.out_path, self.conf.gr_iter_E))


    def print_iter_point_fitter(self, iter, goal_close, E_tlist, t_list, nt):
        try:
            if iter % self.conf.imod_plotout == 0 and iter >= self.conf.imin:
                self.plot_fitter(iter, goal_close)
                self.plot_E(E_tlist, iter, t_list, nt)

        except ValueError as err:
            print_err("A nasty error has occurred during the reporting: ", err)
            print_err("Hopefully that doesn't affect the calculations, so the application is going on...")


class MultipleFitterReporter(FitterReporter):
    def __init__(self, conf_rep_table, conf_rep_plot):
        super(MultipleFitterReporter, self).__init__()

        self.reps = []
        if not conf_rep_plot.is_empty():
            self.reps.append(PlotFitterReporter(conf_rep_plot))
        if not conf_rep_table.is_empty():
            self.reps.append(TableFitterReporter(conf_rep_table))
        self.conf_rep_table = conf_rep_table
        self.conf_rep_plot = conf_rep_plot

    def create_propagation_reporter(self, prop_id: str):
        prop_conf_rep_table = copy.deepcopy(self.conf_rep_table)
        prop_conf_rep_plot = copy.deepcopy(self.conf_rep_plot)

        assert(prop_conf_rep_plot.out_path == prop_conf_rep_table.out_path)
        out_path = prop_conf_rep_table.out_path
        prop_out_path = os.path.join(out_path, prop_id)
        if not os.path.exists(prop_out_path):
            os.makedirs(prop_out_path)

        return MultiplePropagationReporter(out_path=prop_out_path,
                                           conf_rep_table=prop_conf_rep_table.propagation,
                                           conf_rep_plot=prop_conf_rep_plot.propagation)

    def open(self):
        for rep in self.reps:
            rep.open()
        return self

    def close(self):
        pass

    def print_iter_point_fitter(self, iter, goal_close, E_tlist, t_list, nt):
        for rep in self.reps:
            rep.print_iter_point_fitter(iter, goal_close, E_tlist, t_list, nt)
