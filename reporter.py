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
    def __init__(self, out_path: str, nlvls: int):
        self._out_path = out_path
        self._nlvls = nlvls

    def open(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def print_time_point_prop(self, l, psi: Psi, t, x, np, moms, ener, overlp0, overlpf, overlp_tot, ener_tot,
                              psi_max_abs, psi_max_rel, E, freq_mult):
        raise NotImplementedError()


class TablePropagationReporter(PropagationReporter):
    def __init__(self, out_path: str, nlvls: int,
                 conf: config.ReportRootConfiguration.ReportFitterConfiguration.ReportTablePropagationConfiguration):
        super().__init__(out_path=out_path, nlvls=nlvls)
        self.conf = conf
        self.f_abs = [None] * nlvls
        self.f_real = [None] * nlvls
        self.f_prop = [None] * nlvls
        self.f_fit = None

    @staticmethod
    def name_template(name: str, level: int):
        return name.replace("{level}", str(level))

    def open(self):
        if not os.path.exists(self._out_path):
            os.mkdir(self._out_path)

        for n in range(self._nlvls):
            self.f_abs[n] = open(os.path.join(self._out_path, self.name_template(self.conf.tab_abs, n)), 'w')
            self.f_real[n] = open(os.path.join(self._out_path, self.name_template(self.conf.tab_real, n)), 'w')
            self.f_prop[n] = open(os.path.join(self._out_path, self.name_template(self.conf.tab_tvals, n)), 'w')
        self.f_fit = open(os.path.join(self._out_path, self.conf.tab_tvals_fit), 'w')
        return self


    def close(self):
        for n in range(self._nlvls):
            self.f_abs[n].close()
            self.f_abs[n] = None
            self.f_real[n].close()
            self.f_real[n] = None
            self.f_prop[n].close()
            self.f_prop[n] = None
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
    def __plot_t_file_prop(t, momx, momx2, momp, momp2, ener, overlp0, overlpf, psi_max_abs, psi_max_real, file_prop):
        """ Plots expectation values of the current x, x*x, p and p*p, and other values,
        which are modified by propagation, as a function of time """
        file_prop.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
            t * 1e+15, momx.real, momx2.real, momp.real, momp2.real, ener.real, abs(overlp0), abs(overlpf), psi_max_abs,
            psi_max_real))
        file_prop.flush()


    @staticmethod
    def __plot_t_file_fitter(t, E, freq_mult, ener_tot, overlp_tot, file_fit):
        """ Plots the values, which are modified by fitter, as a function of time """
        file_fit.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(t * 1e+15, abs(E), freq_mult, ener_tot.real, abs(overlp_tot[0]), abs(overlp_tot[1])))
        file_fit.flush()

    def plot(self, psi: Psi, t, x, np):
        for n in range(self._nlvls):
            self.__plot_file(psi.f[n], t, x, np, self.f_abs[n], self.f_real[n])

    def plot_prop(self, t, moms, ener, overlp0, overlpf, psi_max_abs, psi_max_real):
        for n in range(self._nlvls):
            self.__plot_t_file_prop(t, moms.x[n], moms.x2[n], moms.p[n], moms.p2[n], ener[n], overlp0[n], overlpf[n],
                             psi_max_abs[n], psi_max_real[n], self.f_prop[n])

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


    def print_time_point_prop(self, l, psi: Psi, t, x, np, moms, ener, overlp0, overlpf, overlp_tot, ener_tot,
                              psi_max_abs, psi_max_real, E, freq_mult):
        if l % self.conf.mod_fileout == 0 and l >= self.conf.lmin:
            self.plot(psi, t, x, np)
            self.plot_prop(t, moms, ener, overlp0, overlpf, psi_max_abs, psi_max_real)
            self.plot_fitter(t, E, freq_mult, ener_tot, overlp_tot)


class PlotPropagationReporter(PropagationReporter):
    def __init__(self, out_path: str, nlvls: int,
                 conf: config.ReportRootConfiguration.ReportFitterConfiguration.ReportPlotPropagationConfiguration):
        super().__init__(out_path=out_path, nlvls=nlvls)
        self.conf = conf

    def open(self):
        if not os.path.exists(self._out_path):
            os.mkdir(self._out_path)

        # Time
        self.t_list = []
        self.t_moms_list = [[] for i in range(self._nlvls)]
        self.t_fit_list = []

        # Coordinate
        self.x_list = []

        # X = Time
        self.x_list = [[] for i in range(self._nlvls)]
        self.x2_list = [[] for i in range(self._nlvls)]
        self.p_list = [[] for i in range(self._nlvls)]
        self.p2_list = [[] for i in range(self._nlvls)]

        self.ener_list = []
        self.overlp0_list = []
        self.overlpf_list = []
        self.abs_psi_max_list = []
        self.real_psi_max_list = []

        self.ener_tot_list = []
        self.overlp0_tot_list = []
        self.overlpf_tot_list = []
        self.E_list = []
        self.freq_mult_list = []

        # X = Coordinate
        self.psi_abs = [dict() for i in range(self._nlvls)]  # key: t, value: {'x': [], 'y': []}
        self.psi_real = [dict() for i in range(self._nlvls)]  # key: t, value: {'x': [], 'y': []}

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


    @staticmethod
    def __plot_tvals_mult_update_graph(t_list, vals_list, nlvls, title_plot, title_y, plot_name):
        fig_vals = go.Figure()

        vals_t_list = [[0.0] * len(t_list) for i in range(nlvls)]
        for nt in range(len(t_list)):
            for n in range(nlvls):
                vals_t_list[n][nt] = vals_list[nt][n]

        for n in range(nlvls):
            sc = go.Scatter(x=t_list, y=vals_t_list[n], name="level #" + str(n), mode="lines")
            fig_vals.add_trace(sc)

        fig_vals.update_layout(
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

        fig_vals.write_image(plot_name)


    def plot(self, psi:Psi, t, x, np, n):
        psi0_abs = []
        psi0_real = []
        for i in range(np):
            psi0_abs.append(abs(psi.f[n][i]))
            psi0_real.append(psi.f[n][i].real)

        self.psi_abs[n][t] = {'x': x, 'y': psi0_abs}
        self.psi_real[n][t] = {'x': x, 'y': psi0_real}

        if self.i % self.conf.mod_update == 0:
            # Updating the graph for psi_abs
            self.__plot_update_graph(self.psi_abs[n], self.conf.number_plotout,
                                    "Absolute value of the wave function on the state #%d" % n,
                                    "abs(Ψ)", os.path.join(self._out_path, self.conf.gr_abs.replace("{level}", str(n))))

            # Updating the graph for psi_real
            self.__plot_update_graph(self.psi_real[n], self.conf.number_plotout,
                                    "Real value of the wave function on the state #%d" % n,
                                    "Re(Ψ)", os.path.join(self._out_path, self.conf.gr_real.replace("{level}", str(n))))

    def plot_moms_prop(self, t, moms, n):
        self.t_moms_list[n].append(t)
        self.x_list[n].append(moms.x[n].real)
        self.x2_list[n].append(moms.x2[n].real)
        self.p_list[n].append(moms.p[n].real)
        self.p2_list[n].append(moms.p2[n].real)

        namem = ["<x>", "<x^2>", "<p>", "<p^2>"]
        moms_list = [self.x_list[n], self.x2_list[n], self.p_list[n]]

        if self.i % self.conf.mod_update == 0:
            # Updating the graph for moms without <p^2>
            self.__plot_moms_update_graph(self.t_moms_list[n], moms_list, namem,
                                          "Expectation values for the state #%d" % n, "",
                                          os.path.join(self._out_path, self.conf.gr_moms_low.replace("{level}", str(n))))

            moms_list.append(self.p2_list[n])
            # Updating the graph for moms
            self.__plot_moms_update_graph(self.t_moms_list[n], moms_list, namem,
                                          "Expectation values for the state #%d" % n, "",
                                          os.path.join(self._out_path, self.conf.gr_moms.replace("{level}", str(n))))

    def plot_prop(self, t, ener, overlp0, overlpf, psi_max_abs, psi_max_real):
        self.t_list.append(t)
        self.ener_list.append([el.real for el in ener])
        self.overlp0_list.append([abs(el) for el in overlp0])
        self.overlpf_list.append([abs(el) for el in overlpf])
        self.abs_psi_max_list.append(psi_max_abs)
        self.real_psi_max_list.append(psi_max_real)

        if self.i % self.conf.mod_update == 0:
            # Updating the graph for ener
            self.__plot_tvals_mult_update_graph(self.t_list, self.ener_list, self._nlvls,
                                                "State energies", "Energy",
                                                os.path.join(self._out_path, self.conf.gr_ener))

            # Updating the graph for lower state population
            self.__plot_tvals_mult_update_graph(self.t_list, self.overlp0_list, self._nlvls,
                                                "Overlaps with initial state", "abs((psi0, psi))",
                                                os.path.join(self._out_path, self.conf.gr_overlp0))

            self.__plot_tvals_mult_update_graph(self.t_list, self.overlpf_list, self._nlvls,
                                                "Overlaps with goal state", "abs((psif, psi))",
                                                os.path.join(self._out_path, self.conf.gr_overlpf))

            # Updating the graph for maximum absolute value of ground state wavefunction
            self.__plot_tvals_mult_update_graph(self.t_list, self.abs_psi_max_list, self._nlvls,
                                                "Time dependencies of max|Ψ|", "max|Ψ|",
                                                os.path.join(self._out_path, self.conf.gr_abs_max))

            # Updating the graph for maximum real value of ground state wavefunction
            self.__plot_tvals_mult_update_graph(self.t_list, self.real_psi_max_list, self._nlvls,
                                           "Time dependencies of maximum value of real(Ψ)",
                                           "max(real(Ψ))", os.path.join(self._out_path, self.conf.gr_real_max))

    def plot_fitter(self, t, E, freq_mult, ener_tot, overlp_tot):
        self.t_fit_list.append(t)
        self.E_list.append(E)
        self.freq_mult_list.append(freq_mult)
        self.ener_tot_list.append(ener_tot.real)
        self.overlp0_tot_list.append(abs(overlp_tot[0]))
        self.overlpf_tot_list.append(abs(overlp_tot[1]))

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


    def print_time_point_prop(self, l, psi: Psi, t, x, np, moms, ener, overlp0, overlpf, overlp_tot, ener_tot,
                              psi_max_abs, psi_max_real, E, freq_mult):
        try:
            if l % self.conf.mod_plotout == 0 and l >= self.conf.lmin:
                for n in range(self._nlvls):
                    self.plot(psi, t, x, np, n)
                    self.plot_moms_prop(t, moms, n)

                self.plot_prop(t, ener, overlp0, overlpf, psi_max_abs, psi_max_real)
                self.plot_fitter(t, E, freq_mult, ener_tot, overlp_tot)
                self.i += 1
        except ValueError as err:
            print_err("A nasty error has occurred during the reporting: ", err)
            print_err("Hopefully that doesn't affect the calculations, so the application is going on...")


class MultiplePropagationReporter(PropagationReporter):
    reps: List[PropagationReporter]

    def __init__(self, out_path: str, nlvls: int, conf_rep_table, conf_rep_plot):
        super(MultiplePropagationReporter, self).__init__(out_path=out_path, nlvls=nlvls)
        self.reps = []
        if not conf_rep_plot.is_empty():
            self.reps.append(PlotPropagationReporter(conf=conf_rep_plot, out_path=os.path.join(self._out_path, "plots"), nlvls=self._nlvls))
        if not conf_rep_table.is_empty():
            self.reps.append(TablePropagationReporter(conf=conf_rep_table, out_path=os.path.join(self._out_path, "tables"), nlvls=self._nlvls))

    def open(self):
        for rep in self.reps:
            rep.open()
        return self

    def close(self):
        pass

    def print_time_point_prop(self, l, psi: Psi, t, x, np, moms, ener, overlp0, overlpf, overlp_tot, ener_tot,
                              psi_max_abs, psi_max_real, E, freq_mult):
        for rep in self.reps:
            rep.print_time_point_prop(l, psi, t, x, np, moms, ener, overlp0, overlpf, overlp_tot, ener_tot,
                                      psi_max_abs, psi_max_real, E, freq_mult)


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

    def create_propagation_reporter(self, prop_id: str, nlvls: int):
        raise NotImplementedError()


class TableFitterReporter(FitterReporter):
    def __init__(self, conf: config.ReportRootConfiguration.ReportTableFitterConfiguration):
        super().__init__()
        self.conf = conf
        self.f_ifit = None
        self.f_ifit_E = None

    def create_propagation_reporter(self, prop_id: str, nlvls: int):
        prop_conf_output = copy.deepcopy(self.conf)
        return TablePropagationReporter(out_path=os.path.join(prop_conf_output.out_path, prop_id), nlvls=nlvls,
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

    def create_propagation_reporter(self, prop_id: str, nlvls: int):
        prop_conf_output = copy.deepcopy(self.conf)
        return PlotPropagationReporter(out_path=os.path.join(prop_conf_output.out_path, prop_id), nlvls=nlvls,
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
        #E_abs = []
        #for i in range(nt + 1):
        #    E_abs.append(E[i])

        self.E_abs[iter] = {'t': t, 'y': E}

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

    def create_propagation_reporter(self, prop_id: str, nlvls: int):
        prop_conf_rep_table = copy.deepcopy(self.conf_rep_table)
        prop_conf_rep_plot = copy.deepcopy(self.conf_rep_plot)

        assert(prop_conf_rep_plot.out_path == prop_conf_rep_table.out_path)
        out_path = prop_conf_rep_table.out_path
        prop_out_path = os.path.join(out_path, prop_id)
        if not os.path.exists(prop_out_path):
            os.makedirs(prop_out_path)

        return MultiplePropagationReporter(out_path=prop_out_path, nlvls=nlvls,
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
