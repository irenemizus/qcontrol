""" A Python script for solving a controlled propagation
task in a two-potential quantum system using a Newtonian polynomial
algorithm with a Chebyshev-based interpolation scheme.

Usage: python newcheb.py [options]

Options:
    -h, --help
        print this usage info and exit
    --json_rep
        input json file name, in which all the following reporting options should be provided.
        If something is not provided in the file or this option is missing at all,
        the following default values will be used
    --json_task
        input json file name, in which all the following calculation parameters should be provided.
        If something is not provided in the file or this option is missing at all,
        the following default values will be used
    --json_create
        instead of running calculation, create a set of input_task json files from the source file also given
        by option "--json_task" to use them in a subsequent batch run, and exit


    Content of the json_rep file
    run_id
        An ID parameter (or a set if IDs for different runs in case of batch execution), which is augmented to the name
        of output folder specified in the "out_path" key of the json_rep file.
        For 3 runs its values can be specified as { "@SUBST:LIST": [ "ut/run1", "ut/run2", "ut/run3" ] }.
        By default, is "no_id"

    In key "fitter":
        out_path
            a path name for the output files.
            If some of calculation parameters are varying, the corresponding name structure can be used:
            "output_ut/T={input:$.fitter.propagation.T}__w_list={input:$.fitter.w_list}"
            (for varying of fitter.propagation.T and fitter.w_list values).
            By default, is "output"
        table_glob_path
            a path name for the file with global table, which contains the results in case of varying calculation parameters.
            Shouldn't be specified otherwise.
            By default, is ""
        plotting_flag
            a flag that indicates the type of output
            "all"           --  to print both plots and tables (option by default)
            "tables"        --  to print only tables
            "tables_iter"   --  to print only "global" tables with dependency on iteration number
            "plots"         --  to print only plots
            "none"          --  no printing to files at all

        parameters, which has to be specified if writing of the resulting tables for "global" values as a function
        of iteration number is needed.
        table.imin
            number of iteration, from which the result should be written to a file.
            A negative value will be considered as 0.
            By default, is equal to 0
        table.imod_fileout
            step of writing to file (to write to file at each <val>-th iteration).
            By default, is equal to 1
        table.tab_iter
            output file name, to which the iteration dependencies of the "global" values should be written.
            By default, is equal to "tab_iter.csv"
        table.tab_iter_E
            output file name, to which the iteration dependency of the laser field energy envelope should be written.
            By default, is equal to "tab_iter_E.csv"

        parameters, which has to be specified if plotting of the resulting figures for controlled values is needed.
        plot.imin
            number of a time step, from which the result should be plotted.
            A negative value will be considered as 0.
            By default, is equal to 0
        plot.imod_plotout
            step of plotting graphs with x-axis = number of iteration (to plot to file each <val>-th iteration).
            By default, is equal to 1
        plot.gr_iter
            output file name, to which the iteration dependencies of the "global" values should be plotted.
            By default, is equal to "fig_iter.pdf"
        plot.gr_iter_E
            output file name, to which the iteration dependency of the laser field energy envelope should be plotted.
            By default, is equal to "fig_iter_E.pdf"

        In key "propagation":
            parameters, which has to be specified if writing of the resulting propagation tables is needed.
            table.lmin
                number of a time step, from which the result should be written to a file.
                A negative value will be considered as 0.
                By default, is equal to 0
            table.mod_fileout
                step of writing to file (to write to file each <val>-th time step).
                By default, is equal to 100
            table.tab_abs
                output file name, to which absolute values of wavefunctions should be written.
                By default, is equal to "tab_abs_{level}.csv"
                ({level} is replaced automatically by "0" for the ground state and by "1" for the excited one)
            table.tab_real
                output file name, to which real parts of wavefunctions should be written.
                By default, is equal to "tab_real_{level}.csv"
                ({level} is replaced automatically by "0" for the ground state and by "1" for the excited one)
            table.tab_tvals
                output file name, to which expectation values of x, x*x, p, p*p and other
                time-dependent values should be written.
                By default, is equal to "tab_tvals_{level}.csv"
                ({level} is replaced automatically by "0" for the ground state and by "1" for the excited one)
            table.tab_tvals_fit
                output file name, to which the time dependencies of the controlled values should be written.
                By default, is equal to "tab_tvals_fit.csv"

            parameters, which has to be specified if plotting of the resulting propagation figures is needed.
            plot.lmin
                number of a time step, from which the result should be plotted.
                A negative value will be considered as 0.
                By default, is equal to 0
            plot.mod_plotout
                step of plotting graphs with x-axis = time (to plot to file each <val>-th time step).
                By default, is equal to 100
            plot.mod_update
                step for updating the plots.
                By default, is equal to 20
            plot.number_plotout
                maximum number of graphs for different time points to plot on one canvas
                for the absolute and real values of wavefunctions. Must be larger than 1.
                By default, is equal to 15
            plot.gr_*
                output file name, to which the corresponding result should be plotted.
                By default, is equal to "fig_*.pdf"


    Content of the json_task file

    In key "fitter":
    task_type
        type of the calculation task:
        "trans_wo_control"               - calculation of transition from the ground state
                                           to the excited one under the influence of external
                                           non-controlled laser field with gaussian envelope and a constant
                                           chirp (by default)
        "single_pot"                     - simple propagation of an initial harmonic / morse wavefunction
                                           shifted from the potential minima in a harmonic / morse potential
        "filtering"                      - filtering task
                                           in this case E0 and nu_L are zeroing mandatory
        "intuitive_control"              - calculation of transitions from the ground state
                                           to the excited state and back to the ground one
                                           under the influence of a sequence of equal laser pulses
                                           with gaussian envelopes and a constant chirps
        "local_control_population"       - calculation of transition from the ground state
                                           to the excited one under the influence of external
                                           laser field with controlled envelope form
                                           by the local control algorithm, when the goal operator is A = / 0  0 \
                                                                                                   \ 0  1 /
        "local_control_projection"       - calculation of transition from the ground state
                                           to the excited one under the influence of external
                                           laser field with controlled chirp
                                           by the local control algorithm, when the goal operator is A = P_g + P_e
        "optimal_control_krotov"         - calculation of transition from the ground state
                                           to the excited one under the influence of a controlled external
                                           laser field with an iterative Krotov algorithm,
                                           when the propagation on a current time step
                                           is partially under the old field and partially - under the new field,
                                           which is calculated "on the fly"
        "optimal_control_unit_transform" - calculation of transition from the pure ground and excited states
                                           under the influence of a Hadamard H1 unitary transformation
                                           using a controlled external laser field with an iterative Krotov algorithm
                                           and the squared modulus functional Fsm
    k_E
        aspect ratio for the inertial "force" in equation for the laser field energy in sec^(-2).
        Applicable for the task_type = "local_control", only. For all other cases is a dummy variable.
        By default, is equal to 1e29 1/s**2
    lamb
        aspect ratio for the decay term in equation for the laser field energy in 1 / sec.
        Applicable for the task_type = "local_control", only. For all other cases is a dummy variable.
        By default, is equal to 4e14 1/s
    pow
        power value in the decay term in equation for the laser field energy.
        Applicable for the task_type = "local_control", only. For all other cases is a dummy variable.
        By default, is equal to 0.8
    epsilon
        small parameter, which is used for cutting of an imaginary part in dA/dt
        (applicable for the task_type = "local_control"), or as a divergence criteria for
        the task_type = "optimal_control_...". For all other cases is a dummy variable.
        By default, is equal to 1e-15
    impulses_number
        number of laser pulses in the "intuitive_control" task type.
        The values more than 1 are applicable for the task_type = "intuitive_control", only.
        In this case if a value less than 2 provided, it will be replaced by 2.
        For the task_type = "filtering"  / "single_morse" / "single_harmonic" it will be replaced by 0.
        For the task_type = "trans_wo_control" or "local_control" it will be replaced by 1.
        By default, is equal to 1
    delay
        time delay between the laser pulses in sec.
        Is a dummy variable for impulses_number less than 2.
        By default, is equal to 600e-15 s
    iter_max
        maximum iteration number for the "optimal_control_..." task_type in case if a divergence with the
        given criteria hasn't been reached. Is a dummy variable for all other task types.
        If iter_max = -1, there is no limitation on the maximum number of iterations.
        By default, is equal to -1
    h_lambda
        parameter, which is applicable for the task_type = "optimal_control_..." only.
        For all other cases is a dummy variable.
        By default, is equal to 0.0066
    init_guess
        type of initial guessed laser field envelope used in propagation tasks.
        Available options:
        "zero"      - without external laser field (by default)
        "gauss"     - a gaussian-like envelope
        "sqrsin"    - a squared sinus envelope
        "maxwell"   - a maxwell distribution-like envelope
    init_guess_hf
        type of the high-frequency part of the initial guessed laser field used in propagation tasks.
        Available options:
        "exp"       - exponential type exp(i omega_L t) (by default)
        "cos"       - cos-type cos(omega_L t)
        "cos_set"   - a sequence of cos-type terms [w_0 * cos(omega_L t) + sum(w_i * cos(omega_L t * k) + w_j * cos(omega_L t / k))]
    w_list
        a list of 2 * pcos - 1 amplitudes for separate harmonics of the laser field high-frequency part of type "cos_set".
        Is a dummy variable for all other types of "init_guess_hf".
        If not specified or is empty, the amplitude values are generated randomly.
        By default, is equal to []
    lf_aug_type
        a way of adding the controlling laser field to "BH_model"-type Hamiltonian.
        For all other variants of "hamil_type" variables is a dummy variable.
        Available options:
        "z" - H = H0 + 2E(t)Jz (by default)
        "x" - H = H0 + 2E(t)Jx
    nb
        number of basis vectors of the Hilbert space used in the calculation task.
        By default, is equal to 1
    pcos
        maximum frequency multiplier for a sum [w_0 * cos(omega_L t) + sum(w_i * cos(omega_L t * k) + w_j * cos(omega_L t / k))]
        with k = 2 ... pcos in the case "init_guess_hf" = "cos_set", or just a frequency multiplier itself for
        the "init_guess_hf" = "cos" case, which is used in a high-frequency part for the laser field initial guess.
        For the "init_guess_hf" = "exp" case is a dummy variable.
        In the case "init_guess_hf" = "cos_set" the maximum frequency multiplier equal to floor(pcos) will be used;
        it has to be greater than 1 then.
        By default, is equal to 1
    Em
        a multiplier used for evaluation of the laser field energy maximum value (E_max = E0 * Em),
        which can be reached during the controlling procedure.
        By default, is equal to 1.5
    mod_log
        step of output to stdout (to write to stdout each <val>-th time step).
        By default, is equal to 500

    propagation
        a dictionary that contains the parameters, which are used in a simple propagation task with laser field:
        m
            reduced mass value of the considered system.
            By default, is equal to 0.5 Dalton
        pot_type
            type of the potentials ("morse" or "harmonic").
            By default, the "morse" type is used
        a
            ground state scaling coefficient.
            By default, is equal to 1.0 1/a_0 -- for "morse" potential,
                                        a_0 -- for "harmonic" potential
        De
            ground state dissociation energy value.
            By default, is equal to 20000.0 1/cm for "morse" potential,
                        is a dummy variable for "harmonic" potential
        x0p
            shift of the upper potential relative to the ground one.
            By default, is equal to -0.17 a_0 for double potentials,
                        is identically equated to zero for filtering / single morse / single harmonic tasks
        a_e
            excited state scaling coefficient.
            By default, is equal to 1.0 1/a_0 -- for "morse" potential,
                        is equal to 1.0 a_0 -- for "harmonic" potential,
                        is identically equated to zero for filtering / single morse / single harmonic tasks
        De_e
            excited state dissociation energy value.
            By default, is equal to 10000.0 1/cm for "morse" potential,
                        is a dummy variable for "harmonic" potential,
                        is identically equated to zero for filtering / single morse / single harmonic tasks
        Du
            energy shift between the minima of the upper potential and the ground one.
            By default, is equal to 20000.0 1/cm for double potentials,
                        is identically equated to zero for filtering / single morse / single harmonic tasks
        wf_type
            type of the wavefunctions ("morse" or "harmonic").
            By default, the "morse" type is used
        hamil_type
            type of the Hamiltonian operator used ("ntriv", "two_levels" or "BH_model").
            By default, the "ntriv" type is used
        U, delta
            parameters of angular momentum-type Hamiltonian (applicable for 'hamil_type' = 'BH_model' only),
            U and delta are in units of 1 / cm.
            By default, both are equal to 0.0
        x0
            coordinate initial condition.
            By default, is equal to 0.0
        p0
            momentum initial condition.
            By default, is equal to 0.0
        L
            spatial range of the problem in a_0.
            By default, is equal to 5.0 a_0
        T
            time range of the problem in sec.
            By default, is equal to 600e15 s
        np
            number of collocation points; must be a power of 2.
            By default, is equal to 1024
        nch
            number of Chebyshev interpolation points; must be a power of 2.
            By default, is equal to 64
        nt
            number of time grid points.
            By default, is equal to 420000
        E0
            amplitude value of the laser field energy envelope in 1 / cm.
            By default, is equal to 71.54 1 / cm,
                        is identically equated to zero for filtering / single morse / single harmonic tasks
        t0
            initial time, when the laser field reaches its maximum value, in sec.
            Is a dummy variable for filtering / single morse / single harmonic tasks.
            By default, is equal to 300e-15 s
        sigma
            scaling parameter of the laser field envelope in sec.
            Is a dummy variable for filtering / single morse / single harmonic tasks.
            By default, is equal to 50e-15 s
        nu_L
            basic frequency of the laser field in Hz.
            By default, is equal to 0.29297e15 Hz,
                        is identically equated to zero for filtering / single morse / single harmonic tasks

    There is a possibility of varying any input parameter specified in the json_task file.
    The key words for that:

    @SUBST:LIST
        possible values for the given calculation parameter to be looked over are specified as a list.
        Example: "T": { "@SUBST:LIST": [ 2.9E-13, 3.1E-13, 3.3E-13, 3.5E-13 ] }

Examples:
    python newcheb.py --json_task "input_task.json" --json_rep "input_report.json"
        perform a propagation task using the parameter values specified in the json files
        "input_task.json" and "input_report.json" or the default ones if something wasn't provided in the file
"""

__author__ = "Irene Mizus (irenem@hit.ac.il)"
__license__ = "Python"

import random
import re
from multiprocessing import Lock
from pprint import pprint, pformat

from jsonpath2 import Path

from json_substitutions import JsonSubstitutions
from tools import print_err

import sys
import traceback
import os.path
import getopt
import json
import math

import grid_setup
import fitter
import reporter
import task_manager
from config import *

from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def usage():
    """ Print usage information """
    print (__doc__)


def _warning_collocation_points(np, np_min):
    print_err("WARNING: The number of collocation points np = {} should be more than an estimated initial value {}. "
          "You've got a divergence!".format(np, np_min))


def _warning_time_steps(nt, nt_min):
    print_err("WARNING: The number of time steps nt = {} should be more than an estimated value {}. "
          "You've got a divergence!".format(nt, nt_min))

def print_json_input_task(conf_task, run_id,  id):
    file_name = f"input_task_ut_ang_mom_H_{run_id}_var{id}.json"
    pretty_print_json = pformat(conf_task).replace("'", '"')
    with open(file_name, "w") as finp:
        finp.write(pretty_print_json)

def print_input(conf_rep_plot, conf_task, file_name):
    with open(os.path.join(conf_rep_plot.fitter.out_path, file_name), "w") as finp:
        finp.write("task_type:\t\t"   f"{conf_task.fitter.task_type}\n")
        finp.write("iter_max:\t\t"   f"{conf_task.fitter.iter_max}\n")
        finp.write("epsilon:\t\t"   f"{conf_task.fitter.epsilon:.1E}\n")

        finp.write("nb:\t\t\t"   f"{conf_task.fitter.nb}\n")
        finp.write("wf_type:\t\t"   f"{conf_task.fitter.propagation.wf_type}\n")

        finp.write("impulses_number:\t"   f"{conf_task.fitter.impulses_number}\n")
        finp.write("Em:\t\t\t"   f"{conf_task.fitter.Em}\n")
        finp.write("E0:\t\t\t"   f"{conf_task.fitter.propagation.E0}\n")
        finp.write("t0:\t\t\t"   f"{conf_task.fitter.propagation.t0}\n")
        finp.write("sigma:\t\t\t"   f"{conf_task.fitter.propagation.sigma:.6E}\n")
        finp.write("nu_L:\t\t\t"   f"{conf_task.fitter.propagation.nu_L:.6E}\n")
        finp.write("h_lambda:\t\t"   f"{conf_task.fitter.h_lambda}\n")
        finp.write("init_guess:\t\t"   f"{conf_task.fitter.init_guess}\n")
        finp.write("init_guess_hf:\t\t"   f"{conf_task.fitter.init_guess_hf}\n")
        finp.write("pcos:\t\t\t"   f"{conf_task.fitter.pcos}\n")
        finp.write("w_list:\t\t\t"   f"{conf_task.fitter.w_list}\n")
        finp.write("lf_aug_type:\t\t"   f"{conf_task.fitter.lf_aug_type}\n")

        finp.write("hamil_type:\t\t"   f"{conf_task.fitter.propagation.hamil_type}\n")
        finp.write("U:\t\t\t"   f"{conf_task.fitter.propagation.U}\n")
        finp.write("delta:\t\t\t"   f"{conf_task.fitter.propagation.delta}\n")

        finp.write("pot_type:\t\t"   f"{conf_task.fitter.propagation.pot_type}\n")
        finp.write("Du:\t\t\t"   f"{conf_task.fitter.propagation.Du}\n")

        finp.write("np:\t\t\t"   f"{conf_task.fitter.propagation.np}\n")
        finp.write("L:\t\t\t"   f"{conf_task.fitter.propagation.L}\n")
        finp.write("nch:\t\t\t"   f"{conf_task.fitter.propagation.nch}\n")
        finp.write("nt:\t\t\t"   f"{conf_task.fitter.propagation.nt}\n")
        finp.write("T:\t\t\t"   f"{conf_task.fitter.propagation.T:.6E}\n")


def process_input_templates_in_report(data_task, data_rep_node):
    if isinstance(data_rep_node, dict):
        for k in data_rep_node.keys():
            data_rep_node[k] = process_input_templates_in_report(data_task, data_rep_node[k])
            #print(data_rep_node[k])

    elif isinstance(data_rep_node, list):
        for i in range(len(data_rep_node)):
            data_rep_node[i] = process_input_templates_in_report(data_task, data_rep_node[i])

    elif isinstance(data_rep_node, str):
        # Processing templates in the report config
        input_pat = re.compile("\\{input:([^\\}]*)\\}")

        res = input_pat.finditer(data_rep_node)
        if res:
            try:
                while True:
                    r = res.__next__()

                    path = r.group(1)
                    found_subst = r.group(0)
                    # Finding the value with a path
                    jsonpath_expression = Path.parse_str(path)
                    match = jsonpath_expression.match(data_task)

                    m = match.__next__()

                    val = m.current_value
                    oldval = data_rep_node
                    data_rep_node = oldval.replace(found_subst, str(val))
            except StopIteration:
                # Do the nothing!
                pass
            #print(data_rep_node)
    return data_rep_node

def main(argv):
    """ The main() function """
    # analyze cmdline:
    try:
        options, arguments = getopt.getopt(argv, 'h', ['help', 'json_create', 'json_rep=', 'json_task='])
    except getopt.GetoptError:
        print_err("\tThere are unrecognized options!")
        print_err("\tRun this script with '-h' option to see the usage info and available options.")
        sys.exit(2)

    file_json_rep = None
    file_json_task = None
    json_create = False

    # analyze provided options and their values (if any):
    for opt, val in options:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("--json_create"):
            json_create = True
        elif opt in ("--json_rep"):
            file_json_rep = val
        elif opt in ("--json_task"):
            file_json_task = val

    if 'file_json_rep' not in locals():
        print("\tNo input json file with reporting options was provided. The default values of options will be used")
    else:
        with open(file_json_rep, "r") as read_file:
            data_rep_template = json.load(read_file)

    if 'file_json_task' not in locals():
        print("\tNo input json file with calculation parameters was provided. The default values of parameters will be used")
    else:
        with open(file_json_task, "r") as read_file:
            data_task_src = json.load(read_file)

    first_pass = { "val": False }

    substs = JsonSubstitutions(data_task_src)

    job_mutex = Lock()

    cpu_count = multiprocessing.cpu_count()
    print(f"Found {cpu_count} cores to torture")

    executor = ThreadPoolExecutor(cpu_count)

    futures = []
    job_id: int = 0
    json_id: int = 0
    for dt in substs:
        if json_create:
            run_id = dt["run_id"].split("/")[-1]
            print_json_input_task(dt, run_id, json_id)
            json_id += 1
            continue

        def job(id: int, data_task):
            try:
                print(f"Running variant {id}:")
                pprint(data_task)

                conf_task = TaskRootConfiguration()
                conf_task.load(data_task)

                data_rep = copy.deepcopy(data_rep_template)

                # # Processing templates in the report config
                process_input_templates_in_report(data_task, data_rep)

                conf_rep_table = ReportTableRootConfiguration()
                conf_rep_table.load(data_rep)

                conf_rep_plot = ReportPlotRootConfiguration()
                conf_rep_plot.load(data_rep)

                # analyze provided json data
                if conf_rep_table.fitter.propagation.lmin < 0 or conf_rep_plot.fitter.propagation.lmin < 0:
                    raise ValueError(
                        "The number 'lmin' of time iteration, from which the result"
                        "should be written to a file or plotted, has to be be positive or 0")

                if conf_rep_table.fitter.imin < -1 or conf_rep_plot.fitter.imin < -1:
                    raise ValueError(
                        "The number 'imin' of iteration number, from which the result"
                        "should be written to a file or plotted, has to be positive, -1 or 0")

                if conf_rep_table.fitter.propagation.mod_fileout < 0 or conf_task.fitter.mod_log < 0 or \
                        conf_rep_table.fitter.imod_fileout < 0:
                    raise ValueError(
                        "The numbers 'mod_fileout', 'imod_fileout', and 'mod_log' have to be positive or 0")

                if conf_rep_plot.fitter.propagation.mod_plotout < 0 or conf_rep_plot.fitter.propagation.mod_update < 0 or \
                    conf_rep_plot.fitter.imod_plotout < 0:
                    raise ValueError(
                        "The step for plotting graphs with x-axis = time, 'mod_plotout', "
                        "the step for plotting graphs with x-axis = number of iteration, 'imod_plotout', "
                        "and for updating the plots, 'mod_update', have to be positive or 0")

                if conf_rep_plot.fitter.propagation.number_plotout < 2:
                    raise ValueError(
                        "The maximum number of graphs, 'number_plotout', to be plotted on one canvas has to be larger than 1")

                if not math.log2(conf_task.fitter.propagation.np).is_integer() or not math.log2(
                        conf_task.fitter.propagation.nch).is_integer():
                    raise ValueError(
                        "The number of collocation points, 'np', and of Chebyshev "
                        "interpolation points, 'nch', have to be positive integers and powers of 2")

                if conf_task.fitter.impulses_number < 0:
                    raise ValueError(
                        "The number of laser pulses, 'impulses_number', has to be positive or 0")

                if conf_task.fitter.nb < 0:
                    raise ValueError(
                        "The number of basis vectors of the Hilbert space, 'nb', has to be positive or 0")

                if conf_task.fitter.Em < 0:
                    raise ValueError(
                        "The multiplier 'Em' used for evaluation of the laser field energy maximum value, "
                        "which can be reached during the controlling procedure, has to be positive")

                if conf_task.fitter.iter_max < -1 and (conf_task.fitter.task_type == conf_task.fitter.TaskType.OPTIMAL_CONTROL_KROTOV or
                        conf_task.fitter.task_type == conf_task.fitter.TaskType.OPTIMAL_CONTROL_GRADIENT):
                    raise ValueError(
                        "The maximum number of iterations in the optimal control task, 'iter_max', has to be positive, 0 or -1")

                if conf_task.fitter.propagation.L <= 0.0 or conf_task.fitter.propagation.T <= 0.0:
                    raise ValueError(
                        "The value of spatial range, 'L', and of time range, 'T', of the problem have to be positive")

                if conf_task.fitter.propagation.m <= 0.0 or conf_task.fitter.propagation.a <= 0.0 or conf_task.fitter.propagation.De <= 0.0:
                    raise ValueError(
                        "The value of a reduced mass, 'm/mass', of a scaling factor, 'a', and of a dissociation energy, 'De', have to be positive")

                if not conf_task.fitter.propagation.E0 >= 0.0 or not conf_task.fitter.propagation.sigma > 0.0 or not conf_task.fitter.propagation.nu_L >= 0.0:
                    raise ValueError(
                        "The amplitude value of the laser field energy envelope, 'E0',"
                        "of a scaling parameter of the laser field envelope, 'sigma',"
                        "and of a basic frequency of the laser field, 'nu_L', have to be positive")


                if conf_task.fitter.task_type == conf_task.FitterConfiguration.TaskType.FILTERING or \
                        conf_task.fitter.task_type == conf_task.FitterConfiguration.TaskType.SINGLE_POT:
                    print("A '%s' task begins..." % str(conf_task.fitter.task_type).split(".")[-1].lower())

                    if conf_task.fitter.propagation.E0 != 0.0:
                        raise ValueError(
                            "For the 'task_type' = '%s' the amplitude value of the laser field energy envelope, 'E0', has to be equal to zero"
                            % str(conf_task.fitter.task_type).split(".")[-1].lower())

                    if conf_task.fitter.propagation.nu_L != 0.0:
                        raise ValueError(
                            "For the 'task_type' = '%s' the value of a basic frequency of the laser field, 'nu_L', has to be equal to zero"
                            % str(conf_task.fitter.task_type).split(".")[-1].lower())

                    if conf_task.fitter.init_guess != conf_task.fitter.InitGuess.ZERO:
                        raise ValueError(
                            "For the 'task_type' = '%s' the initial guess type for the laser field envelope, 'init_guess', has to be 'zero'"
                            % str(conf_task.fitter.task_type).split(".")[-1].lower())

                    if conf_task.fitter.impulses_number != 0:
                        raise ValueError(
                            "For the 'task_type' = '%s' the 'impulses_number' value has to be equal to zero"
                            % str(conf_task.fitter.task_type).split(".")[-1].lower())
                else:
                    if conf_task.fitter.task_type == conf_task.fitter.TaskType.TRANS_WO_CONTROL:
                        print("An ordinary transition task begins...")

                        if conf_task.fitter.impulses_number != 1:
                            raise ValueError(
                                "For the 'task_type' = 'trans_wo_control' the 'impulses_number' value has to be equal to 1")

                    elif conf_task.fitter.task_type == conf_task.FitterConfiguration.TaskType.INTUITIVE_CONTROL:
                        print("An intuitive control task begins...")

                        if conf_task.fitter.init_guess == "zero":
                            raise ValueError(
                                "For the 'task_type' = 'intuitive_control' the initial guess type for the laser field envelope, "
                                "'init_guess', mustn't be 'zero'")

                        if conf_task.fitter.impulses_number < 2:
                            raise ValueError(
                                "For the 'task_type' = 'intuitive_control' the 'impulses_number' value has to be larger than 1")

                    elif conf_task.fitter.task_type == conf_task.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION:
                        print("A local control with goal population task begins...")

                        if conf_task.fitter.init_guess == "zero":
                            raise ValueError(
                                "For the 'task_type' = 'local_control_population' the initial guess type for the laser field envelope, "
                                "'init_guess', mustn't be 'zero'")

                        if conf_task.fitter.impulses_number != 1:
                            raise ValueError(
                                "For the 'task_type' = 'local_control_population' the 'impulses_number' value has to be equal to 1")

                    elif conf_task.fitter.task_type == conf_task.FitterConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
                        print("A local control with goal projection task begins...")

                        if conf_task.fitter.init_guess == "zero":
                            raise ValueError(
                                "For the 'task_type' = 'local_control_projection' the initial guess type for the laser field envelope, "
                                "'init_guess', mustn't be 'zero'")

                        if conf_task.fitter.impulses_number != 1:
                            raise ValueError(
                                "For the 'task_type' = 'local_control_projection' the 'impulses_number' value has to be equal to 1")

                    elif conf_task.fitter.task_type == conf_task.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV:
                        print("An optimal control task with Krotov method begins...")

                        if conf_task.fitter.init_guess == "zero":
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_krotov' the initial guess type for the laser field envelope, "
                                "'init_guess', mustn't be 'zero'")

                        if conf_task.fitter.impulses_number != 1:
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_krotov' the 'impulses_number' value has to be equal to 1")

                    elif conf_task.fitter.task_type == conf_task.FitterConfiguration.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                        print("An optimal control task with unitary quantum Fourier transformation begins...")

                        if conf_task.fitter.init_guess == "zero":
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_unit_transform' the initial guess type for the laser field envelope, "
                                "'init_guess', mustn't be 'zero'")

                        if conf_task.fitter.propagation.np > 1:
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_unit_transform' the number of collocation points, 'np', has to be equal to 1")

                        if conf_task.fitter.impulses_number != 1:
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_unit_transform' the 'impulses_number' value has to be equal to 1")

                        if conf_task.fitter.propagation.L == 1.0:
                            pass
                        else:
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_unit_transform' the spatial range of the problem, 'L', has to be equal to 1.0")

                    else:
                        raise RuntimeError("Impossible case in the TaskType class")

                nw = int(2 * conf_task.fitter.pcos - 1)
                if not conf_task.fitter.w_list:
                    # conf_task.fitter.w_list = [float(x) / 100.0 for x in random.sample(range(1, 101), nw)]
                    conf_task.fitter.w_list = [float(x) / 10.0 - 2.0 for x in random.sample(range(0, 40), nw)]
                else:
                    assert len(conf_task.fitter.w_list) == nw

                task_manager_imp = task_manager.create(conf_task.fitter)

                # setup of the grid
                grid = grid_setup.GridConstructor(conf_task.fitter.propagation)
                dx, x = grid.grid_setup()

                # evaluating of initial wavefunction (of type PsiBasis)
                psi0 = task_manager_imp.psi_init(x, conf_task.fitter.propagation.np, conf_task.fitter.propagation.x0,
                                                 conf_task.fitter.propagation.p0, conf_task.fitter.propagation.x0p,
                                                 conf_task.fitter.propagation.m, conf_task.fitter.propagation.De,
                                                 conf_task.fitter.propagation.De_e, conf_task.fitter.propagation.Du,
                                                 conf_task.fitter.propagation.a, conf_task.fitter.propagation.a_e,
                                                 conf_task.fitter.propagation.L, conf_task.fitter.nb)

                # evaluating of the final goal (of type PsiBasis)
                psif = task_manager_imp.psi_goal(x, conf_task.fitter.propagation.np, conf_task.fitter.propagation.x0,
                                                 conf_task.fitter.propagation.p0, conf_task.fitter.propagation.x0p,
                                                 conf_task.fitter.propagation.m, conf_task.fitter.propagation.De,
                                                 conf_task.fitter.propagation.De_e, conf_task.fitter.propagation.Du,
                                                 conf_task.fitter.propagation.a, conf_task.fitter.propagation.a_e,
                                                 conf_task.fitter.propagation.L, conf_task.fitter.nb)

                # initial propagation direction
                init_dir = task_manager_imp.init_dir
                # checking of triviality of the system
                ntriv = task_manager_imp.ntriv
                step = -1

                if not os.path.exists(conf_rep_plot.fitter.out_path):
                    os.makedirs(conf_rep_plot.fitter.out_path, exist_ok=True)

                #     T_ac = conf_task.fitter.propagation.T #conf_task.fitter.propagation.Du / phys_base.Hz_to_cm
                #     T_start = T_ac / 2.0
                #     T0_step = pow(2.0, 1.0 / 100) #2 * T_ac / 800 #pow(1.01, 1.0 / 200)
                #     T_cur = T_start
                #     for step in range(200):
                #         conf_task.fitter.propagation.T = round(T_cur, 19)
                # T_cur *= T0_step

                if conf_task.fitter.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.SQRSIN and \
                   conf_task.run_id != "no_id":
                    conf_task.fitter.propagation.sigma = 2.0 * conf_task.fitter.propagation.T #TODO: to add a possibility to vary groups of parameters

                print_input(conf_rep_plot, conf_task, "table_inp_" + str(step) + ".txt")

                # setup of the time grid
                forw_time_grid = grid_setup.ForwardTimeGridConstructor(conf_prop=conf_task.fitter.propagation)
                t_step, t_list = forw_time_grid.grid_setup()

                # main calculation part
                fit_reporter_imp = reporter.MultipleFitterReporter(conf_rep_table=conf_rep_table.fitter, conf_rep_plot=conf_rep_plot.fitter)
                fit_reporter_imp.open()

                fitting_solver = fitter.FittingSolver(conf_task.fitter, init_dir, ntriv, psi0, psif, task_manager_imp.pot, task_manager_imp.laser_field, task_manager_imp.laser_field_hf, fit_reporter_imp,
                                                      _warning_collocation_points,
                                                      _warning_time_steps
                                                      )
                fitting_solver.time_propagation(dx, x, t_step, t_list)
                fit_reporter_imp.close()

                with job_mutex:
                    if conf_rep_table.fitter.table_glob_path != "":
                        if first_pass["val"]:
                            wa = "w"
                            first_pass["val"] = True
                        else:
                            wa = "a"

                        with open(conf_rep_table.fitter.table_glob_path, wa) as fout:
                            # Printing the last value into a table
                            F_cur = 0.0
                            step = -1
                            with open(os.path.join(conf_rep_plot.fitter.out_path, "tab_iter.csv"), "r") as f:
                                lines = f.readlines()
                                # F_cur = float(lines[-1].strip().split(" ")[-1])
                                for line in lines:
                                    F_last = float(line.strip().split(" ")[-1])
                                    step_last = int(line.strip().split(" ")[0])
                                    if F_last < F_cur:
                                        F_cur = F_last
                                        step = step_last

                            fout.write(f"{step}\t{conf_rep_table.fitter.out_path}\t{F_cur}\n")
                            fout.flush()
            except Exception as e:
                print_err(f"An exception interrupted the job #{job_id}: {str(e)}")
                print_err(traceback.format_exc())
            return id

        future = executor.submit(job, job_id, dt)
        job_id += 1
        futures.append(future)

    for f in futures:
        res = f.result()
        print(f"Job #{res} is done")


if __name__ == "__main__":
    main(sys.argv[1:])