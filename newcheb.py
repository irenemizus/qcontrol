""" A Python script for solving a controlled propagation
task in a two-potential quantum system using a Newtonian polynomial
algorithm with a Chebyshev-based interpolation scheme.

Usage: python newcheb.py [options]

Options:
    -h, --help
        print this usage info and exit
    -f, --jsonfile
        input json file name, in which all the following data should be provided
        if something is not provided in the file or this option is missing at all,
        the following default values will be used


    Content of the json file

    in key "output":
    key "table"
        a subsection, which has to be specified if writing of the resulting tables is needed.
        In the key "table":
        lmin
            number of a time step, from which the result should be written to a file.
            A negative value will be considered as 0.
            By default, is equal to 0
        mod_stdout
            step of output to stdout (to write to stdout each <val>-th time step).
            By default, is equal to 500
        mod_fileout
            step of writing in file (to write in file each <val>-th time step).
            By default, is equal to 100
        tab_abs
            output file name, to which absolute values of wavefunctions should be written.
            By default, is equal to "output/tab_abs.csv"
        tab_real
            output file name, to which real parts of wavefunctions should be written.
            By default, is equal to "output/tab_real.csv"
        tab_mom
            output file name, to which expectation values of x, x*x, p, p*p should be written.
            By default, is equal to "output/tab_mom.csv"
        out_path
            a path name for the output tables.
            By default, is equal to "output"

    key "plot"
        a subsection, which has to be specified if plotting of the resulting figures is needed.
        In the key "plot":
        lmin
            number of a time step, from which the result should be plotted.
            A negative value will be considered as 0.
            By default, is equal to 0
        mod_plotout
            step of plotting graphs with x-axis = time (to plot to file each <val>-th time step).
            By default, is equal to 500
        mod_update
            step for updating the plots
            By default, is equal to 50
        number_plotout
            maximum number of graphs for different time points to plot on one canvas
            for the absolute and real values of wavefunctions. Must be larger than 1
            By default, is equal to 15
        out_path
            a path name for the output plots.
            By default, is equal to "output/plots"
        gr_*
            output file name, to which the corresponding result should be plotted.
            By default, is equal to "output/plots/fig_*.pdf"

    in key "fitter":
    task_type
        type of the calculation task:
        "trans_wo_control"  - calculation of transition from the ground state
                              to the excited one under the influence of external
                              non-controlled laser field with gaussian envelope and a constant
                              chirp (by default)
        "single_pot"        - simple propagation of an initial harmonic / morse wavefunction
                              shifted from the potential minima in a harmonic / morse potential
        "filtering"         - filtering task
                              in this case E0 and nu_L are zeroing mandatory
        "intuitive_control" - calculation of transitions from the ground state
                              to the excited state and back to the ground one
                              under the influence of a sequence of equal laser pulses
                              with gaussian envelopes and a constant chirps
        "local_control_population" - calculation of transition from the ground state
                                     to the excited one under the influence of external
                                     laser field with controlled envelope form
                                     by the local control algorithm, when the goal operator is A = / 0  0 \
                                                                                                   \ 0  1 /
        "local_control_projection" - calculation of transition from the ground state
                                     to the excited one under the influence of external
                                     laser field with controlled chirp
                                     by the local control algorithm, when the goal operator is A = P_g + P_e
        "optimal_control_krotov"   - calculation of transition from the ground state
                                     to the excited one under the influence of a controlled external
                                     laser field with an iterative Krotov algorithm,
                                     when the propagation on a current time step
                                     is partially under the old field and partially - under the new field,
                                     which is calculated "on the fly"
        "optimal_control_gradient" - calculation of transition from the ground state
                                     to the excited one under the influence of a controlled external
                                     laser field with an iterative gradient algorithm,
                                     when the propagation on a current time step
                                     is under an old field, calculated on the previous step
    k_E
        aspect ratio for the inertial "force" in equation for the laser field energy in sec^(-2).
        Applicable for the task_type = "local_control", only. For all other cases is a dummy variable.
        By default, is equal to 1e29
    lamb
        aspect ratio for the decay term in equation for the laser field energy in 1 / sec.
        Applicable for the task_type = "local_control", only. For all other cases is a dummy variable.
        By default, is equal to 4e14
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
        By default, is equal to 600e-15
    iter_max
        maximum iteration number for the "optimal_control_..." task_type in case if a divergence with the
        given criteria hasn't been reached. Is a dummy variable for all other task types.
        By default, is equal to 5
    propagation
        a dictionary that contains the parameters, which are used in a simple propagation task with laser field:
        m
            reduced mass value of the considered system
            by default, is equal to 0.5 Dalton
        pot_type
            type of the potentials ("morse" or "harmonic")
            by default, the "morse" type is used
        a
            ground state scaling coefficient
            by default, is equal to 1.0 1/a_0 -- for "morse" potential,
                                        a_0 -- for "harmonic" potential
        De
            ground state dissociation energy value
            by default, is equal to 20000.0 1/cm for "morse" potential,
                        is a dummy variable for "harmonic" potential
        x0p
            shift of the upper potential relative to the ground one
            by default, is equal to -0.17 a_0 for double potentials,
                        is identically equated to zero for filtering / single morse / single harmonic tasks
        a_e
            excited state scaling coefficient
            by default, is equal to 1.0 1/a_0 -- for "morse" potential,
                        is equal to 1.0 a_0 -- for "harmonic" potential,
                        is identically equated to zero for filtering / single morse / single harmonic tasks
        De_e
            excited state dissociation energy value
            by default, is equal to 10000.0 1/cm for "morse" potential,
                        is a dummy variable for "harmonic" potential,
                        is identically equated to zero for filtering / single morse / single harmonic tasks
        Du
            energy shift between the minima of the upper potential and the ground one
            by default, is equal to 20000.0 1/cm for double potentials,
                        is identically equated to zero for filtering / single morse / single harmonic tasks
        wf_type
            type of the wavefunctions ("morse" or "harmonic")
            by default, the "morse" type is used
        x0
            coordinate initial condition
            by default, is equal to 0.0
        p0
            momentum initial condition
            by default, is equal to 0.0
        L
            spatial range of the problem in a_0
            by default, is equal to 5.0 a_0
        T
            time range of the problem in sec
            by default, is equal to 600e15 s
        np
            number of collocation points; must be a power of 2
            by default, is equal to 1024
        nch
            number of Chebyshev interpolation points; must be a power of 2
            by default, is equal to 64
        nt
            number of time grid points
            by default, is equal to 420000
        E0
            amplitude value of the laser field energy envelope in 1 / cm
            by default, is equal to 71.54 1 / cm,
                        is identically equated to zero for filtering / single morse / single harmonic tasks
        t0
            initial time, when the laser field reaches its maximum value, in sec
            is a dummy variable for filtering / single morse / single harmonic tasks
            by default, is equal to 300e-15 s
        sigma
            scaling parameter of the laser field envelope in sec
            is a dummy variable for filtering / single morse / single harmonic tasks
            by default, is equal to 50e-15 s
        nu_L
            basic frequency of the laser field in Hz
            by default, is equal to 0.29297e15 Hz,
                        is identically equated to zero for filtering / single morse / single harmonic tasks

Examples:
    python newcheb.py --jsonfile "input.json"
            perform a propagation task using the parameter values specified in the json file
            "input.json" or the default ones if something wasn't provided in the file
"""

__author__ = "Irene Mizus (irenem@hit.ac.il)"
__license__ = "Python"

from tools import print_err

import sys
import getopt
import json
import math

import grid_setup
import fitter
import reporter
import task_manager
from config import *


def usage():
    """ Print usage information """
    print (__doc__)


def _warning_collocation_points(np, np_min):
    print_err("WARNING: The number of collocation points np = {} should be more than an estimated initial value {}. "
          "You've got a divergence!".format(np, np_min))


def _warning_time_steps(nt, nt_min):
    print_err("WARNING: The number of time steps nt = {} should be more than an estimated value {}. "
          "You've got a divergence!".format(nt, nt_min))

def main(argv):
    """ The main() function """
    # analyze cmdline:
    try:
        options, arguments = getopt.getopt(argv, 'hf:', ['help', 'jsonfile='])
    except getopt.GetoptError:
        print_err("\tThere are unrecognized options!")
        print_err("\tRun this script with '-h' option to see the usage info and available options.")
        sys.exit(2)

    file_json = None
    # analyze provided options and their values (if any):
    for opt, val in options:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-f", "--jsonfile"):
            file_json = val

    if 'file_json' not in locals():
        print("\tNo input json file was provided. The default values of calculation parameters will be used")
    else:
        with open(file_json, "r") as read_file:
            json_data = {}
            data = json.load(read_file)

    conf = RootConfiguration()
    conf.load(data)

    # analyze provided json data
    if not math.log2(conf.fitter.propagation.np).is_integer() or not math.log2(conf.fitter.propagation.nch).is_integer():
        raise ValueError("The number of collocation points 'np' and of Chebyshev "
                         "interpolation points 'nch' must be positive integers and powers of 2")

    if conf.output.table.lmin < 0 or conf.output.plot.lmin < 0:
        raise ValueError("The number 'lmin' of time iteration, from which the result"
                         "should be written to a file or plotted should be positive or 0")

    if conf.output.table.mod_fileout < 0 or conf.fitter.mod_log < 0:
        raise ValueError("The numbers 'mod_fileout' and 'mod_log' should be positive or 0")

    if conf.output.plot.mod_plotout < 0 or conf.output.plot.mod_update < 0:
        raise ValueError("The step for plotting graphs with x-axis = time 'mod_plotout' "
                         "and for updating the plots 'mod_update' should be positive or 0")

    if conf.output.plot.number_plotout < 2:
        raise ValueError("The maximum number of graphs 'number_plotout' to be plotted on one canvas"
                         "must be larger than 1!")

    if conf.fitter.impulses_number < 0:
        raise ValueError("The number of laser pulses 'impulses_number' should be positive or 0")

    if conf.fitter.iter_max < 0 and (conf.fitter.task_type == conf.fitter.TaskType.OPTIMAL_CONTROL_KROTOV or
            conf.fitter.task_type == conf.fitter.TaskType.OPTIMAL_CONTROL_GRADIENT):
        raise ValueError("The maximum number of iterations in the optimal control task 'iter_max' should be positive or 0")

    if conf.fitter.propagation.L <= 0.0 or conf.fitter.propagation.T <= 0.0:
        raise ValueError("The value of spatial range 'L' and of time range 'T' of the problem"
                         "must be positive")

    if conf.fitter.propagation.m <= 0.0 or conf.fitter.propagation.a <= 0.0 or conf.fitter.propagation.De <= 0.0:
        raise ValueError("The value of a reduced mass 'm/mass', of a scaling factor 'a'"
                         "and of a dissociation energy 'De' must be positive")

    if not conf.fitter.propagation.E0 >= 0.0 or not conf.fitter.propagation.sigma > 0.0 or not conf.fitter.propagation.nu_L >= 0.0:
        raise ValueError("The value of an amplitude value of the laser field energy envelope 'E0',"
                         "of a scaling parameter of the laser field envelope 'sigma'"
                         "and of a basic frequency of the laser field 'nu_L' must be positive")


    if conf.fitter.task_type == conf.FitterConfiguration.TaskType.FILTERING or \
            conf.fitter.task_type == conf.FitterConfiguration.TaskType.SINGLE_POT:

        print("A '%s' task begins. E0 and nu_L values are zeroed..."
              % str(conf.fitter.task_type).split(".")[-1].lower())
        conf.fitter.propagation.E0 = 0.0
        conf.fitter.propagation.nu_L = 0.0

        if conf.fitter.impulses_number != 0:
            print("For the task_type = '%s' the impulses_number value will be replaced by zero"
                  % str(conf.fitter.task_type).split(".")[-1].lower())
            conf.fitter.impulses_number = 0
    else:
        if conf.fitter.task_type == conf.fitter.TaskType.TRANS_WO_CONTROL:
            print("An ordinary transition task begins...")
            if conf.fitter.impulses_number != 1:
                print("For the task_type = 'trans_wo_control' the impulses_number value will be replaced by 1")
                conf.fitter.impulses_number = 1
        elif conf.fitter.task_type == conf.FitterConfiguration.TaskType.INTUITIVE_CONTROL:
            print("An intuitive control task begins...")
            if conf.fitter.impulses_number < 2:
                print("For the task_type = 'intuitive_control' the impulses_number value will be replaced by 2")
                conf.fitter.impulses_number = 2
        elif conf.fitter.task_type == conf.FitterConfiguration.TaskType.LOCAL_CONTROL_POPULATION:
            print("A local control with goal population task begins...")
            if conf.fitter.impulses_number != 1:
                print("For the task_type = 'local_control_population' the impulses_number value will be replaced by 1")
                conf.fitter.impulses_number = 1
        elif conf.fitter.task_type == conf.FitterConfiguration.TaskType.LOCAL_CONTROL_PROJECTION:
            print("A local control with goal projection task begins...")
            if conf.fitter.impulses_number != 1:
                print("For the task_type = 'local_control_projection' the impulses_number value will be replaced by 1")
                conf.fitter.impulses_number = 1
        elif conf.fitter.task_type == conf.FitterConfiguration.TaskType.OPTIMAL_CONTROL_KROTOV:
            print("An optimal control task with Krotov method begins...")
            if conf.fitter.impulses_number != 1:
                print("For the task_type = 'optimal_control_krotov' the impulses_number value will be replaced by 1")
                conf.fitter.impulses_number = 1
        elif conf.fitter.task_type == conf.FitterConfiguration.TaskType.OPTIMAL_CONTROL_GRADIENT:
            print("An optimal control task with gradient method begins...")
            if conf.fitter.impulses_number != 1:
                print("For the task_type = 'optimal_control_gradient' the impulses_number value will be replaced by 1")
                conf.fitter.impulses_number = 1
        else:
            raise RuntimeError("Impossible case in the TaskType class")


    task_manager_imp = task_manager.create(conf.fitter)

    # setup of the grid
    grid = grid_setup.GridConstructor(conf.fitter.propagation)
    dx, x = grid.grid_setup()

    # evaluating of initial wavefunction
    psi0 = task_manager_imp.psi_init(x, conf.fitter.propagation.np, conf.fitter.propagation.x0,
                                     conf.fitter.propagation.p0, conf.fitter.propagation.m,
                                     conf.fitter.propagation.De, conf.fitter.propagation.a)

    # evaluating of the final goal
    psif = task_manager_imp.psi_goal(x, conf.fitter.propagation.np, conf.fitter.propagation.x0,
                                     conf.fitter.propagation.p0, conf.fitter.propagation.x0p,
                                     conf.fitter.propagation.m, conf.fitter.propagation.De,
                                     conf.fitter.propagation.De_e, conf.fitter.propagation.Du,
                                     conf.fitter.propagation.a, conf.fitter.propagation.a_e)

    # main calculation part
    fit_reporter_imp = reporter.MultipleFitterReporter(conf.output)
    fit_reporter_imp.open()

    fitting_solver = fitter.FittingSolver(conf.fitter, psi0, psif, task_manager_imp.pot, fit_reporter_imp,
                                          _warning_collocation_points,
                                          _warning_time_steps
                                          )
    fitting_solver.time_propagation(dx, x)
    fit_reporter_imp.close()


if __name__ == "__main__":
    main(sys.argv[1:])