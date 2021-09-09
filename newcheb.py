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
    lmin
        number of a time step, from which the result should be written to a file.
        A negative value will be considered as 0
        by default, is equal to 0
    mod_stdout
        step of output to stdout (to write to stdout each <val>-th time step).
        By default, is equal to 500
    mod_fileout
        step of writing in file (to write in file each <val>-th time step).
        By default, is equal to 100
    file_abs
        output file name, to which absolute values of wavefunctions should be written
        by default, is equal to "fort.21"
    file_real
        output file name, to which real parts of wavefunctions should be written
        by default, is equal to "fort.22"
    file_mom
        output file name, to which expectation values of x, x*x, p, p*p should be written
        by default, is equal to "fort.23"

    in key "fitter":
    task_type
        type of the calculation task:
        "trans_wo_control"  - calculation of transition from the ground state
                              to the excited one under the influence of external
                              non-controlled laser field with gaussian envelope and a constant
                              chirp (by default)
        "filtering"         - filtering task
                              in this case E0 and nu_L are zeroing mandatory
        "intuitive_control" - calculation of transitions from the ground state
                              to the excited state and back to the ground one
                              under the influence of a sequence of equal laser pulses
                              with gaussian envelopes and a constant chirps
        "local_control"     - calculation of transition from the ground state
                              to the excited one under the influence of external
                              laser field with controlled envelope form (by the local control
                              algorithm) and a constant chirp
    k_E
        aspect ratio for the inertial "force" in equation for the laser field energy in sec^(-2).
        Applicable for the task_type = "local_control", only. For all other cases is a dummy variable
        by default, is equal to 1e29
    lamb
        aspect ratio for the decay term in equation for the laser field energy in 1 / sec.
        Applicable for the task_type = "local_control", only. For all other cases is a dummy variable
        by default, is equal to 4e14
    pow
        power value in the decay term in equation for the laser field energy.
        Applicable for the task_type = "local_control", only. For all other cases is a dummy variable
        by default, is equal to 0.8
    epsilon
        small parameter for cutting of an imaginary part in dA/dt.
        Applicable for the task_type = "local_control", only. For all other cases is a dummy variable
        by default, is equal to 1e-15
    impulses_number
        number of laser pulses in the "intuitive_control" task type
        the values more than 1 are applicable for the task_type = "intuitive_control", only.
        In this case if a value less than 2 provided, it will be replaced by 2
        for the task_type = "filtering" it will be replaced by 0
        for the task_type = "trans_wo_control" or "local_control" it will be replaced by 1
        by default, is equal to 1
    delay
        time delay between the laser pulses in sec
        is a dummy variable for impulses_number less than 2
        by default, is equal to 600e-15
    propagation
        a dictionary that contains the parameters, which are used in a simple propagation task with laser field:
        m
            reduced mass value of the considered system
            for dimensionless problem, should be equal to 1.0
            by default, is equal to 0.5 Dalton
        pot_type
            type of the potentials ("morse" or "harmonic")
            by default, the "morse" type is used
        a
            scaling coefficient for dimensional problem
            by default, is equal to 1.0 1/a_0 -- for "morse" potential,
                                        a_0 -- for "harmonic" potential
        De
            dissociation energy value for dimensional problem
            by default, is equal to 20000.0 1/cm for "morse" potential
                        is a dummy variable for "harmonic" potential
        x0p
            shift of the upper potential relative to the ground one
            by default, is equal to -0.17 a_0
        wf_type
            type of the wavefunctions ("morse" or "harmonic")
            by default, the "morse" type is used
        x0
            coordinate initial conditions for dimensionless problem
            by default, is equal to 0.0
        p0
            momentum initial conditions for dimensionless problem
            by default, is equal to 0.0
        L
            spatial range of the problem (in a_0 if applicable)
            for dimensionless problem, should be equal to 15.0
            by default, is equal to 5.0 a_0
        T
            time range of the problem in sec or in pi (half periods) units
            for dimensionless problem, should be equal to 0.1
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
            by default, is equal to 71.54 1 / cm
        t0
            initial time, when the laser field reaches its maximum value, in sec
            by default, is equal to 300e-15 s
        sigma
            scaling parameter of the laser field envelope in sec
            by default, is equal to 50e-15 s
        nu_L
            basic frequency of the laser field in Hz
            by default, is equal to 0.29297e15 Hz

Examples:
    python newcheb.py --jsonfile "input.json"
            perform a propagation task using the parameter values specified in the json file
            "input.json" or the default ones if something wasn't provided in the file
"""

__author__ = "Irene Mizus (irenem@hit.ac.il)"
__license__ = "Python"


OUT_PATH="output"

import os.path
import sys
import getopt
import json
import math

import double_morse
import harmonic
import phys_base
import fitter
from config import *


def usage():
    """ Print usage information """

    print (__doc__)


def _warning_collocation_points(np, np_min):
    print("WARNING: The number of collocation points np = {} should be more than an estimated initial value {}. "
          "You've got a divergence!".format(np, np_min), file=sys.stderr)


def _warning_time_steps(nt, nt_min):
    print("WARNING: The number of time steps nt = {} should be more than an estimated value {}. "
          "You've got a divergence!".format(nt, nt_min), file=sys.stderr)


def plot_file(psi, t, x, np, f_abs, f_real):
    """ Plots absolute and real values of the current wavefunction """
    for i in range(np):
        f_abs.write("{:.6f} {:.6f} {:.6e}\n".format(t * 1e+15, x[i], abs(psi[i])))
        f_real.write("{:.6f} {:.6f} {:.6e}\n".format(t * 1e+15, x[i], psi[i].real))
        f_abs.flush()
        f_real.flush()


def plot_mom_file(t, momx, momx2, momp, momp2, ener, E, overlp, tot, abs_psi_max, real_psi_max, file_mom):
    """ Plots expectation values of the current x, x*x, p and p*p """
    file_mom.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    t * 1e+15, momx.real, momx2.real, momp.real, momp2.real, ener, E, abs(overlp), tot, abs_psi_max, real_psi_max))
    file_mom.flush()


def plot_test_file(l, phi_l, phi_u, f):
    f.write("Step number: {0}\n".format(l))
    f.write("Lower state wavefunction:")
    for i in range(len(phi_l)):
        f.write("{0}\n".format(phi_l[i]))
    f.write("Upper state wavefunction:")
    for i in range(len(phi_u)):
        f.write("{0}\n".format(phi_u[i]))


def main(argv):
    """ The main() function """
    # analyze cmdline:
    try:
        options, arguments = getopt.getopt(argv, 'hf:', ['help', 'jsonfile='])
    except getopt.GetoptError:
        print("\tThere are unrecognized options!", sys.stderr)
        print("\tRun this script with '-h' option to see the usage info and available options.", sys.stderr)
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

    if conf.output.lmin < 0 or conf.output.mod_fileout < 0 or conf.output.mod_stdout < 0 or conf.fitter.impulses_number < 0:
        raise ValueError("The number of laser pulses 'impulses_number', "
                         "the number 'lmin' of time iteration, from which the result"
                         "should be written to a file, as well as steps of output "
                         "'mod_stdout' and 'mod_fileout' should be positive or 0")

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


    if conf.fitter.propagation.pot_type == conf.FitterConfiguration.PropagationConfiguration.PotentialType.MORSE:
        print("Morse potentials are used")
        pot = double_morse.pot
    elif conf.fitter.propagation.pot_type == conf.FitterConfiguration.PropagationConfiguration.PotentialType.HARMONIC:
        print("Harmonic potentials are used")
        pot = harmonic.pot
    else:
        raise RuntimeError("Impossible case in the PotentialType class")

    if conf.fitter.propagation.wf_type == conf.FitterConfiguration.PropagationConfiguration.WaveFuncType.MORSE:
        print("Morse wavefunctions are used")
        psi_init = double_morse.psi_init
    elif conf.fitter.propagation.wf_type == conf.FitterConfiguration.PropagationConfiguration.WaveFuncType.HARMONIC:
        print("Harmonic wavefunctions are used")
        psi_init = harmonic.psi_init
    else:
        raise RuntimeError("Impossible case in the WaveFuncType class")

    if conf.fitter.task_type == conf.FitterConfiguration.TaskType.FILTERING:
        print("A filtering task begins. E0 ans nu_L values are zeroed...")
        conf.fitter.propagation.E0 = 0.0
        conf.fitter.propagation.nu_L = 0.0
        if conf.fitter.impulses_number != 0:
            print("For the task_type = 'filtering' the impulses_number value will be replaced by zero")
            conf.fitter.impulses_number = 0
    elif conf.fitter.task_type == conf.FitterConfiguration.TaskType.TRANS_WO_CONTROL:
        print("An ordinary transition task begins...")
        if conf.fitter.impulses_number != 1:
            print("For the task_type = 'trans_wo_control' the impulses_number value will be replaced by 1")
            conf.fitter.impulses_number = 1
    elif conf.fitter.task_type == conf.FitterConfiguration.TaskType.INTUITIVE_CONTROL:
        print("An intuitive control task begins...")
        if conf.fitter.impulses_number < 2:
            print("For the task_type = 'intuitive_control' the impulses_number value will be replaced by 2")
            conf.fitter.impulses_number = 2
    elif conf.fitter.task_type == conf.FitterConfiguration.TaskType.LOCAL_CONTROL:
        if conf.fitter.task_subtype == conf.FitterConfiguration.TaskSubType.GOAL_POPULATION:
            print("A local control with goal population task begins...")
        elif conf.fitter.task_subtype == conf.FitterConfiguration.TaskSubType.GOAL_MOMENTUM:
            print("A local control with goal momentum task begins...")
        else:
            raise RuntimeError("Impossible case in the TaskSubType class")
        if conf.fitter.impulses_number != 1:
            print("For the task_type = 'local_control' the impulses_number value will be replaced by 1")
            conf.fitter.impulses_number = 1
    else:
        raise RuntimeError("Impossible case in the TaskType class")

    # main calculation part
    with open(os.path.join(OUT_PATH, conf.output.file_abs), 'w') as f_abs, \
         open(os.path.join(OUT_PATH, conf.output.file_real), 'w') as f_real, \
         open(os.path.join(OUT_PATH, conf.output.file_mom), 'w') as f_mom, \
         open(os.path.join(OUT_PATH, conf.output.file_abs) + "_exc", 'w') as f_abs_up, \
         open(os.path.join(OUT_PATH, conf.output.file_real + "_exc"), 'w') as f_real_up, \
         open(os.path.join(OUT_PATH, conf.output.file_mom + "_exc"), 'w') as f_mom_up:

        def plot(psi, t, x, np):
            plot_file(psi, t, x, np, f_abs, f_real)


        def plot_mom(t, moms: phys_base.ExpectationValues, ener, E, overlp, ener_tot, abs_psi_max, real_psi_max):
            plot_mom_file(t, moms.x_l, moms.x2_l, moms.p_l, moms.p2_l, ener, E, overlp, ener_tot,
                          abs_psi_max, real_psi_max, f_mom)


        def plot_up(psi, t, x, np):
            plot_file(psi, t, x, np, f_abs_up, f_real_up)


        def plot_mom_up(t, moms, ener, E, overlp, overlp_tot, abs_psi_max, real_psi_max):
            plot_mom_file(t, moms.x_u, moms.x2_u, moms.p_u, moms.p2_u, ener, E, overlp, overlp_tot,
                          abs_psi_max, real_psi_max, f_mom_up)


        fitting_solver = fitter.FittingSolver(conf, psi_init, pot,
                                              _warning_collocation_points,
                                              _warning_time_steps,
                                              plot,
                                              plot_up,
                                              plot_mom,
                                              plot_mom_up
                                              )

        fitting_solver.time_propagation()


if __name__ == "__main__":
    main(sys.argv[1:])