"""
A software package aimed for numerical calculation of different quantum system types.
Namely, it models molecular dynamics for small quantum systems, as well as propagation dynamics
for general unitary transformations, and their behaviour under a controlled laser field excitation.
The package is developed in Python, uses a Newtonian polynomial algorithm with a Chebyshev-based
interpolation scheme, is designed to be modular and easily expendable,
and also supports different types of quantum systems and different methods of quantum control.
The implementation has a flexible object-oriented structure, includes a simple configuration DSL allowing
feeding separate configurations to specific modules with changing input parameters,
an automated plotting system for making output reports and graphs, and also an automated unit-testing mechanism.

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
            a path name for the file with global table, which contains the results in case of varying
            calculation parameters.
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
        plot.inumber_plotout
            maximum number of graphs with x-axis = "distance" to be plotted on a single canvas.
            By default, is equal to 15
        plot.gr_iter
            output file name, to which the iteration dependencies of the "global" values should be plotted.
            By default, is equal to "fig_iter.html"
        plot.gr_iter_E
            output file name, to which the iteration dependency of the laser field energy envelope should be plotted.
            By default, is equal to "fig_iter_E.html"
        plot.gr_iter_F
            output file name, to which the iteration dependency of the F operator value, which reflects closeness to
            the goal state in the unitary transformation algorithm (task_type = "optimal_control_unit_transform") as
            in the article [J.P. Palao, R. Kosloff, Phys. Rev. A, 68, 062308 (2003)], should be plotted.
            By default, is equal to "fig_iter_F.html"
        plot.gr_iter_E_int
            output file name, to which the iteration dependency of the integral value of laser field strength
            E_int = sum(E(t) * conjugate(E(t)) * time_step) should be plotted.
            By default, is equal to "fig_iter_E_int.html"
        plot.gr_iter_J
            output file name, to which the iteration dependency of the J = F - h_lambda**2 * E_int operator value
            should be plotted.
            By default, is equal to "fig_iter_J.html"

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
            plot.gr_abs
                output file name, to which absolute values of wavefunctions should be plotted.
                By default, is equal to "fig_abs{level}.html"
                ({level} is replaced automatically by "0" for the ground state and by "1" for the excited one)
            plot.gr_real
                output file name, to which real parts of wavefunctions should be plotted.
                By default, is equal to "fig_real{level}.html"
                ({level} is replaced automatically by "0" for the ground state and by "1" for the excited one)
            plot.gr_moms
                output file name, to which the expectation values <x>, <x^2>, <p>, and <p^2> should be plotted.
                By default, is equal to "fig_moms{level}.html"
                ({level} is replaced automatically by "0" for the ground state and by "1" for the excited one)
            plot.gr_*
                output file name, to which the corresponding result should be plotted.
                By default, is equal to "fig_*.html"
                Currently, the following names are supported:
                * = ener, norm, overlp0, overlpf, abs_max, real_max, smoms,
                    ener_tot, overlp0_tot, overlpf_tot, lf_en, lf_fr


    Content of the json_task file

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
                                           with gaussian envelopes and constant chirps
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
        "optimal_control_unit_transform" - calculation of a unitary transformation of a system with either Hadamard H1
                                           Hamiltonian or a Bose-Hubbard one using a controlled external laser field
                                           with an iterative Krotov algorithm and the squared modulus functional Fsm
    pot_type
        type of the potentials ("morse", "harmonic" or "none").
        By default, the "morse" type is used
    wf_type
        type of the wavefunctions ("morse", "harmonic" or "const").
        By default, the "morse" type is used
    hamil_type
        type of the Hamiltonian operator used ("ntriv", "two_levels" (Hadamard H1 Hamiltonian) or
        "BH_model" (Bose-Hubbard Hamiltonian)).
        By default, the "ntriv" type is used
    lf_aug_type
        a way of adding the controlling laser field to "BH_model"-type Hamiltonian.
        For all other variants of "hamil_type" variables is a dummy variable.
        Available options:
        "z" - H = -2 * delta * Jx + 2 * U * Jx**2 + 2 * E(t) * Jz (by default)
        "x" - H = -2 * delta * Jx * E(t) + 2 * U * Jz + 2 * W * Jz^2
    init_guess
        type of initial guessed laser field envelope used in propagation tasks.
        Available options:
        "zero"      - without external laser field (by default)
        "const"     - a constant envelope
        "gauss"     - a gaussian-like envelope
        "sqrsin"    - a squared sinus envelope
        "maxwell"   - a maxwell distribution-like envelope
    init_guess_hf
        type of the high-frequency part of the initial guessed laser field used in propagation tasks.
        Available options:
        "exp"       - exponential type exp(i omega_L t) (by default)
        "cos"       - cos-type cos(omega_L t)
        "sin"       - sin-type sin(omega_L t)
        "cos_set"   - a sequence of cos-type terms [w_0 * cos(omega_L t) + sum(w_i * cos(omega_L t * k) + w_j * cos(omega_L t / k))]
        "sin_set"   - a sequence of sin-type terms [w_0 * sin(omega_L t) + w_1 * sin(omega_L t * 3) + w_2 * sin(omega_L t * 5) + ...)]
                     (should be used together with a sqrsin- or gauss-type envelope if a symmetric initial guess is needed,
                     for example, for an Hadamard-like Hamiltonian in a unitary transformation task)
    nb
        number of basis vectors of the Hilbert space used in the calculation task.
        By default, is equal to 1
    nlevs
        number of levels in basis vectors of the Hilbert space used in the calculation task.
        By default, is equal to 2
    T
        time range of the problem in sec.
        By default, is equal to 600e-15 s
    L
        spatial range of the problem in a_0.
        By default, is equal to 5.0 a_0
    np
        number of collocation points; must be a power of 2.
        By default, is equal to 1024
    De
        ground state dissociation energy value.
        By default, is equal to 20000.0 1/cm for "morse" potential,
                    is a dummy variable for "harmonic" and "none" potentials
    De_e
        excited state dissociation energy value.
        By default, is equal to 10000.0 1/cm for "morse" potential,
                    is a dummy variable for "harmonic" and "none" potentials,
                    is identically equated to zero for filtering / single morse / single harmonic tasks
    Du
        energy shift between the minima of the upper potential and the ground one.
        By default, is equal to 20000.0 1/cm for double potentials,
                    is identically equated to zero for filtering / single morse / single harmonic tasks
    x0p
        shift of the upper potential relative to the ground one.
        By default, is equal to -0.17 a_0 for double potentials,
                    is identically equated to zero for filtering / single morse / single harmonic tasks
    x0
        coordinate initial condition.
        By default, is equal to 0.0
    p0
        momentum initial condition.
        By default, is equal to 0.0
    a
        ground state scaling coefficient.
        By default, is equal to 1.0 1/a_0 -- for "morse" potential,
                                1.0 a_0 -- for "harmonic" potential,
                                is a dummy variable for a "none" potential
    a_e
        excited state scaling coefficient.
        By default, is equal to 1.0 1/a_0 -- for "morse" potential,
                    is equal to 1.0 a_0 -- for "harmonic" potential,
                    is a dummy variable for a "none" potential,
                    is identically equated to zero for filtering / single morse / single harmonic tasks
    U, W, delta
        parameters of angular momentum-type Hamiltonian (applicable for 'hamil_type' = 'BH_model' only),
        U, W and delta are in units of 1 / cm.
        For lf_aug_type = "x" and nb <= 2: W value should be equal to U or 0.0.
        For lf_aug_type = "x" and nb = 4: W value should be equal to 2 * U.
        For lf_aug_type = "z": W is a dummy variable
        By default, U, W are equal to 0.0; delta is equal to 1.0
    t0_auto
        parameter that controls the way of using t0 parameter.
        If specified as "True", is calculated automatically as a function of T as follows:
            for  init_guess = "sqrsin"  -- t0 = 0.0 s
            for  init_guess = "maxwell" -- t0 = 0.0 s
            for  init_guess = "gauss"   -- t0 = T / 2 s
        Must be explicitly set to "True" if a batch calculation with init_guess = "gauss" is running!
        By default, is "False"
    nt_auto
        parameter that controls the way of using nt parameter.
        If specified as "True", is calculated automatically as a function of T:
            nt = floor(T / 2.0)
        By default, is "False"
    sigma_auto
        parameter that controls the way of using sigma parameter.
        If specified as "True", is calculated automatically as a function of T as follows:
            for  init_guess = "sqrsin"  -- sigma = 2 * T
            for  init_guess = "maxwell" -- sigma = T / 5
            for  init_guess = "gauss"   -- sigma = T / 8
        Must be explicitly set to "True" if a batch calculation is running!
        By default, is "False"
    nu_L_auto
        parameter that controls the way of using nu_L parameter.
        If specified as "True", is calculated automatically as a function of T: nu_L = 1 / (2 * T).
        By default, is "False"

    In key "fitter":
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
        By default, is equal to 2
    delay
        time delay between the laser pulses in sec.
        Is a dummy variable for impulses_number less than 2.
        By default, is equal to 600e-15 s
    iter_max
        maximum iteration number for the "optimal_control_..." task_type in case if a divergence with the
        given criteria hasn't been reached. Is a dummy variable for all other task types.
        If iter_max = -1, there is no limitation on the maximum number of iterations.
        By default, is equal to -1
    iter_mid_1
        iteration number for the "optimal_control_..." task_type, which is used to predict if the calculation converges
        or not. If a current value of the operator F at the corresponding iteration is less that its value at
        the iteration iter_mis_2 or is larger that the value nb * nb * q,
        the calculation is considered as divergent, and the calculation stops.
        Is a dummy variable for all other task types.
        By default, is equal to 0
    iter_mid_2
        iteration number for the "optimal_control_..." task_type, which is used to predict if the calculation converges
        or not. If a current value of the operator F at the corresponding iteration is larger that the value nb * nb * q,
        the calculation is considered as divergent, and the calculation stops.
        Is a dummy variable for all other task types.
        By default, is equal to 1
    q
        a coefficient for the "optimal_control_..." task_type, which is used to predict if the calculation converges
        or not. If a current value of the operator F at two prespecified iterations iter_mid_1 and iter_mid_1
        is larger that the value nb * nb * q,
        the calculation is considered as divergent, and the calculation stops.
        Is a dummy variable for all other task types.
        By default, is equal to 0.0 (the check is switched off)
    h_lambda
        parameter, which is applicable for the task_type = "optimal_control_..." only.
        For all other cases is a dummy variable.
        By default, is equal to 0.0066
    h_lambda_mode
        type of using the h_lambda parameter. Applicable for the
        task_type = "optimal_control_unit_transform" only.
        For all other cases is a dummy variable.
        Available options:
        "const"     - constant value given in input json file (by default)
        "dynamical" - dynamically changeable parameter obtained as
                      h_lambda * nb / (nb - abs(sum(<psi_k_goal|psi_k(T)>)))
    w_list
        a list of 2 * pcos - 1 amplitudes for separate harmonics of the laser field high-frequency part of type "cos_set"
        or "sin_set".
        Is a dummy variable for all other types of "init_guess_hf".
        If not specified or is empty, the amplitude values are generated randomly.
        By default, is equal to []
    w_min
        a minimum value of an element from w_list, which can be randomly generated.
        By default, is equal to -2.0.
    w_max
        a maximum value of an element from w_list, which can be randomly generated.
        By default, is equal to 2.0.
    hf_hide
        parameter that specifies if we should get rid of the high-frequency part of laser field during the propagation part.
        By default, is "True"
    pcos
        maximum frequency multiplier for a sum [w_0 * cos(omega_L t) + sum(w_i * cos(omega_L t * k) + w_j * cos(omega_L t / k))]
        or [w_0 * sin(omega_L t) + w_1 * sin(omega_L t * 3) + w_2 * sin(omega_L t * 5) + ... + w_(2 * pcos - 2) * sin(omega_L t * (4 * pcos - 3)))]
        with k = 2 ... pcos in the cases "init_guess_hf" = "cos_set" and "sin_set", or just a frequency multiplier itself for
        the "init_guess_hf" = "cos" and "sin" cases, which is used in a high-frequency part for the laser field initial guess.
        For the "init_guess_hf" = "exp" case is a dummy variable.
        In the cases "init_guess_hf" = "cos_set" and "sin_set" the maximum frequency multiplier equal to floor(pcos) will be used;
        it has to be greater than 1 then.
        By default, is equal to 1
    Em
        a multiplier used for evaluation of the laser field energy maximum value (E_max = E0 * Em),
        which can be reached during the controlling procedure
        (for the "BH_model"-type Hamiltonian with lf_aug_type = "z", only).
        For all other variants of Hamiltonian is a dummy variable.
        By default, is equal to 1.5
    F_type
        a type of the F functional, which is used to evaluate closeness to the goal states in the unitary transformation
        algorithm (task_type = "optimal_control_unit_transform") as in the article
        [J.P. Palao, R. Kosloff, Phys. Rev. A, 68, 062308 (2003)]. Is a dummy variable for all other task_type cases.
         Available options:
         "sm" - "squared module" type of the functional (F_sm) (by default)
         "re" - "real" type of the functional (F_re)
         "ss" - "state-to-state" type of the functional (F_ss)
    mod_log
        step of output to stdout (to write to stdout each <val>-th time step).
        By default, is equal to 500

    propagation
        a dictionary that contains the parameters, which are used in a simple propagation task with laser field:
        m
            reduced mass value of the considered system.
            By default, is equal to 0.5 Dalton
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
            Is a dummy variable for filtering / single morse / single harmonic tasks and a "const" type of the envelope.
            By default, is equal to 300e-15 s
        sigma
            scaling parameter of the laser field envelope in sec.
            Is a dummy variable for filtering / single morse / single harmonic tasks and a "const" type of the envelope.
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

For batch runs:
    - Update script "generate_jsons.sh" by changing the paths for --json_rep and --json_task options and run it:
      ./generate_jsons.sh
      It generates a set of json input files and puts it into a directory "./batch_jsons"
    - Inside the directory ./input_batch run one of the scripts "launch_run*.sh". The output and error files will be
      put into a directory "../batch_jsons_out", the calculation results will be put into the directory specified in the
      json_rep file, in option fitter.out_path
    - Update a script "launch_pp*.sh" by providing the correct input/output folder paths and run it to get the
      postprocessing statistics for the obtained results; the output and error files will be put into a directory
      "../batch_jsons_out"
    - Scripts "runs_reordering*.sh" can be used to rename the folders with calculation results if more than
      a single batch run was made; usage:
      ./runs_reordering*.sh N, where N is the number of runs made during the previous batch run. Then the folders with
      all the results can be placed into a single output folder
"""

__author__ = "Irene Mizus (irenem@hit.ac.il)"
__license__ = "Python"

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
    pretty_print_json = pformat(conf_task).replace("'", '"').replace("True", "true").replace("False", "false")
    with open(file_name, "w") as finp:
        finp.write(pretty_print_json)

def print_input(conf_rep_plot, conf_task, file_name):
    with open(os.path.join(conf_rep_plot.fitter.out_path, file_name), "w") as finp:
        finp.write("task_type:\t\t"   f"{conf_task.task_type}\n")
        finp.write("iter_max:\t\t"   f"{conf_task.fitter.iter_max}\n")
        finp.write("iter_mid_1:\t\t"   f"{conf_task.fitter.iter_mid_1}\n")
        finp.write("iter_mid_2:\t\t"   f"{conf_task.fitter.iter_mid_2}\n")
        finp.write("q:\t\t\t"   f"{conf_task.fitter.q}\n")
        finp.write("epsilon:\t\t"   f"{conf_task.fitter.epsilon:.1E}\n")

        finp.write("nb:\t\t\t"   f"{conf_task.nb}\n")
        finp.write("nlevs:\t\t\t"   f"{conf_task.nlevs}\n")
        finp.write("wf_type:\t\t"   f"{conf_task.wf_type}\n")

        finp.write("impulses_number:\t"   f"{conf_task.fitter.impulses_number}\n")
        finp.write("Em:\t\t\t"   f"{conf_task.fitter.Em}\n")
        finp.write("E0:\t\t\t"   f"{conf_task.fitter.propagation.E0}\n")
        finp.write("t0:\t\t\t"   f"{conf_task.fitter.propagation.t0}\n")
        finp.write("t0_auto:\t\t"   f"{conf_task.t0_auto}\n")
        finp.write("sigma:\t\t\t"   f"{conf_task.fitter.propagation.sigma:.6E}\n")
        finp.write("sigma_auto:\t\t"   f"{conf_task.sigma_auto}\n")
        finp.write("nu_L:\t\t\t"   f"{conf_task.fitter.propagation.nu_L:.6E}\n")
        finp.write("nu_L_auto:\t\t"   f"{conf_task.nu_L_auto}\n")
        finp.write("h_lambda:\t\t"   f"{conf_task.fitter.h_lambda}\n")
        finp.write("h_lambda_mode:\t\t"   f"{conf_task.fitter.h_lambda_mode}\n")
        finp.write("init_guess:\t\t"   f"{conf_task.init_guess}\n")
        finp.write("init_guess_hf:\t\t"   f"{conf_task.init_guess_hf}\n")
        finp.write("F_type:\t\t\t"   f"{conf_task.fitter.F_type}\n")
        finp.write("pcos:\t\t\t"   f"{conf_task.fitter.pcos}\n")
        finp.write("hf_hide:\t\t"   f"{conf_task.fitter.hf_hide}\n")
        finp.write("w_list:\t\t\t"   f"{conf_task.fitter.w_list}\n")
        finp.write("w_min:\t\t\t"   f"{conf_task.fitter.w_min}\n")
        finp.write("w_max:\t\t\t"   f"{conf_task.fitter.w_max}\n")

        finp.write("hamil_type:\t\t"   f"{conf_task.hamil_type}\n")
        finp.write("U:\t\t\t"   f"{conf_task.U}\n")
        finp.write("W:\t\t\t"   f"{conf_task.W}\n")
        finp.write("delta:\t\t\t"   f"{conf_task.delta}\n")
        finp.write("lf_aug_type:\t\t"   f"{conf_task.lf_aug_type}\n")

        finp.write("pot_type:\t\t"   f"{conf_task.pot_type}\n")
        finp.write("Du:\t\t\t"   f"{conf_task.Du}\n")

        finp.write("np:\t\t\t"   f"{conf_task.np}\n")
        finp.write("L:\t\t\t"   f"{conf_task.L}\n")
        finp.write("nch:\t\t\t"   f"{conf_task.fitter.propagation.nch}\n")
        finp.write("nt:\t\t\t"   f"{conf_task.fitter.propagation.nt}\n")
        finp.write("nt_auto:\t\t"   f"{conf_task.nt_auto}\n")
        finp.write("T:\t\t\t"   f"{conf_task.T:.6E}\n")


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

                if not math.log2(conf_task.np).is_integer() or not math.log2(
                        conf_task.fitter.propagation.nch).is_integer():
                    raise ValueError(
                        "The number of collocation points, 'np', and of Chebyshev "
                        "interpolation points, 'nch', have to be positive integers and powers of 2")

                if conf_task.fitter.impulses_number < 0:
                    raise ValueError(
                        "The number of laser pulses, 'impulses_number', has to be positive or 0")

                if conf_task.nb < 1:
                    raise ValueError(
                        "The number of basis vectors of the Hilbert space, 'nb', has to be positive")

                if conf_task.nlevs < 2:
                    raise ValueError(
                        "The number of levels in basis vectors of the Hilbert space, 'nlevs', has to be larger than 1")

                if conf_task.nlevs < conf_task.nb:
                    raise ValueError(
                        "The number of levels in basis vectors of the Hilbert space, 'nlevs', must be larger or equal to "
                        "the number of basis vectors themselves, 'nb'")

                if conf_task.fitter.Em <= 0:
                    raise ValueError(
                        "The multiplier 'Em' used for evaluation of the laser field energy maximum value, "
                        "which can be reached during the controlling procedure, has to be positive")

                if conf_task.fitter.q < 0:
                    raise ValueError(
                        "The multiplier 'q', which is used to predict if the calculation converges or not, "
                        "has to be positive or 0.0")

                if (conf_task.fitter.iter_max < -1 or conf_task.fitter.iter_mid_1 < 0 or conf_task.fitter.iter_mid_2 < 0) \
                        and (conf_task.task_type == conf_task.TaskType.OPTIMAL_CONTROL_KROTOV or
                        conf_task.task_type == conf_task.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM):
                    raise ValueError(
                        "The maximum and two threshold numbers of iterations in the optimal control task, "
                        "'iter_max', 'iter_mid_1' and 'iter_mid_2', "
                        "have to be positive or 0; 'iter_max' can also be equal to -1")

                if conf_task.fitter.iter_mid_2 <= conf_task.fitter.iter_mid_1:
                    raise ValueError(
                        "Parameter 'iter_mid_1' must be less than 'iter_mid_2'")

                if conf_task.L <= 0.0 or conf_task.T <= 0.0:
                    raise ValueError(
                        "The value of spatial range, 'L', and of time range, 'T', of the problem have to be positive")

                if conf_task.fitter.propagation.m <= 0.0 or conf_task.a <= 0.0 or conf_task.De <= 0.0:
                    raise ValueError(
                        "The value of a reduced mass, 'm/mass', of a scaling factor, 'a', and of a dissociation energy, 'De', have to be positive")

                if not conf_task.fitter.propagation.E0 >= 0.0 or not conf_task.fitter.propagation.sigma > 0.0 or not conf_task.fitter.propagation.nu_L >= 0.0:
                    raise ValueError(
                        "The amplitude value of the laser field energy envelope, 'E0',"
                        "of a scaling parameter of the laser field envelope, 'sigma',"
                        "and of a basic frequency of the laser field, 'nu_L', have to be positive")

                if conf_task.run_id != "no_id" and conf_task.sigma_auto != True:
                    raise ValueError("A batch calculation is running. 'sigma_auto' parameter must be set to 'True'!")

                if conf_task.run_id != "no_id" and \
                   conf_task.t0_auto != True and \
                   conf_task.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.GAUSS:
                    raise ValueError("A batch calculation with init_guess = 'gauss' is running. 't0_auto' parameter must be set to 'True'!")

                if conf_task.task_type == conf_task.TaskType.FILTERING or \
                        conf_task.task_type == conf_task.TaskType.SINGLE_POT:
                    print("A '%s' task begins..." % str(conf_task.task_type).split(".")[-1].lower())

                    if conf_task.fitter.propagation.E0 != 0.0:
                        raise ValueError(
                            "For the 'task_type' = '%s' the amplitude value of the laser field energy envelope, 'E0', has to be equal to zero"
                            % str(conf_task.task_type).split(".")[-1].lower())

                    if conf_task.fitter.propagation.nu_L != 0.0:
                        raise ValueError(
                            "For the 'task_type' = '%s' the value of a basic frequency of the laser field, 'nu_L', has to be equal to zero"
                            % str(conf_task.task_type).split(".")[-1].lower())

                    if conf_task.init_guess != conf_task.fitter.InitGuess.ZERO:
                        raise ValueError(
                            "For the 'task_type' = '%s' the initial guess type for the laser field envelope, 'init_guess', has to be 'zero'"
                            % str(conf_task.task_type).split(".")[-1].lower())

                    if conf_task.fitter.impulses_number != 0:
                        raise ValueError(
                            "For the 'task_type' = '%s' the 'impulses_number' value has to be equal to zero"
                            % str(conf_task.task_type).split(".")[-1].lower())
                else:
                    if conf_task.task_type == conf_task.TaskType.TRANS_WO_CONTROL:
                        print("An ordinary transition task begins...")

                        if conf_task.fitter.impulses_number != 1:
                            raise ValueError(
                                "For the 'task_type' = 'trans_wo_control' the 'impulses_number' value has to be equal to 1")

                    elif conf_task.task_type == conf_task.TaskType.INTUITIVE_CONTROL:
                        print("An intuitive control task begins...")

                        if conf_task.init_guess == "zero":
                            raise ValueError(
                                "For the 'task_type' = 'intuitive_control' the initial guess type for the laser field envelope, "
                                "'init_guess', mustn't be 'zero'")

                        if conf_task.fitter.impulses_number < 2:
                            raise ValueError(
                                "For the 'task_type' = 'intuitive_control' the 'impulses_number' value has to be larger than 1")

                        if not conf_task.fitter.hf_hide:
                            raise ValueError(
                                "For the 'task_type' = 'intuitive_control' the 'hf_hide' value has to be equal to 'true'")

                    elif conf_task.task_type == conf_task.TaskType.LOCAL_CONTROL_POPULATION:
                        print("A local control with goal population task begins...")

                        if conf_task.init_guess == "zero":
                            raise ValueError(
                                "For the 'task_type' = 'local_control_population' the initial guess type for the laser field envelope, "
                                "'init_guess', mustn't be 'zero'")

                        if conf_task.fitter.impulses_number != 1:
                            raise ValueError(
                                "For the 'task_type' = 'local_control_population' the 'impulses_number' value has to be equal to 1")

                        if not conf_task.fitter.hf_hide:
                            raise ValueError(
                                "For the 'task_type' = 'local_control_population' the 'hf_hide' value has to be equal to 'true'")

                    elif conf_task.task_type == conf_task.TaskType.LOCAL_CONTROL_PROJECTION:
                        print("A local control with goal projection task begins...")

                        if conf_task.init_guess == "zero":
                            raise ValueError(
                                "For the 'task_type' = 'local_control_projection' the initial guess type for the laser field envelope, "
                                "'init_guess', mustn't be 'zero'")

                        if conf_task.fitter.impulses_number != 1:
                            raise ValueError(
                                "For the 'task_type' = 'local_control_projection' the 'impulses_number' value has to be equal to 1")

                        if not conf_task.fitter.hf_hide:
                            raise ValueError(
                                "For the 'task_type' = 'local_control_projection' the 'hf_hide' value has to be equal to 'true'")

                    elif conf_task.task_type == conf_task.TaskType.OPTIMAL_CONTROL_KROTOV:
                        print("An optimal control task with Krotov method begins...")

                        if conf_task.init_guess == "zero":
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_krotov' the initial guess type for the laser field envelope, "
                                "'init_guess', mustn't be 'zero'")

                        if conf_task.fitter.impulses_number != 1:
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_krotov' the 'impulses_number' value has to be equal to 1")

                        if not conf_task.fitter.hf_hide:
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_krotov' the 'hf_hide' value has to be equal to 'true'")

                    elif conf_task.task_type == conf_task.TaskType.OPTIMAL_CONTROL_UNIT_TRANSFORM:
                        print("An optimal control task with unitary quantum Fourier transformation begins...")

                        if conf_task.init_guess == "zero":
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_unit_transform' the initial guess type for the laser field envelope, "
                                "'init_guess', mustn't be 'zero'")

                        if conf_task.np > 1:
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_unit_transform' the number of collocation points, 'np', has to be equal to 1")

                        if conf_task.fitter.impulses_number != 1:
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_unit_transform' the 'impulses_number' value has to be equal to 1")

                        if conf_task.L == 1.0:
                            pass
                        else:
                            raise ValueError(
                                "For the 'task_type' = 'optimal_control_unit_transform' the spatial range of the problem, 'L', has to be equal to 1.0")

                    else:
                        raise RuntimeError("Impossible case in the TaskType class")

                # defining the automatically obtained parameters
                nw = int(2 * conf_task.fitter.pcos - 1)
                if not conf_task.fitter.w_list:
                    conf_task.fitter.w_list = [x for x in numpy.random.default_rng().uniform(conf_task.fitter.w_min,
                                                                                             conf_task.fitter.w_max,
                                                                                             nw)]
                else:
                    assert len(conf_task.fitter.w_list) == nw

                if conf_task.sigma_auto == True:
                    if conf_task.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.SQRSIN:
                        conf_task.fitter.propagation.sigma = numpy.float64(2.0 * conf_task.T) #TODO: to add a possibility to vary groups of parameters
                    elif conf_task.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.MAXWELL:
                        conf_task.fitter.propagation.sigma = numpy.float64(conf_task.T / 5.0)
                    elif conf_task.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.GAUSS:
                        conf_task.fitter.propagation.sigma = numpy.float64(conf_task.T / 8.0)
                    else:
                        pass

                if conf_task.t0_auto == True:
                    if conf_task.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.SQRSIN:
                        conf_task.fitter.propagation.t0 = numpy.float64(0.0) #TODO: to add a possibility to vary groups of parameters
                    elif conf_task.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.MAXWELL:
                        conf_task.fitter.propagation.t0 = numpy.float64(0.0)
                    elif conf_task.init_guess == TaskRootConfiguration.FitterConfiguration.InitGuess.GAUSS:
                        conf_task.fitter.propagation.t0 = numpy.float64(conf_task.T / 2.0)
                    else:
                        pass

                if conf_task.nt_auto == True:
                    conf_task.fitter.propagation.nt = math.floor(conf_task.T / 2.0E-15)
                else:
                    pass

                # printing final values of all the calculation parameters to the file
                step = -1
                if not os.path.exists(conf_rep_plot.fitter.out_path):
                    os.makedirs(conf_rep_plot.fitter.out_path, exist_ok=True)
                print_input(conf_rep_plot, conf_task, "table_inp_" + str(step) + ".txt")

                # setup of the task
                task_manager_imp = task_manager.create(conf_task)

                # main calculation part
                fit_reporter_imp = reporter.MultipleFitterReporter(conf_rep_table=conf_rep_table.fitter, conf_rep_plot=conf_rep_plot.fitter)
                fit_reporter_imp.open()

                fitting_solver = fitter.FittingSolver(conf_task.fitter, conf_task.task_type, conf_task.T,
                                                      conf_task.np, conf_task.L,
                                                      task_manager_imp.init_dir, task_manager_imp.ntriv,
                                                      task_manager_imp.psi0, task_manager_imp.psif,
                                                      task_manager_imp.v, task_manager_imp.akx2,
                                                      task_manager_imp.F_goal,
                                                      task_manager_imp.laser_field, task_manager_imp.laser_field_hf,
                                                      task_manager_imp.F_type, task_manager_imp.aF_type,
                                                      task_manager_imp.hamil2D, fit_reporter_imp,
                                                      _warning_collocation_points,
                                                      _warning_time_steps
                                                      )
                fitting_solver.time_propagation(task_manager_imp.dx,
                                                task_manager_imp.x,
                                                task_manager_imp.t_step,
                                                task_manager_imp.t_list)
                fit_reporter_imp.close()
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
