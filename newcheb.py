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

    in key "phys_syst_pars":
    m
        reduced mass value of the considered system
        for dimensionless problem, should be equal to 1.0
        by default, is equal to 0.5 Dalton

    in key "potential_pars":
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

    in key "laser_field_pars":
    impulses_number
        number of laser pulses in the "intuitive_control" task type
        the values more than 1 are applicable for the task_type = "intuitive_control", only.
        In this case if a value less than 2 provided, it will be replaced by 2
        for the task_type = "filtering" it will be replaced by 0
        for the task_type = "trans_wo_control" or "local_control" it will be replaced by 1
        by default, is equal to 1
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
    delay
        time delay between the laser pulses in sec
        is a dummy variable for impulses_number less than 2
        by default, is equal to 600e-15

    in key "phys_calc_pars":
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
    wf_type
        type of the wavefunctions ("morse" or "harmonic")
        by default, the "morse" type is used
    L
        spatial range of the problem (in a_0 if applicable)
        for dimensionless problem, should be equal to 15.0
        by default, is equal to 5.0 a_0
    T
        time range of the problem in sec or in pi (half periods) units
        for dimensionless problem, should be equal to 0.1
        by default, is equal to 600e15 s

    in key "alg_calc_pars":
    np
        number of collocation points; must be a power of 2
        by default, is equal to 1024
    nch
        number of Chebyshev interpolation points; must be a power of 2
        by default, is equal to 64
    nt
        number of time grid points
        by default, is equal to 420000
    epsilon
        small parameter for cutting of an imaginary part in dA/dt.
        Applicable for the task_type = "local_control", only. For all other cases is a dummy variable
        by default, is equal to 1e-15

    in key "init_conditions":
    x0
        coordinate initial conditions for dimensionless problem
        by default, is equal to 0.0
    p0
        momentum initial conditions for dimensionless problem
        by default, is equal to 0.0

    in key "print_pars":
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
from enum import Enum

import double_morse
import harmonic
import propagation
import phys_base


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


class PotentialType(Enum):
    MORSE = 0
    HARMONIC = 1


class WaveFuncType(Enum):
    MORSE = 0
    HARMONIC = 1


class TaskType(Enum):
    TRANS_WO_CONTROL = 0
    FILTERING = 1
    INTUITIVE_CONTROL = 2
    LOCAL_CONTROL = 3


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

    # Default json data values
    m = 0.5  # Dalton
    L = 5.0  # a_0
    # 5.0 a_0 -- for the working transition between PECs; # 0.2 -- for a model harmonic oscillator with a = 1.0; # 4.0 a_0 -- for morse oscillator; # 6.0 a_0 -- for dimensional harmonic oscillator
    np = 1024
    # 1024 -- for the working transition between PECs and two laser pulses; # 128 -- for a model harmonic oscillator with a = 1.0; # 2048 -- for morse oscillator and filtering on the ground PEC (99.16% quality); # 512 -- for dimensional harmonic oscillator
    nch = 64
    T = 600e-15  # s
    # 1200 fs -- for two laser pulses; # 280 (600) fs -- for the working transition between PECs; # 2240 fs -- for filtering on the ground PEC (99.16% quality)
    nt = 420000
    # 840000 -- for two laser pulses; 200000 (420000) -- for the working transition between PECs; # 900000 -- for filtering on the ground PEC (99.16% quality)
    x0 = 0  # TODO: to fix x0 != 0
    p0 = 0  # TODO: to fix p0 != 0
    a = 1.0  # 1/a_0 -- for morse oscillator, a_0 -- for harmonic oscillator
    De = 20000.0  # 1/cm
    x0p = -0.17  # a_0
    E0 = 71.54  # 1/cm
    t0 = 300e-15  # s
    sigma = 50e-15  # s
    nu_L = 0.29297e15  # Hz
    # 0.29297e15 -- for the working transition between PECs; # 0.5879558e15 -- analytical difference b/w excited and ground energies; # 0.5859603e15 -- calculated difference b/w excited and ground energies !!; # 0.599586e15 = 20000 1/cm
    delay = 600e-15  # s
    impulses_number = 1
    lmin = 0
    mod_stdout = 500
    mod_fileout = 100
    epsilon = 1e-15

    file_abs = "fort.21"
    file_real = "fort.22"
    file_mom = "fort.23"

    pot_type = PotentialType.MORSE
    wf_type = WaveFuncType.MORSE
    task = TaskType.TRANS_WO_CONTROL

    if 'file_json' not in locals():
        print("\tNo input json file was provided. The default values of calculation parameters will be used")
    else:
        with open(file_json, "r") as read_file:
            json_data = {}
            data = json.load(read_file)

        # analyze provided input json file
        if "phys_syst_pars" in data:
            if "m" in data["phys_syst_pars"]: m = float(data["phys_syst_pars"]["m"])
        else:
            print("Section 'phys_syst_pars' wasn't provided in the input json file. The default values of calculation parameters will be used")
        if "phys_calc_pars" in data:
            if "L" in data["phys_calc_pars"]: L = float(data["phys_calc_pars"]["L"])
            if "T" in data["phys_calc_pars"]: T = float(data["phys_calc_pars"]["T"])
            if "wf_type" in data["phys_calc_pars"]: wf_type = WaveFuncType[data["phys_calc_pars"]["wf_type"].upper()]
            if "task_type" in data["phys_calc_pars"]: task = TaskType[data["phys_calc_pars"]["task_type"].upper()]
        else:
            print("Section 'phys_calc_pars' wasn't provided in the input json file. The default values of calculation parameters will be used")
        if "alg_calc_pars" in data:
            if "np" in data["alg_calc_pars"]: np = int(data["alg_calc_pars"]["np"])
            if "nch" in data["alg_calc_pars"]: nch = int(data["alg_calc_pars"]["nch"])
            if "nt" in data["alg_calc_pars"]: nt = int(data["alg_calc_pars"]["nt"])
            if "epsilon" in data["alg_calc_pars"]: epsilon = float(data["alg_calc_pars"]["epsilon"])
        else:
            print("Section 'alg_calc_pars' wasn't provided in the input json file. The default values of calculation parameters will be used")
        if "init_conditions" in data:
            if "x0" in data["init_conditions"]: x0 = float(data["init_conditions"]["x0"])
            if "p0" in data["init_conditions"]: p0 = float(data["init_conditions"]["p0"])
        else:
            print("Section 'init_conditions' wasn't provided in the input json file. The default values of calculation parameters will be used")
        if "potential_pars" in data:
            if "a" in data["potential_pars"]: a = float(data["potential_pars"]["a"])
            if "De" in data["potential_pars"]: De = float(data["potential_pars"]["De"])
            if "x0p" in data["potential_pars"]: x0p = float(data["potential_pars"]["x0p"])
            if "pot_type" in data["potential_pars"]: pot_type = PotentialType[data["potential_pars"]["pot_type"].upper()]
        else:
            print("Section 'potential_pars' wasn't provided in the input json file. The default values of calculation parameters will be used")
        if "laser_field_pars" in data:
            if "E0" in data["laser_field_pars"]: E0 = float(data["laser_field_pars"]["E0"])
            if "t0" in data["laser_field_pars"]: t0 = float(data["laser_field_pars"]["t0"])
            if "sigma" in data["laser_field_pars"]: sigma = float(data["laser_field_pars"]["sigma"])
            if "nu_L" in data["laser_field_pars"]: nu_L = float(data["laser_field_pars"]["nu_L"])
            if "delay" in data["laser_field_pars"]: delay = float(data["laser_field_pars"]["delay"])
            if "impulses_number" in data["laser_field_pars"]: impulses_number = int(data["laser_field_pars"]["impulses_number"])
        else:
            print("Section 'laser_field_pars' wasn't provided in the input json file. The default values of calculation parameters will be used")
        if "print_pars" in data:
            if "lmin" in data["print_pars"]: lmin = int(data["print_pars"]["lmin"])
            if "mod_stdout" in data["print_pars"]: mod_stdout = int(data["print_pars"]["mod_stdout"])
            if "mod_fileout" in data["print_pars"]: mod_fileout = int(data["print_pars"]["mod_fileout"])
            if "file_abs" in data["print_pars"]: file_abs = data["print_pars"]["file_abs"]
            if "file_real" in data["print_pars"]: file_real = data["print_pars"]["file_real"]
            if "file_mom" in data["print_pars"]: file_mom = data["print_pars"]["file_mom"]
        else:
            print("Section 'print_pars' wasn't provided in the input json file. The default values of calculation parameters will be used")

    # analyze provided json data
    if pot_type == PotentialType.MORSE:
        print("Morse potentials are used")
        pot = double_morse.pot
    elif pot_type == PotentialType.HARMONIC:
        print("Harmonic potentials are used")
        pot = harmonic.pot
    else:
        raise RuntimeError("Impossible case in the PotentialType class")

    if wf_type == WaveFuncType.MORSE:
        print("Morse wavefunctions are used")
        psi_init = double_morse.psi_init
    elif wf_type == WaveFuncType.HARMONIC:
        print("Harmonic wavefunctions are used")
        psi_init = harmonic.psi_init
    else:
        raise RuntimeError("Impossible case in the WaveFuncType class")

    if not math.log2(np).is_integer() or not math.log2(nch).is_integer():
        raise ValueError("The number of collocation points 'np' and of Chebyshev "
                         "interpolation points 'nch' must be positive integers and powers of 2")

    if lmin < 0 or mod_fileout < 0 or mod_stdout < 0 or impulses_number < 0:
        raise ValueError("The number of laser pulses 'impulses_number', "
                         "the number 'lmin' of time iteration, from which the result"
                         "should be written to a file, as well as steps of output "
                         "'mod_stdout' and 'mod_fileout' should be positive or 0")

    if not L > 0.0 or not T > 0.0:
        raise ValueError("The value of spatial range 'L' and of time range 'T' of the problem"
                         "must be positive")

    if not m > 0.0 or not a > 0.0 or not De > 0.0:
        raise ValueError("The value of a reduced mass 'm/mass', of a scaling factor 'a'"
                         "and of a dissociation energy 'De' must be positive")

    if not E0 >= 0.0 or not sigma > 0.0 or not nu_L >= 0.0:
        raise ValueError("The value of an amplitude value of the laser field energy envelope 'E0',"
                         "of a scaling parameter of the laser field envelope 'sigma'"
                         "and of a basic frequency of the laser field 'nu_L' must be positive")

    if task == TaskType.FILTERING:
        print("A filtering task begins. E0 ans nu_L values are zeroed...")
        E0 = 0.0
        nu_L = 0.0
        if impulses_number != 0:
            print("For the task_type = 'filtering' the impulses_number value will be replaced by zero")
            impulses_number = 0
    elif task == TaskType.TRANS_WO_CONTROL:
        print("An ordinary transition task begins...")
        if impulses_number != 1:
            print("For the task_type = 'trans_wo_control' the impulses_number value will be replaced by 1")
            impulses_number = 1
    elif task == TaskType.INTUITIVE_CONTROL:
        print("An intuitive control task begins...")
        if impulses_number < 2:
            print("For the task_type = 'intuitive_control' the impulses_number value will be replaced by 2")
            impulses_number = 2
    elif task == TaskType.LOCAL_CONTROL:
        print("A local control task begins...")
        if impulses_number != 1:
            print("For the task_type = 'local_control' the impulses_number value will be replaced by 1")
            impulses_number = 1
    else:
        raise RuntimeError("Impossible case in the TaskType class")


    # main propagation loop
    with open(os.path.join(OUT_PATH, file_abs), 'w') as f_abs, \
         open(os.path.join(OUT_PATH, file_real), 'w') as f_real, \
         open(os.path.join(OUT_PATH, file_mom), 'w') as f_mom, \
         open(os.path.join(OUT_PATH, file_abs) + "_exc", 'w') as f_abs_up, \
         open(os.path.join(OUT_PATH, file_real + "_exc"), 'w') as f_real_up, \
         open(os.path.join(OUT_PATH, file_mom + "_exc"), 'w') as f_mom_up, \
         open("test", 'w') as f:


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


        def plot_test(l, phi_l, phi_u):
            plot_test_file(l, phi_l, phi_u, f)


        dt = 0
        stat_saved = propagation.PropagationSolver.StaticState()
        dyn_ref = propagation.PropagationSolver.DynamicState()
        milliseconds_full = 0
        res_saved = propagation.PropagationSolver.StepReaction.OK
        E_patched = 0.0
        dAdt_happy = 0.0

        def report_static(stat: propagation.PropagationSolver.StaticState):
            nonlocal stat_saved
            stat_saved = stat

            nonlocal dt
            dt = stat.dt

            # check if input data are correct in terms of the given problem
            # calculating the initial energy range of the Hamiltonian operator H
            emax0 = stat.v[0][1][0] + abs(stat.akx2[int(np / 2 - 1)]) + 2.0
            emin0 = stat.v[0][0]

            # calculating the initial minimum number of collocation points that is needed for convergence
            np_min0 = int(
                math.ceil(L *
                          math.sqrt(
                              2.0 * m * (emax0 - emin0) * phys_base.dalt_to_au / phys_base.hart_to_cm) /
                          math.pi
                          )
            )

            # calculating the initial minimum number of time steps that is needed for convergence
            nt_min0 = int(
                math.ceil((emax0 - emin0) * T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h
                          )
            )

            if np < np_min0:
                _warning_collocation_points(np, np_min0)
            if nt < nt_min0:
                _warning_time_steps(nt, nt_min0)

            cener0_tot = stat.cener0 + stat.cener0_u
            overlp0_abs = abs(stat.overlp00) + abs(stat.overlpf0)

            # plotting initial values
            plot(stat.psi0[0], 0.0, stat.x, np)
            plot_up(stat.psi0[1], 0.0, stat.x, np)

            plot_mom(0.0, stat.moms0, stat.cener0.real, stat.E00.real, stat.overlp00, cener0_tot.real,
                     abs(stat.psi0[0][520]), stat.psi0[0][520].real)
            plot_mom_up(0.0, stat.moms0, stat.cener0_u.real, stat.E00.real, stat.overlpf0, overlp0_abs,
                     abs(stat.psi0[1][520]), stat.psi0[1][520].real)

            print("Initial emax = ", emax0)

            print(" Initial state features: ")
            print("Initial normalization: ", abs(stat.cnorm0))
            print("Initial energy: ", abs(stat.cener0))

            print(" Final goal features: ")
            print("Final goal normalization: ", abs(stat.cnormf))
            print("Final goal energy: ", abs(stat.cenerf))


        def report_dynamic(dyn: propagation.PropagationSolver.DynamicState):
            nonlocal dyn_ref
            dyn_ref = dyn


        def process_instrumentation(instr: propagation.PropagationSolver.InstrumentationOutputData):
            t = dt * dyn_ref.l

            # calculating the minimum number of collocation points and time steps that are needed for convergence
            nt_min = int(math.ceil((instr.emax - instr.emin) * T * phys_base.cm_to_erg / 2.0 / phys_base.Red_Planck_h))
            np_min = int(math.ceil(
                L * math.sqrt(
                    2.0 * m * (instr.emax - instr.emin) * phys_base.dalt_to_au / phys_base.hart_to_cm) / math.pi))

            cener = instr.cener_l + instr.cener_u
            overlp_abs = abs(instr.overlp0) + abs(instr.overlpf)

            time_span = instr.time_after - instr.time_before
            milliseconds_per_step = time_span.microseconds / 1000
            nonlocal milliseconds_full
            milliseconds_full += milliseconds_per_step

            # local control algorithm
            coef = 2.0 * phys_base.cm_to_erg / phys_base.Red_Planck_h
            dAdt = dyn_ref.E * instr.psigc_psie.imag * coef

            nonlocal dAdt_happy
            nonlocal E_patched
            nonlocal epsilon

            if task != TaskType.LOCAL_CONTROL:
                res = propagation.PropagationSolver.StepReaction.OK
            else:
                if dAdt >= 0.0:
                    res = propagation.PropagationSolver.StepReaction.OK
                    dAdt_happy = dAdt
                else:
                    if abs(instr.psigc_psie.imag) > epsilon:
                        E_patched = -dAdt_happy / (instr.psigc_psie.imag * coef)
                    else:
                        print("Imaginary part in dA/dt is too small and has been replaces by epsilon")
                        E_patched = dAdt_happy / (epsilon * coef)
                    res = propagation.PropagationSolver.StepReaction.REPEAT

            # plotting the result
            if dyn_ref.l % mod_fileout == 0 and res == propagation.PropagationSolver.StepReaction.OK:
                if dyn_ref.l >= lmin:
                    plot(dyn_ref.psi[0], t, stat_saved.x, np)
                    plot_up(dyn_ref.psi[1], t, stat_saved.x, np)

                if dyn_ref.l >= lmin:
                    plot_mom(t, instr.moms, instr.cener_l.real, dyn_ref.E, instr.overlp0, cener.real,
                             abs(dyn_ref.psi[0][520]), dyn_ref.psi[0][520].real)
                    plot_mom_up(t, instr.moms, instr.cener_u.real, instr.E_full.real, instr.overlpf, overlp_abs,
                                abs(dyn_ref.psi[1][520]), dyn_ref.psi[1][520].real)

            if dyn_ref.l % mod_stdout == 0:
                if np < np_min:
                    _warning_collocation_points(np, np_min)
                if nt < nt_min:
                    _warning_time_steps(nt, nt_min)

                print("l = ", dyn_ref.l)
                print("t = ", t * 1e15, "fs")

                print("emax = ", instr.emax)
                print("emin = ", instr.emin)
                print("normalized scaled time interval = ", instr.t_sc)
                print("normalization on the lower state = ", abs(instr.cnorm_l))
                print("normalization on the upper state = ", abs(instr.cnorm_u))
                print("overlap with initial wavefunction = ", abs(instr.overlp0))
                print("overlap with final goal wavefunction = ", abs(instr.overlpf))
                print("energy on the lower state = ", instr.cener_l.real)
                print("energy on the upper state = ", instr.cener_u.real)
                print("Time derivation of the expectation value from the goal operator A = ", dAdt)

                print(
                    "milliseconds per step: " + str(milliseconds_per_step) + ", on average: " + str(
                        milliseconds_full / dyn_ref.l))

                if res !=  propagation.PropagationSolver.StepReaction.OK:
                    print("REPEATNG THE ITERATION")

            nonlocal res_saved
            res_saved = res
            return res


        # Calculating envelope of the laser field energy at the given time value
        def LaserFieldEnvelope(stat: propagation.PropagationSolver.StaticState,
                               dyn: propagation.PropagationSolver.DynamicState):
            if res_saved == propagation.PropagationSolver.StepReaction.OK:
                t = stat.dt * dyn.l
                E = phys_base.laser_field(E0, t, t0, sigma)
                if task == TaskType.INTUITIVE_CONTROL:
                    for npul in range(1, impulses_number):
                        E += phys_base.laser_field(E0, t, t0 + (npul * delay), sigma)
            elif res_saved == propagation.PropagationSolver.StepReaction.REPEAT:
                nonlocal E_patched
                E = E_patched
            else:
                raise RuntimeError("Impossible case")

            return E


        solver = propagation.PropagationSolver(
            psi_init, pot,
            report_static=report_static,
            report_dynamic=report_dynamic,
            process_instrumentation=process_instrumentation,
            laser_field_envelope=LaserFieldEnvelope,
            m=m, L=L, np=np, nch=nch, T=T, nt=nt, x0=x0, p0=p0, a=a, De=De, x0p=x0p, E0=E0,
            t0=t0, sigma=sigma, nu_L=nu_L, delay=delay)

        solver.time_propagation()


if __name__ == "__main__":
    main(sys.argv[1:])