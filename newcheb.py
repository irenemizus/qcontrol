""" A Python script for solving a controlled propagation
task in a two-potential quantum system using a Newtonian polynomial
algorithm with a Chebyshev-based interpolation scheme.

Usage: python newcheb.py [options]

Options:
    -h, --help
        print this usage info and exit
    -m, --mass
        reduced mass value of the considered system
        by default, is equal to 0.5 Dalton for dimensional problem
                    is equal to 1.0 for dimensionless problem
    -L
        spatial range of the problem (in a_0 if applicable)
        by default, is equal to 5 a_0 for dimensional problem
                    is equal to 15 for dimensionless problem
    --np
        number of collocation points; must be a power of 2
        by default, is equal to 1024
    --nch
        number of Chebyshev interpolation points; must be a power of 2
        by default, is equal to 64
    --T
        time range of the problem in femtosec or in pi (half periods) units
        by default, is equal to 280.0 fs for dimensional problem
                    is equal to 0.1 for dimensionless problem
    --nt
        number of time grid points
        by default, is equal to 100000
    --x0
        coordinate initial conditions for dimensionless problem
        by default, is equal to 0
    --p0
        momentum initial conditions for dimensionless problem
        by default, is equal to 0
    -a
        scaling coefficient for dimensional problem
        by default, is equal to 1.0 1/a_0 -- for morse oscillator, a_0 -- for harmonic oscillator
    --De
        dissociation energy value for dimensional problem
        by default, is equal to 20000.0 1 / cm
    --E0
        amplitude value of the laser field energy envelope in 1 / cm
        by default, is equal to 71.68 1 / cm
    --t0
        initial time, when the laser field is switched on, in femtosec
        by default, is equal to 140 fs
    --sigma
        scaling parameter of the laser field envelope in femtosec
        by default, is equal to 50 fs
    --nu_L
        basic frequency of the laser field in PHz
        by default, is equal to 0.293 PHz
    --lmin
        number of a time step, from which the result should be written to a file.
        A negative value will be considered as 0
        by default, is equal to 0
    --mod_stdout
        step of output to stdout (to write to stdout each <val>-th time step).
        By default, is equal to 500
    --mod_fileout
        step of writing in file (to write in file each <val>-th time step).
        By default, is equal to 100
    --file_abs
        output file name, to which absolute values of wavefunctions should be written
        by default, is equal to "fort.21"
    --file_real
        output file name, to which real parts of wavefunctions should be written
        by default, is equal to "fort.22"
    --file_mom
        output file name, to which expectation values of x, x*x, p, p*p should be written
        by default, is equal to "fort.23"

Examples:
    python newcheb.py  --file_abs "res_abs" -L 30
            perform a propagation task using spatial range of the dimensionless problem equal to 30,
            the name "res_abs" for the absolute wavefunctions values output file, and
            default values for other parameters
"""

__author__ = "Irene Mizus (irenem@hit.ac.il)"
__license__ = "Python"


import double_morse

OUT_PATH="output"

import os
import os.path
import sys
import getopt
import propagation


def usage():
    """ Print usage information """

    print (__doc__)


def plot_file(psi, t, x, np, f_abs, f_real):
    """ Plots absolute and real values of the current wavefunction """
    for i in range(np):
        f_abs.write("{:.6f} {:.6f} {:.6e}\n".format(t * 1e+15, x[i], abs(psi[i])))
        f_real.write("{:.6f} {:.6f} {:.6e}\n".format(t * 1e+15, x[i], psi[i].real))


def plot_mom_file(t, momx, momx2, momp, momp2, ener, E, overlp, ener_tot, file_mom):
    """ Plots expectation values of the current x, x*x, p and p*p """
    file_mom.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(t * 1e+15, momx.real, momx2.real, momp.real, momp2.real, ener, E, abs(overlp), ener_tot))


def plot_test_file(l, phi_l, phi_u, f):
    f.write("Step number: {0}\n".format(l))
    f.write("Lower state wavefunction:")
    for i in range(len(phi_l)):
        f.write("{0}\n".format(phi_l[i]))
    f.write("Upper state wavefunction:")
    for i in range(len(phi_u)):
        f.write("{0}\n".format(phi_u[i]))

def main(argv):
    import ctypes;
    mkl_rt = ctypes.CDLL('libmkl_rt.so');
    print(mkl_rt.mkl_get_max_threads())

    """ The main() function """
    # analyze cmdline:
    try:
        options, arguments = getopt.getopt(argv, 'hm:L:a:T:', ['help', 'mass=', '', '', '', 'np=', 'nch=', 'nt=', 'x0=', 'p0=', \
                                          'x0p=', 'De=', 'E0=', 't0=', 'sigma=', 'nu_L=', 'delay=', 'lmin=', 'mod_stdout=', \
                                          'mod_fileout', 'file_abs=', 'file_real=', 'file_mom='])
    except getopt.GetoptError:
        print("\tThere are unrecognized options!", sys.stderr)
        print("\tRun this script with '-h' option to see the usage info and available options.", sys.stderr)
        sys.exit(2)

    # default filenames
    file_abs = "fort.21"
    file_real = "fort.22"
    file_mom = "fort.23"

    # Default argument values
    m = 0.5  # Dalton
    L = 5.0  # a_0   0.2 -- for a model harmonic oscillator with a = 1.0 # 4.0 a_0 -- for morse oscillator # 6.0 a_0 -- for dimensional harmonic oscillator
    np = 2048  # 128 -- for a model harmonic oscillator with a = 1.0 # 2048 -- for morse oscillator # 512 -- for dimensional harmonic oscillator
    nch = 64
    T = 280e-15  # s -- for morse oscillator
    nt = 100000
    x0 = 0  # TODO: to fix x0 != 0
    p0 = 0  # TODO: to fix p0 != 0
    a = 1.0  # 1/a_0 -- for morse oscillator, a_0 -- for harmonic oscillator
    De = 20000.0  # 1/cm
    x0p = 1.0 # a_0
    E0 = 71.68  # 1/cm
    t0 = 140e-15  # s
    sigma = 50e-15  # s
    nu_L = 0.293e15 #0.5879558e15  # 0.5859603e15 - calculated difference b/w excited and ground energies !!, #0.599586e15, # Hz
    delay = 300e-15 #s
    lmin = 0
    mod_stdout = 500
    mod_fileout = 100

    # analyze provided options and their values (if any):
    for opt, val in options:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-m", "--mass"):
            m = float(val)
        elif opt == "-L":
            L = float(val)
        elif opt == "--np":
            np = int(val)
        elif opt == "--nch":
            nch = int(val)
        elif opt == "-T":
            T = float(val)
        elif opt == "-a":
            a = float(val)
        elif opt == "--nt":
            nt = int(val)
        elif opt == "--x0":
            x0 = float(val)
        elif opt == "--p0":
            p0 = float(val)
        elif opt == "--De":
            De = float(val)
        elif opt == "--x0p":
            x0p = float(val)
        elif opt == "--E0":
            E0 = float(val)
        elif opt == "--t0":
            t0 = float(val)
        elif opt == "--sigma":
            sigma = float(val)
        elif opt == "--nu_L":
            nu_L = float(val)
        elif opt == "--delay":
            delay = float(val)
        elif opt == "--lmin":
            lmin = int(val)
        elif opt == "--mod_stdout":
            mod_stdout = int(val)
        elif opt == "--mod_fileout":
            mod_fileout = int(val)
        elif opt == "--file_abs":
            file_abs = val
        elif opt == "--file_real":
            file_real = val
        elif opt == "--file_mom":
            file_mom = val

    psi_init = double_morse.psi_init
    pot = double_morse.pot

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

        def plot_mom(t, momx, momx2, momp, momp2, ener, E, overlp, ener_tot):
            plot_mom_file(t, momx, momx2, momp, momp2, ener, E, overlp, ener_tot, f_mom)

        def plot_up(psi, t, x, np):
            plot_file(psi, t, x, np, f_abs_up, f_real_up)

        def plot_mom_up(t, momx, momx2, momp, momp2, ener, E, overlp, ener_tot):
            plot_mom_file(t, momx, momx2, momp, momp2, ener, E, overlp, ener_tot, f_mom_up)

        def plot_test(l, phi_l, phi_u):
            plot_test_file(l, phi_l, phi_u, f)

        solver = propagation.PropagationSolver(
            psi_init, pot, plot, plot_mom, plot_test, plot_up, plot_mom_up,
            m=m, L=L, np=np, nch=nch, T=T, nt=nt, x0=x0, p0=p0, a=a, De=De, x0p=x0p, E0=E0,
            t0=t0, sigma=sigma, nu_L=nu_L, delay=delay, lmin=lmin, mod_stdout=mod_stdout, mod_fileout=mod_fileout)
        solver.time_propagation()
        #solver.filtering()


if __name__ == "__main__":
    main(sys.argv[1:])