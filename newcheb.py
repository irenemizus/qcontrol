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
        by default, is equal to 2048
    --nch
        number of Chebyshev interpolation points; must be a power of 2
        by default, is equal to 64
    --T
        time range of the problem in femtosec or in pi (half periods) units
        by default, is equal to 30.0 fs for dimensional problem
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
        by default, is equal to 1.0 1 / a_0
    --De
        dissociation energy value for dimensional problem
        by default, is equal to 20000.0 1 / cm
    --E0
        amplitude value of the laser field energy envelope in 1 / cm
        by default, is equal to 1.0 / 1200.0 1 / cm
    --t0
        initial time, when the laser field is switched on, in femtosec
        by default, is equal to 25 fs
    --sigma
        scaling parameter of the laser field envelope in femtosec
        by default, is equal to 10 fs
    --nu_L
        basic frequency of the laser field in PHz
        by default, is equal to 0.6 PHz
    --lmin
        number of a time step, from which the result should be written to a file.
        A negative value will be considered as 0
        by default, is equal to 0
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


import harmonic
import single_morse
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


def plot_mom_file(t, momx, momx2, momp, momp2, ener, E, file_mom):
    """ Plots expectation values of the current x, x*x, p and p*p """
    file_mom.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(t * 1e+15, momx.real, momx2.real, momp.real, momp2.real, ener, E))


def main(argv):
    """ The main() function """
    # analyze cmdline:
    try:
        options, arguments = getopt.getopt(argv, 'hm:L:a:T:', ['help', 'mass=', '', '', '', 'np=', 'nch=', 'nt=', 'x0=', 'p0=', \
                                          'De=', 'E0=', 't0=', 'sigma=', 'nu_L', 'lmin=', 'file_abs=', 'file_real=', 'file_mom='])
    except getopt.GetoptError:
        print("\tThere are unrecognized options!", sys.stderr)
        print("\tRun this script with '-h' option to see the usage info and available options.", sys.stderr)
        sys.exit(2)

    # default filenames
    file_abs = "fort.21"
    file_real = "fort.22"
    file_mom = "fort.23"

    # analyze provided options and their values (if any):
    for opt, val in options:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-m", "--mass"):
            m = float(val)
        elif opt == "L":
            L = float(val)
        elif opt == "np":
            np = int(val)
        elif opt == "nch":
            nch = int(val)
        elif opt == "T":
            T = float(val)
        elif opt == "a":
            a = float(val)
        elif opt == "nt":
            nt = int(val)
        elif opt == "x0":
            x0 = float(val)
        elif opt == "p0":
            p0 = float(val)
        elif opt == "De":
            De = float(val)
        elif opt == "E0":
            E0 = float(val)
        elif opt == "t0":
            t0 = float(val)
        elif opt == "sigma":
            sigma = float(val)
        elif opt == "nu_L":
            nu_L = float(val)
        elif opt == "lmin":
            lmin = int(val)
        elif opt == "file_abs":
            file_abs = val
        elif opt == "file_real":
            file_real = val
        elif opt == "file_mom":
            file_mom = val

    psi_init = double_morse.psi_init
    pot = double_morse.pot

    # main propagation loop
    with open(os.path.join(OUT_PATH, file_abs), 'w') as f_abs, \
         open(os.path.join(OUT_PATH, file_real), 'w') as f_real, \
         open(os.path.join(OUT_PATH, file_mom), 'w') as f_mom:

        def plot(psi, t, x, np):
            plot_file(psi, t, x, np, f_abs, f_real)

        def plot_mom(t, momx, momx2, momp, momp2, ener, E):
            plot_mom_file(t, momx, momx2, momp, momp2, ener, E, f_mom)

        solver = propagation.PropagationSolver(psi_init, pot, plot, plot_mom)
        solver.time_propagation()


if __name__ == "__main__":
    main(sys.argv[1:])