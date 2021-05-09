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
        by default, is equal to 0.2 a_0 for dimensional problem
                    is equal to 15 for dimensionless problem
    --np
        number of collocation points; must be a power of 2
        by default, is equal to 128
    --nch
        number of Chebyshev interpolation points; must be a power of 2
        by default, is equal to 64
    --T
        time range of the problem in femtosec or in pi (half periods) units
        by default, is equal to 10.0 fs for dimensional problem
                    is equal to 0.1 for dimensionless problem
    --nt
        number of time grid points
        by default, is equal to 10
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

from harmonic import pot
from harmonic import psi_init
#from single_morse import pot
#from single_morse import psi_init
from math_base import coord_grid, cprod, cprod2, initak
from phys_base import diff, hamil, prop
from phys_base import hart_to_cm, dalt_to_au, Red_Planck_h, cm_to_erg

OUT_PATH="output"

import os
import os.path
import sys
import getopt
import math


def usage():
    """ Print usage information """

    print (__doc__)


def plot(psi, t, x, np, file_abs, file_real):
    """ Plots absolute and real values of the current wavefunction """
    for i in range(np):
        file_abs.write("{:.6f} {:.6f} {:.6e}\n".format(t * 1e+15, x[i], abs(psi[i])))
        file_real.write("{:.6f} {:.6f} {:.6e}\n".format(t * 1e+15, x[i], psi[i].real))


def plot_mom(t, momx, momx2, momp, momp2, ener, file_mom):
    """ Plots expectation values of the current x, x*x, p and p*p """
    file_mom.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}  {:.6f}\n".format(t * 1e+15, momx.real, momx2.real, momp.real, momp2.real, ener))


def main(argv):
    """ The main() function """
    # analyze cmdline:
    try:
        options, arguments = getopt.getopt(argv, 'hm:L:a:T:', ['help', 'mass=', '', '', '', 'np=', 'nch=', 'nt=', 'x0=', 'p0=', \
                                          'De=', 'lmin=', 'file_abs=', 'file_real=', 'file_mom='])
    except getopt.GetoptError:
        print >> sys.stderr, "\tThere are unrecognized options!"
        print >> sys.stderr, "\tRun this script with '-h' option to see the usage info and available options."
        sys.exit(2)

    # default values
    m = 0.5
    L = 6.0 # 0.2 -- for a model harmonic oscillator with a = 1.0 # 4.0 a_0 -- for single morse oscillator
    np = 512 #8192
    nch = 64
    T = 400e-15 # s -- for single morse oscillator
    nt = 40000
    x0 = 1
    p0 = 0
    a = 1.0
    De = 20000.0
    lmin = 0
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
        elif opt == "lmin":
            lmin = int(val)
        elif opt == "file_abs":
            file_abs = val
        elif opt == "file_real":
            file_real = val
        elif opt == "file_mom":
            file_mom = val

    # analyze provided arguments
    if not math.log2(np).is_integer() or not math.log2(nch).is_integer():
        print("The number of collocation points 'np' and of Chebyshev \
interpolation points 'nch' must be positive integers and powers of 2. Exiting", sys.stderr)
        sys.exit(1)

    if lmin < 0:
        print("The number 'lmin' of time iteration, from which the result \
should be written to a file, is negative and will be changed to zero", sys.stderr)
        lmin = 0

    if not L > 0.0 or not T > 0.0:
        print("The value of spatial range 'L' and of time range 'T' of the problem \
must be positive. Exiting", sys.stderr)
        sys.exit(1)

    if not m > 0.0 or not a > 0.0 or not De > 0.0:
        print("The value of a reduced mass 'm/mass', of a scaling factor 'a' \
and of a dissociation energy 'De' must be positive. Exiting", sys.stderr)
        sys.exit(1)

    # creating a directory for output files
    os.makedirs(OUT_PATH, exist_ok=True)

    # calculating coordinate step of the problem
    dx = L / (np - 1)

    # setting the coordinate grid
    x = coord_grid(dx, np)
#    print(x)

    # evaluating of potential(s)
    v = pot(x, m, De, a)
#    print(v)

    # evaluating of initial wavefunction
    psi0 = psi_init(x, x0, p0, m, De, a)
#    abs_psi0 = [abs(i) for i in psi0]
#    print(abs_psi0)

    # initial normalization check
    cnorm0 = cprod(psi0, psi0, dx, np)
    print("Initial normalization: ", abs(cnorm0))

#    cx1 = []
#    for i in range(np):
#        cx1.append(complex(1.0, 0.0))
#    cnorm00 = cprod2(psi0, cx1, dx, np)
#    print(abs(cnorm00))

    # evaluating of k vector
    akx2 = initak(np, dx, 2)
#    print(akx2)

    # evaluating of kinetic energy
    coef_kin = -hart_to_cm / (2.0 * m * dalt_to_au)
    akx2 = [ak * coef_kin for ak in akx2]
 #   print(akx2)

#    phi0_kin = diff(psi0, akx2, np)
#    print(phi0_kin)

    # calculating of initial energy
    phi0 = hamil(psi0, v, akx2, np)

    cener0 = cprod(phi0, psi0, dx, np)
    print("Initial energy: ", abs(cener0))

    # check if input data are correct in terms of the given problem
    # calculating the initial energy range of the Hamiltonian operator H
    emax0 = v[0] + abs(akx2[int(np / 2 - 1)]) + 2.0
    print("Initial emax = ", emax0)

    # calculating the initial minimum number of collocation points that is needed for convergence
    np_min0 = int(math.ceil(L * math.sqrt(2 * m * emax0 * dalt_to_au / hart_to_cm) / math.pi))

    if np < np_min0:
        print("The number of collocation points np = {} should be more than an estimated initial value {}. \
You've got a divergence!".format(np, np_min0))

    # time propagation
    dt = T / (nt - 1)
    psi = []
    psi[:] = psi0[:]

    # main propagation loop
    with open(os.path.join(OUT_PATH, file_abs), 'w') as f_abs, \
         open(os.path.join(OUT_PATH, file_real), 'w') as f_real, \
         open(os.path.join(OUT_PATH, file_mom), 'w') as f_mom:

        for l in range(1, nt + 1):
            # calculating the energy range of the Hamiltonian operator H
            emax = v[0] + abs(akx2[int(np / 2 - 1)]) + 2.0
            t_sc = dt * emax * cm_to_erg / 4.0 / Red_Planck_h

            if l % 10 == 0:
                print("emax = ", emax)
                print("Normalized scaled time interval = ", t_sc)

            # calculating the minimum number of collocation points and time steps that are needed for convergence
            nt_min = int(math.ceil(emax * T * cm_to_erg / 2.0 / Red_Planck_h))
            np_min = int(math.ceil(L * math.sqrt(2 * m * emax * dalt_to_au / hart_to_cm) / math.pi))

            if np < np_min and l % 10 == 0:
                print("The number of collocation points np = {} should be more than an estimated value {}. \
You've got a divergence!".format(np, np_min))
            if nt < nt_min and l % 10 == 0:
                print("The number of time steps nt = {} should be more than an estimated value {}. \
You've got a divergence!".format(nt, nt_min))

            psi = prop(psi, t_sc, nch, np, v, akx2, emax)

            cnorm = cprod(psi, psi, dx, np)
            overlp = cprod(psi0, psi, dx, np)

            t = dt * l
            if l % 10 == 0:
                print("l = ", l)
                print("t = ", t * 1e15, "fs")
                print("normalization = ", cnorm)
                print("overlap = ", overlp)

            # renormalization
            psi = [el / math.sqrt(abs(cnorm)) for el in psi]

            # calculating of a current energy
            phi = hamil(psi, v, akx2, np)
            cener = cprod(psi, phi, dx, np)
            if l % 10 == 0:
                print("energy = ", cener.real)

            # calculating of expectation values
            # for x
            momx = cprod2(psi, x, dx, np)
            # for x^2
            x2 = [el * el for el in x]
            momx2 = cprod2(psi, x2, dx, np)
            # for p^2
            phi_kin = diff(psi, akx2, np)
#            phi_p2 = [el / (-coef_kin) for el in phi_kin]
            phi_p2 = [el * 2.0 * m for el in phi_kin]
            momp2 = cprod(psi, phi_p2, dx, np)
            # for p
            akx = initak(np, dx, 1)
            akx = [el * hart_to_cm / (-1j) / dalt_to_au for el in akx]
            phip = diff(psi, akx, np)
            momp = cprod(psi, phip, dx, np)

            # plotting the result
            if (l >= lmin):
                plot(psi, t, x, np, f_abs, f_real)
            if (l >= lmin):
                plot_mom(t, momx, momx2, momp, momp2, cener.real, f_mom)


if __name__ == "__main__":
    main(sys.argv[1:])