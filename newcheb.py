""" A Python script for solving a controlled propagation
task in a two-potential quantum system using a Newtonian polynomial
algorithm with a Chebyshev-based interpolation scheme.

Usage: python newcheb.py [options]

Options:
    -h, --help
        print this usage info and exit
    --dx
        coordinate grid step value in a_0
        by default, is equal to 0.2 a_0
    --np
        number of collocation points; must be a power of 2
        by default, is equal to 128
    --nch
        number of Chebyshev interpolation points; must be a power of 2
        by default, is equal to 64
    --dt
        time step in pi (half periods) units
        by default, is equal to 0.1
    --nt
        number of time grid points
        by default, is equal to 10
    --x0
        coordinate initial conditions in a_0
        by default, is equal to 0
    --p0
        momentum initial conditions in 1 / a_0
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
    python newcheb.py  --file_abs "res_abs" --dx 0.12
            perform a propagation task using coordinate grid step equal to 0.12 a_0,
            the name "res_abs" for the absolute wavefunctions values output file, and
            default values for other parameters
"""

__author__ = "Irene Mizus (irenem@hit.ac.il)"
__license__ = "Python"

from harmonic import pot, psi_init
from math_base import coord_grid, cprod, cprod2, initak
from phys_base import diff, hamil, prop

OUT_PATH="output"

# ----------------------------------------------------------
import os
import os.path
import sys
import getopt
import math


# ----------------------------------------------------------
def usage():
    """ Print usage information """

    print (__doc__)



#xp, dv = points(8, 0.1)
#print(xp)
#print(dv)


def plot(psi, t, x, np, file_abs, file_real):
    """ Plots absolute and real values of the current wavefunction """
    for i in range(np):
        file_abs.write("{:.6f} {:.6f} {:.6e}\n".format(t, x[i], abs(psi[i])))
        file_real.write("{:.6f} {:.6f} {:.6e}\n".format(t, x[i], psi[i].real))


def plot_mom(t, momx, momx2, momp, momp2, file_mom):
    """ Plots expectation values of the current x, x*x, p and p*p """
    file_mom.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(t, momx.real, momx2.real, momp.real, momp2.real))


def main(argv):
    """ The main() function """
    # analyze cmdline:
    try:
        options, arguments = getopt.getopt(argv, 'h', ['help', 'dx=', 'np=', 'nch=', 'dt=', 'nt=', 'x0=', 'p0=', \
                                            'lmin=', 'file_abs=', 'file_real=', 'file_mom='])
    except getopt.GetoptError:
        print >> sys.stderr, "\tThere are unrecognized options!"
        print >> sys.stderr, "\tRun this script with '-h' option to see the usage info and available options."
        sys.exit(2)

    #default values
    dx = 0.2
    np = 128
    nch = 64
    dt = 0.1
    nt = 10
    x0 = 0
    p0 = 0
    lmin = 0
    file_abs = "fort.21"
    file_real = "fort.22"
    file_mom = "fort.23"

    # analyze provided options and their values (if any):
    for opt, val in options:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt == "dx":
            dx = float(val)
        elif opt == "np":
            np = int(val)
        elif opt == "nch":
            nch = int(val)
        elif opt == "dt":
            dt = float(val)
        elif opt == "nt":
            nt = int(val)
        elif opt == "x0":
            x0 = float(val)
        elif opt == "p0":
            p0 = float(val)
        elif opt == "lmin":
            lmin = int(val)
        elif opt == "file_abs":
            file_abs = val
        elif opt == "file_real":
            file_real = val
        elif opt == "file_mom":
            file_mom = val

    # analyze provided arguments
    if (not math.log2(np).is_integer() or not math.log2(nch).is_integer()):
        print("The number of collocation points 'np' and of Chebyshev \
interpolation points 'nch' must be positive integers and powers of 2. Exiting", sys.stderr)
        sys.exit(1)

    if lmin < 0:
        print("The number 'lmin' of time iteration, from which the result \
should be written to a file, is negative and will be changed to zero", sys.stderr)
        lmin = 0

    if (not dx > 0.0 or not dt > 0.0):
        print("The value of coordinate step 'dx' and of time step 'dt' \
must be positive. Exiting", sys.stderr)
        sys.exit(1)

    # creating a directory for output files
    os.makedirs(OUT_PATH, exist_ok=True)

    # setting the coordinate grid
    x = coord_grid(dx, np)
#    print(x)

    # evaluating of potential(s)
    v = pot(x)
#    print(v)

    # evaluating of initial wavefunction
    psi0 = psi_init(x, x0, p0)
#   abs_psi0 = [abs(i) for i in psi0]
#   print(abs_psi0)

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
    m = 1.0
    coef_kin = -1.0 / (2.0 * m)
    akx2 = [ak * coef_kin for ak in akx2]
 #   print(akx2)

#    phi0_kin = diff(psi0, akx2, np)
#    print(phi0_kin)

    # calculating of initial energy
    phi0 = hamil(psi0, v, akx2, np)
    cener0 = cprod(phi0, psi0, dx, np)
    print("Initial energy: ", abs(cener0))

#    jj = reorder(nch)
#    for i in range(nch):
#        print(i, jj[i])

    # time propagation
    dt *= math.pi
    psi = []
    psi[:] = psi0[:]

    # main propagation loop
    with open(os.path.join(OUT_PATH, file_abs), 'w') as f_abs, \
         open(os.path.join(OUT_PATH, file_real), 'w') as f_real, \
         open(os.path.join(OUT_PATH, file_mom), 'w') as f_mom:

        for l in range(1, nt + 1):
            psi = prop(psi, dt, nch, np, v, akx2)

            cnorm = cprod(psi, psi, dx, np)
            overlp = cprod(psi0, psi, dx, np)

            t = dt * l
            print("t = ", t)
            print("normalization = ", cnorm)
            print("overlap = ", overlp)

            # renormalization
            psi = [el / math.sqrt(abs(cnorm)) for el in psi]

            # calculating of a current energy
            phi = hamil(psi, v, akx2, np)
            cener = cprod(psi, phi, dx, np)
            print("energy = ", cener.real)

            # calculating of expectation values
            # for x
            momx = cprod2(psi, x, dx, np)
            # for x^2
            x2 = [el * el for el in x]
            momx2 = cprod2(psi, x2, dx, np)
            # for p^2
            phi_kin = diff(psi, akx2, np)
            phi_p2 = [el / (-coef_kin) for el in phi_kin]
            momp2 = cprod(psi, phi_p2, dx, np)
            # for p
            akx = initak(np, dx, 1)
            akx = [el / (-1j) for el in akx]
            phip = diff(psi, akx, np)
            momp = cprod(psi, phip, dx, np)

            # plotting the result
            if (l >= lmin):
                plot(psi, t, x, np, f_abs, f_real)
            if (l >= lmin):
                plot_mom(t, momx, momx2, momp, momp2, f_mom)


if __name__ == "__main__":
    main(sys.argv[1:])