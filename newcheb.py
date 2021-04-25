""" A Python script for solving of a controlled propagation
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

Examples:
    python newcheb.py  --file_abs "res_abs" --dx 0.12
            perform a propagation task using coordinate grid step equal to 0.12 a_0,
            the name "res_abs" for the absolute wavefunctions values output file, and
            default values for other parameters
"""

__author__ = "Irene Mizus (irenem@hit.ac.il)"
__license__ = "Python"

# ----------------------------------------------------------
import os
import os.path
import sys
import getopt
import math
import cmath
import numpy
from numpy.fft import fft, ifft

# ----------------------------------------------------------
def usage():
    """ Print usage information """

    print (__doc__)

# ----------------------------------------------------------
def coord_grid(dx, np):
    """ Setting of the coordinate grid; it should be symmetric,
        equidistant and centered at about minimum of the potential
        INPUT
        dx  coordinate grid step
        np  number of grid points
        OUTPUT
        x  vector of length np defining positions of grid points """

    shift = float(np - 1) * dx / 2.0
    x = [float(i) * dx - shift for i in range(np)]
    return x

# ----------------------------------------------------------
def pot(x):
    """ Potential energy vector
        INPUT
        x  vector of length np defining positions of grid points				  							  ^
        OUTPUT
        v real vector of length np describing the potential V(X) """

    # Single harmonic potential
    v = [xi * xi / 2.0 for xi in x]
    return v

# ----------------------------------------------------------
def psi_init(x, x0, p0):
    """ Initial wave function generator
        INPUT
        x    vector of length np defining positions of grid points
        x0   initial coordinate
        p0   initial momentum
        OUTPUT
        psi  complex vector of length np describing the wavefunction """

    psi = [cmath.exp(-(xi - x0) * (xi - x0) / 2.0 + 1j * p0 * xi) / pow(math.pi, 0.25) for xi in x]
    return psi

# ----------------------------------------------------------
def cprod(cx1, cx2, dx, np):
    """ Calculates scalar product of cx1 and cx2
        INPUT
        cx1 complex vector of length np
        cx2 complex vector of length np
        dx coordinate grid step
        np number of grid points
        OUTPUT
        cnorm = < cx1 | cx2 > """

    cnorm = complex(0.0, 0.0)
    for i in range(np):
        cnorm += cx1[i] * cx2[i].conjugate()

    return cnorm * dx

# ----------------------------------------------------------
def cprod2(cx1, cx, dx, np):
    """ Calculates expectation value of cx with wavefunction vector cx1
        INPUT
        cx1 complex vector of length np
        cx complex vector of length np
        dx coordinate grid step
        np number of grid points
        OUTPUT
        cnorm2 = < cx1 | cx | cx1> """

    cnorm2 = complex(0.0, 0.0)
    for i in range(np):
        cnorm2 += cx1[i] * cx1[i].conjugate() * cx[i]

    return cnorm2 * dx

# ----------------------------------------------------------
def initak(n, dx, iorder):
    """ Initializes an array ak, which can be used for
        multiplication in the frequency domain of an FFT.
        The array will contain the values (1j*k)^iorder,
        where the real variable k is the variable in the frequency domain
        INPUT
        n       length of the ak-array. n is a power of 2
        dx      coordinate grid step in the time domain
        iorder  the power of 1j*k (equivalent to the order of the
                derivative when the FFT is used for differentiating)
        OUTPUT
        ak      complex one dimensional array of length n """

    dk = 2.0 * math.pi / (n - 1) / dx

    ak = []
    for j in range(n):
        ak.append(0.0)

    for i in range(int(n / 2)):
        ak[i + 1] = pow(1j * dk * float(i + 1), iorder)
        ak[n - i - 1] = pow(-1, iorder) * ak[i + 1]

    return ak

# ----------------------------------------------------------
def diff(psi, akx2, np):
    """ Calculates kinetic energy mapping carried out in momentum space
        INPUT
        psi   complex vector of length np
        akx2  complex vector of length np, = k^2/2m
        np    number of grid points
        OUTPUT
        phi   complex vector of length np describing the mapping
              of kinetic energy phi = P^2/2m psi """

    psi_freq = fft(numpy.array(psi))

    phi_freq = []
    for i in range(np):
        phi_freq.append(psi_freq[i] * akx2[i])

    phi = ifft(numpy.array(phi_freq))

    return phi

# ----------------------------------------------------------
def hamil(psi, v, akx2, np):
    """ Calculates Hamiltonian mapping of vector psi
        INPUT
        psi   complex vector of length np
        v     potential energy real vector of length np
        akx2  complex kinetic energy vector of length np, = k^2/2m
        np    number of grid points
        OUTPUT
        phi = H psi complex vector of length np """

    # kinetic energy mapping
    phi = diff(psi, akx2, np)

    # potential energy mapping and accumulation phi = H psi
    for i in range(np):
        phi[i] += v[i] * psi[i]

    return phi

# ----------------------------------------------------------
def main(argv):
    """ The main() function """
    # analyze cmdline:
    try:
        options, arguments = getopt.getopt(argv, 'h', ['help', 'dx=', 'np=', 'nch=', 'dt=', 'nt=', 'x0=', 'p0=', \
                                            'lmin=', 'file_abs=', 'file_real='])
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

    # setting the coordinate grid
    x = coord_grid(dx, np)
    print(x)

    # evaluating of potential(s)
    v = pot(x)
    print(v)

    # evaluating of initial wavefunction
    psi0 = psi_init(x, x0, p0)
    abs_psi0 = [abs(i) for i in psi0]
    print(abs_psi0)

    # initial normalization check
    cnorm0 = cprod(psi0, psi0, dx, np)
    print("Initial normalization: %f" % abs(cnorm0))

#    cx1 = []
#    for i in range(np):
#        cx1.append(complex(1.0, 0.0))
#    cnorm00 = cprod2(psi0, cx1, dx, np)
#    print(abs(cnorm00))

    # evaluating of k vector
    akx2 = initak(np, dx, 2)
    print(akx2)

    # evaluating of kinetic energy
    m = 1.0
    coef_kin = -1.0 / (2.0 * m)
    akx2 = [ak * coef_kin for ak in akx2]
    print(akx2)

#    phi0_kin = diff(psi0, akx2, np)
#    print(phi0_kin)

    # calculating of initial energy
    phi0 = hamil(psi0, v, akx2, np)
    cener0 = cprod(phi0, psi0, dx, np)
    print("Initial energy: %f" % abs(cener0))



# ----------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv[1:])




