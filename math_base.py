import math
import time
from typing import List
from typing import Dict
import numpy


def cprod(cx1, cx2, dx, np):
    """ Calculates scalar product of cx1 and cx2
        INPUT
        cx1 complex vector of length np
        cx2 complex vector of length np
        dx coordinate grid step
        np number of grid points
        OUTPUT
        cnorm = < cx1 | cx2 > """

    assert cx1.size == np
    assert cx2.size == np

    return numpy.vdot(cx2, cx1) * dx


def cprod2(cx1, cx, dx, np):
    """ Calculates expectation value of cx with wavefunction vector cx1
        INPUT
        cx1 complex vector of length np
        cx complex vector of length np
        dx coordinate grid step
        np number of grid points
        OUTPUT
        cnorm2 = < cx1 | cx | cx1> """

    assert cx1.size == np
    assert cx.size == np

    cx1cx = numpy.multiply(cx1, cx)
    return numpy.vdot(cx1, cx1cx) * dx


def cprod3(cx1, cx, cx2, dx, np):
    """ Calculates cnorm3 = < cx1 | cx | cx2>
        INPUT
        cx1 complex vector of length np
        cx complex vector of length np
        cx2 complex vector of length np
        dx coordinate grid step
        np number of grid points
        OUTPUT
        cnorm3 = < cx1 | cx | cx2> """

    assert cx1.size == np
    assert cx.size == np
    assert cx2.size == np

    cx1cx = numpy.multiply(cx1, cx)
    return numpy.vdot(cx2, cx1cx) * dx


def initak(n, dx, iorder, ntriv):
    """ Initializes an array ak, which can be used for
        multiplication in the frequency domain of an FFT.
        The array will contain the values (1j*k)^iorder,
        where the real variable k is the variable in the frequency domain
        INPUT
        n       length of the ak-array. n is a power of 2
        dx      coordinate grid step in the time domain
        iorder  the power of 1j*k (equivalent to the order of the
                derivative when the FFT is used for differentiating)
        ntriv   constant parameter; 1 -- an ordinary non-trivial diatomic-like system
                                    0 -- a trivial 2-level system
                                   -1 -- a trivial n-level system with angular momentum Hamiltonian and
                                         with external laser field augmented inside a Jz term
                                   -2 -- a trivial n-level system with angular momentum Hamiltonian and
                                         with external laser field augmented inside a Jx term
        OUTPUT
        ak      complex one dimensional array of length n """

    ak = numpy.zeros(n, numpy.complex128)

    if ntriv > 0:
        dk = 2.0 * math.pi / (n - 1) / dx

        for i in range(int(n / 2)):
            ak[i + 1] = pow(1j * dk * float(i + 1), iorder)
            ak[n - i - 1] = pow(-1, iorder) * ak[i + 1]

    return ak


def fold(src):
    """ Takes 2-D numpy array with even number of columns.
        Creating array of twice the height and half width by copying
        the right half of the input array under the left half """

    init_height = src.shape[0]
    init_width = src.shape[1]

    assert init_width % 2 == 0
    res_height = init_height * 2
    res_width = init_width // 2

    res = numpy.zeros((res_height, res_width))
    res[:init_height,:res_width] = src[:init_height,:res_width]
    res[init_height:res_height, :res_width] = src[:init_height, res_width:init_width]

    return res


# All the possible values of reorder() function
# are gonna be kept in this dictionary in order
# to avoid repeating calculation
reorder_CACHE: Dict[int, List[int]] = {}


def reorder(nch):
    """ Shuffles the order of points in the Chebyshev interpolation scheme
        INPUT
        nch   number of interpolation points (must be a power of 2)
        OUTPUT
        jj    integer vector of length nch containing order of interpolation points """

    # Checking the cache
    if nch in reorder_CACHE.keys():
        return reorder_CACHE[nch]

    assert (nch & (nch - 1) == 0) and nch > 0  # nch is a positive power of two
    folds_count = int(math.log2(nch))

    jj = numpy.zeros((nch))
    for i in range(nch):
        jj[i] = i

    jj = numpy.reshape(jj, (1,nch))
    for i in range(folds_count):
        jj = fold(jj)
    numpy.reshape(jj, (nch))

    res = []
    for i in range(nch):
        res.append(int(jj[i]))

    # Putting the result into cache
    reorder_CACHE[nch] = res

    return res


def points(nch, t, func):
    """ Calculation of interpolation points and divided difference coefficients
        INPUT
        nch   number of interpolation points (must be a power of 2 if reorder is necessary)
        t     scaled time interval (dimensionless)
        func  a function to be approximated by Chebyshev interpolation scheme
        OUTPUT
        xp    real vector of length nch defining the positions of the interpolation points				^
        dv    complex vector of length nch defining the divided differences """

    # defining the order of the interpolation points
    jj = reorder(nch)

    # calculating of the Chebyshev interpolation points
    xp = numpy.zeros(nch, numpy.float64)
    for i in range(nch):
        phase = float(2 * jj[i] + 1) / float(2 * nch) * math.pi
        xp[i] = 2.0 * math.cos(phase)

    dv = numpy.zeros(nch, numpy.complex128)
    # calculating the first two terms of the divided difference
    dv[0] = func(xp[0], t)
    dv[1] = (func(xp[1], t) - dv[0]) / (xp[1] - xp[0])

    # recursion for the divided difference
    for i in range(2, nch):
        res = 1.0
        sum = complex(0.0, 0.0)
        for j in range(i - 1):
            res *= (xp[i] - xp[j])
            sum += dv[j + 1] * res
        res *= (xp[i] - xp[i - 1])
        dv[i] = (func(xp[i], t) - dv[0] - sum) / res

    return xp, dv



np = 1024000000

print("1")
v1 = numpy.ones(np, numpy.float64)
v2 = numpy.ones(np, numpy.float64)

print("2")
t1 = time.time()
v3 = cprod(v1, v2, 0.1, np)
t2 = time.time()
print(t2 - t1)

print(v3)