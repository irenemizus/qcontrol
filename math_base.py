import math
import numpy


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


def reorder(nch):
    """ Shuffles the order of points in the Chebyshev interpolation scheme
        INPUT
        nch   number of interpolation points (must be a power of 2)
        OUTPUT
        jj    integer vector of length nch containing order of interpolation points """

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

    return res


def points(nch, t, func):
    """ Calculation of interpolation points and divided difference coefficients
        INPUT
        nch   number of interpolation points (must be a power of 2 if reorder is necessary)
        t     scaled time interval
        func  a function to be approximated by Chebyshev interpolation scheme
        OUTPUT
        xp    real vector of length nch defining the positions of the interpolation points				^
        dv    complex vector of length nch defining the divided differences """

    # defining the order of the interpolation points
    jj = reorder(nch)

    # calculating of the Chebyshev interpolation points
    xp = []
    for i in range(nch):
        phase = float(2 * jj[i] + 1) / float(2 * nch) * math.pi
        xp.append(2.0 * math.cos(phase))

    dv = []
    # calculating the first two terms of the divided difference
    dv.append(func(xp[0], t))
    dv.append((func(xp[1], t) - dv[0]) / (xp[1] - xp[0]))

    # recursion for the divided difference
    for i in range(2, nch):
        res = 1.0
        sum = complex(0.0, 0.0)
        for j in range(i - 1):
            res *= (xp[i] - xp[j])
            sum += dv[j + 1] * res
        res *= (xp[i] - xp[i - 1])
        dv.append((func(xp[i], t) - dv[0] - sum) / res)

    return xp, dv