import math

import numpy

import phys_base
from psi_basis import Psi


class _Hamil2D:
    """ Calculates two-dimensional Hamiltonian mapping of vector psi
        INPUT
        psi         list of complex vectors of length np
        v           list of potential energy real vectors of length np
        akx2        complex kinetic energy vector of length np, = k^2/2m
        np          number of grid points
        E           a real value of external laser field
        eL          a laser field energy shift = h * nu_L / 2.0
        E_full      a complex value of external laser field
        ntriv       constant parameter; 1 -- an ordinary non-trivial diatomic-like system
                                        0 -- a trivial 2-level system
                                       -1 -- a trivial n-level system with angular momentum Hamiltonian and
                                             with external laser field augmented inside a Jz term
                                       -2 -- a trivial n-level system with angular momentum Hamiltonian and
                                             with external laser field augmented inside a Jx term
        orig        a boolean parameter that depends
                    if an original form of the Hamiltonian should be used (orig = True) or
                    the shifted real version (orig = False -- by default)
        U, W, delta parameters of angular momentum-type Hamiltonian

        OUTPUT
        phi = H psi list of complex vectors of length np """

    @staticmethod
    def non_trivial(psi: Psi, v, akx2, np, E, eL, U, W, delta, ntriv, E_full, orig):
        for i in range(len(psi.f)):
            assert psi.f[i].size == np
            assert v[i][1].size == np
        assert akx2.size == np

        phi: Psi = Psi(lvls=len(psi.f))

        if orig:
            # without laser field energy shift
            # diagonal terms
            psieL_l = 0.0
            psieL_u = 0.0
            # non-diagonal terms
            psiE_l = psi.f[0] * E_full.conjugate()
            psiE_u = psi.f[1] * E_full
        else:
            # with laser field energy shift
            # diagonal terms
            psieL_l = psi.f[0] * eL
            psieL_u = psi.f[1] * eL
            # non-diagonal terms
            psiE_l = psi.f[0] * E
            psiE_u = psi.f[1] * E

        # diagonal terms
        # 1D Hamiltonians mapping for the corresponding states
        phi_dl = phys_base.hamil_cpu(psi.f[0], v[0][1], akx2, np, ntriv)
        phi_du = phys_base.hamil_cpu(psi.f[1], v[1][1], akx2, np, ntriv)

        # diagonal terms
        # adding of the laser field energy shift
        numpy.add(phi_dl, psieL_l, out=phi_dl)
        numpy.subtract(phi_du, psieL_u, out=phi_du)

        # adding non-diagonal terms
        phi.f[0] = numpy.subtract(phi_dl, psiE_u)
        phi.f[1] = numpy.subtract(phi_du, psiE_l)

        return phi

    @staticmethod
    def two_levels(psi: Psi, v, akx2, np, E, eL, U, W, delta, ntriv, E_full, orig):
        for i in range(len(psi.f)):
            assert psi.f[i].size == np
            assert v[i][1].size == np
        assert akx2.size == np

        phi: Psi = Psi(lvls=len(psi.f))

        # non-diagonal terms
        psiE_l = psi.f[0] * E_full.conjugate()
        psiE_u = psi.f[1] * E_full

        # diagonal terms
        # 1D Hamiltonians mapping for the corresponding states
        phi_dl = phys_base.hamil_cpu(psi.f[0], v[0][1], akx2, np, ntriv)
        phi_du = phys_base.hamil_cpu(psi.f[1], v[1][1], akx2, np, ntriv)

        # adding non-diagonal terms
        phi.f[0] = numpy.subtract(phi_dl, psiE_u)
        phi.f[1] = numpy.subtract(phi_du, psiE_l)

        return phi

    @staticmethod
    def BH_X(psi: Psi, v, akx2, np, E, eL, U, W, delta, ntriv, E_full, orig):
        for i in range(len(psi.f)):
            assert psi.f[i].size == np
            assert v[i][1].size == np
        assert akx2.size == np

        phi: Psi = Psi(lvls=len(psi.f))
        nlvls = len(psi.f)
        l = (nlvls - 1) / 2.0
        H = numpy.zeros((nlvls, nlvls), dtype=numpy.complex128)

        H.itemset((0, 0), 2.0 * l * U + 2.0 * l * l * W)
        for vi in range(1, nlvls):
            Q = 2.0 * (l - vi) * U + 2.0 * (l - vi) * (l - vi) * W # U, W ~ 1 / cm
            P = -delta * E * math.sqrt(l * (l + 1) - (l - vi + 1) * (l - vi)) # delta ~ 1 / cm
            R = -delta * E * math.sqrt(l * (l + 1) - (l - vi + 1) * (l - vi)) # delta ~ 1 / cm
            H.itemset((vi, vi), Q)
            H.itemset((vi - 1, vi), P)
            H.itemset((vi, vi - 1), R)

        for gl in range(nlvls):
            phi_gl = numpy.zeros(np, dtype=numpy.complex128)
            for il in range(nlvls):
                H_psi_el_mult = H.item(gl, il) * psi.f[il]
                phi_gl = numpy.add(phi_gl, H_psi_el_mult)
            phi.f[gl] = phi_gl

        return phi

    @staticmethod
    def BH_Z(psi: Psi, v, akx2, np, E, eL, U, W, delta, ntriv, E_full, orig):
        for i in range(len(psi.f)):
            assert psi.f[i].size == np
            assert v[i][1].size == np
        assert akx2.size == np

        phi: Psi = Psi(lvls=len(psi.f))
        nlvls = len(psi.f)
        l = (nlvls - 1) / 2.0
        H = numpy.zeros((nlvls, nlvls), dtype=numpy.complex128)

        H.itemset((0, 0), 2.0 * l ** 2 * U + 2.0 * l * E)
        for vi in range(1, nlvls):
            Q = 2.0 * (l - vi) ** 2 * U + 2.0 * (l - vi) * E  # U ~ 1 / cm
            P = -delta * math.sqrt(l * (l + 1) - (l - vi + 1) * (l - vi))  # delta ~ 1 / cm
            R = -delta * math.sqrt(l * (l + 1) - (l - vi + 1) * (l - vi))  # delta ~ 1 / cm
            H.itemset((vi, vi), Q)
            H.itemset((vi - 1, vi), P)
            H.itemset((vi, vi - 1), R)

        for gl in range(nlvls):
            phi_gl = numpy.zeros(np, dtype=numpy.complex128)
            for il in range(nlvls):
                H_psi_el_mult = H.item(gl, il) * psi.f[il]
                phi_gl = numpy.add(phi_gl, H_psi_el_mult)
            phi.f[gl] = phi_gl

        return phi


class Hamil2D:
    def __init__(self, v, akx2, np, U, W, delta, ntriv):
        self._v = v
        self._akx2 = akx2
        self._np = np
        self._U = U
        self._W = W
        self._delta = delta
        self._ntriv = ntriv

    def __call__(self, orig: bool, psi, E, eL, E_full):
        raise NotImplemented("Use an inherited class, stupid monkey.")


class Hamil2DNonTrivial(Hamil2D):
    def __call__(self, orig: bool, psi, E, eL, E_full):
        return _Hamil2D.non_trivial(psi, self._v, self._akx2, self._np, E, eL, self._U, self._W, self._delta, self._ntriv, E_full, orig)

class Hamil2DBHZ(Hamil2D):
    def __call__(self, orig: bool, psi, E, eL, E_full):
        return _Hamil2D.BH_Z(psi, self._v, self._akx2, self._np, E, eL, self._U, self._W, self._delta, self._ntriv, E_full, orig)

class Hamil2DBHX(Hamil2D):
    def __call__(self, orig: bool, psi, E, eL, E_full):
        return _Hamil2D.BH_X(psi, self._v, self._akx2, self._np, E, eL, self._U, self._W, self._delta, self._ntriv, E_full, orig)

class Hamil2DTwoLevels(Hamil2D):
    def __call__(self, orig: bool, psi, E, eL, E_full):
        return _Hamil2D.two_levels(psi, self._v, self._akx2, self._np, E, eL, self._U, self._W, self._delta, self._ntriv, E_full, orig)
