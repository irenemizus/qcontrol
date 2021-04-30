import unittest
from math_base import *
from phys_base import *
from harmonic import *


class diff_Tests(unittest.TestCase):
    def test_4(self):
        x = coord_grid(0.2, 4)
        akx2 = initak(4, 0.2, 2)
        psi = psi_init(x, 0, 0)
        self.assertTrue(numpy.allclose(diff(psi, akx2, 4), \
                                       numpy.array([ 1.60683526+0.j, -1.60683526+0.j, \
                                                     -1.60683526+0.j,  1.60683526+0.j])))


class hamil_Tests(unittest.TestCase):
    def test_4(self):
        x = coord_grid(0.2, 4)
        v = pot(x)
        akx2 = initak(4, 0.2, 2)
        psi = psi_init(x, 0, 0)
        self.assertTrue(numpy.allclose(hamil(psi, v, akx2, 4), \
                                       numpy.array([1.6391486+0.j, -1.60309837+0.j, \
                                                    -1.60309837+0.j, 1.6391486+0.j])))


class residum_Tests(unittest.TestCase):
    def test_4(self):
        x = coord_grid(0.2, 4)
        v = pot(x)
        akx2 = initak(4, 0.2, 2)
        psi = psi_init(x, 0, 0)
        emax = v[0] + abs(akx2[1]) + 2.0
        self.assertTrue(numpy.allclose(residum(psi, v, akx2, 1.8477590650225735, 4, emax), \
                                       numpy.array([(-2.7042818112956+0j), (-2.933138984712612+0j), \
                                                    (-2.933138984712612+0j), (-2.7042818112956+0j)])))


class func_Tests(unittest.TestCase):
    def test_4(self):
        self.assertTrue(numpy.allclose(func(1.8477590650225735, 1), \
                                       complex(-0.2734354014243603, -0.9618903686220686)))


class prop_Tests(unittest.TestCase):
    def test_4(self):
        x = coord_grid(0.2, 4)
        v = pot(x)
        akx2 = initak(4, 0.2, 2)
        psi = psi_init(x, 0, 0)
        self.assertTrue(numpy.allclose(prop(psi, 1, 4, 4, v, akx2), \
                                       numpy.array([(-0.44826755424439957+0.7735703919393955j), \
                                                    (0.053328648117942135+0.7248514848475337j), \
                                                    (0.053328648117942135+0.7248514848475337j), \
                                                    (-0.44826755424439957+0.7735703919393955j)])))


if __name__ == '__main__':
    unittest.main()
