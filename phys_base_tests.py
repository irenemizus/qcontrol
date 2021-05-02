import unittest
from math_base import *
from phys_base import *
from harmonic import *


class diff_Tests(unittest.TestCase):
    def test_4(self):
        x = coord_grid(0.2, 4)
        akx2 = initak(4, 0.2, 2)
        psi = psi_init(x, 0, 0, 1, 1)
#        print(diff(psi, akx2, 4))
        self.assertTrue(numpy.allclose(diff(psi, akx2, 4), \
                                       numpy.array([0.0041298+0.j, -0.0041298+0.j, \
                                                    -0.0041298+0.j,  0.0041298+0.j])))


class hamil_Tests(unittest.TestCase):
    def test_4(self):
        x = coord_grid(0.2, 4)
        v = pot(x, 1, 1)
        akx2 = initak(4, 0.2, 2)
        psi = psi_init(x, 0, 0, 1, 1)
#        print(hamil(psi, v, akx2, 4))
        self.assertTrue(numpy.allclose(hamil(psi, v, akx2, 4), \
                                       numpy.array([0.00429924+0.j, -0.00411097+0.j, \
                                                    -0.00411097+0.j,  0.00429924+0.j])))


class residum_Tests(unittest.TestCase):
    def test_4(self):
        x = coord_grid(0.2, 4)
        v = pot(x, 1, 1)
        akx2 = initak(4, 0.2, 2)
        psi = psi_init(x, 0, 0, 1, 1)
        emax = v[0] + abs(akx2[1]) + 2.0
#        print(residum(psi, v, akx2, 1.8477590650225735, 4, emax))
        self.assertTrue(numpy.allclose(residum(psi, v, akx2, 1.8477590650225735, 4, emax), \
                                       numpy.array([(-0.8720177032294409+0j), \
                                                    (-0.8726087818833681+0j), \
                                                    (-0.8726087818833681+0j), \
                                                    (-0.8720177032294409+0j)])))


class func_Tests(unittest.TestCase):
    def test_4(self):
        self.assertTrue(numpy.allclose(func(1.8477590650225735, 1), \
                                       complex(-0.2734354014243603, -0.9618903686220686)))


class prop_Tests(unittest.TestCase):
    def test_4(self):
        x = coord_grid(0.2, 4)
        v = pot(x, 1, 1)
        akx2 = initak(4, 0.2, 2)
        psi = psi_init(x, 0, 0, 1, 1)
#        print(prop(psi, 1, 4, 4, v, akx2))
        self.assertTrue(numpy.allclose(prop(psi, 1, 4, 4, v, akx2), \
                                       numpy.array([(-0.06122372015953172+0.23231184536052218j), \
                                                    (-0.05986793665812845+0.23217668866816255j), \
                                                    (-0.05986793665812845+0.23217668866816255j), \
                                                    (-0.06122372015953172+0.23231184536052218j)])))


if __name__ == '__main__':
    unittest.main()
