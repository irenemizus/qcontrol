import unittest
from math_base import *
from phys_base import func

class coord_grid_Tests(unittest.TestCase):
    def test_16(self):
        self.assertTrue(numpy.allclose(coord_grid(0.2, 16), numpy.array([-1.5, -1.3, -1.1, -0.9,
                                                             -0.7, -0.5, -0.3, -0.1,
                                                              0.1,  0.3,  0.5,  0.7,
                                                              0.9,  1.1,  1.3,  1.5])))


class cprod_Tests(unittest.TestCase):
    def test_4norm(self):
        sss = numpy.array([1, 2, 3, 4])
        self.assertTrue(cprod(sss, sss, 0.2, 4) == 6)


class cprod2_Tests(unittest.TestCase):
    def test_4(self):
        sss = numpy.array([1, 1, 1, 1])
        x = numpy.array([1, 2, 3, 4])
        self.assertTrue(cprod2(sss, x, 0.2, 4) == 2)


class initak_Tests(unittest.TestCase):
    def test_4(self):
        dk = 10.0 * math.pi / 3.0
        dk2 = complex(dk * dk, 0)
        self.assertTrue(numpy.allclose(initak(4, 0.2, 2), numpy.array([0, -dk2, -4.0 * dk2, -dk2])))


class fold_Tests(unittest.TestCase):
    def test_4x2(self):
        sss = numpy.array([[1, 2, 3, 4], [5, 6, 7, 123]])
        self.assertTrue((fold(sss) == numpy.array([[1, 2], [5, 6], [3, 4], [7, 123]])).all())


class reorder_Tests(unittest.TestCase):
    def test_2(self):
        jj1 = reorder(2)
        self.assertTrue((jj1 == numpy.array([0, 1])).all())

    def test_8(self):
        jj2 = reorder(8)
        self.assertTrue((jj2 == numpy.array([0, 4, 2, 6, 1, 5, 3, 7])).all())


class points_Tests(unittest.TestCase):
    def test_4(self):
        self.assertTrue(numpy.allclose(points(4, 1, func)[0],
                                       numpy.array([1.8477590650225735, -0.7653668647301795,
                                                    0.7653668647301797, -1.8477590650225735])))
        self.assertTrue(numpy.allclose(points(4, 1, func)[1],
                                       numpy.array([complex(-0.2734354014243603, -0.9618903686220686),
                                                    complex(-0.38060302857900286, -0.6332232027087304),
                                                    complex(-0.3516313481159491, 0.25126355493174846),
                                                    complex(-2.1243197887715325e-17, 0.135982856037932)])))


class fft_Tests(unittest.TestCase):
    def test_fft(self):
        test = []
        for i in range(128):
            test.append(i / 128 + 0j)

        spectr = numpy.fft.fft(numpy.array(test))
        test2 = numpy.fft.ifft(spectr)

        self.assertTrue(numpy.allclose(test, test2))


if __name__ == '__main__':
    unittest.main()
