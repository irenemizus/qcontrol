import unittest
import numpy
from math_base import *


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


if __name__ == '__main__':
    unittest.main()
