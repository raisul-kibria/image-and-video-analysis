import unittest

import numpy as np

from lesson01 import task02_histogram as task02
from lesson01 import task03_intensity_transformations as task03


__unittest = True     # Hide traceback for failed tests.


class TestLesson01Task02(unittest.TestCase):
    longMessage = False
    input = np.array([[0, 0, 20, 20, 100, 100]])

    def test_find_cuts(self):
        out = task02.histogram_find_cuts(nbins=2)
        self.assertIsNotNone(out, msg='Must be implemented')
        self.assertEqual(len(out), 3, msg='histogram_find_cuts(nbins=2) must have 3 elements.')
        np.testing.assert_array_equal(out, [0,  127.5, 255])
        out = task02.histogram_find_cuts(nbins=6)
        self.assertIsNotNone(out, msg='Must be implemented')
        self.assertEqual(len(out), 7, msg='histogram_find_cuts(nbins=6) must have 7 elements.')
        np.testing.assert_array_equal(out, [0, 42.5, 85, 127.5, 170, 212.5, 255])

    def test_count_values(self):
        out = task02.histogram_count_values(self.input, nbins=2)
        self.assertIsNotNone(out, msg='Must be implemented')
        self.assertEqual(len(out), 2, msg=f'histogram_count_values({self.input}, nbins=2) must have 2 elements.')
        np.testing.assert_array_equal(out, [6, 0])
        out = task02.histogram_count_values(self.input, nbins=6)
        self.assertIsNotNone(out, msg='Must be implemented')
        self.assertEqual(len(out), 6, msg=f'histogram_count_values({self.input}, nbins=6) must have 6 elements.')
        np.testing.assert_array_equal(out, [4, 0, 2, 0, 0, 0])


class TestLesson01Task03(unittest.TestCase):
    longMessage = False
    input_A = np.array([[0, 255, 100, 200]])
    input_B = np.array([[0, 0, 20, 20, 100, 100]])

    def test_negative(self):
        out = task03.negative(self.input_A)
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_A), np.nditer(out), [255, 0, 155, 55]):
            self.assertAlmostEqual(val, exp, msg=f'negative({inp:.7g}) -> {val:.7g} (but expected {exp:.7g})')

    def test_log_transform(self):
        out = task03.log_transform(self.input_A)
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_A), np.nditer(out), [0, 255, 212.2304910127135, 243.87727265632836]):
            self.assertAlmostEqual(val, exp, msg=f'log_transform({inp:.7g}) -> {val:.7g} (but expected {exp:.7g})')

    def test_exp_transform(self):
        out = task03.exp_transform(self.input_A)
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_A), np.nditer(out), [0, 255, 7.798483614, 76.4133139]):
            self.assertAlmostEqual(val, exp, msg=f'exp_transform({inp:.7g}) -> {val:.7g} (but expected {exp:.7g})')

    def test_gamma_transform(self):
        out = task03.gamma_transform(self.input_A, 0.5)
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_A), np.nditer(out), [0, 255, 159.6871942, 225.8317958]):
            self.assertAlmostEqual(val, exp, msg=f'gamma_transform({inp:.7g}, 0.5) -> {val:.7g} (but expected {exp:.7g})')
        out = task03.gamma_transform(self.input_A, 2)
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_A), np.nditer(out), [0, 255, 39.21568627, 156.8627451]):
            self.assertAlmostEqual(val, exp, msg=f'gamma_transform({inp:.7g}, 2) -> {val:.7g} (but expected {exp:.7g})')

    def test_windowing(self):
        out = task03.windowing(self.input_A, 0, 255)
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_A), np.nditer(out), [0, 255, 100, 200]):
            self.assertAlmostEqual(val, exp, msg=f'windowing({self.input_A}): {inp:.7g} -> {val:.7g} (but expected {exp:.7g})')
        out = task03.windowing(self.input_A, 10, 100)
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_A), np.nditer(out), [0, 255, 255, 255]):
            self.assertAlmostEqual(val, exp, msg=f'windowing({self.input_A}): {inp:.7g} -> {val:.7g} (but expected {exp:.7g})')

    def test_minmax_normalization(self):
        out = task03.minmax_normalization(self.input_A)
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_A), np.nditer(out), [0, 255, 100, 200]):
            self.assertAlmostEqual(val, exp, msg=f'minmax_normalization({self.input_A}): {inp:.7g} -> {val:.7g} (but expected {exp:.7g})')
        out = task03.minmax_normalization(self.input_B)
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_B), np.nditer(out), [0, 0, 51, 51, 255, 255]):
            self.assertAlmostEqual(val, exp, msg=f'minmax_normalization({self.input_B}): {inp:.7g} -> {val:.7g} (but expected {exp:.7g})')

    def test_histogram_equalization(self):
        out = task03.histogram_equalization(self.input_B)
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_B), np.nditer(out), [0, 0, 127.5, 127.5, 255, 255]):
            self.assertAlmostEqual(val, exp, msg=f'histogram_equalization({self.input_B}): {inp:.7g} -> {val:.7g} (but expected {exp:.7g})')

    def test_clahe(self):
        out = task03.clahe(self.input_B.astype('uint8'), 0, (1, 1))
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_B), np.nditer(out), [85, 85, 170, 170, 255, 255]):
            self.assertAlmostEqual(val, exp, msg=f'clahe({self.input_B}, 0, (1,1) ): {inp:.7g} -> {val:.7g} (but expected {exp:.7g})')
        out = task03.clahe(self.input_B.astype('uint8'), 10, (1, 1))
        self.assertIsNotNone(out, msg='Must be implemented')
        for inp, val, exp in zip(np.nditer(self.input_B), np.nditer(out), [85, 85, 128, 128, 212, 212]):
            self.assertAlmostEqual(val, exp, msg=f'clahe({self.input_B}, 0, (1,1) ): {inp:.7g} -> {val:.7g} (but expected {exp:.7g})')


if __name__ == '__main__':
    unittest.main()