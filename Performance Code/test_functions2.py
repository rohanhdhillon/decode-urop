import unittest

import numpy

import functions
import functions2


class CanonicalAngularGridTests(unittest.TestCase):
    def test_pole_crossings_adjust_relative_azimuth(self):
        points = numpy.array([
            [-1.0, 40.0, 30.0],
            [181.0, 40.0, 30.0],
            [20.0, -1.0, 30.0],
            [20.0, 181.0, 30.0],
            [-1.0, -1.0, 30.0],
        ])
        expected = numpy.array([
            [1.0, 40.0, 150.0],
            [179.0, 40.0, 150.0],
            [20.0, 1.0, 150.0],
            [20.0, 179.0, 150.0],
            [1.0, 1.0, 30.0],
        ])
        numpy.testing.assert_allclose(
            functions2.canonicalize_angular_grid_degrees(points),
            expected,
        )

    def test_relative_azimuth_wraps_at_both_boundaries(self):
        points = numpy.array([
            [20.0, 40.0, -1.0],
            [20.0, 40.0, 181.0],
            [20.0, 40.0, 359.0],
        ])
        expected = numpy.array([
            [20.0, 40.0, 1.0],
            [20.0, 40.0, 179.0],
            [20.0, 40.0, 1.0],
        ])
        numpy.testing.assert_allclose(
            functions2.canonicalize_angular_grid_degrees(points),
            expected,
        )

    def test_single_coordinate_is_supported_without_mutation(self):
        point = numpy.array([-1.0, 40.0, 30.0])
        original = point.copy()
        canonical = functions2.canonicalize_angular_grid_degrees(point)
        numpy.testing.assert_allclose(canonical, [1.0, 40.0, 150.0])
        numpy.testing.assert_array_equal(point, original)


class DerivativeComplexityRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit_result = {
            "basis_terms": functions2.generate_terms(4, 4, 8),
            "coefficients": numpy.array([0.8, -0.3, 0.2, 0.1, -0.05, 0.04, 0.03, -0.02]),
        }

    def test_interior_values_match_original_estimator(self):
        grid = numpy.array([
            [30.0, 45.0, 60.0],
            [75.0, 105.0, 90.0],
            [150.0, 135.0, 120.0],
        ])
        original = functions.derivative_complexity_for_fit(self.fit_result, grid)
        corrected = functions2.derivative_complexity_for_fit(self.fit_result, grid)
        for key in ("values", "gradient_norm", "curvature_norm"):
            numpy.testing.assert_allclose(corrected[key], original[key], rtol=1e-12, atol=1e-12)

    def test_boundary_stencils_are_finite(self):
        grid = numpy.array([
            [0.0, 40.0, 0.0],
            [180.0, 40.0, 180.0],
            [20.0, 0.0, 90.0],
            [20.0, 180.0, 90.0],
        ])
        result = functions2.derivative_complexity_for_fit(self.fit_result, grid)
        self.assertTrue(numpy.all(numpy.isfinite(result["gradient_norm"])))
        self.assertTrue(numpy.all(numpy.isfinite(result["curvature_norm"])))


if __name__ == "__main__":
    unittest.main()
