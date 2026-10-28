"""Mechanics tests for the new linear/nonlinear modelling pipeline."""
import unittest
import numpy as np
from scipy import sparse

from analysis.dynamics import build_input_share_matrix, simulate_batch
from analysis.data import aggregate_country_sector, industry_group, SECTORS
from analysis.structural import weighted_pagerank


class LinearNonlinearTests(unittest.TestCase):
    def test_industry_groups_cover_expected_codes(self):
        self.assertEqual(len(SECTORS), 12)
        self.assertEqual(industry_group("B06"), "Mining_quarrying")
        self.assertEqual(industry_group("C20"), "Chemicals_petroleum_minerals")
        self.assertEqual(industry_group("G"), "Wholesale_retail_trade")

    def test_input_share_column_normalization(self):
        Z = sparse.csr_matrix(np.array([[5., 6., 0.], [4., 3., 2.], [0., 1., 0.]]))
        P = build_input_share_matrix(Z)
        # Supplier-to-buyer ratios use the full buyer input total as denominator.
        self.assertAlmostEqual(float(P[0, 1]), 6 / 10)
        self.assertAlmostEqual(float(P[1, 1]), 3 / 10)
        # The diagonal flow is retained in the denominator but does not propagate.
        self.assertEqual(float(P[0, 0]), 0.0)
        self.assertLessEqual(float(np.asarray(P.sum(axis=0)).max()), 1.0 + 1e-12)

    def test_nonlinear_model_saturates_and_linear_model_is_clipped(self):
        # Node 0 supplies node 1. Source is held for steps 0, 1, 2.
        Z = sparse.csr_matrix(np.array([[0., 10.], [0., 0.]]))
        P = build_input_share_matrix(Z)
        Q = P.T.tocsr()
        output = np.array([1., 1.])
        sources = np.array([0])
        severities = np.array([0.2])
        beta = np.array([0.3])
        rho = np.array([0.1])
        linear, _ = simulate_batch(Q, output, sources, severities, beta, rho, "linear", 3, 10, 1e-9, False)
        nonlinear, traces = simulate_batch(Q, output, sources, severities, beta, rho, "nonlinear", 3, 10, 1e-9, True)
        self.assertGreater(linear[0]["peak_output_weighted_disruption"], 0)
        self.assertGreater(nonlinear[0]["peak_output_weighted_disruption"], 0)
        self.assertLessEqual(linear[0]["peak_output_weighted_disruption"], 1.0)
        self.assertLessEqual(nonlinear[0]["peak_output_weighted_disruption"], 1.0)
        self.assertEqual(min(r["step"] for r in traces), 0)
        self.assertEqual(max(r["step"] for r in traces), nonlinear[0]["actual_steps"])

    def test_pagerank_is_probability_vector(self):
        A = sparse.csr_matrix(np.array([[0., 2., 0.], [0., 0., 1.], [1., 0., 0.]]))
        pr = weighted_pagerank(A)
        self.assertEqual(len(pr), 3)
        self.assertAlmostEqual(float(pr.sum()), 1.0, places=8)
        self.assertTrue(np.all(pr >= 0))


if __name__ == "__main__":
    unittest.main()
