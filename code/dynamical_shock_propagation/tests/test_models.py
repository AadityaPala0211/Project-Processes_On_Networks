import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy import sparse
from dynamical_shock_propagation.models import UPDATES
from dynamical_shock_propagation.models.model3_mmca_sis import infection_probability, update_batch
from dynamical_shock_propagation.data_loader import build_dependency_matrix
from dynamical_shock_propagation.simulator import simulate_source, simulate_batch, prepare_model3_edges
from dynamical_shock_propagation.stability_analysis import analyze
from dynamical_shock_propagation.structural_metrics import core_numbers_undirected
from dynamical_shock_propagation.intervention_analysis import simulate_intervention_batch
from dynamical_shock_propagation.config import Parameters

class DynamicalModelTests(unittest.TestCase):
    def setUp(self):
        # Directed supplier-row -> buyer-column toy network; intentionally asymmetric.
        self.P = sparse.csr_matrix(np.array([
            [0.0, 0.4, 0.2],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0],
        ]))
        self.d = np.array([0.7, 0.3, 0.1])

    def test_all_models_are_bounded(self):
        for name, fn in UPDATES.items():
            result = fn(self.d, self.P, beta=1.8, recovery_rate=0.1)
            self.assertTrue(np.all((result >= 0) & (result <= 1)), name)

    def test_pointwise_one_step_ordering(self):
        names = ("model1_linear", "model2_bilinear_sis", "model3_mmca_sis", "model4_exponential_sis")
        updates = [UPDATES[name](self.d, self.P, beta=0.3, recovery_rate=0.1) for name in names]
        for upper, lower in zip(updates, updates[1:]):
            self.assertTrue(np.all(upper >= lower - 1e-12))

    def test_mmca_product_matches_direct_product(self):
        beta = 0.7
        expected = []
        for buyer in range(self.P.shape[1]):
            suppliers = self.P[:, buyer].nonzero()[0]
            q = np.clip(beta * self.P[suppliers, buyer].toarray().ravel() * self.d[suppliers], 0, 1)
            expected.append(1.0 - np.prod(1.0 - q))
        actual = infection_probability(self.d, self.P, beta)
        self.assertTrue(np.allclose(actual, expected, atol=1e-12))

    def test_mmca_batch_matches_column_updates(self):
        states = np.column_stack([self.d, self.d[::-1], np.array([0.0, 0.5, 1.0])])
        got = update_batch(states, self.P, 0.4, 0.1, edge_arrays=prepare_model3_edges(self.P), edge_chunk_size=2)
        expected = np.column_stack([UPDATES["model3_mmca_sis"](states[:, j], self.P, 0.4, 0.1)
                                    for j in range(states.shape[1])])
        self.assertTrue(np.allclose(got, expected, atol=1e-12))

    def test_shared_zero_state_jacobian(self):
        eps = 1e-7
        v = np.array([0.2, 0.6, 0.4])
        expected = (1.0 - 0.1) * v + 0.3 * (self.P.T @ v)
        for name, fn in UPDATES.items():
            got = fn(eps * v, self.P, 0.3, 0.1) / eps
            self.assertTrue(np.allclose(got, expected, atol=2e-6), name)

    def test_input_share_denominator_includes_self_use(self):
        W = sparse.csr_matrix([[0.0, 20.0], [0.0, 0.0]])
        # Buyer 1's total inputs are 20 cross-node + 80 self-use => P[0,1] = 0.2.
        P = build_dependency_matrix(W, np.array([50.0, 100.0]))
        self.assertAlmostEqual(P[0, 1], 0.2)

    def test_source_holding_and_state_bounds(self):
        row, path = simulate_source(self.P, np.ones(3), 0, "model4_exponential_sis",
            severity=0.2, beta=0.3, recovery_rate=0.1, source_hold_steps=2,
            max_steps=8, keep_trajectories=True)
        self.assertAlmostEqual(path[1]["states"][0], 0.2)
        self.assertAlmostEqual(path[2]["states"][0], 0.2)
        states = np.asarray([p["states"] for p in path])
        self.assertTrue(np.all((states >= 0) & (states <= 1)))
        self.assertIn("converged", row)

    def test_zero_spectral_radius_has_no_finite_threshold(self):
        P = sparse.csr_matrix((2, 2))
        result = analyze(P, beta=0.3, recovery_rate=0.1)
        self.assertEqual(result["dependency_matrix_spectral_radius"], 0.0)
        self.assertIsNone(result["critical_beta"])

    def test_k_core_on_triangle_with_a_tail(self):
        # Nodes 0,1,2 form a triangle; node 3 is attached only to node 0.
        A = sparse.csr_matrix(np.array([
            [0,1,1,1], [1,0,1,0], [1,1,0,0], [1,0,0,0]
        ], dtype=float))
        core = core_numbers_undirected(A)
        self.assertEqual(core.tolist(), [2,2,2,1])

    def test_batched_intervention_protects_downstream_nodes(self):
        P = sparse.csr_matrix(np.array([
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 0.9],
            [0.0, 0.0, 0.0],
        ]))
        params = Parameters(initial_disruption=0.2, beta=0.9, recovery_rate=0.1,
                            source_hold_steps=1, max_steps=12, convergence_tolerance=1e-10,
                            affected_threshold=0.01)
        sources = np.array([0])
        masks_none = np.zeros((3, 1), dtype=bool)
        masks_downstream = np.array([[False], [True], [True]])
        edge_arrays = prepare_model3_edges(P)
        baseline = simulate_intervention_batch(P, np.ones(3), sources, "model2_bilinear_sis",
            params, masks_none, 1.0, edge_arrays, 10)[0]
        protected = simulate_intervention_batch(P, np.ones(3), sources, "model2_bilinear_sis",
            params, masks_downstream, 0.0, edge_arrays, 10)[0]
        self.assertLess(protected["cumulative_disruption_share_steps"],
                        baseline["cumulative_disruption_share_steps"])

if __name__ == "__main__":
    unittest.main()
