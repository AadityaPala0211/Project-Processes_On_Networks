"""Fast unit tests on a small illustrative signed production network."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT = Path(__file__).parents[1] / "code" / "signed_shock_propagation.py"
spec = importlib.util.spec_from_file_location("signed_shock", SCRIPT)
model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(model)

class SignedShockTests(unittest.TestCase):
    def setUp(self):
        self.nodes = ["RUS_B06", "SAU_B06", "USA_B06", "USA_C20", "DEU_C20", "CHN_G"]
        self.Z = np.array([
            [0, 0, 0, 7, 4, 0],
            [0, 0, 0, 3, 5, 0],
            [0, 0, 0, 1, 0, 3],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 4, 0, 2],
            [0, 0, 0, 0, 0, 0],
        ], dtype=float)
        self.Zsp, self.input_totals, self.out_strength = model.flow_matrices(self.Z)
        self.D = model.dependency_matrix(self.Zsp, self.input_totals)
        self.C = model.substitution_matrix(self.Zsp, self.nodes, self.out_strength)
        self.output = np.array([40, 30, 40, 30, 20, 15], dtype=float)

    def test_dependency_weights_are_input_shares(self):
        self.assertAlmostEqual(self.D[0, 3], 7 / 15)
        self.assertAlmostEqual(self.D[1, 3], 3 / 15)
        self.assertEqual(self.D[0, 0], 0)  # self-use is not a spreading edge

    def test_substitution_only_connects_same_detailed_activity(self):
        self.assertAlmostEqual(float(self.C[:, 0].sum()), 1.0)
        self.assertGreater(self.C[1, 0], 0)  # Saudi B06 can be a substitute for Russian B06
        self.assertEqual(self.C[3, 0], 0)  # C20 is not a substitute for B06
        self.assertEqual(self.C[0, 0], 0)

    def test_one_shock_generates_both_signs(self):
        records, path, _ = model.simulate(self.D, self.C, self.output, 0, steps=12)
        self.assertEqual(path.shape, (13, len(self.nodes)))
        self.assertEqual(path[0, 0], -1.0)
        self.assertTrue(np.any(path < 0))
        self.assertTrue(np.any(path > 0))
        self.assertGreater(records[1]["output_weighted_positive_opportunity_share"], 0)
        self.assertGreater(records[1]["output_weighted_negative_pressure_share"], 0)

    def test_half_shock_has_half_initial_state(self):
        _, full, _ = model.simulate(self.D, self.C, self.output, 0, severity=1.0, steps=4)
        _, half, _ = model.simulate(self.D, self.C, self.output, 0, severity=0.5, steps=4)
        self.assertAlmostEqual(full[0, 0], -1.0)
        self.assertAlmostEqual(half[0, 0], -0.5)

if __name__ == "__main__":
    unittest.main()
