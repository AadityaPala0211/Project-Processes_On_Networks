"""Integration tests for the adjacency builder using a tiny synthetic ICIO CSV."""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "full_data_network_structure_analysis" / "01_build_adjacency_matrix.py"


class FullDataAdjacencyBuilderTests(unittest.TestCase):
    def test_direction_diagonal_and_node_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "toy_icio.csv"
            output_dir = root / "output"
            # Row i -> column j is the intermediate flow supplied by i to j.
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["row_label", "AAA_A01", "BBB_B06", "OUT"])
                writer.writerow(["AAA_A01", 10, 2, 12])
                writer.writerow(["BBB_B06", 3, 20, 23])
                writer.writerow(["OUT", 12, 23, 35])

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), "--out", str(output_dir)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            weighted = sparse.load_npz(output_dir / "adjacency_weighted_full.npz").toarray()
            binary = sparse.load_npz(output_dir / "adjacency_binary_full.npz").toarray()
            raw = sparse.load_npz(output_dir / "intermediate_flows_positive_full.npz").toarray()
            np.testing.assert_allclose(weighted, np.array([[0.0, 2.0], [3.0, 0.0]]))
            np.testing.assert_array_equal(binary, np.array([[0, 1], [1, 0]], dtype=np.uint8))
            # Raw flow matrix intentionally retains accounting self-flows.
            np.testing.assert_allclose(raw, np.array([[10.0, 2.0], [3.0, 20.0]]))

            with (output_dir / "node_index.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["node"] for row in rows], ["AAA_A01", "BBB_B06"])
            self.assertEqual([row["matrix_index"] for row in rows], ["0", "1"])

    def test_negative_flows_are_clipped_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "toy_icio.csv"
            output_dir = root / "output"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["row_label", "AAA_A01", "BBB_B06", "OUT"])
                writer.writerow(["AAA_A01", 0, -2, 0])
                writer.writerow(["BBB_B06", 3, 0, 3])
                writer.writerow(["OUT", 3, 0, 3])

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), "--out", str(output_dir)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            weighted = sparse.load_npz(output_dir / "adjacency_weighted_full.npz").toarray()
            np.testing.assert_allclose(weighted, np.array([[0.0, 0.0], [3.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
