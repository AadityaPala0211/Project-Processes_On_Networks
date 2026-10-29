"""Shared defaults for full-resolution dynamics."""
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "data" / "2022_SML.csv"
DEFAULT_MATRIX_DIR = ROOT / "results_full_network"
DEFAULT_OUTPUT_DIR = ROOT / "results_dynamics"
MODEL_NAMES = (
    "model1_linear", "model2_bilinear_sis",
    "model3_mmca_sis", "model4_exponential_sis",
)

@dataclass(frozen=True)
class Parameters:
    initial_disruption: float = 0.20
    beta: float = 0.30
    recovery_rate: float = 0.10
    source_hold_steps: int = 3
    max_steps: int = 50
    convergence_tolerance: float = 1e-7
    affected_threshold: float = 0.01

@dataclass(frozen=True)
class SensitivityGrid:
    severities: tuple = (0.10, 0.20, 0.40)
    betas: tuple = (0.10, 0.30, 0.50)
    recovery_rates: tuple = (0.05, 0.10, 0.20)

def parameters_dict(value):
    return asdict(value)
