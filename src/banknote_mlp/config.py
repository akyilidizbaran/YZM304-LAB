from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "banknote_authentication.csv"


@dataclass(frozen=True)
class BaselineConfig:
    test_size: float = 0.20
    random_state: int = 42
    hidden_units: int = 6
    n_steps: int = 500
    learning_rate: float = 0.01
    threshold: float = 0.5
    optimizer: str = "SGD"
    loss_function: str = "binary_cross_entropy"
    hidden_activation: str = "tanh"
    output_activation: str = "sigmoid"


BASELINE_CONFIG = BaselineConfig()
