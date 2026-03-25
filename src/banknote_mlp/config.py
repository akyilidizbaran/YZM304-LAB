from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "banknote_authentication.csv"


@dataclass(frozen=True)
class BaselineConfig:
    test_size: float = 0.20
    validation_size: float = 0.20
    random_state: int = 42
    hidden_layers: tuple[int, ...] = (6,)
    n_steps: int = 500
    learning_rate: float = 0.01
    standardize: bool = True
    l2_lambda: float = 0.0
    threshold: float = 0.5
    optimizer: str = "SGD"
    loss_function: str = "binary_cross_entropy"
    hidden_activation: str = "tanh"
    output_activation: str = "sigmoid"
    selection_rule: str = "highest_validation_accuracy_then_lowest_steps"


BASELINE_CONFIG = BaselineConfig()
