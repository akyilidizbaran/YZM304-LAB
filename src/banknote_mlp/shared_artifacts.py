from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InitialParameters:
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    npz_path: Path
    metadata_path: Path


def architecture_key(input_size: int, hidden_layers: tuple[int, ...], output_size: int = 1) -> str:
    return "-".join([str(input_size), *(str(unit) for unit in hidden_layers), str(output_size)])


def generate_initial_parameters(
    *,
    layer_sizes: tuple[int, ...],
    random_state: int,
    weight_scale: float = 1.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(random_state)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []

    for n_inputs, n_outputs in zip(layer_sizes[:-1], layer_sizes[1:]):
        scale = weight_scale * np.sqrt(2.0 / (n_inputs + n_outputs))
        weights.append(rng.normal(loc=0.0, scale=scale, size=(n_inputs, n_outputs)).astype(np.float64))
        biases.append(np.zeros((n_outputs,), dtype=np.float64))

    return weights, biases


def save_initial_parameters(
    directory: Path,
    *,
    key: str,
    layer_sizes: tuple[int, ...],
    random_state: int,
    weight_scale: float = 1.0,
) -> InitialParameters:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"{key}.npz"
    metadata_path = directory / f"{key}.json"

    weights, biases = generate_initial_parameters(
        layer_sizes=layer_sizes,
        random_state=random_state,
        weight_scale=weight_scale,
    )
    np.savez(
        npz_path,
        **{f"W{idx}": weight for idx, weight in enumerate(weights)},
        **{f"b{idx}": bias for idx, bias in enumerate(biases)},
    )
    metadata_path.write_text(
        json.dumps(
            {
                "key": key,
                "layer_sizes": list(layer_sizes),
                "random_state": random_state,
                "weight_scale": weight_scale,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return InitialParameters(weights=weights, biases=biases, npz_path=npz_path, metadata_path=metadata_path)


def load_initial_parameters(npz_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    archive = np.load(npz_path)
    weights = [archive[key].astype(np.float64) for key in sorted(name for name in archive.files if name.startswith("W"))]
    biases = [archive[key].astype(np.float64) for key in sorted(name for name in archive.files if name.startswith("b"))]
    return weights, biases
