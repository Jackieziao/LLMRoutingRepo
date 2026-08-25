"""Grid-exact discrete inverse optimization over a normalized objective."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True, slots=True)
class ChoiceObservation:
    chosen: Mapping[str, float]
    alternatives: Sequence[Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class InverseSolution:
    weights: dict[str, float]
    pairwise_accuracy: float
    minimum_margin: float


def infer_weights(
    observations: Sequence[ChoiceObservation], features: Sequence[str], resolution: int = 20
) -> InverseSolution:
    """Infer non-negative, sum-to-one utility weights from discrete choices."""

    if not observations or not features or resolution < 1:
        raise ValueError("observations/features and positive resolution are required")
    best: tuple[tuple[float, float], tuple[float, ...]] | None = None
    for units in product(range(resolution + 1), repeat=len(features)):
        if sum(units) != resolution:
            continue
        weights = tuple(unit / resolution for unit in units)
        margins: list[float] = []
        for observation in observations:
            chosen = _utility(observation.chosen, features, weights)
            margins.extend(
                chosen - _utility(alternative, features, weights)
                for alternative in observation.alternatives
            )
        accuracy = sum(margin >= -1e-12 for margin in margins) / len(margins) if margins else 1.0
        score = (accuracy, min(margins, default=0.0))
        if best is None or score > best[0]:
            best = (score, weights)
    if best is None:  # pragma: no cover - guarded by validation
        raise RuntimeError("simplex enumeration failed")
    return InverseSolution(dict(zip(features, best[1], strict=True)), best[0][0], best[0][1])


def _utility(
    values: Mapping[str, float], features: Sequence[str], weights: tuple[float, ...]
) -> float:
    try:
        return sum(values[name] * weight for name, weight in zip(features, weights, strict=True))
    except KeyError as error:
        raise ValueError(f"missing feature {error.args[0]}") from error


def inverse_example() -> InverseSolution:
    """Working reproducible example with quality/economy/latency objectives."""

    observations = [
        ChoiceObservation(
            {"quality": 0.90, "economy": 0.40, "speed": 0.60},
            [{"quality": 0.60, "economy": 0.70, "speed": 0.70}],
        ),
        ChoiceObservation(
            {"quality": 0.70, "economy": 0.90, "speed": 0.60},
            [{"quality": 0.80, "economy": 0.60, "speed": 0.70}],
        ),
        ChoiceObservation(
            {"quality": 0.70, "economy": 0.60, "speed": 0.90},
            [{"quality": 0.80, "economy": 0.70, "speed": 0.50}],
        ),
    ]
    return infer_weights(observations, ("quality", "economy", "speed"), resolution=20)
