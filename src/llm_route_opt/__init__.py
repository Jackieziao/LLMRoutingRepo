"""Provider-neutral LLM routing and deployment optimization."""

from .benchmark import BenchmarkDataset, EvaluationResult, evaluate
from .deployment import DeploymentOptimizer, DeploymentPlan, Workload
from .inverse import ChoiceObservation, InverseSolution, infer_weights
from .optimization import RoutingOptimizationResult, maximize_quality
from .routers import CascadeRouter, DAGRouter, SingleModelRouter, TopKRouter
from .schemas import ModelProfile, QueryFeatures, RouteDecision, RouteMeasurement

__all__ = [
    "BenchmarkDataset",
    "CascadeRouter",
    "ChoiceObservation",
    "DAGRouter",
    "DeploymentOptimizer",
    "DeploymentPlan",
    "EvaluationResult",
    "InverseSolution",
    "ModelProfile",
    "QueryFeatures",
    "RouteDecision",
    "RouteMeasurement",
    "RoutingOptimizationResult",
    "SingleModelRouter",
    "TopKRouter",
    "Workload",
    "evaluate",
    "infer_weights",
    "maximize_quality",
]

__version__ = "0.1.0"
