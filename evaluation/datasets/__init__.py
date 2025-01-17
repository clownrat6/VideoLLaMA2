from .base import BaseEvalDataset
from .mvbench import MVBenchDataset
from .videomme import VideoMMEDataset
from .mlvu import MLVUDataset
from .tomato import TOMATODataset
from .egoschema import EgoSchemaDataset
from .perception_test import PerceptionTestDataset
from .tempcompass import TempCompassDataset

__all__ = ["build_dataset", "MVBenchDataset", "VideoMMEDataset", "MLVUDataset", "TOMATODataset", "EgoSchemaDataset", "PerceptionTestDataset", "TempCompassDataset"]


DATASET_REGISTRY = {
    "mvbench": MVBenchDataset,
    "videomme": VideoMMEDataset,
    "mlvu": MLVUDataset,
    "tomato": TOMATODataset,
    "egoschema": EgoSchemaDataset,
    "perceptiontest": PerceptionTestDataset,
    "tempcompass": TempCompassDataset,
}


def build_dataset(benchmark_name: str, **kwargs) -> BaseEvalDataset:
    assert benchmark_name in DATASET_REGISTRY, f"Unknown benchmark: {benchmark_name}, available: {DATASET_REGISTRY.keys()}"
    return DATASET_REGISTRY[benchmark_name](**kwargs)
