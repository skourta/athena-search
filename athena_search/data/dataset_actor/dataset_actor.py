from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal

import ray

from .config import DatasetConfig, DatasetFormat
from .services.hybrid_data_service import HybridDataService
from .services.pickle_data_service import PickleDataService

if TYPE_CHECKING:
    from athena.tiramisu.tiramisu_program import TiramisuProgram
# Frequency at which the dataset is saved to disk
SAVING_FREQUENCY = 10000


class DatasetActor:
    """
    DatasetActor is a class that is used to read the dataset and update it.
    It is used to read the dataset from disk and update it with the new functions.
    It is also used to save the dataset to disk.

    """

    def __init__(self, config: DatasetConfig):
        if config.dataset_format == DatasetFormat.PICKLE:
            self.dataset_service = PickleDataService(
                config.dataset_path,
                config.cpps_path,
                config.save_path,
                config.shuffle,
                config.seed,
                config.saving_frequency,
                config.suffix,
            )
        elif config.dataset_format == DatasetFormat.HYBRID:
            self.dataset_service = HybridDataService(
                config.dataset_path,
                config.cpps_path,
                config.save_path,
                config.shuffle,
                config.seed,
                config.saving_frequency,
            )
        else:
            raise ValueError("Unknown dataset format")

    def get_next_function(self, random=False) -> TiramisuProgram | None:
        return self.dataset_service.get_next_function(random)

    # Update the dataset with the new function
    def update_dataset(self, function_name: str, function_dict: dict) -> bool:
        return self.dataset_service.update_dataset(function_name, function_dict)

    # Get dataset size
    def get_dataset_size(self) -> int:
        return self.dataset_service.dataset_size

    # Get function by name
    def get_function_by_name(self, function_name: str) -> TiramisuProgram:
        if function_name not in self.dataset_service.function_names:
            raise ValueError(f"Function {function_name} not in dataset")
        return self.dataset_service.get_function_by_name(function_name)


@ray.remote
class DatasetActorDistributed(DatasetActor):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)

    def get_dataset(self):
        return self.dataset_service.dataset
