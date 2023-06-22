import logging
import pickle
import random
from typing import Literal

import numpy as np

from athena.tiramisu.tiramisu_program import TiramisuProgram
from .base_data_service import (
    BaseDataService,
)


class PickleDataService(BaseDataService):
    def __init__(
        self,
        dataset_path: str,
        cpps_path: str,
        path_to_save_dataset: str,
        shuffle: bool = False,
        seed: int = None,
        saving_frequency: int = 10000,
        training_mode: Literal["model", "cpu"] = "model",
    ):
        super().__init__(
            dataset_path=dataset_path,
            path_to_save_dataset=path_to_save_dataset,
            shuffle=shuffle,
            seed=seed,
            saving_frequency=saving_frequency,
            training_mode=training_mode,
        )
        self.cpps_path = cpps_path
        self.cpps = {}

        logging.info(
            f"reading dataset in full pkl format: dataset pkl from {self.dataset_path} and cpps pkl from {self.cpps_path}"
        )

        with open(self.dataset_path, "rb") as f:
            self.dataset = pickle.load(f)
            self.function_names = list(self.dataset.keys())

        with open(self.cpps_path, "rb") as f:
            self.cpps = pickle.load(f)

        # Shuffle the dataset (can be used with random sampling turned off to get a random order)
        if self.shuffle:
            # Set the seed if specified (for reproducibility)
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.function_names)

        self.dataset_size = len(self.function_names)

    # Returns next function name, function data, and function cpps
    def get_next_function(self, random=False) -> TiramisuProgram:
        if random:
            function_name = np.random.choice(self.function_names)
        # Choose the next function sequentially
        else:
            function_name = self.function_names[
                self.current_function_index % self.dataset_size
            ]
            self.current_function_index += 1

        # print(
        #     f"Selected function with index: {self.current_function_index}, name: {function_name}"
        # )

        return TiramisuProgram.from_dict(
            function_name,
            self.dataset[function_name],
            self.cpps[function_name],
        )

    # Returns function data and function cpps by name
    def get_function_by_name(self, function_name: str) -> TiramisuProgram:
        return TiramisuProgram.from_dict(
            function_name,
            self.dataset[function_name],
            self.cpps[function_name],
        )
