from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Literal

import yaml
from athena.data.dataset_actor.config import DatasetConfig


@dataclass
class TiramisuConfig:
    tiramisu_path: str = ""
    env_type: Literal["model", "cpu"] = "model"
    tags_model_weights: str = ""
    is_new_tiramisu: bool = False
    old_tiramisu_path: str = ""
    max_runs: int = 30

    def __post_init__(self):
        if not self.is_new_tiramisu:
            self.tiramisu_path = self.old_tiramisu_path


@dataclass
class AthenaSearchConfig:
    tiramisu: TiramisuConfig
    dataset: DatasetConfig
    machine: str = "greene"
    workspace: str = "workspace"
    env_vars: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.tiramisu, dict):
            self.tiramisu = TiramisuConfig(**self.tiramisu)
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig(self.dataset)


def read_yaml_file(path):
    with open(path) as yaml_file:
        return yaml_file.read()


def parse_yaml_file(yaml_string: str) -> Dict[Any, Any]:
    return yaml.safe_load(yaml_string)


def dict_to_config(parsed_yaml: Dict[Any, Any]) -> AthenaSearchConfig:
    tiramisu = TiramisuConfig(**parsed_yaml["tiramisu"])
    dataset = DatasetConfig(parsed_yaml["dataset"])

    return AthenaSearchConfig(
        **parsed_yaml["athena"],
        env_vars=parsed_yaml["env_vars"] if "env_vars" in parsed_yaml else {},
        dataset=dataset,
        tiramisu=tiramisu,
    )


class BaseConfig:
    base_config = None

    @classmethod
    def init(cls, config_yaml="config.yaml", logging_level=logging.DEBUG):
        parsed_yaml_dict = parse_yaml_file(read_yaml_file(config_yaml))
        BaseConfig.base_config = dict_to_config(parsed_yaml_dict)
        logging.basicConfig(
            level=logging_level,
            format="|%(asctime)s|%(levelname)s| %(message)s",
        )
