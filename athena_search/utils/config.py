from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Literal

import yaml
from athena.utils.config import AthenaConfig
from athena.utils.config import BaseConfig as AthenaBaseConfig
from athena_search.data.dataset_actor.config import DatasetConfig


@dataclass
class AthenaSearchConfig:
    athena: AthenaConfig
    dataset: DatasetConfig
    machine: str = "jubail"

    def __post_init__(self):
        if isinstance(self.athena, dict):
            self.athena = AthenaConfig(**self.athena)
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig(self.dataset)


def read_yaml_file(path):
    with open(path) as yaml_file:
        return yaml_file.read()


def parse_yaml_file(yaml_string: str) -> Dict[Any, Any]:
    return yaml.safe_load(yaml_string)


def dict_to_config(parsed_yaml: Dict[Any, Any]) -> AthenaSearchConfig:
    athena = AthenaConfig(**parsed_yaml["athena"])
    dataset = DatasetConfig(parsed_yaml["dataset"])

    return AthenaSearchConfig(
        athena=athena,
        dataset=dataset,
    )


class BaseConfig:
    base_config = None

    @classmethod
    def init(
        cls,
        config_yaml="config.yaml",
        logging_level=logging.DEBUG,
        log_file: str | None = "athena_search.log",
        worker_id: int | str | None = None,
    ):
        parsed_yaml_dict = parse_yaml_file(read_yaml_file(config_yaml))
        BaseConfig.base_config = dict_to_config(parsed_yaml_dict)

        if worker_id is not None:
            log_format = f"|worker{worker_id}|%(asctime)s|%(levelname)s| %(message)s"
        else:
            log_format = "|%(asctime)s|%(levelname)s| %(message)s"

        if log_file is None:
            logging.basicConfig(
                level=logging_level,
                format=log_format,
            )
        else:
            logging.basicConfig(
                filename=log_file,
                filemode="a",
                level=logging_level,
                format=log_format,
            )
        AthenaBaseConfig.from_athena_config(BaseConfig.base_config.athena)
