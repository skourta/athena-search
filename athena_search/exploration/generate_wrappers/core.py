import logging
import os
import shutil
from typing import List, Literal

import ray
from athena.tiramisu import Schedule, TiramisuProgram

from athena_search.utils.config import BaseConfig

from .progress_actor import ProgressActorDistributed


def execute_program_empty_schedule(
    node_name: str,
    tiramisu_program: TiramisuProgram,
):
    assert tiramisu_program.name
    assert BaseConfig.base_config

    schedule = Schedule(tiramisu_program=tiramisu_program)

    logging.info(f"{node_name} executing {tiramisu_program.name}")

    try:
        exec_times = schedule.execute(nb_exec_tiems=1, delete_files=False)

        src_wrapper = os.path.join(
            BaseConfig.base_config.athena.workspace, f"{tiramisu_program.name}_wrapper"
        )
        print(f"Copying wrapper {src_wrapper}")
        # copy wrapper from workspace to wrappers folder
        shutil.copy(src_wrapper, "./wrappers")
    except Exception as e:
        logging.error(f"Skipping this schedule with Error: {e}")
        return False

    logging.info(f"{tiramisu_program.name} executed")
    logging.info(f"Execution times: {exec_times}")

    return True


@ray.remote
def execute_program_empty_schedule_distributed(
    node_name: str,
    programs_to_execute: List[str],
    progress_actor: ProgressActorDistributed,  # type: ignore
    dataset_actor,
    logging_level: int = logging.INFO,
    log_file: str | None = None,
):
    print(f"Node {node_name} starting")
    BaseConfig.init(logging_level=logging_level, worker_id=node_name, log_file=log_file)
    print(f"Node {node_name} logging to {log_file}")
    assert BaseConfig.base_config
    print(f"Node {node_name} started")

    for program_name in programs_to_execute:
        tiramisu_program = ray.get(
            dataset_actor.get_function_by_name.remote(program_name)
        )
        print(tiramisu_program)

        if execute_program_empty_schedule(
            node_name=node_name,
            tiramisu_program=tiramisu_program,
        ):
            progress_actor.report_progress.remote(node_name, program_name, True)
        else:
            progress_actor.report_progress.remote(node_name, program_name, False)
