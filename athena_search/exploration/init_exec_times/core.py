import logging
from typing import List, Literal

import ray
from athena.tiramisu import Schedule, TiramisuProgram

from athena_search.utils.config import BaseConfig

from .progress_actor import ProgressActorDistributed


def execute_program_empty_schedule(
    node_name: str,
    tiramisu_program: TiramisuProgram,
    machine_name: str = "local",
    nbr_to_execute: int = 10,
    max_mins_per_schedule: int = 60,
):
    assert tiramisu_program.name
    assert BaseConfig.base_config

    schedule = Schedule(tiramisu_program=tiramisu_program)

    logging.info(f"{node_name} executing {tiramisu_program.name}")

    try:
        exec_times = schedule.apply_schedule(
            nb_exec_tiems=nbr_to_execute,
            max_mins_per_schedule=max_mins_per_schedule,
        )
    except Exception as e:
        logging.error(f"Skipping this schedule with Error: {e}")
        return False

    logging.info(f"{tiramisu_program.name} executed")
    logging.info(f"Execution times: {exec_times}")

    tiramisu_program.initial_execution_times[machine_name] = exec_times

    return True


@ray.remote
def execute_program_empty_schedule_distributed(
    node_name: str,
    programs_to_execute: List[str],
    progress_actor: ProgressActorDistributed,  # type: ignore
    dataset_actor,
    nbr_to_execute: int = 10,
    max_mins_per_schedule: int = 60,
    logging_level: int = logging.INFO,
    log_file: str | None = None,
):
    print(f"Node {node_name} starting")
    BaseConfig.init(logging_level=logging_level, worker_id=node_name, log_file=log_file)
    print(f"Node {node_name} logging to {log_file}")
    assert BaseConfig.base_config
    print(f"Node {node_name} started")

    machine_name = BaseConfig.base_config.machine

    for program_name in programs_to_execute:
        tiramisu_program = ray.get(
            dataset_actor.get_function_by_name.remote(program_name)
        )
        print(tiramisu_program)

        if execute_program_empty_schedule(
            node_name=node_name,
            tiramisu_program=tiramisu_program,
            machine_name=machine_name,
            nbr_to_execute=nbr_to_execute,
            max_mins_per_schedule=max_mins_per_schedule,
        ):
            progress_actor.report_progress.remote(node_name, program_name, True)
            dataset_actor.update_dataset.remote(
                program_name,
                {"initial_execution_times": tiramisu_program.initial_execution_times},
            )
        else:
            progress_actor.report_progress.remote(node_name, program_name, False)
