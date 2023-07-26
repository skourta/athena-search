import logging
from typing import List, Literal

import ray
from athena.tiramisu import Schedule, TiramisuProgram

from athena_search.utils.config import BaseConfig

from .progress_actor import ProgressActor, ProgressActorDistributed


def execute_schedules(
    node_name: str,
    tiramisu_program: TiramisuProgram,
    schedules: List[str],
    progress_actor: ProgressActor,
    machine_name: str = "local",
    nbr_to_execute: int = 10,
    max_mins_per_schedule: int = 60,
    mode: Literal["SEQ", "DIST"] = "SEQ",
):
    assert tiramisu_program.name
    assert BaseConfig.base_config

    nbr_executed = 0
    nbr_skipped_schedules = 0
    skipped_schedules = []
    count = len(schedules)

    for index, schedule_str in enumerate(schedules):
        schedule_dict = tiramisu_program.schedules_dict[schedule_str]

        if "execution_times" not in schedule_dict:
            schedule_dict["execution_times"] = {}

        logging.info(
            f"Node {node_name} executing schedule {index} out of {count} for {tiramisu_program.name}"
        )
        schedule = Schedule.from_sched_str(
            tiramisu_program=tiramisu_program, sched_str=schedule_str
        )

        logging.info(f"Applying Schedule: {schedule_str} for {tiramisu_program.name}")

        try:
            exec_times = schedule.apply_schedule(
                nb_exec_tiems=nbr_to_execute,
                max_mins_per_schedule=max_mins_per_schedule,
            )
        except Exception as e:
            logging.error(f"Skipping this schedule with Error: {e}")
            skipped_schedules.append((tiramisu_program.name, schedule_str))
            continue

        logging.info(f"Schedule: {schedule_str} for {tiramisu_program.name} executed")
        logging.info(f"Execution times: {exec_times}")

        schedule_dict["execution_times"][machine_name] = exec_times

        nbr_executed += 1

        if progress_actor:
            if mode == "SEQ":
                progress_actor.report_progress(
                    node_name, nbr_executed, skipped_schedules, tiramisu_program
                )
            else:
                progress_actor.report_progress.remote(
                    node_name, nbr_executed, skipped_schedules, tiramisu_program
                )

    return nbr_executed, skipped_schedules


@ray.remote
def execute_schedules_distributed(
    node_name: str,
    tiramisu_program: TiramisuProgram,
    schedules: List[str],
    progress_actor: ProgressActorDistributed,  # type: ignore
    nbr_to_execute: int = 10,
    max_mins_per_schedule: int = 60,
    logging_level: int = logging.INFO,
    log_file: str | None = None,
):
    BaseConfig.init(logging_level=logging_level, worker_id=node_name, log_file=log_file)
    print(f"Node {node_name} logging to {log_file}")
    assert BaseConfig.base_config
    print(f"Node {node_name} started")

    machine_name = BaseConfig.base_config.machine

    execute_schedules(
        node_name=node_name,
        tiramisu_program=tiramisu_program,
        schedules=schedules,
        machine_name=machine_name,
        nbr_to_execute=nbr_to_execute,
        max_mins_per_schedule=max_mins_per_schedule,
        progress_actor=progress_actor,
        mode="DIST",
    )
