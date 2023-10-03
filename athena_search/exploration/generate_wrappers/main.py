import argparse
import json
import logging
import os
import random
import socket
import time
from typing import Dict, List, Tuple

import athena.tiramisu as tiramisu
import athena.tiramisu.tiramisu_actions as tiramisu_actions
import ray
from athena.tiramisu import TiramisuProgram
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from athena_search.data.dataset_actor.dataset_actor import (
    DatasetActor,
    DatasetActorDistributed,
)
from athena_search.exploration.generate_wrappers.core import (
    execute_program_empty_schedule,
    execute_program_empty_schedule_distributed,
)
from athena_search.exploration.generate_wrappers.progress_actor import (
    ProgressActor,
    ProgressActorDistributed,
)
from athena_search.utils.config import BaseConfig


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="datasets/final_dataset_updated_2_executioner_1878945.pkl",
        type=str,
    )
    parser.add_argument("--suffix", default=socket.gethostname(), type=str)
    parser.add_argument("--num-workers", default=-1, type=int)
    parser.add_argument("--replace", default=False, type=bool)

    return parser.parse_args()


# Import placement group APIs.
from ray.util.placement_group import placement_group, placement_group_table


def launch_distributed(
    programs_to_execute: List[str],
    dataset_actor,
    progress_actor,
    num_workers: int = 1,
    logging_level: int = logging.INFO,
    log_file: str | None = None,
):
    assert BaseConfig.base_config

    if num_workers == -1:
        num_workers = int(ray.cluster_resources()["CPU"])

    num_programs_per_task = len(programs_to_execute) // num_workers
    programs_remaining = len(programs_to_execute) % num_workers

    tasks = []
    i = 0
    for i in range(num_workers):
        start = i * num_programs_per_task
        num_programs_to_do = num_programs_per_task

        # Add remaining programs to the last worker
        if i == num_workers - 1:
            num_programs_to_do += programs_remaining

        print(f"worker {i} will execute {num_programs_to_do} programs")
        task = execute_program_empty_schedule_distributed.remote(
            node_name=i,  # type: ignore
            programs_to_execute=programs_to_execute[start : start + num_programs_to_do],
            progress_actor=progress_actor,  # type: ignore
            dataset_actor=dataset_actor,
            logging_level=logging_level,
            log_file=log_file,
        )

        tasks.append(task)
        print(tasks)

        i += 1

    print("Waiting for all tasks to finish")
    nbr_done = 0
    while len(tasks) > 0:
        done, tasks = ray.wait(tasks, timeout=5)
        nbr_done += len(done)
        print(f"Tasks done: {len(done)}")
        # print error if any in done
        for task in done:
            try:
                ray.get(task)
            except Exception as e:
                print(f"Error in task {task}: {e}")
        progress_actor.print_progress.remote()

    return ray.get(progress_actor.get_total_progress.remote()), ray.get(
        progress_actor.get_skipped_schedules.remote()
    )


if __name__ == "__main__":
    args = get_arguments()
    suffix = f"generate_wrappers_{args.suffix}"
    log_file = f"outputs/{suffix}.log"
    print(f"Logging to {log_file}")
    BaseConfig.init(
        logging_level=logging.INFO,
        log_file=log_file,
        # log_file=None,
    )
    assert BaseConfig.base_config

    BaseConfig.base_config.dataset.suffix = suffix

    num_workers = args.num_workers

    if args.dataset:
        BaseConfig.base_config.dataset.dataset_path = args.dataset

    if num_workers == 1:
        dataset_actor = DatasetActor(BaseConfig.base_config.dataset)
        dataset = dataset_actor.dataset_service.dataset
    else:
        dataset_actor = DatasetActorDistributed.remote(BaseConfig.base_config.dataset)
        dataset = ray.get(dataset_actor.get_dataset.remote())

    programs_to_execute = list(dataset.keys())

    # count existing wrappers in ./wrappers
    existing_wrappers = [
        f.split("_wrapper")[0]
        for f in os.listdir("./wrappers")
        if f.endswith("_wrapper")
    ]

    programs_to_execute = [
        program_name
        for program_name in programs_to_execute
        if program_name not in existing_wrappers
    ]

    total_to_execute = len(programs_to_execute)
    print(f"Number of programs with no wrapper: {total_to_execute:,}")
    print(f"Number of wrappers already generated: {len(existing_wrappers):,}")

    if num_workers != 1:
        progress_actor = ProgressActorDistributed.remote(total_to_execute)
        total_done, total_skipped_schedules = launch_distributed(
            programs_to_execute=programs_to_execute,
            num_workers=num_workers,
            progress_actor=progress_actor,
            dataset_actor=dataset_actor,
            log_file=log_file,
        )
    else:
        progress_actor = ProgressActor(total_to_execute)
        for program_name in programs_to_execute:
            assert type(dataset_actor) is DatasetActor
            tiramisu_program = dataset_actor.get_function_by_name(program_name)

            assert tiramisu_program

            if execute_program_empty_schedule(
                node_name="local",
                tiramisu_program=tiramisu_program,
            ):
                progress_actor.report_progress(
                    node_name="local",
                    program_name=program_name,
                    is_executed=True,
                )
            else:
                progress_actor.report_progress(
                    node_name="local",
                    program_name=program_name,
                    is_executed=False,
                )

            progress_actor.print_progress()

        progress_actor.print_progress()
        total_done, _ = progress_actor.get_total_progress()
        total_skipped_schedules = progress_actor.get_skipped_schedules()

    print("Done")
    print(f"Total done: {total_done:,}")
    print(f"Total skipped schedules: {len(total_skipped_schedules):,}")
    print(
        f"Total Progress: {(total_done + len(total_skipped_schedules))/total_to_execute:,}"
    )

    # save skipped schedules to json
    with open(f"outputs/{suffix}_skipped_schedules.json", "w") as f:
        json.dump(total_skipped_schedules, f)
