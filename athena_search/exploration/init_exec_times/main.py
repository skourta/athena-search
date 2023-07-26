import argparse
import json
import logging
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
from athena_search.exploration.init_exec_times.core import (
    execute_program_empty_schedule,
    execute_program_empty_schedule_distributed,
)
from athena_search.exploration.init_exec_times.progress_actor import (
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
    parser.add_argument("--num-nodes", default=1, type=int)
    parser.add_argument("--replace", default=False, type=bool)
    parser.add_argument("--saving-frequency", default=5, type=int)

    return parser.parse_args()


# Import placement group APIs.
from ray.util.placement_group import placement_group, placement_group_table
from ray.util.state import list_nodes


def launch_distributed(
    programs_to_execute: List[str],
    dataset_actor,
    progress_actor,
    num_nodes: int = 1,
    logging_level: int = logging.INFO,
    log_file: str | None = None,
):
    assert BaseConfig.base_config

    # ray.init(address="auto")

    list_of_nodes = list_nodes()

    assert len(list_of_nodes) == num_nodes

    print("List of nodes in the cluster")
    print(list_of_nodes)

    node_names = [node.node_name for node in list_of_nodes]
    # for each node, create a placement group
    placement_groups = {}
    for node in list_of_nodes:
        placement_groups[node.node_name] = placement_group(
            bundles=[{"CPU": node.resources_total["CPU"]}],
            name=node.node_name,
            strategy="STRICT_SPREAD",
        )

    print("Placement groups scheduled")
    print(placement_groups)
    for pg in placement_groups.values():
        print(placement_group_table(pg))

    # wait for placement groups to be created
    unready_placement_groups = [pg.ready() for pg in placement_groups.values()]

    print("Waiting for placement groups to be created")
    ray.wait(unready_placement_groups, num_returns=len(unready_placement_groups))

    print("Placement groups created")
    for pg in placement_groups.values():
        print(placement_group_table(pg))

    # Launch a task on each placement group.
    tasks = []

    num_programs_per_task = len(programs_to_execute) // num_nodes
    programs_remaining = len(programs_to_execute) % num_nodes

    i = 0
    for node_name, pg in placement_groups.items():
        start = i * num_programs_per_task
        num_programs_to_do = num_programs_per_task

        # Add remaining programs to the last worker
        if i == num_nodes - 1:
            num_programs_to_do += programs_remaining

        print(f"Node {node_name} task {i} will execute {num_programs_to_do} programs")
        task = execute_program_empty_schedule_distributed.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_groups[node_name],
            )
        ).remote(
            node_name=node_name,  # type: ignore
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
    suffix = f"init_executioner_{args.suffix}"
    log_file = f"outputs/{suffix}.log"
    print(f"Logging to {log_file}")
    BaseConfig.init(
        logging_level=logging.INFO,
        log_file=log_file,
        # log_file=None,
    )
    assert BaseConfig.base_config

    BaseConfig.base_config.dataset.suffix = suffix

    if args.saving_frequency:
        BaseConfig.base_config.dataset.saving_frequency = args.saving_frequency

    num_nodes = args.num_nodes

    if args.dataset:
        BaseConfig.base_config.dataset.dataset_path = args.dataset

    if num_nodes == 1:
        dataset_actor = DatasetActor(BaseConfig.base_config.dataset)
        dataset = dataset_actor.dataset_service.dataset
    else:
        dataset_actor = DatasetActorDistributed.remote(BaseConfig.base_config.dataset)
        dataset = ray.get(dataset_actor.get_dataset.remote())

    programs_to_execute = []

    for program_name in dataset:
        if (
            "initial_execution_times" in dataset[program_name]
            and BaseConfig.base_config.machine
            in dataset[program_name]["initial_execution_times"]
            and not args.replace
        ):
            continue
        else:
            programs_to_execute.append(program_name)

    total_to_execute = len(programs_to_execute)
    print(f"Number of programs to execute: {total_to_execute:,}")

    if num_nodes != 1:
        progress_actor = ProgressActorDistributed.remote(total_to_execute)
        total_done, total_skipped_schedules = launch_distributed(
            programs_to_execute=programs_to_execute,
            num_nodes=num_nodes,
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
                machine_name=BaseConfig.base_config.machine,
            ):
                progress_actor.report_progress(
                    node_name="local",
                    program_name=program_name,
                    is_executed=True,
                )
                dataset_actor.update_dataset(
                    program_name,
                    {
                        "initial_execution_times": tiramisu_program.initial_execution_times
                    },
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
