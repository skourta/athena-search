import argparse
import json
import logging
import socket
from typing import Dict, List
import ray
from athena_search.data.dataset_actor.dataset_actor import (
    DatasetActor,
    DatasetActorDistributed,
)
from athena_search.utils.config import BaseConfig

from ray.util.state import list_nodes

# Import placement group APIs.
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from athena_search.exploration.executioner.progress_actor import (
    ProgressActor,
    ProgressActorDistributed,
)
from athena_search.exploration.executioner.core import (
    execute_schedules,
    execute_schedules_distributed,
)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="datasets/final/final_dataset.pkl", type=str
    )
    parser.add_argument("--suffix", default=socket.gethostname(), type=str)
    parser.add_argument("--num-nodes", default=1, type=int)
    parser.add_argument("--replace", default=False, type=bool)
    parser.add_argument("--saving-frequency", default=5, type=int)

    return parser.parse_args()


def launch_distributed(
    schedules_to_execute: Dict[str, List[str]],
    dataset_actor,
    progress_actor,
    num_nodes: int = 1,
    logging_level: int = logging.INFO,
    log_file: str | None = None,
):
    assert BaseConfig.base_config

    # ray.init(address="auto")

    list_of_nodes = list_nodes()

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

    total_schedules = sum(
        [len(schedules) for schedules in schedules_to_execute.values()]
    )

    # Launch a task on each placement group.
    tasks = []
    id_dict = {}

    for program_name, schedules in schedules_to_execute.items():
        if len(tasks) == num_nodes:
            print("Waiting for a node to be freed")
            finished_task, tasks = ray.wait(tasks)

            node_to_use = id_dict[finished_task[0]]
            del id_dict[finished_task[0]]

            print(f"Node {node_to_use} freed")
            progress_actor.print_progress.remote()
        else:
            node_to_use = [node for node in node_names if node not in id_dict.values()][
                0
            ]

        tiramisu_program = dataset_actor.get_function_by_name.remote(program_name)
        print(f"Launching {program_name} on {node_to_use}")

        task = execute_schedules_distributed.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_groups[node_to_use],
            )
        ).remote(
            node_name=node_to_use,  # type: ignore
            tiramisu_program=tiramisu_program,
            schedules=schedules,
            progress_actor=progress_actor,
            logging_level=logging_level,
            log_file=log_file,
        )

        tasks.append(task)

        id_dict[task] = node_to_use

        progress_actor.print_progress.remote()

    print("Waiting for all tasks to finish")
    ray.wait(tasks, num_returns=len(tasks))

    return ray.get(progress_actor.get_progress.remote()), ray.get(
        progress_actor.get_skipped_schedules.remote()
    )


if __name__ == "__main__":
    args = get_arguments()
    suffix = f"executioner_{args.suffix}"
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

    schedules_to_execute = {}

    for program_name in dataset:
        if "schedules_dict" in dataset[program_name]:
            schedules_dict = dataset[program_name]["schedules_dict"]
            for schedule, schedule_dict in schedules_dict.items():
                if not schedule_dict["legality"]:
                    continue

                if (
                    "execution_times" in schedule_dict
                    and BaseConfig.base_config.machine
                    in schedule_dict["execution_times"]
                    and not args.replace
                ):
                    continue

                if program_name not in schedules_to_execute:
                    schedules_to_execute[program_name] = []

                schedules_to_execute[program_name].append(schedule)

    total_to_execute = sum(
        [
            len(schedules_to_execute[program_name])
            for program_name in schedules_to_execute
        ]
    )
    print(f"Number of schedules to execute: {total_to_execute:,}")

    if num_nodes != 1:
        progress_actor = ProgressActorDistributed.remote(
            total_to_execute, dataset_actor
        )
        total_done, total_skipped_schedules = launch_distributed(
            schedules_to_execute=schedules_to_execute,
            num_nodes=num_nodes,
            progress_actor=progress_actor,
            dataset_actor=dataset_actor,
            log_file=log_file,
        )
    else:
        progress_actor = ProgressActor(total_to_execute, dataset_actor)
        for program_name in schedules_to_execute:
            assert type(dataset_actor) is DatasetActor
            tiramisu_program = dataset_actor.get_function_by_name(program_name)

            execute_schedules(
                node_name=socket.gethostname(),
                tiramisu_program=tiramisu_program,
                schedules=schedules_to_execute[program_name],
                machine_name=BaseConfig.base_config.machine,
                progress_actor=progress_actor,
            )

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
