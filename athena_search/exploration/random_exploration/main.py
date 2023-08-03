import argparse
import logging
import socket
from typing import Dict, List, Tuple

import ray

from athena_search.data.dataset_actor.dataset_actor import (
    DatasetActor,
    DatasetActorDistributed,
)
from athena_search.exploration.random_exploration.core import (
    generate_legalities_random_schedules,
    generate_legalities_random_schedules_dist,
)
from athena_search.exploration.random_exploration.progress_actor import ProgressActor
from athena_search.utils.config import BaseConfig


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--suffix", default=socket.gethostname(), type=str)
    parser.add_argument("--num-nodes", default=1, type=int)
    parser.add_argument(
        "--dataset-path", default="datasets/final/final_dataset.pkl", type=str
    )

    return parser.parse_args()


@ray.remote
def function_to_run(dataset_actor: DatasetActor, id: int, progress_actor: "ProgressActor", suffix: str = None):  # type: ignore
    while True:
        next_program = ray.get(dataset_actor.get_next_function.remote())
        logging.info(f"Worker {id} got next program {next_program.name}")
        schedules_dict, num_schedule = ray.get(
            generate_legalities_random_schedules_dist.remote(
                next_program, worker_id=id, log_file=f"athena_search_{suffix}.log"
            )
        )

        dataset_actor.update_dataset.remote(
            next_program.name, {"schedules_dict": schedules_dict}
        )

        progress_actor.report_progress.remote(id, num_schedule)
        print(f"Progress Report: worker {id} generated {num_schedule}")
        logging.info(f"Progress Report: worker {id} generated {num_schedule}")


def launch_distributed(
    num_workers: int = -1, suffix: str = "random_exploration_", num_nodes: int = 1
):
    assert BaseConfig.base_config

    if num_nodes > 1:
        ray.init(address="auto")
    else:
        ray.init()
    print(ray.available_resources())
    progress_actor = ProgressActor.remote()

    if num_workers == -1:
        num_workers = int(ray.available_resources()["CPU"])

    print(f"Launching {num_workers} workers")

    dataset_actor = DatasetActorDistributed.remote(BaseConfig.base_config.dataset)

    workers = []
    id_dict = {}

    for i in range(num_workers):
        ray_worker = function_to_run.remote(
            dataset_actor, id=i, progress_actor=progress_actor, suffix=args.suffix
        )
        workers.append(ray_worker)
        id_dict[ray_worker] = i

    while len(workers) > 0:
        crashed_workers, workers = ray.wait(workers, timeout=30)
        if crashed_workers:
            print(
                f"\n\n Crashed workers: {[id_dict[crashed] for crashed in crashed_workers]}, remaining {len(workers)}\n\n"
            )

            try:
                ray.get(crashed_workers[0])
            except Exception as e:
                logging.error(f"Worker {id_dict[crashed_workers[0]]} crashed: {e}")
                print(f"Worker {id_dict[crashed_workers[0]]} crashed: {e}")

            print("Restarting crashed workers ...")
            for crashed in crashed_workers:
                ray_worker = function_to_run.remote(
                    dataset_actor,
                    id=id_dict[crashed],
                    progress_actor=progress_actor,
                    suffix=args.suffix,
                )
                workers.append(ray_worker)
                id_dict[ray_worker] = id_dict[crashed]

            print("Restarted workers")

        print(f"{len(workers)} workers generating schedules")
        print(f"Schedule generated: {ray.get(progress_actor.get_progress.remote())}")
        print(f"Dict of progress: {ray.get(progress_actor.get_prgress_list.remote())}")


if __name__ == "__main__":
    args = get_arguments()
    suffix = f"random_exploration_{args.suffix}"

    BaseConfig.init(
        logging_level=logging.INFO,
        log_file=f"outputs/{suffix}.log",
        # log_file=None,
    )
    assert BaseConfig.base_config
    BaseConfig.base_config.dataset.suffix = suffix

    num_workers = args.num_workers

    if args.dataset_path:
        BaseConfig.base_config.dataset.dataset_path = args.dataset_path

    if num_workers != 1:
        launch_distributed(num_workers, suffix, args.num_nodes)
    else:
        logging.info("Launching single worker ...")

        logging.info("Initializing dataset actor ...")

        dataset_actor = DatasetActor(BaseConfig.base_config.dataset)
        print(
            BaseConfig.base_config.dataset.saving_frequency,
            "saving frequency",
            BaseConfig.base_config.dataset.save_path,
        )

        nbr_schedules_generated = 0

        logging.info("Starting to generate schedules ...")
        while True:
            next_program = dataset_actor.get_next_function()
            assert next_program

            schedules_dict, nbr_schedules = generate_legalities_random_schedules(
                next_program
            )

            next_program.schedules_dict.update(schedules_dict)
            nbr_schedules_generated += nbr_schedules

            assert next_program.name

            dataset_actor.update_dataset(
                next_program.name,
                {
                    "schedules_dict": next_program.schedules_dict,
                },
            )

            logging.info(
                f"Progress Report: worker {id} generated {nbr_schedules} schedules for {next_program.name}"
            )
