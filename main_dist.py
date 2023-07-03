import argparse
import logging
from time import sleep
import time
from typing import Tuple
from athena.tiramisu.tiramisu_actions.fusion import Fusion
from athena.tiramisu import TiramisuProgram, tiramisu_actions
import ray
from athena_search.data.dataset_actor.dataset_actor import DatasetActor
from athena_search.exploration.random_exploration import (
    generate_legalities_random_schedules,
    generate_legalities_random_schedules_dist,
)

from athena_search.utils.config import BaseConfig

# from athena.utils.config import BaseConfig
import tests.utils as test_utils
import socket


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=-1, type=int)
    parser.add_argument("--suffix", default=socket.gethostname(), type=str)

    return parser.parse_args()


@ray.remote
class ProgressActor:
    def __init__(self):
        self.num_schedules_generated_per_task = {}

    def report_progress(self, task_id: int, num_schedules: int) -> None:
        if task_id not in self.num_schedules_generated_per_task:
            self.num_schedules_generated_per_task[task_id] = 0
        self.num_schedules_generated_per_task[task_id] += num_schedules

    def get_progress(self) -> int:
        return sum(self.num_schedules_generated_per_task.values())

    def get_prgress_list(self):
        return self.num_schedules_generated_per_task


@ray.remote
def function_to_run(dataset_actor: DatasetActor, id: int, progress_actor: "ProgressActor", suffix: str = None):  # type: ignore
    while True:
        next_program = ray.get(dataset_actor.get_next_function.remote())
        schedules_dict, num_schedule = ray.get(
            generate_legalities_random_schedules_dist.remote(
                next_program, worker_id=id, log_file=f"athena_search_{suffix}.log"
            )
        )

        dataset_actor.update_dataset.remote(
            next_program.name,
            {"schedules_dict": schedules_dict},
            suffix=f"{suffix}",
        )

        progress_actor.report_progress.remote(id, num_schedule)
        print(f"Progress Report: worker {id} generated {num_schedule}")
        logging.info(f"Progress Report: worker {id} generated {num_schedule}")


if __name__ == "__main__":
    BaseConfig.init(
        logging_level=logging.INFO, log_file=f"athena_search_{socket.gethostname()}.log"
    )
    assert BaseConfig.base_config

    args = get_arguments()
    ray.init()
    print(ray.available_resources())
    progress_actor = ProgressActor.remote()
    # suffix = time.time()

    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = int(ray.available_resources()["CPU"])

    print(f"Launching {num_workers} workers")

    dataset_actor = DatasetActor.remote(BaseConfig.base_config.dataset)

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
