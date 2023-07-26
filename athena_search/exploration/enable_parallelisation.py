import argparse
import logging
import random
import socket
from athena.tiramisu import Schedule, TiramisuProgram, tiramisu_actions
import ray
from athena_search.data.dataset_actor.dataset_actor import (
    DatasetActor,
    DatasetActorDistributed,
)

from athena_search.exploration.random_exploration import get_random_action
from athena_search.utils.config import BaseConfig
from tests.utils import load_test_data


class ParallelisationExplorer:
    def __init__(self, tiramisu_function: TiramisuProgram, max_tries: int) -> None:
        self.tiramisu_function = tiramisu_function
        self.schedules_dict = tiramisu_function.schedules_dict
        self.parallelisations_to_enable = []
        self.max_tries = max_tries
        self.num_parallelisations_enabled = 0

    def explore(self) -> None:
        candidates = tiramisu_actions.Parallelization.get_candidates(
            self.tiramisu_function.tree
        )

        for root in candidates:
            for candidate in candidates[root]:
                self.parallelisations_to_enable.extend(candidate)
        logging.info(f"Parallelisations to enable: {self.parallelisations_to_enable}")
        for iterator in self.parallelisations_to_enable:
            schedule = Schedule(self.tiramisu_function)

            assert schedule.tree

            try:
                schedule.add_optimizations(
                    [tiramisu_actions.Parallelization([iterator], schedule.tree)]
                )
            except tiramisu_actions.CannotApplyException as e:
                logging.error(f"Cannot apply action: {e}")
                logging.error(f"Schedule: {schedule}")
                logging.error(f"Tree: {schedule.tree}")
                logging.error(f"renaming: {schedule.tree.renamed_iterators}")
                continue
            except KeyError as e:
                logging.error(f"KeyError: {e}")
                logging.error(f"Schedule: {schedule}")
                logging.error(f"Tree: {schedule.tree}")
                logging.error(f"renaming: {schedule.tree.renamed_iterators}")
                logging.error(f"iterator: {iterator}")
                logging.error(
                    f"action {tiramisu_actions.Parallelization(iterator, schedule.tree)}"
                )
                raise e

            try:
                schedule_is_legal = self.get_legality(schedule)
            except Exception as e:
                logging.error(
                    f"Skipping iterator cannot get legality of schedule{schedule} due to: {e}"
                )
                continue

            if not schedule_is_legal:
                logging.info(f"Parallelisation on {iterator} is not legal. Enabling...")
                schedule = self.enable(iterator)
                if schedule is not None:
                    logging.info(f"Enabled parallelisation on {iterator}")
                    logging.info(f"Enabling Schedule: {schedule}")
                    self.num_parallelisations_enabled += 1
            else:
                logging.info(f"Parallelisation on {iterator} is legal.")

    def enable(self, iterator: str) -> Schedule | None:
        schedule = Schedule(self.tiramisu_function)

        assert schedule.tree

        nb_tries = 0

        parallelisation_is_legal = False

        schedules_seen = []

        while parallelisation_is_legal == False and nb_tries < self.max_tries:
            logging.info(f"Try {nb_tries} to enable parallelisation on {iterator}")
            tmp_schedule = schedule.copy()

            assert tmp_schedule.tree

            random_action = get_random_action(tmp_schedule)
            logging.info(f"Random action: {random_action}")
            if (
                random_action is not None
                and random_action[0].is_parallelization() == False
            ):
                random_action = random_action[0]
                try:
                    tmp_schedule.add_optimizations([random_action])
                except tiramisu_actions.CannotApplyException as e:
                    logging.error(f"Cannot apply action: {e}")
                    logging.error(f"Schedule: {schedule}")
                    logging.error(f"Tree: {schedule.tree}")
                    logging.error(f"renaming: {schedule.tree.renamed_iterators}")
                    nb_tries += 1
                    continue
                except KeyError as e:
                    logging.error(f"KeyError: {e}")
                    logging.error(f"Schedule: {schedule}")
                    logging.error(f"Tree: {schedule.tree}")
                    logging.error(f"renaming: {schedule.tree.renamed_iterators}")
                    logging.error(f"iterator: {iterator}")

                if str(tmp_schedule) in schedules_seen:
                    logging.info(f"Schedule {tmp_schedule} already seen")
                    nb_tries += 1
                    continue
                else:
                    schedules_seen.append(str(tmp_schedule))

                try:
                    schedule_is_legal = self.get_legality(tmp_schedule)
                except Exception as e:
                    logging.error(
                        f"Cannot get legality of schedule{tmp_schedule} due to: {e}"
                    )
                    nb_tries += 1
                    continue

                logging.info(
                    f"Schedule {tmp_schedule} is: {'legal' if schedule_is_legal else 'illegal'}"
                )

                if schedule_is_legal:
                    logging.info(f"Checking if parallelisation is legal")
                    schedule = tmp_schedule.copy()

                    assert tmp_schedule.tree

                    try:
                        tmp_schedule.add_optimizations(
                            [
                                tiramisu_actions.Parallelization(
                                    [iterator], tmp_schedule.tree
                                )
                            ]
                        )
                    except tiramisu_actions.CannotApplyException as e:
                        logging.error(f"Cannot apply action: {e}")
                        logging.error(f"Schedule: {tmp_schedule}")
                        logging.error(f"Tree: {tmp_schedule.tree}")
                        logging.error(
                            f"renaming: {tmp_schedule.tree.renamed_iterators}"
                        )
                        nb_tries += 1
                        continue
                    except KeyError as e:
                        logging.error(f"KeyError: {e}")
                        logging.error(f"Schedule: {tmp_schedule}")
                        logging.error(f"Tree: {tmp_schedule.tree}")
                        logging.error(
                            f"renaming: {tmp_schedule.tree.renamed_iterators}"
                        )
                        logging.error(f"iterator: {iterator}")
                        logging.error(
                            f"action {tiramisu_actions.Parallelization([iterator], tmp_schedule.tree)}"
                        )
                    schedule_is_legal = None
                    try:
                        schedule_is_legal = self.get_legality(tmp_schedule)
                    except Exception as e:
                        logging.error(
                            f"cannot get legality of schedule{tmp_schedule} due to: {e}"
                        )
                        nb_tries += 1
                        continue
                    logging.info(
                        f"Schedule {tmp_schedule} is: {'legal' if schedule_is_legal else 'illegal'}"
                    )
                    if schedule_is_legal:
                        parallelisation_is_legal = True
                        schedule = tmp_schedule.copy()
                        logging.info(f"Found an enabling schedule: {schedule}")
                    else:
                        logging.info(
                            f"Parallelisation on {iterator} still not legal with schedule {tmp_schedule}"
                        )
                schedule_is_legal = None
            else:
                logging.info(
                    f"Random action is None or is not a parallelisation action: {random_action}"
                )
                continue
            nb_tries += 1
        if parallelisation_is_legal:
            return schedule
        else:
            return None

    def get_legality(self, schedule: Schedule):
        if str(schedule) in self.schedules_dict:
            legality = self.schedules_dict[str(schedule)]["legality"]
        else:
            legality = schedule.is_legal()
            self.schedules_dict[str(schedule)] = {
                "legality": legality,
            }
        return legality


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--suffix", default=socket.gethostname(), type=str)
    parser.add_argument("--num-nodes", default=1, type=int)

    return parser.parse_args()


@ray.remote
class ProgressActor:
    def __init__(self):
        self.num_parallelizations_explored = {}
        self.num_parallelizations_enabled = {}

    def report_progress(
        self, task_id: int, num_explored: int, num_enabled: int
    ) -> None:
        if task_id not in self.num_parallelizations_explored:
            self.num_parallelizations_explored[task_id] = 0
            self.num_parallelizations_enabled[task_id] = 0
        self.num_parallelizations_explored[task_id] += num_explored
        self.num_parallelizations_enabled[task_id] += num_enabled

    def get_progress(self):
        return sum(self.num_parallelizations_explored.values()), sum(
            self.num_parallelizations_enabled.values()
        )

    def get_prgress_list(self):
        return self.num_parallelizations_explored, self.num_parallelizations_enabled


@ray.remote
def function_to_run(dataset_actor: DatasetActorDistributed, id: int, progress_actor: "ProgressActor", suffix: str = None, max_tries: int = 10):  # type: ignore
    BaseConfig.init(
        logging_level=logging.INFO,
        log_file=f"outputs/{suffix}.log",
        worker_id=id,
    )

    while True:
        next_program = ray.get(dataset_actor.get_next_function.remote())
        explorer = ParallelisationExplorer(next_program, max_tries=max_tries)
        explorer.explore()

        dataset_actor.update_dataset.remote(
            next_program.name,
            {"schedules_dict": explorer.schedules_dict},
            suffix=f"{suffix}",
        )

        progress_actor.report_progress.remote(
            id,
            len(explorer.parallelisations_to_enable),
            explorer.num_parallelisations_enabled,
        )
        print(
            f"Progress Report: worker {id} explored {len(explorer.parallelisations_to_enable)}, enabled {explorer.num_parallelisations_enabled}"
        )
        logging.info(
            f"Progress Report: worker {id} explored {len(explorer.parallelisations_to_enable)}, enabled {explorer.num_parallelisations_enabled}"
        )


def launch_distributed(
    num_workers: int = -1, suffix: str = "parallelization_explorer", num_nodes: int = 1
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
            dataset_actor, id=i, progress_actor=progress_actor, suffix=suffix
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
                print(f"Worker {id_dict[crashed_workers[0]]} crashed")

            print("Restarting crashed workers ...")
            for crashed in crashed_workers:
                ray_worker = function_to_run.remote(
                    dataset_actor,
                    id=id_dict[crashed],
                    progress_actor=progress_actor,
                    suffix=suffix,
                )
                workers.append(ray_worker)
                id_dict[ray_worker] = id_dict[crashed]

            print("Restarted workers")

        print(f"{len(workers)} workers generating schedules")
        print(
            f"Iterators explored, enabled: {ray.get(progress_actor.get_progress.remote())}"
        )
        print(f"Dict of progress: {ray.get(progress_actor.get_prgress_list.remote())}")


if __name__ == "__main__":
    args = get_arguments()
    suffix = f"parallelization_explorer_{args.suffix}"

    BaseConfig.init(
        logging_level=logging.INFO,
        log_file=f"outputs/{suffix}.log",
        # log_file=None,
    )
    assert BaseConfig.base_config

    num_workers = args.num_workers

    if num_workers != 1:
        launch_distributed(num_workers, suffix, args.num_nodes)
    else:
        dataset_actor = DatasetActor(BaseConfig.base_config.dataset)

        while True:
            next_program = dataset_actor.get_next_function()

            assert next_program.name

            explorer = ParallelisationExplorer(next_program, max_tries=10)
            explorer.explore()

            dataset_actor.update_dataset(
                next_program.name,
                {"schedules_dict": explorer.schedules_dict},
                suffix=f"{suffix}",
            )

            print(
                f"Progress Report: explored {len(explorer.parallelisations_to_enable)}, enabled {explorer.num_parallelisations_enabled}"
            )
            logging.info(
                f"Progress Report: explored {len(explorer.parallelisations_to_enable)}, enabled {explorer.num_parallelisations_enabled}"
            )
