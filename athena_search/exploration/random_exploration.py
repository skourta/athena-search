import argparse
import logging
import random
import socket
import time
from typing import Dict, List, Tuple
from athena.tiramisu import TiramisuProgram
import athena.tiramisu as tiramisu
import athena.tiramisu.tiramisu_actions as tiramisu_actions
import ray
from athena_search.data.dataset_actor.dataset_actor import (
    DatasetActor,
    DatasetActorDistributed,
)

from athena_search.utils.config import BaseConfig

MAX_TRIES = 50


@ray.remote
def generate_legalities_random_schedules_dist(
    tiramisu_program: TiramisuProgram,
    seed: int = 0,
    tiling_factors: List[int] = [16, 32, 64, 128, 256, 512],
    unrolling_factors: List[int] = [2, 4, 8, 16, 32, 64],
    max_depth: int = 10,
    worker_id: int = 0,
    logging_level: int = logging.INFO,
    log_file: str = None,
):
    BaseConfig.init(logging_level=logging_level, worker_id=worker_id, log_file=log_file)
    assert BaseConfig.base_config

    logging.info(
        f"Worker {worker_id} Starting random exploration: {tiramisu_program.name}"
    )

    return generate_legalities_random_schedules(
        tiramisu_program=tiramisu_program,
        seed=seed,
        tiling_factors=tiling_factors,
        unrolling_factors=unrolling_factors,
        max_depth=max_depth,
    )


# @ray.remote
def generate_legalities_random_schedules(
    tiramisu_program: TiramisuProgram,
    seed: int = 0,
    tiling_factors: List[int] = [16, 32, 64, 128, 256, 512],
    unrolling_factors: List[int] = [2, 4, 8, 16, 32, 64],
    max_depth: int = 10,
):
    # BaseConfig.init(logging_level=logging_level)
    assert BaseConfig.base_config

    schedule = tiramisu.Schedule(tiramisu_program)

    if tiramisu_program.schedules_dict:
        schedules_dict = tiramisu_program.schedules_dict
    else:
        schedules_dict = {}

    num_schedule = 0

    assert schedule.tree

    nbr_tries = 0

    while len(schedule.optims_list) < max_depth and nbr_tries < MAX_TRIES:
        action_type = None
        candidate = None
        tmp_schedule = schedule.copy()

        while (action_type is None or candidate is None) and nbr_tries < MAX_TRIES:
            nbr_tries += 1

            logging.info(f"nbr_tries: {nbr_tries}")
            logging.info(f"current schedule: {tmp_schedule}")

            candidate = None

            # Randomly select an Action to apply
            action_type = random.choice(
                tiramisu.tiramisu_actions.TiramisuAction.get_types()
            )

            logging.info(f"action_type: {action_type}")

            candidates = get_candidates(action_type, tmp_schedule)

            logging.info(f"candidates: {candidates}")

            if len(candidates) == 0:
                candidate = None
                logging.info("No candidates found")
                continue

            # Randomly select a candidate
            if type(candidates) == list:
                candidate = random.choice(candidates)
                if type(candidate) is not list:
                    candidate = [candidate]
            elif type(candidates) == dict:
                # Choose a random root (key)
                root = random.choice(list(candidates.keys()))
                # choose a random candidate from the root's list of candidates
                if candidates[root]:
                    candidate = random.choice(candidates[root])

            logging.info(f"chose candidate: {candidate}")

        if nbr_tries > MAX_TRIES or candidate is None or action_type is None:
            break

        try:
            assert action_type
            tmp_schedule.add_optimizations(
                initialize_actions(
                    action_type,
                    candidate,
                    tmp_schedule,
                    tiling_factors,
                    unrolling_factors,
                    schedules_dict,
                )
            )
        except SkipActionException as e:
            logging.info(f"Skipping action: {e}")
            continue
        except tiramisu_actions.CannotApplyException as e:
            logging.info(f"Skipping action: {e}")
            continue
        except Exception as e:
            logging.info(f"Skipping action due to an Exception: {e}")
            continue

        logging.info(f"schedule: { tmp_schedule}")

        if tmp_schedule in schedules_dict:
            logging.info("Schedule already in dataset. Skipping.")
            continue

        try:
            if tmp_schedule.is_legal():
                if str(tmp_schedule) not in schedules_dict:
                    schedules_dict[str(tmp_schedule)] = {}
                logging.info("Schedule is legal")
                schedules_dict[str(tmp_schedule)]["legality"] = True

                # exec_times = tmp_schedule.apply_schedule(
                #     nb_exec_tiems=10, max_mins_per_schedule=30
                # )
                # schedules_dict[str(tmp_schedule)]["execution_times"][
                #     BaseConfig.base_config.machine
                # ] = exec_times
                # logging.info(f"Execution times: {exec_times}")

                schedule = tmp_schedule
                num_schedule += 1
            else:
                logging.info("Schedule is not legal")
                if str(tmp_schedule) not in schedules_dict:
                    schedules_dict[str(tmp_schedule)] = {}
                schedules_dict[str(tmp_schedule)]["legality"] = False
                num_schedule += 1

        except SkipActionException as e:
            logging.info(f"Skipping action: {e}")
            continue
        except Exception as e:
            logging.info(f"Skipping action due to an Exception: {e}")
            continue

    logging.info(schedule)
    return schedules_dict, num_schedule


def initialize_actions(
    action_type: tiramisu_actions.TiramisuActionType,
    candidate_params,
    schedule: tiramisu.Schedule,
    tiling_factors: List[int] = [32, 64, 128],
    unrolling_factors: List[int] = [4, 8, 16],
    schedules_dict: Dict = {},
) -> List[tiramisu_actions.TiramisuAction]:
    # Check if the schedule tree is None
    assert schedule.tree
    # Get the computations associated with the candidate parameters (works for most computations)

    # Interchange
    if action_type == tiramisu_actions.TiramisuActionType.INTERCHANGE:
        return [tiramisu_actions.Interchange(candidate_params, schedule.tree)]

    # Tiling 2D
    elif action_type == tiramisu_actions.TiramisuActionType.TILING_2D:
        assert len(candidate_params) == 2
        params = [
            candidate_params[0],
            candidate_params[1],
            random.choice(tiling_factors),
            random.choice(tiling_factors),
        ]
        return [tiramisu_actions.Tiling2D(params, schedule.tree)]

    # Tiling 3D
    elif action_type == tiramisu_actions.TiramisuActionType.TILING_3D:
        assert len(candidate_params) == 3
        params = [
            candidate_params[0],
            candidate_params[1],
            candidate_params[2],
            random.choice(tiling_factors),
            random.choice(tiling_factors),
            random.choice(tiling_factors),
        ]
        return [tiramisu_actions.Tiling3D(params, schedule.tree)]

    # Parallelization
    elif action_type == tiramisu_actions.TiramisuActionType.PARALLELIZATION:
        # We parallelize all the iterators of the level
        actions = []
        for node in candidate_params:
            actions.append(
                tiramisu_actions.Parallelization(
                    [node],
                    schedule.tree,
                )
            )
        return actions

    # Skewing
    elif action_type == tiramisu_actions.TiramisuActionType.SKEWING:
        assert len(candidate_params) == 2
        assert schedule.tree

        if str(schedule) not in schedules_dict:
            schedules_dict[str(schedule)] = {}

        computations = schedule.tree.get_iterator_subtree_computations(
            candidate_params[0]
        )
        loop_levels = schedule.tree.get_iterator_levels(candidate_params)

        if str(schedule) not in schedules_dict:
            schedules_dict[str(schedule)] = {}
        if "solver_values" not in schedules_dict[str(schedule)]:
            schedules_dict[str(schedule)]["solver_values"] = {}

        try:
            factors = tiramisu_actions.Skewing.get_factors(
                schedule=schedule,
                loop_levels=loop_levels,
                comps_skewed_loops=computations,
            )

        except ValueError:
            schedules_dict[str(schedule)]["solver_values"][
                f"{loop_levels[0]},{loop_levels[1]}"
            ] = None
            raise SkipActionException("Skewing factors not found")

        schedules_dict[str(schedule)]["solver_values"][
            f"{loop_levels[0]},{loop_levels[1]}"
        ] = factors

        params = [candidate_params[0], candidate_params[1], factors[0], factors[1]]

        return [tiramisu_actions.Skewing(params, schedule.tree)]

    # Unrolling
    elif action_type == tiramisu_actions.TiramisuActionType.UNROLLING:
        assert len(candidate_params) == 1
        params = [candidate_params[0], random.choice(unrolling_factors)]
        return [tiramisu_actions.Unrolling(params, schedule.tree)]

    # Fusion
    elif action_type == tiramisu_actions.TiramisuActionType.FUSION:
        assert len(candidate_params) == 1
        assert type(candidate_params[0]) == tuple
        return [tiramisu_actions.Fusion(candidate_params[0], schedule.tree)]

    elif action_type == tiramisu_actions.TiramisuActionType.REVERSAL:
        assert type(candidate_params) == str
        return [
            tiramisu_actions.Reversal(
                [candidate_params],
                schedule.tree,
            )
        ]
    elif action_type == tiramisu_actions.TiramisuActionType.DISTRIBUTION:
        assert type(candidate_params) == list

        return [
            tiramisu_actions.Distribution(
                candidate_params,
                schedule.tree,
            )
        ]


def get_candidates(
    action_type: tiramisu_actions.TiramisuActionType,
    schedule: tiramisu.Schedule,
):
    if schedule.tree is None:
        raise ValueError("The schedule tree is None")
    if action_type == tiramisu_actions.TiramisuActionType.INTERCHANGE:
        return tiramisu_actions.Interchange.get_candidates(schedule.tree)
    elif action_type == tiramisu_actions.TiramisuActionType.TILING_2D:
        return tiramisu_actions.Tiling2D.get_candidates(schedule.tree)
    elif action_type == tiramisu_actions.TiramisuActionType.TILING_3D:
        return tiramisu_actions.Tiling3D.get_candidates(schedule.tree)
    elif action_type == tiramisu_actions.TiramisuActionType.PARALLELIZATION:
        return tiramisu_actions.Parallelization.get_candidates(schedule.tree)
    elif action_type == tiramisu_actions.TiramisuActionType.SKEWING:
        return tiramisu_actions.Skewing.get_candidates(schedule.tree)
    elif action_type == tiramisu_actions.TiramisuActionType.UNROLLING:
        return tiramisu_actions.Unrolling.get_candidates(schedule.tree)
    elif action_type == tiramisu_actions.TiramisuActionType.FUSION:
        return tiramisu_actions.Fusion.get_candidates(schedule.tree)
    elif action_type == tiramisu_actions.TiramisuActionType.REVERSAL:
        return tiramisu_actions.Reversal.get_candidates(schedule.tree)
    elif action_type == tiramisu_actions.TiramisuActionType.DISTRIBUTION:
        return tiramisu_actions.Distribution.get_candidates(schedule.tree)
    else:
        raise NotImplementedError


def get_random_action(schedule: tiramisu.Schedule):
    action_type = None
    candidate = None
    nbr_tries = 0

    action = None

    while (action_type is None or candidate is None) and nbr_tries < MAX_TRIES:
        nbr_tries += 1

        logging.info(f"selecting random action nbr_tries: {nbr_tries}")
        logging.info(f"current schedule: {schedule}")

        candidate = None

        # Randomly select an Action to apply
        action_type = random.choice(
            tiramisu.tiramisu_actions.TiramisuAction.get_types()
        )

        logging.info(f"action_type: {action_type}")

        candidates = get_candidates(action_type, schedule)

        logging.info(f"candidates: {candidates}")

        if len(candidates) == 0:
            candidate = None
            logging.info("No candidates found")
            continue

        # Randomly select a candidate
        if type(candidates) == list:
            candidate = random.choice(candidates)
            if type(candidate) is not list:
                candidate = [candidate]
        elif type(candidates) == dict:
            # Choose a random root (key)
            root = random.choice(list(candidates.keys()))
            # choose a random candidate from the root's list of candidates
            if candidates[root]:
                candidate = random.choice(candidates[root])

        logging.info(f"chose candidate: {candidate}")

        if candidate is None:
            continue

        try:
            action = initialize_actions(
                action_type=action_type,
                candidate_params=candidate,
                schedule=schedule,
            )
        except SkipActionException as e:
            logging.error(f"Skipping action due to {e}")
            continue
        except tiramisu_actions.CannotApplyException as e:
            logging.error(f"Cannot apply this action: {e}")
            continue

    logging.info(f"initialized action: {action}")
    return action


class SkipActionException(Exception):
    pass


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--suffix", default=socket.gethostname(), type=str)
    parser.add_argument("--num-nodes", default=1, type=int)
    parser.add_argument("--dataset-path", default=None, type=str)

    return parser.parse_args()


@ray.remote
class ProgressActor:
    def __init__(self):
        self.nbr_schedules_generated_per_task = {}

    def report_progress(self, task_id: int, num_schedules: int) -> None:
        if task_id not in self.nbr_schedules_generated_per_task:
            self.nbr_schedules_generated_per_task[task_id] = 0
        self.nbr_schedules_generated_per_task[task_id] += num_schedules

    def get_progress(self) -> int:
        return sum(self.nbr_schedules_generated_per_task.values())

    def get_prgress_list(self):
        return self.nbr_schedules_generated_per_task


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

    num_workers = args.num_workers

    if args.dataset_path:
        BaseConfig.base_config.dataset.dataset_path = args.dataset_path

    if num_workers != 1:
        launch_distributed(num_workers, suffix, args.num_nodes)
    else:
        logging.info("Launching single worker ...")

        logging.info("Initializing dataset actor ...")
        dataset_actor = DatasetActor(BaseConfig.base_config.dataset)

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
                suffix=f"{suffix}",
            )

            logging.info(
                f"Progress Report: worker {id} generated {nbr_schedules} schedules for {next_program.name}"
            )
