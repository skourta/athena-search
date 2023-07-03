import logging
import random
import time
from typing import Dict, List, Tuple
from athena.tiramisu import TiramisuProgram
import athena.tiramisu as tiramisu
import athena.tiramisu.tiramisu_actions as tiramisu_actions
import ray

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

        try:
            if tmp_schedule.is_legal():
                if str(tmp_schedule) not in schedules_dict:
                    schedules_dict[str(tmp_schedule)] = {}
                logging.info("Schedule is legal")
                schedules_dict[str(tmp_schedule)]["legality"] = True
                # logging.info(tmp_schedule.apply_schedule(nb_exec_tiems=10))
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

        try:
            factors = tiramisu_actions.Skewing.get_factors(
                schedule=schedule,
                loop_levels=loop_levels,
                comps_skewed_loops=computations,
            )

        except ValueError:
            schedules_dict[str(schedule)]["solver_values"][",".join(loop_levels)] = None
            raise SkipActionException("Skewing factors not found")

        schedules_dict[str(schedule)]["solver_values"][",".join(loop_levels)] = factors

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
    else:
        raise NotImplementedError


class SkipActionException(Exception):
    pass
