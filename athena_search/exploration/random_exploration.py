import logging
import random
from typing import List
from athena.tiramisu import TiramisuProgram
import athena.tiramisu as tiramisu
import athena.tiramisu.tiramisu_actions as tiramisu_actions


class RandomExploration:
    def __init__(
        self,
        tiramisu_program: TiramisuProgram,
        seed: int = 0,
        tiling_factors: List[int] = [32, 64, 128],
        unrolling_factors: List[int] = [4, 8, 16],
    ):
        self.tiramisu_program = tiramisu_program
        self.schedule = tiramisu.Schedule(self.tiramisu_program)
        self.seed = seed
        self.tiling_factors = tiling_factors
        self.unrolling_factors = unrolling_factors

    def run(self, max_depth: int = 10):
        assert self.schedule.tree

        while len(self.schedule.optims_list) < max_depth:
            action_type = None
            candidate = None
            tmp_schedule = self.schedule.copy()

            while action_type is None or candidate is None:
                # Randomly select an Action to apply
                action_type = random.choice(
                    tiramisu.tiramisu_actions.TiramisuAction.get_types()
                )
                # # Force an action
                # if action_type != tiramisu_actions.TiramisuActionType.REVERSAL:
                #     continue

                logging.info(f"action_type: {action_type}")

                candidates = self.get_candidates(action_type, tmp_schedule)

                logging.info(f"candidates: {candidates}")

                if len(candidates) == 0:
                    raise ValueError("No candidates found")

                candidate = None
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

            try:
                tmp_schedule.add_optimizations(
                    self.initialize_actions(action_type, candidate, tmp_schedule)
                )
            except SkipActionException as e:
                logging.info(f"Skipping action: {e}")
                continue

            logging.info(f"schedule: { tmp_schedule}")

            try:
                if tmp_schedule.is_legal():
                    logging.info("Schedule is legal")
                    logging.info(tmp_schedule.apply_schedule(nb_exec_tiems=10))
                    self.schedule = tmp_schedule
                else:
                    logging.info("Schedule is not legal")

            except SkipActionException as e:
                logging.info(f"Skipping action: {e}")
                continue
            except Exception as e:
                logging.info(f"Skipping action due to an Exception: {e}")
                continue

    def initialize_actions(
        self,
        action_type: tiramisu_actions.TiramisuActionType,
        candidate_params,
        schedule: tiramisu.Schedule,
    ) -> List[tiramisu_actions.TiramisuAction]:
        # Check if the schedule tree is None
        assert schedule.tree
        # Get the computations associated with the candidate parameters (works for most computations)

        # Interchange
        if action_type == tiramisu_actions.TiramisuActionType.INTERCHANGE:
            computations = []
            for node in candidate_params:
                computations.extend(
                    schedule.tree.get_iterator_node(node).computations_list
                )
            return [tiramisu_actions.Interchange(candidate_params, computations)]

        # Tiling 2D
        elif action_type == tiramisu_actions.TiramisuActionType.TILING_2D:
            assert len(candidate_params) == 2
            computations = []
            for node in candidate_params:
                computations.extend(
                    schedule.tree.get_iterator_node(node).computations_list
                )
            params = [
                candidate_params[0],
                candidate_params[1],
                random.choice(self.tiling_factors),
                random.choice(self.tiling_factors),
            ]
            return [tiramisu_actions.Tiling2D(params, computations)]

        # Tiling 3D
        elif action_type == tiramisu_actions.TiramisuActionType.TILING_3D:
            assert len(candidate_params) == 3
            computations = []
            for node in candidate_params:
                computations.extend(
                    schedule.tree.get_iterator_node(node).computations_list
                )
            params = [
                candidate_params[0],
                candidate_params[1],
                candidate_params[2],
                random.choice(self.tiling_factors),
                random.choice(self.tiling_factors),
                random.choice(self.tiling_factors),
            ]
            return [tiramisu_actions.Tiling3D(params, computations)]

        # Parallelization
        elif action_type == tiramisu_actions.TiramisuActionType.PARALLELIZATION:
            # We parallelize all the iterators of the level
            actions = []
            for node in candidate_params:
                actions.append(
                    tiramisu_actions.Parallelization(
                        [node],
                        schedule.tree.get_candidate_computations(node),
                    )
                )
            return actions

        # Skewing
        elif action_type == tiramisu_actions.TiramisuActionType.SKEWING:
            assert len(candidate_params) == 2
            assert schedule.tree
            computations = schedule.tree.get_candidate_computations(candidate_params[0])
            loop_levels = schedule.tree.get_iterator_levels(candidate_params)

            try:
                factors = tiramisu_actions.Skewing.get_factors(
                    schedule=schedule,
                    loop_levels=loop_levels,
                    comps_skewed_loops=computations,
                )

            except ValueError:
                raise SkipActionException("Skewing factors not found")
            params = [candidate_params[0], candidate_params[1], factors[0], factors[1]]

            return [tiramisu_actions.Skewing(params, computations)]

        # Unrolling
        elif action_type == tiramisu_actions.TiramisuActionType.UNROLLING:
            assert len(candidate_params) == 1

            computations = schedule.tree.get_candidate_computations(candidate_params[0])

            params = [candidate_params[0], random.choice(self.unrolling_factors)]
            return [tiramisu_actions.Unrolling(params, computations)]

        # Fusion
        elif action_type == tiramisu_actions.TiramisuActionType.FUSION:
            assert len(candidate_params) == 1
            assert type(candidate_params[0]) == tuple
            computations = []
            for node in candidate_params[0]:
                computations.extend(schedule.tree.get_candidate_computations(node))
            return [tiramisu_actions.Fusion(candidate_params[0], computations)]

        elif action_type == tiramisu_actions.TiramisuActionType.REVERSAL:
            assert type(candidate_params) == str
            computations = schedule.tree.get_candidate_computations(candidate_params)
            return [
                tiramisu_actions.Reversal(
                    [candidate_params],
                    computations,
                )
            ]

    def get_candidates(
        self,
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
