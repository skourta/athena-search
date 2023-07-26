import argparse
import logging
from time import sleep
import time
from typing import Tuple
from athena.tiramisu.tiramisu_actions.fusion import Fusion
from athena.tiramisu import Schedule, TiramisuProgram, tiramisu_actions
import ray
from athena_search.data.dataset_actor.dataset_actor import DatasetActor
from athena_search.exploration.random_exploration import (
    generate_legalities_random_schedules,
)

from athena_search.utils.config import BaseConfig

# from athena.utils.config import BaseConfig
import tests.utils as test_utils


# def function_to_run(dataset_actor: DatasetActor, id: int, progress_actor: "ProgressActor", suffix: str = None):  # type: ignore
#     while True:
#         next_program = dataset_actor.get_next_function()

#         legalities, solver_results = generate_legalities_random_schedules(
#             next_program, id_worker=id, log_file=f"athena_search_{suffix}.log"
#         )

#         next_program.schedules_legality.update(legalities)
#         next_program.schedules_solver.update(solver_results)

#         assert next_program.name

#         dataset_actor.update_dataset(
#             next_program.name,
#             {
#                 "schedules_legality": next_program.schedules_legality,
#                 "schedules_solver": next_program.schedules_solver,
#             },
#             suffix=f"{suffix}",
#         )

#         logging.info(
#             f"Progress Report: worker {id} generated {len(legalities)} schedules and {len(solver_results)} solver results"
#         )


if __name__ == "__main__":
    BaseConfig.init(logging_level=logging.INFO)
    assert BaseConfig.base_config

    suffix = time.time()

    # dataset_actor = DatasetActor(BaseConfig.base_config.dataset)

    # next_program = dataset_actor.get_next_function()

    # schedule = Schedule(next_program)

    # print(schedule.apply_schedule(3, 0.000001))
    # print(schedule.apply_schedule(3))

    matmul = TiramisuProgram.from_file(
        "./_tmp/function_matmul_MEDIUM.cpp", load_annotations=True, load_tree=True
    )

    schedule = Schedule(matmul)

    print(schedule.apply_schedule(3))

    assert schedule.tree

    schedule.add_optimizations(
        [tiramisu_actions.Parallelization(["i00"], schedule.tree)]
    )

    print(schedule.apply_schedule(3))
