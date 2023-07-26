import argparse
import json
import logging
import socket
import traceback
from typing import List, Tuple

import ray
from athena_search.utils.config import BaseConfig
import random
import re
from athena_search.utils.utils import load_dataset, save_dataset
from athena.tiramisu import Schedule, TiramisuProgram, tiramisu_actions

import pickle

SAVE_FREQUENCY = 5


def rl_sched_str_to_schedule(rl_sched_str: str, tiramisu_program: TiramisuProgram):
    schedule = Schedule(tiramisu_program)
    assert schedule.tree

    regex = r"{([\d\w]+)}:([\w\d\(\),\-]+)"
    matches = re.findall(regex, rl_sched_str)

    for match in matches[1:]:
        assert match[1] == matches[0][1]

    comp = matches[0][0]
    schedule_str = matches[0][1]
    schedule_list = [
        f"{optimization.strip()})"
        for optimization in schedule_str.split(")")
        if optimization != ""
    ]

    for optimization_str in schedule_list:
        if optimization_str[0] == "P":
            # extract loop level and comps using P\(L(\d),comps=\[([\w',]*)
            regex = r"P\(L(\d)\)"
            match = re.match(regex, optimization_str)
            if match:
                loop_level = int(match.group(1))

                schedule.add_optimizations(
                    [
                        tiramisu_actions.Parallelization(
                            [schedule.tree.get_comp_iterator(comp, loop_level)],
                            schedule.tree,
                        )
                    ]
                )
        elif optimization_str[0] == "U":
            # extract loop level, factor and comps using U\(L(\d),(\d+),comps=\[([\w',]*)\]\)
            regex = r"U\(L(\d),(\d+)\)"
            match = re.match(regex, optimization_str)
            if match:
                loop_level = int(match.group(1))
                factor = int(match.group(2))

                schedule.add_optimizations(
                    [
                        tiramisu_actions.Unrolling(
                            [
                                schedule.tree.get_comp_iterator(comp, loop_level),
                                factor,
                            ],
                            schedule.tree,
                        )
                    ]
                )
        elif optimization_str[0] == "I":
            regex = r"I\(L(\d),L(\d)\)"
            match = re.match(regex, optimization_str)
            if match:
                first_loop_level = int(match.group(1))
                second_loop_level = int(match.group(2))

                schedule.add_optimizations(
                    [
                        tiramisu_actions.Interchange(
                            [
                                schedule.tree.get_comp_iterator(comp, first_loop_level),
                                schedule.tree.get_comp_iterator(
                                    comp, second_loop_level
                                ),
                            ],
                            schedule.tree,
                        )
                    ]
                )
        elif optimization_str[0] == "R":
            regex = r"R\(L(\d)\)"
            match = re.match(regex, optimization_str)
            if match:
                loop_level = int(match.group(1))

                schedule.add_optimizations(
                    [
                        tiramisu_actions.Reversal(
                            [schedule.tree.get_comp_iterator(comp, loop_level)],
                            schedule.tree,
                        )
                    ]
                )
        elif optimization_str[:2] == "T2":
            regex = r"T2\(L(\d),L(\d),(\d+),(\d+)\)"
            match = re.match(regex, optimization_str)
            if match:
                outer_loop_level = int(match.group(1))
                inner_loop_level = int(match.group(2))
                outer_loop_factor = int(match.group(3))
                inner_loop_factor = int(match.group(4))

                schedule.add_optimizations(
                    [
                        tiramisu_actions.Tiling2D(
                            [
                                schedule.tree.get_comp_iterator(comp, outer_loop_level),
                                schedule.tree.get_comp_iterator(comp, inner_loop_level),
                                outer_loop_factor,
                                inner_loop_factor,
                            ],
                            schedule.tree,
                        )
                    ]
                )
        elif optimization_str[:2] == "T3":
            regex = r"T3\(L(\d),L(\d),L(\d),(\d+),(\d+),(\d+)\)"
            match = re.match(regex, optimization_str)
            if match:
                outer_loop_level = int(match.group(1))
                middle_loop_level = int(match.group(2))
                inner_loop_level = int(match.group(3))
                outer_loop_factor = int(match.group(4))
                middle_loop_factor = int(match.group(5))
                inner_loop_factor = int(match.group(6))

                schedule.add_optimizations(
                    [
                        tiramisu_actions.Tiling3D(
                            [
                                schedule.tree.get_comp_iterator(comp, outer_loop_level),
                                schedule.tree.get_comp_iterator(
                                    comp, middle_loop_level
                                ),
                                schedule.tree.get_comp_iterator(comp, inner_loop_level),
                                outer_loop_factor,
                                middle_loop_factor,
                                inner_loop_factor,
                            ],
                            schedule.tree,
                        )
                    ]
                )
        elif optimization_str[0] == "S":
            regex = r"S\(L(\d),L(\d),(-?\d+),(-?\d+)\)"
            match = re.match(regex, optimization_str)
            if match:
                outer_loop_level = int(match.group(1))
                inner_loop_level = int(match.group(2))
                outer_loop_factor = int(match.group(3))
                inner_loop_factor = int(match.group(4))

                schedule.add_optimizations(
                    [
                        tiramisu_actions.Skewing(
                            [
                                schedule.tree.get_comp_iterator(comp, outer_loop_level),
                                schedule.tree.get_comp_iterator(comp, inner_loop_level),
                                outer_loop_factor,
                                inner_loop_factor,
                            ],
                            schedule.tree,
                        )
                    ]
                )

    return schedule


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=-1, type=int)
    parser.add_argument("--suffix", default=socket.gethostname(), type=str)
    parser.add_argument("--num-nodes", default=1, type=int)

    return parser.parse_args()


def init_logger(log_file: str | None, logging_level: int, worker_id: int | None):
    if worker_id is not None:
        log_format = f"|worker{worker_id}|%(asctime)s|%(levelname)s| %(message)s"
    else:
        log_format = "|%(asctime)s|%(levelname)s| %(message)s"

    if log_file is None:
        logging.basicConfig(
            level=logging_level,
            format=log_format,
        )
    else:
        logging.basicConfig(
            filename=log_file,
            filemode="a",
            level=logging_level,
            format=log_format,
        )


def convert_program(
    tiramisu_program: TiramisuProgram,
    schedules_legality: dict,
):
    assert tiramisu_program.name

    skipped: List[Tuple[str, str, Exception]] = []
    num_done = 0

    new_program_dict = {}
    new_program_dict["program_annotation"] = tiramisu_program.annotations
    new_program_dict["schedules_dict"] = {}
    for schedule in schedules_legality:
        if schedules_legality[schedule] == 1:
            schedule_legality = True
        elif schedules_legality[schedule] == 0:
            schedule_legality = False
        else:
            skipped.append(
                (tiramisu_program.name, str(schedule), ValueError("legality -1"))
            )
            continue
        try:
            schedule_obj = rl_sched_str_to_schedule(schedule, tiramisu_program)
        except Exception as e:
            logging.error(f"Skipped: {tiramisu_program.name}|{schedule}|{e}")
            skipped.append((tiramisu_program.name, str(schedule), e))
            continue
        new_program_dict["schedules_dict"][str(schedule_obj)] = {
            "legality": schedule_legality,
            "execution_times": {},
            "solver_results": {},
        }
        num_done += 1

    return new_program_dict, num_done, skipped


@ray.remote
def convert_program_distributed(
    worker_id: int,
    tiramisu_program: TiramisuProgram,
    schedules_legality: dict,
    log_file: str | None = None,
    logging_level: int = logging.INFO,
):
    assert tiramisu_program.name
    init_logger(log_file=log_file, logging_level=logging_level, worker_id=worker_id)
    logging.info("-----------")
    logging.info(f"worker: {worker_id} working on {tiramisu_program.name} program")
    try:
        new_program_dict, num_done, skipped = convert_program(
            tiramisu_program, schedules_legality
        )
    except Exception as e:
        logging.error(f"worker: {worker_id} failed on {tiramisu_program.name}")
        logging.error(traceback.format_exc())
        return False, tiramisu_program.name, None, 0, None

    return True, tiramisu_program.name, new_program_dict, num_done, skipped


if __name__ == "__main__":
    args = get_arguments()
    suffix = f"rl_data_{args.suffix}"
    log_file = None
    logging_level = logging.INFO
    init_logger(log_file=log_file, logging_level=logging_level, worker_id=None)

    # load cpps
    with open(
        "/scratch/sk10691/workspace/dataset_source/functions_cpps.pkl", "rb"
    ) as f:
        cpps = pickle.load(f)

    rl = load_dataset("/scratch/sk10691/workspace/dataset_source/full_rl.pkl")

    num_workers = args.num_workers

    new_rl = {}
    all_skipped = []
    all_num_done = 0

    num_schedules = 0
    for program_name in rl:
        num_schedules += len(rl[program_name]["schedules_legality"])
    logging.info(
        f"working on converting {num_schedules} schedules for {len(rl)} programs"
    )

    if num_workers == 1:
        for idx, program_name in enumerate(rl.keys()):
            print(f"working on {program_name} {idx}/{len(rl)}", flush=True)
            tiramisu_program = TiramisuProgram.from_dict(
                program_name, data=rl[program_name], original_str=cpps[program_name]
            )
            schedules_legality = rl[program_name]["schedules_legality"]

            new_prog_dict, num_done, skipped = convert_program(
                tiramisu_program, schedules_legality
            )
            new_rl[program_name] = new_prog_dict
            all_skipped.extend(skipped)
            all_num_done += num_done

            if all_num_done % SAVE_FREQUENCY == 0:
                save_dataset(new_rl, f"datasets/{suffix}.pkl")
    else:
        if args.num_nodes > 1:
            ray.init(address="auto")
        else:
            ray.init()

        print(ray.available_resources())

        if num_workers == -1:
            num_workers = int(ray.available_resources()["CPU"])

        print(f"Launching {num_workers} workers")

        programs_to_redo = []
        workers = []
        for idx, program_name in enumerate(rl.keys()):
            print(f"working on {program_name} {idx}/{len(rl)}", flush=True)
            tiramisu_program = TiramisuProgram.from_dict(
                program_name, data=rl[program_name], original_str=cpps[program_name]
            )
            schedules_legality = rl[program_name]["schedules_legality"]

            workers.append(
                convert_program_distributed.remote(
                    len(workers),
                    tiramisu_program,
                    schedules_legality,
                    log_file=log_file,
                    logging_level=logging_level,
                )
            )

            if len(workers) == num_workers:
                finished, workers = ray.wait(workers)

                for finished_worker in finished:
                    result, finshed_program, new_prog_dict, num_done, skipped = ray.get(
                        finished_worker
                    )
                    if result:
                        new_rl[finshed_program] = new_prog_dict
                        all_skipped.extend(skipped)
                        all_num_done += num_done
                        if all_num_done % SAVE_FREQUENCY == 0:
                            save_dataset(new_rl, f"datasets/{suffix}.pkl")
                    else:
                        programs_to_redo.append(finshed_program)

        if len(workers) > 0:
            finsihed, workers = ray.wait(workers, num_returns=len(workers))

            for finished_worker in finsihed:
                result, finshed_program, new_prog_dict, num_done, skipped = ray.get(
                    finished_worker
                )
                if result:
                    new_rl[finshed_program] = new_prog_dict
                    all_skipped.extend(skipped)
                    all_num_done += num_done
                else:
                    programs_to_redo.append(finshed_program)

        save_dataset(new_rl, f"datasets/{suffix}.pkl")
        logging.info(f"Done with {len(new_rl)} programs")
        logging.info(f"Done with {all_num_done} schedules")
        logging.info(f"Skipped {len(all_skipped)} schedules")

        with open(f"outputs/skipped_{suffix}.json", "w") as f:
            json.dump(all_skipped, f)
