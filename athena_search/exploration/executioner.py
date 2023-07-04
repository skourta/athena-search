import logging
import socket
from athena.tiramisu import Schedule, TiramisuProgram
from athena_search.utils.config import BaseConfig
from athena_search.utils.utils import load_dataset, save_dataset

SAVE_FREQUENCY = 20


def exectute_legal_schedules(
    path_to_dataset: str,
    # path_to_annotation: str,
    path_to_cpps: str,
    machine_name: str = "jubail",
    replace=False,
):
    dataset = load_dataset(path_to_dataset)
    # annotations = load_dataset(path_to_annotation)
    cpps = load_dataset(path_to_cpps)

    nbr_executed = 0
    current_schedule = 0
    skipped_schedule = 0

    count = 0

    for program_name in dataset:
        if "schedules_dict" in dataset[program_name]:
            for gen_schedule in dataset[program_name]["schedules_dict"]:
                schedule_dict = dataset[program_name]["schedules_dict"][gen_schedule]
                if "legality" in schedule_dict and schedule_dict["legality"] == True:
                    count += 1

    print(f"Total number of legal schedules: {count}")
    logging.info(f"Total number of legal schedules: {count}")

    for program_name in dataset:
        tiramisu_func = TiramisuProgram.from_dict(
            name=program_name,
            data=dataset[program_name],
            original_str=cpps[program_name],
        )

        if "schedules_dict" not in dataset[program_name]:
            continue

        for gen_schedule in dataset[program_name]["schedules_dict"]:
            schedule_dict = dataset[program_name]["schedules_dict"][gen_schedule]
            if "legality" in schedule_dict and schedule_dict["legality"] == True:
                logging.info(
                    f"Current schedule {current_schedule}: {gen_schedule} for {program_name}"
                )
                print(
                    f"Current schedule {current_schedule}: {gen_schedule} for {program_name}"
                )
                current_schedule += 1

                if "execution_times" not in schedule_dict:
                    schedule_dict["execution_times"] = {}

                if (
                    machine_name in schedule_dict["execution_times"]
                    and replace == False
                ):
                    logging.info(
                        f"Schedule {gen_schedule} for {program_name} already executed on {machine_name}"
                    )
                    skipped_schedule += 1
                    continue

                schedule = Schedule.from_sched_str(
                    tiramisu_program=tiramisu_func, sched_str=gen_schedule
                )
                logging.info(f"Applying Schedule: {gen_schedule} for {program_name}")
                try:
                    exec_times = schedule.apply_schedule(nb_exec_tiems=30)
                except Exception as e:
                    logging.error(f"Skipping this schedule with Error: {e}")
                    skipped_schedule += 1
                    continue
                logging.info(f"Schedule: {gen_schedule} for {program_name} executed")
                logging.info(f"Execution times: {exec_times}")
                schedule_dict["execution_times"][machine_name] = exec_times
                nbr_executed += 1
                if nbr_executed % SAVE_FREQUENCY == 0:
                    logging.info(f"Saving dataset to file")
                    save_dataset(dataset, f"{path_to_dataset}_exec_times.pkl")
                    logging.info(f"Saved dataset to file")

    # write dataset to file
    save_dataset(dataset, f"{path_to_dataset}_exec_times.pkl")


if __name__ == "__main__":
    BaseConfig.init(
        logging_level=logging.INFO,
        log_file=f"outputs/executioner_{socket.gethostname()}.log",
    )
    exectute_legal_schedules(
        path_to_dataset="to_be_executed/142k.pkl",
        # path_to_annotation="/scratch/sk10691/workspace/dataset_source/full_rl.pkl",
        path_to_cpps="/scratch/sk10691/workspace/dataset_source/functions_cpps.pkl",
        machine_name="jubail",
        replace=False,
    )
