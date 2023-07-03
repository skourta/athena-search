import logging
from athena.tiramisu import Schedule, TiramisuProgram
from athena_search.utils.config import BaseConfig
from athena_search.utils.utils import load_dataset, save_dataset


def exectute_legal_schedules(
    path_to_dataset: str,
    path_to_annotation: str,
    path_to_cpps: str,
    machine_name: str = "jubail",
    replace=False,
):
    dataset = load_dataset(path_to_dataset)
    annotations = load_dataset(path_to_annotation)
    cpps = load_dataset(path_to_cpps)

    for program_name in dataset:
        tiramisu_func = TiramisuProgram.from_dict(
            name=program_name,
            data=annotations[program_name],
            original_str=cpps[program_name],
        )

        if "schedules_dict" not in dataset[program_name]:
            continue

        for gen_schedule in dataset[program_name]["schedules_dict"]:
            schedule_dict = dataset[program_name]["schedules_dict"][gen_schedule]
            if "legality" in schedule_dict and schedule_dict["legality"] == True:
                if "execution_times" not in schedule_dict:
                    schedule_dict["execution_times"] = {}

                if (
                    machine_name in schedule_dict["execution_times"]
                    and replace == False
                ):
                    continue

                schedule = Schedule.from_sched_str(
                    tiramisu_program=tiramisu_func, sched_str=gen_schedule
                )
                try:
                    exec_times = schedule.apply_schedule(nb_exec_tiems=30)
                except Exception as e:
                    logging.error(f"Skipping this schedule with Error: {e}")
                    continue
                logging.info(
                    f"Schedule: {gen_schedule} \n Execution times: {exec_times}"
                )
                schedule_dict["execution_times"][machine_name] = exec_times

    # write dataset to file
    save_dataset(dataset, f"{path_to_dataset}_exec_times.pkl")


if __name__ == "__main__":
    BaseConfig.init(logging_level=logging.INFO, log_file=None)
    exectute_legal_schedules(
        path_to_dataset="copy1.pkl",
        path_to_annotation="/scratch/sk10691/workspace/dataset_source/full_rl.pkl",
        path_to_cpps="/scratch/sk10691/workspace/dataset_source/functions_cpps.pkl",
        machine_name="jubail",
        replace=False,
    )
