import pickle

from athena.tiramisu import Schedule


def load_dataset(path_to_dataset: str):
    with open(path_to_dataset, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def save_dataset(dataset, path_to_save: str):
    with open(path_to_save, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


def nbr_schedules(dataset: dict, legality: bool | None = None):
    count = 0
    for program in dataset.values():
        if "schedules_dict" in program:
            if legality is None:
                count += len(program["schedules_dict"])
            else:
                count += sum(
                    [
                        1
                        for schedule in program["schedules_dict"].values()
                        if schedule["legality"] == legality
                    ]
                )

    return count


def minimize_schedule(schedule: Schedule):
    """
    Minimize a schedule by removing all the useless transformations.
    """
    pass
