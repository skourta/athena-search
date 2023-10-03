import pickle

import ray

from athena_search.utils.utils import save_dataset

SAVE_FREQUENCY = 5


@ray.remote
class ProgressActor:
    def __init__(self):
        self.object_files = {}
        self.nbr_obj_generated_per_task = {}

    def report_progress(
        self, task_id: int, function_name: str, obj_file: bytes
    ) -> None:
        if task_id not in self.nbr_obj_generated_per_task:
            self.nbr_obj_generated_per_task[task_id] = 0
        self.nbr_obj_generated_per_task[task_id] += 1
        self.object_files[function_name] = obj_file

        nbr_obj = sum(self.nbr_obj_generated_per_task.values())
        if nbr_obj % SAVE_FREQUENCY == 0:
            self.save_object_files()

    def save_object_files(self):
        print("[START] Saving object files")
        save_dataset(
            dataset=self.object_files, path_to_save="datasets/object_files.pkl"
        )
        print("[END] Saving object files")

    def get_progress(self) -> int:
        return sum(self.nbr_obj_generated_per_task.values())

    def get_prgress_list(self):
        return self.nbr_obj_generated_per_task


# @ray.remote
# class ProgressActorDistributed(ProgressActor):
#     def __init__(self, distributed=False):
#         super().__init__(distributed=distributed)
