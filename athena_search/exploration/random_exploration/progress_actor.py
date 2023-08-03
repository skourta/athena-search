import ray


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


# @ray.remote
# class ProgressActorDistributed(ProgressActor):
#     def __init__(self, distributed=False):
#         super().__init__(distributed=distributed)
