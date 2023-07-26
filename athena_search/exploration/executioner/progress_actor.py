from typing import List

from athena.tiramisu import TiramisuProgram
import ray


class ProgressActor:
    def __init__(self, total_schedules, dataset_actor, distributed=False):
        self.progress = {}
        self.total_scheduels = total_schedules
        self.skip_schedules = []
        self.dataset_actor = dataset_actor
        self.distributed = distributed

    def report_progress(
        self,
        node_name: str,
        num_explored: int,
        skipped_schedules: List[str],
        tiramisu_program: TiramisuProgram,
    ) -> None:
        if node_name not in self.progress:
            self.progress[node_name] = (0, 0)

        self.progress[node_name] = (
            self.progress[node_name][0] + num_explored,
            self.progress[node_name][1] + len(skipped_schedules),
        )

        self.skip_schedules += skipped_schedules
        if self.distributed:
            self.dataset_actor.update_dataset.remote(
                tiramisu_program.name,
                {"schedules_dict": tiramisu_program.schedules_dict},
            )
        else:
            self.dataset_actor.update_dataset(
                tiramisu_program.name,
                {"schedules_dict": tiramisu_program.schedules_dict},
            )

    def get_progress(self):
        return self.progress

    def get_skipped_schedules(self):
        return self.skip_schedules

    def get_total_progress(self):
        return sum([x[0] for x in self.progress.values()]), sum(
            [x[1] for x in self.progress.values()]
        )

    def print_progress(self):
        total_explored, total_skipped = self.get_total_progress()
        print(f"Total Explored: {total_explored:,}")
        print(f"Total Skipped: {total_skipped:,}")
        print(
            f"Total Progress: {((total_explored + total_skipped) / self.total_scheduels):,.2f}"
        )


@ray.remote
class ProgressActorDistributed(ProgressActor):
    def __init__(self, total_schedules, dataset_actor):
        super().__init__(total_schedules, dataset_actor, distributed=True)
