from typing import List

import ray
from athena.tiramisu import TiramisuProgram


class ProgressActor:
    def __init__(self, total_programs, distributed=False):
        self.progress = {}
        self.total_programs = total_programs
        self.skip_schedules = []
        self.distributed = distributed

    def report_progress(
        self,
        node_name: str,
        program_name: str,
        is_executed: bool,
    ) -> None:
        if node_name not in self.progress:
            self.progress[node_name] = (0, 0)

        if is_executed:
            self.progress[node_name] = (
                self.progress[node_name][0] + 1,
                self.progress[node_name][1],
            )
        else:
            self.progress[node_name] = (
                self.progress[node_name][0],
                self.progress[node_name][1] + 1,
            )
            self.skip_schedules.append(program_name)

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
            f"Total Progress: {((total_explored + total_skipped) / self.total_programs):,.2f}"
        )


@ray.remote
class ProgressActorDistributed(ProgressActor):
    def __init__(self, total_schedules):
        super().__init__(total_schedules, distributed=True)
