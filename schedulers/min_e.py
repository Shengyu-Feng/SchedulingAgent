"""
MinE — assign job to the site with the minimum *execution* time (w_j / c_n).
Ignores transfer cost; useful as a pure compute-locality baseline.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from env.interfaces import SchedulerInterface

if TYPE_CHECKING:
    from env.simulator import HEPSimulator
    from env.core import Job


class MinEScheduler(SchedulerInterface):
    def schedule(self, job: "Job", env: "HEPSimulator") -> Optional[int]:
        best_site = min(
            range(len(env.sites)),
            key=lambda i: job.workload_flops / (env.sites[i].cpu_speed_gflops * 1e9),
        )
        return best_site
