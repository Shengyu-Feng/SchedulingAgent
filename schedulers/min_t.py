"""
MinT — find every site that already holds ALL of a job's input files (zero
remote transfer needed); if none do, fall back to every site. Among that
candidate set, break ties using the same rule as MCT: minimum estimated
completion time, queue_backlog(site) + incoming_execution_time(site), both
read live off the simulator; if several candidates tie at the same minimum,
pick uniformly at random among them rather than deterministically.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import random

from env.interfaces import SchedulerInterface

if TYPE_CHECKING:
    from env.simulator import HEPSimulator
    from env.core import Job


class MinTScheduler(SchedulerInterface):
    def schedule(self, job: "Job", env: "HEPSimulator") -> Optional[int]:
        candidates = [
            i for i, site in enumerate(env.sites)
            if all(site.data_node.has_file(fid) for fid in job.input_files)
        ]
        if not candidates:
            candidates = list(range(len(env.sites)))
        
        def completion_time(i: int) -> float:
            site = env.sites[i]
            if site.cpu_speed_gflops <= 0 or site.num_cpus <= 0:
                return float("inf")
            queue_rate = site.cpu_speed_gflops * site.num_cpus * 1e9
            queue_backlog = sum(
                j.workload_flops for j in env.jobs_at_site(site.site_index)
            ) / queue_rate
            incoming_execution_time = job.workload_flops / (site.cpu_speed_gflops * 1e9)
            return queue_backlog + incoming_execution_time

        times = [completion_time(i) for i in candidates]
        best = min(times)
        ties = [i for i, t in zip(candidates, times) if t <= best + 1e-9]
        return random.choice(ties)


