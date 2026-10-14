"""
MCT — Minimum Completion Time, ranked purely by estimated completion time.

Ignores data location and transfer cost entirely (no locality filter, no
remote_transfer_time term) — every site is ranked only by:

    completion_time(site) = queue_backlog(site) + incoming_execution_time(site)

queue_backlog is the total workload of every job currently at that site
(WAITING or RUNNING), read live off the simulator, divided by the site's
aggregate throughput (cpu_speed_gflops * num_cpus) — a parallelism
assumption over the whole backlog. incoming_execution_time is this job's own
workload divided by cpu_speed_gflops alone (no num_cpus) — a single job only
ever runs on one core, matching the simulator's real execution model, so
normalizing it by num_cpus too would understate a job's real cost at
high-core sites.

Ties (e.g. several completely idle sites, all at completion_time == 0) are
broken uniformly at random rather than deterministically — nothing about
completion time distinguishes them, so there's no principled reason to
always prefer the lowest site_index.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import random

from env.interfaces import SchedulerInterface

if TYPE_CHECKING:
    from env.simulator import HEPSimulator
    from env.core import Job


class MCTScheduler(SchedulerInterface):
    def schedule(self, job: "Job", env: "HEPSimulator") -> Optional[int]:
        def completion_time(site) -> float:
            if site.cpu_speed_gflops <= 0 or site.num_cpus <= 0:
                return float("inf")
            queue_rate = site.cpu_speed_gflops * site.num_cpus * 1e9
            queue_backlog = sum(
                j.workload_flops for j in env.jobs_at_site(site.site_index)
            ) / queue_rate
            incoming_execution_time = job.workload_flops / (site.cpu_speed_gflops * 1e9)
            return queue_backlog + incoming_execution_time

        times = [completion_time(s) for s in env.sites]
        best = min(times)
        ties = [i for i, t in enumerate(times) if t <= best + 1e-9]
        return random.choice(ties)
