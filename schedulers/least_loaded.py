"""
Least Loaded (LL) — assign each job to the resource with the smallest
tracked load, then bump that resource's load by the job's own estimated
execution time.

    Maintain load L_j for every resource R_j (persistent scheduler state,
    not read from the live environment).

    Upon arrival of job T_i:
        best_resource <- argmin_j L_j
        Assign T_i to best_resource
        L_best_resource += EstimatedExecutionTime(T_i, best_resource)

L_j is tracked here as a TIME (seconds), so it is comparable across sites
with different core counts/speeds:

    EstimatedExecutionTime(T_i, R_j) = T_i.workload_flops
                                        / (R_j.cpu_speed_gflops * R_j.num_cpus * 1e9)

which is exactly workload divided by (processing power * #cpu cores) as
specified. Summing these increments is equivalent to computing, at any
point, L_j = (total workload ever assigned to R_j) / (R_j.cpu_speed_gflops
* R_j.num_cpus) — same thing, tracked incrementally instead of recomputed.

Faithful to the classical algorithm as given: L_j is monotonically
increasing and is NEVER decremented when a job completes or discounted by
what env.jobs_at_site actually shows right now — it purely reflects everything
this scheduler has ever handed to R_j, not R_j's real present-tense backlog.
That is a deliberate, known limitation of this heuristic (not a bug to fix
here): unlike MCT/MinT's queue_wait, which is read live off the simulator and
naturally shrinks as jobs finish, LL's own bookkeeping has no way to learn
that a job it assigned long ago has already completed.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional

from env.interfaces import SchedulerInterface

if TYPE_CHECKING:
    from env.simulator import HEPSimulator
    from env.core import Job


class LeastLoadedScheduler(SchedulerInterface):
    def __init__(self) -> None:
        self._load: Dict[int, float] = {}

    def schedule(self, job: "Job", env: "HEPSimulator") -> Optional[int]:
        best_resource = min(
            range(len(env.sites)),
            key=lambda j: self._load.get(j, 0.0),
        )

        site = env.sites[best_resource]
        rate = site.cpu_speed_gflops * site.num_cpus * 1e9
        estimated_execution_time = job.workload_flops / rate if rate > 0 else float("inf")
        self._load[best_resource] = self._load.get(best_resource, 0.0) + estimated_execution_time

        return best_resource
