"""
DIANA — Data Intensive and Network Aware scheduler.

Paper: McClatchey et al., "Scheduling in Data Intensive and Network Aware
(DIANA) Grid Environments", arXiv:0707.0862.

Algorithm (as summarised in HEP grid-scheduling surveys)
---------------------------------------------------------
For each incoming job:

    1. Classify the job as **compute-intensive** or **data-intensive** by
       comparing the best-case completion time against the best-case remote
       data-transfer time across all sites.

           if  min_i completion_time(job, i)  >=  ratio * min_i transfer_time(job, i):
               → compute-intensive
           else:
               → data-intensive

    2. Route accordingly:
           compute-intensive → assign to the site with the shortest
                                completion time  (analogous to MCT)
           data-intensive   → assign to the site with the shortest
                                remote data-transfer time  (analogous to MinT)

The `ratio` knob is the surrogate for the paper's cost-matrix weighting:
values > 1 make the scheduler more eager to treat jobs as compute-intensive.

completion_time here is the same rule MinT and MCT both use — this used to be
bare single-core execution_time (job.workload_flops / cpu_speed_gflops), now
replaced everywhere by:

    completion_time(site) = queue_backlog(site) + incoming_execution_time(site)

queue_backlog is the total workload of every job currently at that site
(WAITING or RUNNING), read live off the simulator, divided by the site's
aggregate throughput (cpu_speed_gflops * num_cpus). incoming_execution_time
is this job's own workload divided by cpu_speed_gflops alone (no num_cpus) —
a single job only ever runs on one core.

Tie-breakers keep the policy well-defined when many sites score identically.
The paper itself does not specify one, so this uses the simplest available
signal — site.queue_length(), a raw queued-job COUNT blind to job size —
rather than a flops-weighted queue-wait estimate, which would be an addition
beyond the original method.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from env.interfaces import SchedulerInterface
import numpy as np
if TYPE_CHECKING:
    from env.core import Job
    from env.simulator import HEPSimulator


def _remote_transfer_time(job: "Job", site, env: "HEPSimulator") -> float:
    """Max WAN/tape time over this job's missing input files. Bandwidth is
    the ACTUAL rate a new transfer would get right now — link capacity
    divided by the transfers already sharing it, plus the one being added
    here — not the idle/peak rate."""
    dn = site.data_node
    remote_time = 0.0
    for fid in job.input_files:
        if dn.has_file(fid):
            continue
        f = env.all_files.get(fid)
        if f is None:
            continue
        srcs = [i for i, s in enumerate(env.sites) if s.data_node.has_file(fid)]
        if not srcs:
            remote_time = max(remote_time, f.size_bytes / dn.tape_bandwidth_bytes_s)
            continue
        best_src = env.network.best_source(site.site_index, srcs)
        share_bytes_s = env.link_share_bytes_s(best_src, site.site_index)
        remote_time = max(
            remote_time,
            f.size_bytes / share_bytes_s if share_bytes_s > 0 else float("inf"),
        )
    return remote_time


def _completion_time(job: "Job", site, env: "HEPSimulator") -> float:
    """queue_backlog(site) + incoming_execution_time(site) — the same rule
    MinT and MCT both use."""
    if site.cpu_speed_gflops <= 0 or site.num_cpus <= 0:
        return float("inf")
    queue_rate = site.cpu_speed_gflops * site.num_cpus * 1e9
    queue_backlog = sum(
        j.workload_flops for j in env.jobs_at_site(site.site_index)
    ) / queue_rate
    incoming_execution_time = job.workload_flops / (site.cpu_speed_gflops * 1e9)
    return queue_backlog + incoming_execution_time


class DIANAScheduler(SchedulerInterface):
    def __init__(self, ratio: float = 10.0) -> None:
        """
        Parameters
        ----------
        ratio : float
            Classification threshold.  A job is classed as compute-intensive
            when ``min_completion >= ratio * min_transfer``.  ratio = 1.0
            gives a symmetric split; ratio < 1 biases toward MCT; ratio > 1
            biases toward MinT.
        """
        self.ratio = ratio

    def schedule(self, job: "Job", env: "HEPSimulator") -> Optional[int]:
        features = []
        for site in env.sites:
            features.append({
                "site_index": site.site_index,
                "completion_time": _completion_time(job, site, env),
                "remote_transfer_time": _remote_transfer_time(job, site, env),
                "queue_length": site.queue_length(),
            })

        min_completion = np.mean([f["completion_time"] for f in features])
        min_transfer = np.mean([f["remote_transfer_time"] for f in features])

        compute_intensive = min_completion >= self.ratio * min_transfer

        if compute_intensive:
            best = min(
                features,
                key=lambda f: (
                    f["completion_time"],
                    f["queue_length"],
                    f["remote_transfer_time"],
                ),
            )
        else:
            best = min(
                features,
                key=lambda f: (
                    f["remote_transfer_time"],
                    f["queue_length"],
                    f["completion_time"],
                ),
            )

        return best["site_index"]
