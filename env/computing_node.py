"""ComputingNode: CPU pool + job queue for one site."""
from __future__ import annotations

from typing import Dict, List, Optional
from .core import Job
from .data_node import DataNode


class ComputingNode:
    def __init__(
        self,
        node_id: str,
        site_index: int,
        num_cpus: int,
        cpu_speed_gflops: float,
        data_node: DataNode,
    ):
        self.node_id = node_id
        self.site_index = site_index
        self.num_cpus = num_cpus
        self.cpu_speed_gflops = cpu_speed_gflops  # GFLOPs/s per CPU core
        self.data_node = data_node

        # None  → core is free; str → job_id currently using the core
        self._cpu_busy: List[Optional[str]] = [None] * num_cpus

        # Jobs waiting for files and/or CPUs (unordered; searched on dispatch)
        self._queue: List[Job] = []
        # Running total of workload_flops over self._queue.  Maintained
        # incrementally by enqueue/pop_ready_job so _queue_wait_estimate() is
        # O(1) rather than an O(queue) sum on every call.
        self._queued_flops: float = 0.0

    # ------------------------------------------------------------------ #
    @property
    def free_cpu_ids(self) -> List[int]:
        return [i for i, owner in enumerate(self._cpu_busy) if owner is None]

    @property
    def num_free_cpus(self) -> int:
        return sum(1 for owner in self._cpu_busy if owner is None)

    def cpu_status(self) -> List[Optional[str]]:
        """Per-core snapshot, index = core id: None if that core is idle,
        else the job_id currently occupying it.  A copy — mutating it has no
        effect on the real core assignment.  For how long that job has been
        running, look it up (e.g. via HEPSimulator.all_jobs) and read
        current_time - job.start_time."""
        return list(self._cpu_busy)

    def allocate_cpus(self, job_id: str, count: int) -> List[int]:
        """Grab `count` free cores; return their IDs, or [] if not enough."""
        free = self.free_cpu_ids
        if len(free) < count:
            return []
        chosen = free[:count]
        for c in chosen:
            self._cpu_busy[c] = job_id
        return chosen

    def release_cpus(self, cpu_ids: List[int]) -> None:
        for c in cpu_ids:
            self._cpu_busy[c] = None

    # ------------------------------------------------------------------ #
    def enqueue(self, job: Job) -> None:
        self._queue.append(job)
        self._queued_flops += job.workload_flops

    def pop_ready_job(self, current_time: float) -> Optional[Job]:
        """
        Find and remove the job whose input data is ready and whose CPU
        requirement can be satisfied right now.  Among eligible jobs,
        prefer the one whose data became ready earliest (FIFO on data_ready_time).
        """
        eligible = [
            (i, j)
            for i, j in enumerate(self._queue)
            if j.pending_transfers == 0
            and j.data_ready_time is not None
            and j.data_ready_time <= current_time
            and self.num_free_cpus >= j.num_cpus_required
        ]
        if not eligible:
            return None
        # Earliest data_ready_time first
        eligible.sort(key=lambda x: x[1].data_ready_time)  # type: ignore[return-value]
        idx, job = eligible[0]
        self._queue.pop(idx)
        self._queued_flops -= job.workload_flops
        # Repeated += / -= of ~1e13-magnitude floats leaves residue, which can
        # make an empty queue report a tiny non-zero (even negative) backlog.
        # Snap to exact zero whenever the queue is genuinely empty.
        if not self._queue:
            self._queued_flops = 0.0
        return job

    def queue_length(self) -> int:
        return len(self._queue)

    def queued_jobs(self) -> List[Job]:
        """The actual jobs currently waiting here (not yet dispatched to a
        core) — unordered, same as internal dispatch search order, not a
        priority or wait-time order.  A copy of the list, but the Job
        objects inside are the real ones.  For how long a job has been
        waiting, read current_time - job.data_ready_time (once its inputs
        are ready) or current_time - job.arrival_time (before that)."""
        return list(self._queue)

    def queued_flops(self) -> float:
        """Total workload still owed by the jobs in this site's queue."""
        return self._queued_flops

    def _queue_wait_estimate(self) -> float:
        """
        Estimated seconds to drain the current queue:

            queued_flops / (num_cpus * cpu_speed_gflops * 1e9)

        A lower bound.  It assumes perfect core packing, whereas
        pop_ready_job needs num_cpus_required cores free simultaneously and
        each job runs at single-core speed, so multi-core jobs wait longer
        than this predicts.  It also counts jobs still blocked on their input
        transfers, which cannot occupy cores yet — that overstates near-term
        load, and is the conservative direction.

        Private: this is a modeling estimate, not raw state.  Reserved for the
        system-state diagnostics. Agent-designed schedulers
        must derive any queue-wait signal themselves from queue_length(),
        queued_flops(), or the Job objects returned by
        HEPSimulator.jobs_at_site().
        """
        rate = self.num_cpus * self.cpu_speed_gflops * 1e9
        if rate <= 0:
            return float("inf")
        return max(0.0, self._queued_flops / rate)

    def queued_file_needs(self) -> Dict[str, int]:
        """Count how many queued jobs need each file (for eviction scoring)."""
        counts: Dict[str, int] = {}
        for job in self._queue:
            for fid in job.input_files:
                counts[fid] = counts.get(fid, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    def get_state(self) -> dict:
        return {
            "node_id": self.node_id,
            "site_index": self.site_index,
            "num_cpus": self.num_cpus,
            "cpu_speed_gflops": self.cpu_speed_gflops,
            "num_free_cpus": self.num_free_cpus,
            "queue_length": self.queue_length(),
            "queue_wait_estimate": self._queue_wait_estimate(),
            "data_node": self.data_node.get_state(),
        }
