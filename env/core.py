"""Core data structures for the HEP scheduling environment."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional


class JobStatus(Enum):
    BLOCKED = "blocked"       # waiting for one or more parent jobs to complete
    PENDING = "pending"       # all parents done, submit_time reached — awaiting scheduler
    WAITING = "waiting"       # assigned to site, waiting for files and/or CPU
    RUNNING = "running"       # on CPU (includes staging + execution)
    DONE = "done"


class FileLifecycleState(Enum):
    HOT  = "hot"               # accessed within warm_threshold_s
    WARM = "warm"              # accessed within cold_threshold_s
    COLD = "cold"              # stale — not accessed for a long time
    EVICTION_CANDIDATE = "eviction_candidate"
    TAPE_ONLY = "tape_only"    # evicted from disk; lives only on tape


class EventType(Enum):
    JOB_ARRIVE = auto()
    TRANSFER_COMPLETE = auto()   # WAN/LAN transfer (src == -1 means tape restore)
    JOB_COMPLETE = auto()


@dataclass
class DataFile:
    file_id: str
    size_bytes: float

    is_tape_archived: bool = True    # safe to evict; restorable from tape
    is_raw: bool = False             # raw detector data — never evict
    recomputation_cost: float = 1.0  # relative cost to recompute (higher = protect more)
    creation_time: float = 0.0


@dataclass
class Task:
    """A workflow: a named set of jobs arranged in a dependency DAG."""
    task_id: str
    job_ids: List[str]    # all job IDs belonging to this task


@dataclass
class Job:
    job_id: str
    workload_flops: float
    input_files: List[str]
    output_files: List[str]
    output_file_size_bytes: float
    submit_time: float = 0.0
    num_cpus_required: int = 1

    # DAG membership — set at generation time, read-only during simulation
    task_id: Optional[str] = None
    parent_job_ids: List[str] = field(default_factory=list)

    # Runtime state — set during simulation
    status: JobStatus = field(default=JobStatus.PENDING, compare=False)
    # First moment the job became schedulable: submit_time reached AND every
    # parent DONE.  For a root job this equals submit_time; for a DAG child it
    # is later, and (arrival_time - submit_time) is pure parent-blocking time
    # with no data-placement component.  Set once, on the first JOB_ARRIVE.
    arrival_time: Optional[float] = field(default=None, compare=False)
    assigned_site: Optional[int] = field(default=None, compare=False)
    pending_transfers: int = field(default=0, compare=False)
    data_ready_time: Optional[float] = field(default=None, compare=False)
    start_time: Optional[float] = field(default=None, compare=False)
    completion_time: Optional[float] = field(default=None, compare=False)
    allocated_cpus: List[int] = field(default_factory=list, compare=False)


@dataclass(order=True)
class Event:
    time: float
    seq: int                              # tie-breaker for same-time events
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False, default=None)
