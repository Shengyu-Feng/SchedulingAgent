"""
HEPSimulator: event-driven simulation of an ATLAS-style distributed computing system.

Event pipeline per job
----------------------
1. JOB_ARRIVE     – global scheduler assigns job to a site
2. (optional) remote TRANSFER_COMPLETE for each missing input file
             OR tape restore (src == -1) if no site has the file
3. CPU allocated + staging starts once all files are local
4. JOB_COMPLETE   – CPUs released, output files written to local disk

Disk management
---------------
The disk is a finite cache.  When a write would exceed capacity the simulator
automatically calls EvictionPolicy.select_candidates() to free space.
Files that are tape-archived can be restored via a tape-restore event if they
have been evicted from all sites.
"""
from __future__ import annotations

import heapq
import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

from .computing_node import ComputingNode
from .core import DataFile, Event, EventType, Job, JobStatus, Task
from .data_node import DataNode
from .eviction import EvictionPolicy, EvictionWeights
from .generator import expand_generated_config
from .network import TransferNetwork

logger = logging.getLogger(__name__)


def _fire_hook(obj, name: str, *args) -> None:
    """
    Call an optional policy hook if the object provides one.

    Agent-written policies are only checked for "class exists and imports"
    (see agent/tools.py::_write_scheduler), so they may not subclass the
    interface ABC and therefore may lack the default no-op implementations.
    Missing hooks are skipped; a hook that is present is always called.
    """
    if obj is None:
        return
    fn = getattr(obj, name, None)
    if callable(fn):
        fn(*args)


def _is_blocked_name(name: str) -> bool:
    """True for a single-underscore-prefixed name (simulator internals, e.g.
    _events, _queue, _active_transfers) but not for a dunder (__class__,
    __repr__, ...), which is a Python protocol name, not internal state."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _job_has_arrived(sim: "HEPSimulator", job: Job) -> bool:
    return job.status != JobStatus.BLOCKED and job.submit_time <= sim.current_time


class _ArrivedJobsView:
    """Read-only, arrival-filtered view of HEPSimulator.all_jobs, handed to a
    scheduler in place of the real dict.

    A BLOCKED job, or one whose submit_time is still in the future, is
    invisible here — indexing or iterating past it raises KeyError exactly as
    if the job did not exist yet.  Its input_files are future demand a real
    scheduler could not observe; AGENT_INSTRUCTIONS.md documents this as a
    hard rule, and this view is what makes it impossible to violate by
    accident rather than something merely asked of the model in a prompt.

    A parent job referenced via job.parent_job_ids is always visible: it must
    be DONE (not BLOCKED, and long past its own submit_time) for the child to
    have arrived at all, so this filter never breaks that legitimate lookup.

    __getattribute__ is overridden (not __getattr__) because ``_sim`` is a
    real stored attribute — __getattr__ only fires when normal lookup FAILS,
    so it would never even be consulted for a name that genuinely exists,
    letting external code read ``view._sim`` straight through.  Every method
    below reads its own state via ``object.__getattribute__`` specifically to
    bypass this same override — using ``self._sim`` inside them would recurse
    into the block-list check and reject the object's own internal read.
    """

    def __init__(self, sim: "HEPSimulator") -> None:
        object.__setattr__(self, "_sim", sim)

    def __getattribute__(self, name):
        if _is_blocked_name(name):
            raise AttributeError(
                f"'{name}' is a simulator internal, not observable state — "
                "reading underscore-prefixed attributes is off limits."
            )
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        raise AttributeError("all_jobs is read-only — the scheduler decides placement, nothing else.")

    def __setitem__(self, key, value):
        raise TypeError("all_jobs is read-only — the scheduler decides placement, nothing else.")

    def __getitem__(self, job_id: str) -> Job:
        sim = object.__getattribute__(self, "_sim")
        job = sim.all_jobs[job_id]
        if not _job_has_arrived(sim, job):
            raise KeyError(
                f"job '{job_id}' has not arrived yet (status={job.status.name}) — "
                "a scheduler may not read a BLOCKED or not-yet-submitted job"
            )
        return job

    def get(self, job_id: str, default=None):
        try:
            return self[job_id]
        except KeyError:
            return default

    def __contains__(self, job_id: str) -> bool:
        sim = object.__getattribute__(self, "_sim")
        job = sim.all_jobs.get(job_id)
        return job is not None and _job_has_arrived(sim, job)

    def __iter__(self):
        sim = object.__getattribute__(self, "_sim")
        return (jid for jid, j in sim.all_jobs.items() if _job_has_arrived(sim, j))

    def __len__(self) -> int:
        sim = object.__getattribute__(self, "_sim")
        return sum(1 for j in sim.all_jobs.values() if _job_has_arrived(sim, j))

    def keys(self):
        return iter(self)

    def values(self):
        sim = object.__getattribute__(self, "_sim")
        return (j for j in sim.all_jobs.values() if _job_has_arrived(sim, j))

    def items(self):
        sim = object.__getattribute__(self, "_sim")
        return ((jid, j) for jid, j in sim.all_jobs.items() if _job_has_arrived(sim, j))


class SchedulerEnvView:
    """Read-only, restricted view of HEPSimulator handed to
    SchedulerInterface.schedule() / on_job_assigned() / on_job_complete() in
    place of the real simulator.

    Enforces, at this top level, the two hard rules AGENT_INSTRUCTIONS.md
    documents:
      - all_jobs is arrival-filtered (see _ArrivedJobsView) — a BLOCKED or
        not-yet-submitted job cannot be read through this view at all.
      - read-only, no internals: setting any attribute raises, and any
        single-underscore-prefixed name (env._events, env._inflight, ...) is
        unreachable — AttributeError, not a value.

    Deliberately shallow: this guards the top-level env object and the
    all_jobs mapping only.  Reaching further — env.sites[i].data_node.evict(),
    or mutating a Job/DataFile field once you hold a reference to one — is
    not intercepted.  A full deep sandbox (proxying every nested object) was
    considered and rejected: schedule() runs once per job, thousands of times
    on the larger configs, against a wall-clock budget for the whole run, and
    the two concrete leaks the rules name — future-job visibility and
    reaching simulator internals from the object you were actually handed —
    are what this stops.

    __getattribute__ (not __getattr__) for the same reason as
    _ArrivedJobsView above: ``_sim``/``_jobs_view`` are real stored
    attributes, so __getattr__ would never be consulted for them and
    ``env._sim`` would leak the real simulator straight through.
    """

    def __init__(self, sim: "HEPSimulator") -> None:
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_jobs_view", _ArrivedJobsView(sim))

    def __setattr__(self, name, value):
        raise AttributeError(
            f"scheduler env is read-only — cannot set '{name}'. "
            "The scheduler decides placement, nothing else."
        )

    def __getattribute__(self, name):
        if _is_blocked_name(name):
            raise AttributeError(
                f"'{name}' is a simulator internal, not state a real scheduler "
                "could observe — reading underscore-prefixed attributes is off limits."
            )
        if name == "all_jobs":
            return object.__getattribute__(self, "_jobs_view")
        sim = object.__getattribute__(self, "_sim")
        return getattr(sim, name)


class HEPSimulator:
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        sites: List[ComputingNode],
        network: TransferNetwork,
        all_files: Dict[str, DataFile],
        scheduler,                          # SchedulerInterface
        eviction_policy: Optional[EvictionPolicy] = None,
    ):
        self.sites = sites
        self.network = network
        self.all_files: Dict[str, DataFile] = dict(all_files)
        self.scheduler = scheduler
        self.eviction_policy = eviction_policy or EvictionPolicy()

        self._events: list = []   # min-heap of Event
        self._seq: int = 0
        self.current_time: float = 0.0

        # (file_id, dst_site_idx) → list[Job] waiting on this transfer
        self._transfer_waiters: Dict[Tuple[str, int], List[Job]] = defaultdict(list)
        # set of (file_id, dst_site_idx) currently in-flight (WAN or tape restore)
        self._active_transfers: Set[Tuple[str, int]] = set()

        # ---- fair-share transfer engine ---------------------------------
        # Each link (src, dst) splits its bandwidth equally among the
        # transfers in flight on it.  src == -1 means a tape restore, which
        # shares the destination's tape bandwidth.
        # (file_id, dst) → {file_id, src, dst, remaining, last_update, rate}
        self._inflight: Dict[Tuple[str, int], dict] = {}
        # (src, dst) → set of transfer keys currently on that link
        self._link_transfers: Dict[Tuple[int, int], Set[Tuple[str, int]]] = defaultdict(set)
        # (src, dst) → epoch, bumped on every re-share so that TRANSFER_COMPLETE
        # events issued under an older share are recognised as superseded
        self._link_epoch: Dict[Tuple[int, int], int] = defaultdict(int)

        self.all_jobs: Dict[str, Job] = {}
        self.all_tasks: Dict[str, Task] = {}
        # DAG reverse edges: parent_job_id → [child_job_id, ...]
        self._children: Dict[str, List[str]] = defaultdict(list)

        # How long a job is deferred when scheduler.schedule() returns None.
        # Every deferral costs one JOB_ARRIVE event, so a scheduler that holds
        # jobs for W seconds pays W / defer_retry_s events per job.  It is also
        # the delivery lag for a batching scheduler: a decision made while the
        # job is deferred only takes effect on the job's next retry.
        self.defer_retry_s: float = 1e-6

        # ---- per-site telemetry counters (reset-safe, purely additive) ----
        n = len(sites)
        self._site_compute_seconds: List[float] = [0.0] * n   # CPU-core·seconds used
        self._site_tape_restore_counts: List[int] = [0] * n
        # Job outputs that did not fit on the local disk and went straight to
        # tape.  A non-zero count means this site's disk is the binding
        # constraint, not its CPUs.
        self._site_tape_spill_counts: List[int] = [0] * n
        # Network cost actually paid, per destination site.  Every transfer the
        # simulator starts is counted here — all of it demand-driven staging, so
        # the network price of the scheduler's placements is visible instead of
        # implicit.
        self._site_transfers_in: List[int] = [0] * n
        self._site_bytes_in: List[float] = [0.0] * n
        # Eviction counts live on each DataNode (data_node.eviction_count),
        # covering every eviction the simulator triggers when a write runs out
        # of space.

        # Built once, reused for every schedule()/on_job_assigned/
        # on_job_complete call — see SchedulerEnvView for what it restricts.
        self._scheduler_view = SchedulerEnvView(self)

    # ================================================================== #
    # Factory: build from YAML config
    # ================================================================== #
    @classmethod
    def from_yaml(
        cls,
        config_path: str,
        scheduler,
        eviction_policy: Optional[EvictionPolicy] = None,
    ) -> "HEPSimulator":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cls.from_config(cfg, scheduler, eviction_policy)

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        scheduler,
        eviction_policy: Optional[EvictionPolicy] = None,
    ) -> "HEPSimulator":
        # Expand any *_generation blocks into concrete lists
        cfg = expand_generated_config(cfg)

        # ---- eviction policy -------------------------------------------
        storage_cfg = cfg.get("storage", {})
        tape_bw_bytes = _bw_unit(storage_cfg.get("tape_bandwidth_unit", "MB/s")) * float(
            storage_cfg.get("tape_bandwidth", 500)
        )
        eviction_cfg = storage_cfg.get("eviction", {})
        weights = EvictionWeights(
            age_s=float(eviction_cfg.get("age_s", 3600.0)),
            extra_replicas=float(eviction_cfg.get("extra_replicas", 0.5)),
            downstream=float(eviction_cfg.get("downstream", 3.0)),
            recompute=float(eviction_cfg.get("recompute", 1.5)),
            no_tape_penalty=float(eviction_cfg.get("no_tape_penalty", 1000.0)),
            warm_threshold_s=float(eviction_cfg.get("warm_threshold_s", 3600.0)),
            cold_threshold_s=float(eviction_cfg.get("cold_threshold_s", 86400.0)),
        )
        # A caller-supplied policy wins; it is used exactly as constructed, so
        # a custom policy that wants the YAML thresholds should read them from
        # `EvictionWeights` itself (or be handed them by the caller).
        if eviction_policy is None:
            eviction_policy = EvictionPolicy(weights)


        # ---- data files ------------------------------------------------
        all_files: Dict[str, DataFile] = {}
        for fd in cfg.get("data_files", []):
            fid = fd["id"]
            size = _parse_bytes(fd["size"])
            all_files[fid] = DataFile(
                file_id=fid,
                size_bytes=size,
                is_tape_archived=bool(fd.get("is_tape_archived", True)),
                is_raw=bool(fd.get("is_raw", False)),
                recomputation_cost=float(fd.get("recomputation_cost", 1.0)),
            )

        # ---- sites (computing node + data node) -------------------------
        site_cfgs = cfg["sites"]
        site_id_to_idx: Dict[str, int] = {s["id"]: i for i, s in enumerate(site_cfgs)}
        sites: List[ComputingNode] = []

        for idx, sc in enumerate(site_cfgs):
            dn = DataNode(
                node_id=f"{sc['id']}_storage",
                disk_capacity_bytes=_parse_bytes(sc["disk_capacity"]),
                tape_bandwidth_bytes_s=tape_bw_bytes,
            )
            cn = ComputingNode(
                node_id=sc["id"],
                site_index=idx,
                num_cpus=int(sc["num_cpus"]),
                cpu_speed_gflops=float(sc["cpu_speed_gflops"]),
                data_node=dn,
            )
            sites.append(cn)

        # Place initial file replicas
        for fd in cfg.get("data_files", []):
            fid = fd["id"]
            for loc in fd.get("initial_locations", []):
                if loc not in site_id_to_idx:
                    raise ValueError(f"Unknown site '{loc}' in initial_locations of {fid}")
                site = sites[site_id_to_idx[loc]]
                ok = site.data_node.add_file(all_files[fid], current_time=0.0)
                if not ok:
                    logger.warning(
                        "Disk full at site '%s' — skipping initial placement of '%s' "
                        "(%.1f GB file, %.1f GB free). Increase disk capacity or reduce file sizes.",
                        loc, fid,
                        all_files[fid].size_bytes / 1e9,
                        site.data_node.free_bytes / 1e9,
                    )

        # ---- network ----------------------------------------------------
        net_cfg = cfg["network"]
        n = len(sites)
        wan_raw = net_cfg.get("wan_bandwidth_matrix")
        if wan_raw is not None:
            wan = np.array(wan_raw, dtype=float) * _bw_unit(net_cfg.get("bandwidth_unit", "MB/s"))
            lan_mult = float(net_cfg.get("lan_multiplier", 10.0))
            lan_raw = net_cfg.get("lan_bandwidth")
            lan_arr = (
                np.array(lan_raw, dtype=float) * _bw_unit(net_cfg.get("bandwidth_unit", "MB/s"))
                if lan_raw is not None
                else None
            )
            network = TransferNetwork.from_wan_matrix(wan, lan_arr, lan_mult)
        else:
            bw_raw = net_cfg["bandwidth_matrix"]
            bw = np.array(bw_raw, dtype=float) * _bw_unit(net_cfg.get("bandwidth_unit", "MB/s"))
            network = TransferNetwork(bw)

        # ---- build simulator -------------------------------------------
        sim = cls(sites, network, all_files, scheduler, eviction_policy)

        # Register tasks (optional — not all configs have them)
        for td in cfg.get("tasks", []):
            task = Task(task_id=td["id"], job_ids=list(td["job_ids"]))
            sim.all_tasks[task.task_id] = task

        for jd in cfg.get("jobs", []):
            output_size = _parse_bytes(jd.get("output_file_size", "0B"))
            job = Job(
                job_id=jd["id"],
                workload_flops=_parse_flops(jd["workload"]),
                input_files=list(jd.get("input_files", [])),
                output_files=list(jd.get("output_files", [])),
                output_file_size_bytes=output_size,
                submit_time=float(jd.get("submit_time", 0.0)),
                num_cpus_required=int(jd.get("num_cpus_required", 1)),
                task_id=jd.get("task_id"),
                parent_job_ids=list(jd.get("parent_job_ids", [])),
            )
            sim.submit_job(job)

        return sim

    # ================================================================== #
    # Public API
    # ================================================================== #
    def submit_job(self, job: Job) -> None:
        """
        Register a job.  Root jobs (no parents) are queued immediately.
        Jobs with unfinished parents start BLOCKED and are unblocked
        automatically when their last parent completes.
        """
        self.all_jobs[job.job_id] = job
        for pid in job.parent_job_ids:
            self._children[pid].append(job.job_id)

        parents_done = all(
            self.all_jobs.get(pid, Job("_dummy", 0, [], [], 0)).status == JobStatus.DONE
            for pid in job.parent_job_ids
        )
        if not job.parent_job_ids or parents_done:
            job.status = JobStatus.PENDING
            t = max(job.submit_time, self.current_time)
            self._push(t, EventType.JOB_ARRIVE, job)
        else:
            job.status = JobStatus.BLOCKED

    # ------------------------------------------------------------------ #
    # Link occupancy — the observable side of network contention
    # ------------------------------------------------------------------ #
    def link_load(self, src: int, dst: int) -> int:
        """
        Number of transfers currently sharing the (src, dst) link.  Use src=-1
        for the tape-restore channel into dst.

        Bandwidth is split equally among them, so a link carrying k transfers
        delivers capacity/k to each.  This is the only way to see contention:
        env.network.transfer_time() is computed from raw link bandwidth and
        therefore assumes an idle network.
        """
        return len(self._link_transfers.get((src, dst), ()))

    def link_share_bytes_s(self, src: int, dst: int) -> float:
        """
        Bytes/s a transfer started right now on (src, dst) would receive —
        capacity / (link_load + 1).  Multiply size_bytes / this for a
        contention-aware transfer-time estimate.
        """
        return self._link_capacity(src, dst) / (self.link_load(src, dst) + 1)

    def site_inbound_load(self, site_idx: int) -> int:
        """Total transfers in flight into site_idx, across all its links —
        the count; site_inbound_transfers(site_idx) is the actual list."""
        return sum(1 for (_fid, dst) in self._active_transfers if dst == site_idx)

    def _transfer_snapshot(self, key: Tuple[str, int]) -> Optional[dict]:
        """Live view of one in-flight transfer.  remaining_bytes is computed
        as of right now, not the stale value stored as of the link's last
        re-share (see _reschedule_link) — read-only, does not mutate the
        real transfer state."""
        t = self._inflight.get(key)
        if t is None:
            return None
        remaining_now = max(0.0, t["remaining"] - t["rate"] * (self.current_time - t["last_update"]))
        f = self.all_files.get(t["file_id"])
        return {
            "file_id": t["file_id"],
            "src": t["src"],
            "dst": t["dst"],
            "size_bytes": f.size_bytes if f is not None else None,
            "remaining_bytes": remaining_now,
            "rate_bytes_s": t["rate"],
        }

    def link_transfers(self, src: int, dst: int) -> List[dict]:
        """Live list of every transfer currently on the (src, dst) link
        (src=-1 for the tape-restore channel into dst) — the actual
        transfers link_load(src, dst) only counts.  Each entry: file_id,
        src, dst, size_bytes, remaining_bytes, rate_bytes_s."""
        keys = self._link_transfers.get((src, dst), ())
        out = [self._transfer_snapshot(k) for k in keys]
        return [t for t in out if t is not None]

    def site_inbound_transfers(self, site_idx: int) -> List[dict]:
        """Live list of every transfer currently in flight into site_idx,
        across all its links — the actual transfers site_inbound_load
        (site_idx) only counts.  Same shape as link_transfers()."""
        keys = [key for key in self._active_transfers if key[1] == site_idx]
        out = [self._transfer_snapshot(k) for k in keys]
        return [t for t in out if t is not None]

    # ================================================================== #
    # Transfer engine — fair-share bandwidth per link
    # ================================================================== #
    def _link_capacity(self, src: int, dst: int) -> float:
        """Bytes/s available on a link.  src == -1 is a tape restore, which
        draws on the destination site's tape bandwidth."""
        if src == -1:
            return float(self.sites[dst].data_node.tape_bandwidth_bytes_s)
        return float(self.network.bandwidth(src, dst))

    def _start_transfer(self, file_id: str, src: int, dst: int, size_bytes: float) -> None:
        """Register a new transfer and re-share its link's bandwidth."""
        key = (file_id, dst)
        self._inflight[key] = {
            "file_id": file_id,
            "src": src,
            "dst": dst,
            "remaining": float(size_bytes),
            "last_update": self.current_time,
            "rate": 0.0,
        }
        self._site_transfers_in[dst] += 1
        self._site_bytes_in[dst] += float(size_bytes)
        self._link_transfers[(src, dst)].add(key)
        self._active_transfers.add(key)
        self._reschedule_link((src, dst))

    def _finish_transfer(self, key: Tuple[str, int]) -> None:
        """Retire a completed transfer, returning its share to the link."""
        t = self._inflight.pop(key, None)
        self._active_transfers.discard(key)
        if t is None:
            return
        link = (t["src"], t["dst"])
        self._link_transfers[link].discard(key)
        self._reschedule_link(link)

    def _reschedule_link(self, link: Tuple[int, int]) -> None:
        """
        Credit the bytes moved since this link last changed, then split its
        capacity equally among the remaining transfers.

        Only the earliest finisher gets a TRANSFER_COMPLETE event: with an
        equal share that is simply the transfer with the least bytes left, and
        every completion re-enters here to schedule the next one.  This keeps
        the event count proportional to the number of link changes rather than
        to (changes × transfers on the link).  The link epoch invalidates any
        event issued under a previous share.
        """
        keys = self._link_transfers.get(link)
        self._link_epoch[link] += 1
        if not keys:
            return

        now = self.current_time
        for k in keys:
            t = self._inflight[k]
            t["remaining"] = max(0.0, t["remaining"] - t["rate"] * (now - t["last_update"]))
            t["last_update"] = now

        share = self._link_capacity(*link) / len(keys)
        for k in keys:
            self._inflight[k]["rate"] = share

        nxt = min(keys, key=lambda k: self._inflight[k]["remaining"])
        t = self._inflight[nxt]
        eta = now + (t["remaining"] / share if share > 0 else float("inf"))
        self._push(eta, EventType.TRANSFER_COMPLETE, {
            "file_id": t["file_id"],
            "src": t["src"],
            "dst": t["dst"],
            "epoch": self._link_epoch[link],
        })

    def run(self, until: float = float("inf")) -> None:
        """Run the simulation up to `until` seconds."""
        while self._events:
            ev: Event = heapq.heappop(self._events)
            if ev.time > until:
                heapq.heappush(self._events, ev)
                break
            self.current_time = ev.time
            self._dispatch(ev)

    def step(self) -> Optional[Event]:
        """Process the next single event; return it (or None if empty)."""
        if not self._events:
            return None
        ev: Event = heapq.heappop(self._events)
        self.current_time = ev.time
        self._dispatch(ev)
        return ev

    # ================================================================== #
    # Observation for schedulers / RL agents
    # ================================================================== #
    def jobs_at_site(self, site_idx: int) -> List[Job]:
        """Jobs currently WAITING or RUNNING at site `site_idx` — the raw
        contents of that site's queue and CPUs, for schedulers that want to
        compute their own load/queue-wait signal instead of relying on a
        pre-baked estimate."""
        return [
            j for j in self.all_jobs.values()
            if j.assigned_site == site_idx
            and j.status in (JobStatus.WAITING, JobStatus.RUNNING)
        ]

    def get_system_state(self) -> dict:
        """Snapshot of the full system state."""
        return {
            "current_time": self.current_time,
            "sites": [s.get_state() for s in self.sites],
            "network": self.network.get_state(),
            "num_blocked_jobs": sum(1 for j in self.all_jobs.values() if j.status == JobStatus.BLOCKED),
            "num_pending_jobs": sum(1 for j in self.all_jobs.values() if j.status == JobStatus.PENDING),
            "num_waiting_jobs": sum(1 for j in self.all_jobs.values() if j.status == JobStatus.WAITING),
            "num_running_jobs": sum(1 for j in self.all_jobs.values() if j.status == JobStatus.RUNNING),
            "num_done_jobs": sum(1 for j in self.all_jobs.values() if j.status == JobStatus.DONE),
        }

    def throughput(self) -> float:
        """
        Effective throughput: total workload completed divided by the SUM of
        per-job spans (submission → completion), in GFLOPs/s.

        This is a work-weighted inverse-latency, not a wall-clock system rate.
        Dividing by the sum of per-job spans is deliberate: each job is measured
        from its own submit_time, so the metric is insensitive to the (config-
        fixed) job arrival window and stays discriminative even when every
        policy completes every job.  A wall-clock rate (work / makespan) would
        collapse to a monotone transform of makespan in that case, since the
        numerator is then identical across policies.

        Higher is better; parallelism raises it (serialising work inflates the
        queue wait carried in each job's span, and so lowers it).

        Caveat: like mean_latency_s, this sums over COMPLETED jobs only, so a
        truncated run (see `until`) censors the slow tail and overstates the
        result.  Read it alongside jobs_done, or use throughput_charged().
        """
        done = [j for j in self.all_jobs.values() if j.status == JobStatus.DONE]
        if not done:
            return 0.0
        total_work = sum(j.workload_flops for j in done) / 1e9
        total_span = sum(j.completion_time - j.submit_time for j in done)  # type: ignore[operator]
        return total_work / total_span if total_span > 0 else 0.0

    def throughput_charged(self) -> float:
        """
        Censoring-proof throughput, in GFLOPs/s: work COMPLETED divided by the
        charged span of every job SUBMITTED by the horizon.

            numerator   = GFLOPs of jobs that finished
            denominator = sum over all submitted jobs of
                            (completion - submit)          if finished
                            (horizon    - submit)          if not

        This is to throughput() what mean_latency_charged_s is to
        mean_latency_s.  An unfinished job adds to the denominator and nothing
        to the numerator, so stranding work can only lower the result — whereas
        throughput() drops such a job from both sums and can be *raised* by
        abandoning any job whose work-per-second-of-span is below average.

        Jobs submitted after the horizon are excluded: they never had a chance
        to run, and charging them would penalise a short truncated run.
        """
        horizon = self.current_time
        submitted = [j for j in self.all_jobs.values() if j.submit_time <= horizon]
        if not submitted:
            return 0.0
        total_work = sum(
            j.workload_flops for j in submitted if j.completion_time is not None
        ) / 1e9
        total_span = sum(
            (j.completion_time - j.submit_time)  # type: ignore[operator]
            if j.completion_time is not None
            else max(0.0, horizon - j.submit_time)
            for j in submitted
        )
        return total_work / total_span if total_span > 0 else 0.0

    # ================================================================== #
    # Metrics API (read-only; safe for agent analysis)
    # ================================================================== #

    def makespan(self) -> float:
        """Elapsed time from the first job submission to the last completion (seconds)."""
        done = [j for j in self.all_jobs.values() if j.completion_time is not None]
        if not done:
            return self.current_time
        t_start = min(j.submit_time for j in self.all_jobs.values())
        t_end = max(j.completion_time for j in done)  # type: ignore[type-var]
        return t_end - t_start

    def job_stats(self) -> List[dict]:
        """
        Per-job breakdown (one dict per job, sorted by submit_time).

        Keys
        ----
        job_id, task_id, parent_job_ids, status, submit_time
        assigned_site, assigned_site_id
        arrival_time     – when the job became schedulable (parents DONE and
                           submit_time reached); None if it never arrived
        latency          – completion_time − submit_time          (done only)
        parent_wait      – arrival_time − submit_time             (DAG blocking)
        file_wait        – data_ready_time − arrival_time         (input transfers)
        data_wait        – data_ready_time − submit_time  ==  parent_wait + file_wait
        cpu_wait         – start_time − data_ready_time           (queued for CPU)
        execution_time   – completion_time − start_time           (staging + compute)
        workload_tflops
        is_root          – bool, parent_job_ids is empty
        dag_depth        – 0 for a root, else 1 + max(depth of parents)

        ATTRIBUTION WARNING — do not read `data_wait` as a data-placement cost.
        For a DAG child it is dominated by `parent_wait`, i.e. by how long its
        parents took to run, which is a *scheduling* outcome (and recursively
        the whole upstream chain), not a data-locality outcome.  `file_wait` is
        the only component that measures time actually spent moving this job's
        inputs.  Split by `is_root` before drawing any conclusion:

            latency == parent_wait + file_wait + cpu_wait + execution_time

        holds exactly for every completed job.
        """
        depth_cache: Dict[str, int] = {}

        def _depth(job_id: str, seen: Optional[set] = None) -> int:
            if job_id in depth_cache:
                return depth_cache[job_id]
            j = self.all_jobs.get(job_id)
            if j is None or not j.parent_job_ids:
                depth_cache[job_id] = 0
                return 0
            # Guard against a malformed (cyclic) DAG in a hand-written config.
            seen = seen or set()
            if job_id in seen:
                depth_cache[job_id] = 0
                return 0
            seen.add(job_id)
            d = 1 + max(_depth(pid, seen) for pid in j.parent_job_ids)
            depth_cache[job_id] = d
            return d

        rows = []
        for j in sorted(self.all_jobs.values(), key=lambda x: x.submit_time):
            site_name = (
                self.sites[j.assigned_site].node_id
                if j.assigned_site is not None else None
            )
            latency = (
                float(j.completion_time - j.submit_time)
                if j.completion_time is not None else None
            )
            parent_wait = (
                float(j.arrival_time - j.submit_time)
                if j.arrival_time is not None else None
            )
            data_wait = (
                float(j.data_ready_time - j.submit_time)
                if j.data_ready_time is not None else None
            )
            file_wait = (
                float(j.data_ready_time - j.arrival_time)
                if j.data_ready_time is not None and j.arrival_time is not None else None
            )
            cpu_wait = (
                float(j.start_time - j.data_ready_time)
                if j.start_time is not None and j.data_ready_time is not None else None
            )
            exec_time = (
                float(j.completion_time - j.start_time)
                if j.completion_time is not None and j.start_time is not None else None
            )
            rows.append({
                "job_id": j.job_id,
                "task_id": j.task_id,
                "parent_job_ids": j.parent_job_ids,
                "status": j.status.value,
                "submit_time": j.submit_time,
                "arrival_time": j.arrival_time,
                "assigned_site": j.assigned_site,
                "assigned_site_id": site_name,
                "latency": latency,
                "parent_wait": parent_wait,
                "file_wait": file_wait,
                "data_wait": data_wait,
                "cpu_wait": cpu_wait,
                "execution_time": exec_time,
                "workload_tflops": j.workload_flops / 1e12,
                "is_root": not j.parent_job_ids,
                "dag_depth": _depth(j.job_id),
            })
        return rows

    def site_stats(self) -> List[dict]:
        """
        Per-site utilisation summary (one dict per site).

        Keys
        ----
        site_index, site_id
        num_cpus, cpu_speed_gflops
        jobs_completed      – count of done jobs assigned here
        cpu_utilization     – fraction of CPU-core-seconds used  [0, 1]
        cpu_idle_fraction   – 1 − cpu_utilization
        inbound_transfer_count – transfers that landed (or are landing) here, i.e.
                              inputs staged in because a job was placed away
                              from its data
        inbound_bytes_gb    – total bytes moved into this site over the WAN/tape
        """
        span = self.makespan() or self.current_time
        rows = []
        done_jobs = [j for j in self.all_jobs.values() if j.status == JobStatus.DONE]
        for site in self.sites:
            i = site.site_index
            jobs_here = sum(1 for j in done_jobs if j.assigned_site == i)
            max_cpu_s = span * site.num_cpus
            utilization = (
                self._site_compute_seconds[i] / max_cpu_s if max_cpu_s > 0 else 0.0
            )
            rows.append({
                "site_index": i,
                "site_id": site.node_id,
                "num_cpus": site.num_cpus,
                "cpu_speed_gflops": site.cpu_speed_gflops,
                "jobs_completed": jobs_here,
                "cpu_utilization": round(utilization, 4),
                "cpu_idle_fraction": round(1.0 - utilization, 4),
                "inbound_transfer_count": self._site_transfers_in[i],
                "inbound_bytes_gb": round(self._site_bytes_in[i] / 1e9, 3),
            })
        return rows

    def summary(self) -> dict:
        """
        Top-level scalar metrics — the primary optimisation targets.

        Keys
        ----
        makespan_s           – total elapsed time (s)
        throughput_gflops_s  – effective GFLOPs/s
        throughput_charged_gflops_s – completed GFLOPs divided by the charged
                               span of every SUBMITTED job.  Censoring-proof
                               counterpart of throughput_gflops_s, in the same
                               way mean_latency_charged_s is of mean_latency_s.
        mean_latency_s       – mean job latency (s)
        mean_latency_charged_s – mean over ALL submitted jobs, charging each
                               unfinished job the time it has already waited at
                               the horizon.  Censoring-proof: unlike
                               mean_latency_s it cannot be lowered by stranding
                               the slow tail.  This is the honest objective.
        p95_latency_s        – 95th-percentile job latency (s)
        jobs_done / jobs_total
        total_evictions
        mean_cpu_utilization – averaged across all sites (weighted by num_cpus)

        Latency attribution (means over completed jobs; the four components sum
        to mean_latency_s exactly)
        ----
        mean_parent_wait_s   – blocked on DAG parents.  A *scheduling* cost:
                               it is upstream jobs' runtime, not data placement.
        mean_file_wait_s     – waiting for this job's inputs to arrive.  This,
                               not data_wait, is the locality signal.
        mean_cpu_wait_s      – queued for a free CPU after data was ready
        mean_execution_s     – staging + compute

        Root / child split (a mean over all jobs mixes two different problems)
        ----
        jobs_root_done, jobs_child_done
        mean_latency_root_s  – jobs with no parents
        mean_latency_child_s – jobs with at least one parent
        mean_file_wait_root_s – root-job file wait: the cleanest measure of how
                               well data placement is working, since roots have
                               no parent_wait to swamp it

        Network cost paid
        ----
        total_inbound_transfers  – transfers started (staging + proactive copies)
        total_inbound_bytes_gb

        WARNING — survivorship bias.  mean_latency_s, p95_latency_s and
        throughput_gflops_s are computed over COMPLETED jobs only.  On a
        truncated run (run(until=...)) the slow tail is censored, so a policy
        that strands jobs can report a *better* mean latency than one that
        finishes them all.  Always read these alongside jobs_done/jobs_total;
        a comparison across policies is only meaningful at equal jobs_done.
        The charged variants (mean_latency_charged_s,
        throughput_charged_gflops_s) do not have this problem.
        """
        done = [j for j in self.all_jobs.values() if j.completion_time is not None]
        latencies = sorted(
            j.completion_time - j.submit_time  # type: ignore[operator]
            for j in done
        )
        # Censoring-corrected mean: every job that has been SUBMITTED by the
        # horizon contributes, and one still unfinished is charged the time it
        # has already been waiting.  mean_latency_s alone is computed over
        # completed jobs, so stranding the slow tail *lowers* it; this one can
        # only be lowered by actually finishing work sooner.  Jobs whose
        # submit_time is beyond the horizon are excluded — they never had a
        # chance to run, and counting them as zero-latency would make a short
        # truncated run look artificially good.
        horizon = self.current_time
        submitted = [j for j in self.all_jobs.values() if j.submit_time <= horizon]
        unfinished_wait = sum(
            max(0.0, horizon - j.submit_time)
            for j in submitted
            if j.completion_time is None
        )
        finished_latency = sum(
            j.completion_time - j.submit_time  # type: ignore[operator]
            for j in submitted
            if j.completion_time is not None
        )
        mean_latency_charged = (
            (finished_latency + unfinished_wait) / len(submitted) if submitted else None
        )
        # Nearest-rank percentile: the smallest value with >= 95% of samples
        # at or below it.
        p95 = (
            latencies[min(len(latencies) - 1, max(0, math.ceil(0.95 * len(latencies)) - 1))]
            if latencies else None
        )

        total_cpu_s = sum(self._site_compute_seconds)
        span = self.makespan() or self.current_time
        total_cpu_capacity_s = sum(s.num_cpus for s in self.sites) * span
        mean_util = total_cpu_s / total_cpu_capacity_s if total_cpu_capacity_s > 0 else 0.0

        # ---- latency attribution over completed jobs --------------------
        def _mean(vals: List[float]) -> Optional[float]:
            return sum(vals) / len(vals) if vals else None

        def _parent_wait(j: Job) -> float:
            return float(j.arrival_time - j.submit_time)      # type: ignore[operator]

        def _file_wait(j: Job) -> float:
            return float(j.data_ready_time - j.arrival_time)  # type: ignore[operator]

        # A completed job always has arrival_time, data_ready_time and
        # start_time set, so no None-guards are needed below.
        roots = [j for j in done if not j.parent_job_ids]
        children = [j for j in done if j.parent_job_ids]

        return {
            "makespan_s": self.makespan(),
            "throughput_gflops_s": self.throughput(),
            "throughput_charged_gflops_s": self.throughput_charged(),
            "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
            "mean_latency_charged_s": mean_latency_charged,
            "p95_latency_s": p95,
            "jobs_done": len(done),
            "jobs_total": len(self.all_jobs),
            "mean_parent_wait_s": _mean([_parent_wait(j) for j in done]),
            "mean_file_wait_s": _mean([_file_wait(j) for j in done]),
            "mean_cpu_wait_s": _mean(
                [float(j.start_time - j.data_ready_time) for j in done]  # type: ignore[operator]
            ),
            "mean_execution_s": _mean(
                [float(j.completion_time - j.start_time) for j in done]  # type: ignore[operator]
            ),
            "jobs_root_done": len(roots),
            "jobs_child_done": len(children),
            "mean_latency_root_s": _mean(
                [float(j.completion_time - j.submit_time) for j in roots]  # type: ignore[operator]
            ),
            "mean_latency_child_s": _mean(
                [float(j.completion_time - j.submit_time) for j in children]  # type: ignore[operator]
            ),
            "mean_file_wait_root_s": _mean([_file_wait(j) for j in roots]),
            "total_evictions": sum(s.data_node.eviction_count for s in self.sites),
            "total_inbound_transfers": sum(self._site_transfers_in),
            "total_inbound_bytes_gb": round(sum(self._site_bytes_in) / 1e9, 3),
            "mean_cpu_utilization": round(mean_util, 4),
        }

    def _protected_ids(self, site_idx: int) -> set:
        """
        Files at site_idx that must not be evicted right now:
        - actively being transferred *to* this site
        - currently staged/read by a running job at this site
        - raw detector data (global, checked in eviction policy too)
        """
        protected = {
            fid
            for (fid, dst) in self._active_transfers
            if dst == site_idx
        }
        for job in self.all_jobs.values():
            if job.status == JobStatus.RUNNING and job.assigned_site == site_idx:
                protected.update(job.input_files)
        return protected

    def _replica_counts(self, file_ids=None) -> Dict[str, int]:
        """Replicas per file across all sites.

        ``file_ids`` restricts the scan.  The eviction path only ever looks up
        files resident on the node being reclaimed, and this is O(files ×
        sites) — on a config where disks are tight enough for eviction to be
        routine, scanning the whole catalogue on every write dominates the run.
        """
        ids = self.all_files if file_ids is None else file_ids
        counts: Dict[str, int] = {}
        for fid in ids:
            counts[fid] = sum(1 for s in self.sites if s.data_node.has_file(fid))
        return counts

    def _global_downstream_counts(self) -> Dict[str, int]:
        """Count queued jobs across all sites that need each file."""
        counts: Dict[str, int] = {}
        for site in self.sites:
            for fid, n in site.queued_file_needs().items():
                counts[fid] = counts.get(fid, 0) + n
        return counts

    def _do_write_file(self, f: DataFile, site_idx: int) -> bool:
        """
        Write file f to site_idx's disk.  If disk is full, run eviction first.
        Returns True if the file ends up on disk.
        """
        dn = self.sites[site_idx].data_node
        if dn.add_file(f, self.current_time):
            return True

        # Disk full — run storage reclamation
        protected = self._protected_ids(site_idx)
        # Only files resident here can be evicted, so only they need counting.
        replica_counts = self._replica_counts(list(dn.get_all_meta()))
        downstream = self._global_downstream_counts()

        candidates = self.eviction_policy.select_candidates(
            dn, f.size_bytes, self.current_time,
            protected, self.all_files, replica_counts, downstream,
        )

        for fid in candidates:
            evicted = dn.evict(fid)
            if evicted:
                logger.info(
                    "Evicted '%s' (%.2e B) from site %d to make room for '%s'",
                    fid, evicted.size_bytes, site_idx, f.file_id,
                )

        if dn.add_file(f, self.current_time):
            return True

        logger.warning(
            "Cannot store '%s' at site %d even after eviction (disk %.1f%% full, all remaining files protected).",
            f.file_id, site_idx,
            100 * dn.used_bytes / dn.disk_capacity_bytes,
        )
        return False

    # ================================================================== #
    # Internal event handling
    # ================================================================== #
    def _push(self, time: float, etype: EventType, payload) -> None:
        self._seq += 1
        heapq.heappush(self._events, Event(time, self._seq, etype, payload))

    def _dispatch(self, ev: Event) -> None:
        if ev.event_type == EventType.JOB_ARRIVE:
            self._on_job_arrive(ev)
        elif ev.event_type == EventType.TRANSFER_COMPLETE:
            self._on_transfer_complete(ev)
        elif ev.event_type == EventType.JOB_COMPLETE:
            self._on_job_complete(ev)

    # -- JOB_ARRIVE ---------------------------------------------------- #
    def _on_job_arrive(self, ev: Event) -> None:
        job: Job = ev.payload
        job.status = JobStatus.PENDING

        # First arrival only: this is the instant the job became schedulable
        # (submit_time reached and every parent DONE).  A scheduler that defers
        # by returning None re-enters here, so the guard keeps arrival_time as
        # the eligibility time rather than the assignment time.
        if job.arrival_time is None:
            job.arrival_time = self.current_time

        site_idx = self.scheduler.schedule(job, self._scheduler_view)
        if site_idx is None:
            self._push(self.current_time + self.defer_retry_s,
                       EventType.JOB_ARRIVE, job)
            return

        self._assign_to_site(job, site_idx)

    def _assign_to_site(self, job: Job, site_idx: int) -> None:
        job.assigned_site = site_idx
        job.status = JobStatus.WAITING
        site = self.sites[site_idx]

        missing = [fid for fid in job.input_files if not site.data_node.has_file(fid)]

        if not missing:
            job.pending_transfers = 0
            job.data_ready_time = self.current_time
        else:
            job.pending_transfers = len(missing)
            for fid in missing:
                key = (fid, site_idx)
                self._transfer_waiters[key].append(job)
                if key not in self._active_transfers:
                    self._initiate_transfer(fid, site_idx)

        site.enqueue(job)

        # Optional scheduler hook (default no-op).
        _fire_hook(self.scheduler, "on_job_assigned", job, site_idx, self._scheduler_view)

        if job.pending_transfers == 0:
            self._try_dispatch(site_idx)

    def _initiate_transfer(self, file_id: str, dst_site: int) -> None:
        """Start a transfer of file_id to dst_site.
        Falls back to tape restore (src == -1) when no site holds the file."""
        srcs = [i for i, s in enumerate(self.sites) if s.data_node.has_file(file_id)]

        if srcs:
            src = self.network.best_source(dst_site, srcs)
            f = self.all_files[file_id]
        else:
            # File has been evicted from every site — try tape restore
            f = self.all_files.get(file_id)
            if f is None or not f.is_tape_archived:
                raise RuntimeError(
                    f"File '{file_id}' is unavailable: not on any disk and not tape-archived."
                )
            src = -1
            logger.info("Tape restore of '%s' to site %d at t=%.3f", file_id, dst_site, self.current_time)
            self._site_tape_restore_counts[dst_site] += 1

        self._start_transfer(file_id, src, dst_site, f.size_bytes)

    # -- TRANSFER_COMPLETE --------------------------------------------- #
    def _on_transfer_complete(self, ev: Event) -> None:
        fid: str = ev.payload["file_id"]
        dst: int = ev.payload["dst"]
        src: int = ev.payload["src"]
        key = (fid, dst)
        link = (src, dst)

        # Superseded by a bandwidth re-share on this link — a fresh event for
        # the new earliest finisher has already been queued.
        if ev.payload.get("epoch") != self._link_epoch.get(link):
            return

        # Retire the transfer and hand its share back to the link
        self._finish_transfer(key)

        # Write file to destination disk (with eviction if needed)
        f = self.all_files[fid]
        stored = self._do_write_file(f, dst)
        if not stored:
            # Disk full and every resident file protected.  The bytes did cross
            # the link, so waiting jobs are released and read the copy as
            # scratch — but nothing is cached, so the next job needing this file
            # here pays the transfer again.  Counted as a spill: it is the same
            # "disk is the binding constraint" signal as a lost job output.
            self._site_tape_spill_counts[dst] += 1
            logger.warning(
                "File '%s' could not be cached at site %d (disk full, all "
                "residents protected) — the next reader will re-transfer it.",
                fid, dst,
            )

        # Notify all jobs waiting for this file
        waiters = self._transfer_waiters.pop(key, [])
        for job in waiters:
            job.pending_transfers -= 1
            if job.pending_transfers == 0:
                job.data_ready_time = self.current_time
                logger.debug("Job %s data ready at site %d t=%.3f", job.job_id, dst, self.current_time)

        self._try_dispatch(dst)

    # -- JOB_COMPLETE -------------------------------------------------- #
    def _on_job_complete(self, ev: Event) -> None:
        job: Job = ev.payload["job"]
        site_idx: int = ev.payload["site_idx"]
        site = self.sites[site_idx]

        job.status = JobStatus.DONE
        job.completion_time = self.current_time
        self._site_compute_seconds[site_idx] += (
            self.current_time - job.start_time  # type: ignore[operator]
        ) * job.num_cpus_required

        site.release_cpus(job.allocated_cpus)
        logger.debug("Job %s done at site %d t=%.3f", job.job_id, site_idx, self.current_time)

        # Write output files to local disk (evicting as needed)
        for out_fid in job.output_files:
            out_f = DataFile(
                file_id=out_fid,
                size_bytes=job.output_file_size_bytes,
                is_tape_archived=False,   # freshly created — not yet archived
                creation_time=self.current_time,
            )
            self.all_files[out_fid] = out_f
            if not self._do_write_file(out_f, site_idx):
                # Disk full and nothing evictable.  The output spills straight
                # to the site's tape archive rather than vanishing: a later
                # child job pays a slow tape restore (src == -1) instead of the
                # run aborting on an unreachable file.  Spills are counted and
                # reported so the cost of running disks at the edge is visible.
                out_f.is_tape_archived = True
                self._site_tape_spill_counts[site_idx] += 1
                logger.warning(
                    "Output file '%s' from job %s did not fit at site %d — "
                    "spilled to tape (reads will pay a restore).",
                    out_fid, job.job_id, site_idx,
                )

        # Optionally update file lifecycles at this site
        self.eviction_policy.update_lifecycles(site.data_node, self.current_time)

        # Unblock any child jobs whose last parent just finished
        self._try_unblock_children(job)

        # Optional scheduler hook — fired once the site's state is consistent
        # (outputs written, lifecycles refreshed, children unblocked) but
        # before dispatch, so a stateful scheduler can act on this completion.
        _fire_hook(self.scheduler, "on_job_complete", job, self._scheduler_view)

        self._try_dispatch(site_idx)

    def _try_unblock_children(self, completed_job: Job) -> None:
        """After a job completes, check if any BLOCKED children are now ready."""
        for child_id in self._children.get(completed_job.job_id, []):
            child = self.all_jobs.get(child_id)
            if child is None or child.status != JobStatus.BLOCKED:
                continue
            if all(self.all_jobs[pid].status == JobStatus.DONE for pid in child.parent_job_ids):
                child.status = JobStatus.PENDING
                arrive_t = max(child.submit_time, self.current_time)
                self._push(arrive_t, EventType.JOB_ARRIVE, child)
                logger.debug(
                    "Job %s unblocked (parent %s done), arrives t=%.3f",
                    child_id, completed_job.job_id, arrive_t,
                )

    # -- Dispatch ------------------------------------------------------ #
    def _try_dispatch(self, site_idx: int) -> None:
        site = self.sites[site_idx]
        while site.num_free_cpus > 0:
            job = site.pop_ready_job(self.current_time)
            if job is None:
                break
            self._start_job(job, site_idx)

    def _start_job(self, job: Job, site_idx: int) -> None:
        site = self.sites[site_idx]
        cpus = site.allocate_cpus(job.job_id, job.num_cpus_required)
        if not cpus:
            site.enqueue(job)
            return

        job.allocated_cpus = cpus
        job.status = JobStatus.RUNNING
        job.start_time = self.current_time

        # Mark input files as accessed (HOT)
        for fid in job.input_files:
            site.data_node.touch(fid, self.current_time)

        total_input_bytes = sum(
            self.all_files[fid].size_bytes for fid in job.input_files if fid in self.all_files
        )
        staging = self.network.staging_time(site_idx, total_input_bytes)
        execution = job.workload_flops / (site.cpu_speed_gflops * 1e9)
        t_done = self.current_time + staging + execution

        self._push(t_done, EventType.JOB_COMPLETE, {"job": job, "site_idx": site_idx})
        logger.debug(
            "Job %s started at site %d (cpus %s) staging=%.3f exe=%.3f done=%.3f",
            job.job_id, site_idx, cpus, staging, execution, t_done,
        )


# ====================================================================== #
# Config unit helpers
# ====================================================================== #
_SIZE_UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}
_BW_UNITS   = {"B/s": 1, "KB/s": 1e3, "MB/s": 1e6, "GB/s": 1e9}
_FLOP_UNITS = {"FLOP": 1, "KFLOP": 1e3, "MFLOP": 1e6, "GFLOP": 1e9,
               "TFLOP": 1e12, "FLOPs": 1, "GFLOPs": 1e9, "TFLOPs": 1e12}


def _parse_bytes(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    for unit, mult in sorted(_SIZE_UNITS.items(), key=lambda x: -len(x[0])):
        if str(value).upper().endswith(unit.upper()):
            return float(str(value)[: -len(unit)].strip()) * mult
    raise ValueError(f"Cannot parse size: {value!r}")


def _parse_flops(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    for unit, mult in sorted(_FLOP_UNITS.items(), key=lambda x: -len(x[0])):
        if str(value).upper().endswith(unit.upper()):
            return float(str(value)[: -len(unit)].strip()) * mult
    raise ValueError(f"Cannot parse workload: {value!r}")


def _bw_unit(unit_str: str) -> float:
    for k, v in sorted(_BW_UNITS.items(), key=lambda x: -len(x[0])):
        if unit_str.upper() == k.upper():
            return v
    return 1e6  # default MB/s
