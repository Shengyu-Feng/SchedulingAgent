# HEP Grid Scheduling — Agent Instructions

## 1. The task

A distributed computing grid (ATLAS PanDA style) runs physics jobs across many sites.
Jobs arrive over time; each needs input files that live on only a few sites' disks.
**Write a scheduling policy that assigns each job to a site.**

**You are scored on `throughput_charged_gflops_s`** (§4). It is the
GFLOPs you completed divided by the span of every submitted job, charging each unfinished
one its elapsed wait — so finishing large work fast raises it, and stranding a job can
only lower it.

You control exactly one thing: **which site each job runs on**, returned from
`SchedulerInterface.schedule()`. Everything else is fixed environment — CPU speeds and
core counts, bandwidths, file sizes and their placement, job workloads, disk capacities,
submit times, and the disk eviction policy. The scheduler is read-only with respect to
the environment: it returns a site index and changes nothing else.

A job placed where its inputs already are starts immediately. A job placed elsewhere
pulls its inputs over the WAN (or restores them from tape) and is charged for the wait,
and that transfer shares bandwidth with every other transfer on the same link.

## 2. Jobs

Jobs are grouped into **tasks** — multi-stage DAG workflows where stage *k* reads the
output files of stage *k-1*.

Statuses: `BLOCKED → PENDING → WAITING → RUNNING → DONE`

| Status | Meaning |
|---|---|
| `BLOCKED` | a parent is not `DONE` yet — never passed to `schedule()` |
| `PENDING` | schedulable — `schedule()` is called now |
| `WAITING` | assigned to a site, waiting for input data transfers |
| `RUNNING` | executing (staging + compute) |
| `DONE` | finished; outputs written to the assigned site's disk |

Two mechanics worth knowing:

1. `schedule()` is called at `max(submit_time, last_parent_completion)`, not at
   `submit_time`. The charged span for a job is still measured from `submit_time`.
2. A job's outputs land on the same site it is executed.

## 3. The interface

```python
from env.interfaces import SchedulerInterface
from env.core import Job
from typing import Optional

class MyScheduler(SchedulerInterface):

    def schedule(self, job: Job, env: HEPSimulator) -> Optional[int]:
        """Return a site index in [0, len(env.sites)), or None to defer.

        A deferred job is re-queued and schedule() is called again after
        defer_retry_s (default 1 µs — pass a coarser value if you batch, since
        each retry costs an event).  A job deferred forever never runs.
        """

    # Optional hooks (no-ops by default)
    def on_job_assigned(self, job, site_idx, env) -> None: ...
    def on_job_complete(self, job, env) -> None: ...
```

### 3.1 Job fields

These are all 17 fields a `Job` has. Anything else does not exist and raises `AttributeError` mid-run.

```
job_id                 str
workload_flops         float
input_files            List[str]    file IDs required
output_files           List[str]    file IDs written on completion
output_file_size_bytes float
submit_time            float        seconds
num_cpus_required      int
task_id                str | None
parent_job_ids         List[str]

# set by the simulator
status                 JobStatus    compare with `from env.core import JobStatus`
                                    then e.g. `job.status == JobStatus.DONE` — a
                                    real (non-TYPE_CHECKING) import, since its
                                    members are runtime values, not just a type
                                    hint. .value gives the lowercase string
                                    ("done") if you ever need that instead.
arrival_time           float | None became schedulable; None until it does
assigned_site          int | None
pending_transfers      int
data_ready_time        float | None
start_time             float | None
completion_time        float | None
allocated_cpus         int
```

### 3.2 What you can read off `env`

```python
env.current_time                    # float — simulation clock (seconds)
env.sites                           # List[ComputingNode]
env.all_tasks                       # Dict[str, Task]; task.job_ids in topological order
env.all_jobs                        # Dict[str, Job]
env.all_files                       # Dict[str, DataFile]

# Per site
site = env.sites[i]
site.node_id, site.site_index
site.num_cpus, site.num_free_cpus
site.cpu_speed_gflops               # per-core GFLOPs/s
site.cpu_status()                   # List[str | None], one entry per core: None if
                                    #   idle, else the job_id occupying it. For how
                                    #   long, look the job up (env.all_jobs[jid]) and
                                    #   read env.current_time - job.start_time
site.queue_length()                 # queued-job COUNT, blind to job size
site.queued_jobs()                  # List[Job] actually waiting here, unordered.
                                    #   For how long a job has waited, read
                                    #   env.current_time - job.data_ready_time (once
                                    #   ready) or - job.arrival_time (before that)
site.data_node.get_all_files()      # Dict[str, DataFile] — every file resident on
                                    #   this site's disk, keyed by file_id; live view, do not mutate
site.data_node.free_bytes, .used_bytes, .disk_capacity_bytes
site.data_node.tape_bandwidth_bytes_s  # this site's tape-restore bandwidth (bytes/s);
                                    #   the capacity term for any transfer with src=-1
site.data_node.get_meta(fid)        # FileMeta | None: .last_access_time,
                                    #   .creation_time, .lifecycle (HOT/WARM/COLD)
site.data_node.get_all_meta()       # Dict[str, FileMeta] — live view, do not mutate

# Per-job, per-site feature table
env.get_site_features(job)          # List[dict], all 13 keys:
#   site_index, site_id
#   remote_transfer_time   max WAN/tape time for missing inputs (s), assumes an idle network
#   staging_time           LAN disk→CPU time for all inputs (s)
#   execution_time         compute time at this site (s)
#   queue_wait_estimate    seconds of queued work already at this site
#                          (backlog_flops / site throughput)
#   num_free_cpus, queue_length, cpu_speed_gflops
#   has_all_files          bool
#   disk_used_fraction, disk_free_bytes
#   inbound_transfers      transfers in flight into this site right now

# Live network load
env.link_load(src, dst)             # int — transfers sharing this link (src=-1 → tape)
env.link_transfers(src, dst)        # List[dict] — the actual transfers link_load only
                                    #   counts: {file_id, src, dst, size_bytes,
                                    #   remaining_bytes, rate_bytes_s}; rate_bytes_s IS
                                    #   each one's current instant rate — read it
                                    #   directly, no need to recompute it yourself
env.link_share_bytes_s(src, dst)    # float — bytes/s a NEW transfer started right
                                    #   now would get: capacity / (link_load(src,dst)+1)
                                    #   (an existing transfer's instant rate is instead
                                    #   capacity / link_load(src,dst) — same capacity,
                                    #   no +1, since it's already counted in link_load)
                                    #   capacity = env.network.bandwidth(src, dst) for
                                    #   a real site pair, or the DESTINATION site's
                                    #   site.data_node.tape_bandwidth_bytes_s if src=-1.
                                    #   Do NOT use env.network.bandwidth_matrix[-1, dst]
                                    #   or env.network.bandwidth(-1, dst) for tape —
                                    #   both silently index the LAST row of the matrix
                                    #   (normal Python/numpy negative indexing) instead
                                    #   of meaning "tape", with no error raised.
env.site_inbound_transfers(i)       # int
env.site_inbound_transfer_list(i)   # List[dict] — same shape as link_transfers, the
                                    #   actual transfers site_inbound_transfers only counts

# Files
f.size_bytes, f.is_tape_archived, f.is_raw, f.recomputation_cost

# Network — env.network is a TransferNetwork object; call these, it is not indexable
env.network.bandwidth(src, dst)                     # bytes/s (diagonal = LAN)
env.network.transfer_time(src, dst, size_bytes)     # s, uncontended
env.network.staging_time(site_idx, size_bytes)      # s
env.network.best_source(dst, candidate_sites)       # site index
env.network.bandwidth_matrix                        # np.ndarray [n_sites × n_sites]
```

Two rules:

- **Only read a job that has arrived**, i.e. `j.status != BLOCKED` *and*
  `j.submit_time <= env.current_time`. A `BLOCKED` job's `input_files` is future demand
  you may predict, not look up.
- **Only use the API above.** Underscore-prefixed attributes on `env`, and writes to
  disk, CPU or network state, are off limits — they reveal the future or corrupt the run.

### 3.3 Wiring it up

Every run has a hard 180 s wall-clock limit and is failed if it exceeds it, so
`schedule()` must stay cheap per job. 

## 4. Metrics

Call these after `sim.run()`, never inside a hook.

### `sim.summary()` → dict

`throughput_charged_gflops_s` is **THE SCORE** — completed GFLOPs divided by
the charged span of every submitted job (charging each unfinished one its
elapsed wait at the horizon). It is the only field in this dict you are
scored on; nothing else here is a target to optimise.

The dict also carries a range of other real, raw diagnostic fields (job
completion counts, disk/eviction/tape-transfer counters, per-job timing
breakdowns, and more) — inspect `sim.summary()` yourself (e.g.
`print(sim.summary().keys())`) rather than assuming a fixed schema, since
this may change between environment versions. Whatever you compute from it
to explain or improve a policy's behaviour is up to you.

### `sim.job_stats()` → List[dict]

One dict per job, sorted by `submit_time`. Real per-job fields — `job_id`,
`task_id`, `parent_job_ids`, `status`, `submit_time`, `assigned_site`, and
more — are all present; inspect the keys yourself
(`print(sim.job_stats()[0].keys())`) rather than assuming a fixed set.

### `sim.site_stats()` → List[dict]

One dict per site. Same note: real per-site fields (core counts, disk usage,
transfer/eviction counters, and more) are present — inspect the keys
yourself rather than assuming a fixed set.

`sim.makespan()`, `sim.throughput()` and `sim.throughput_charged()` return
scalars directly; `sim.throughput_charged()` is the same number as
`sim.summary()["throughput_charged_gflops_s"]`.

