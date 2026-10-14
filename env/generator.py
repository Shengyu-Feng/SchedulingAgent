"""
generator.py — procedural generation of large-scale simulation configs.

Converts a compact ``*_generation`` YAML block into a full config dict
compatible with ``HEPSimulator.from_config()``.

Supported distributions
-----------------------
``{dist: randint,   low: A, high: B}``      → int in [A, B]  (inclusive)
``{dist: uniform,   low: A, high: B}``      → float in [A, B)
``{dist: normal,    mean: M, std: S}``      → Gaussian, clipped to ≥ ``clip``
``{dist: lognormal, mean: M, sigma: S}``    → log-normal (M in linear scale)
``{dist: choice,    values: [...], weights: [...]}``  → weighted random choice
``{dist: poisson,   lam: L}``               → Poisson draw
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Distribution sampler
# ---------------------------------------------------------------------------

def _sample(spec: Any, rng: np.random.Generator, n: int = 1, clip: float = 0.0):
    """
    Draw `n` samples from a distribution spec.
    Returns a scalar when n==1, otherwise a 1-D ndarray.
    """
    if isinstance(spec, (int, float)):
        # Constant — same value every time
        result = np.full(n, float(spec))
    else:
        d = spec["dist"]
        if d == "randint":
            result = rng.integers(int(spec["low"]), int(spec["high"]) + 1, size=n)
        elif d == "uniform":
            result = rng.uniform(float(spec["low"]), float(spec["high"]), size=n)
        elif d == "normal":
            result = rng.normal(float(spec["mean"]), float(spec["std"]), size=n)
            result = np.clip(result, clip, None)
        elif d == "lognormal":
            mean, sigma = float(spec["mean"]), float(spec["sigma"])
            mu = np.log(mean) - 0.5 * sigma ** 2
            result = rng.lognormal(mu, sigma, size=n)
        elif d == "choice":
            values = spec["values"]
            weights = spec.get("weights")
            if weights is not None:
                w = np.array(weights, dtype=float)
                w /= w.sum()
            else:
                w = None
            result = rng.choice(values, size=n, p=w)
        elif d == "poisson":
            result = rng.poisson(float(spec["lam"]), size=n)
        else:
            raise ValueError(f"Unknown distribution: {d!r}")

    return result[0] if n == 1 else result


# ---------------------------------------------------------------------------
# Site generation
# ---------------------------------------------------------------------------

def _generate_sites(gen: dict, rng: np.random.Generator) -> List[dict]:
    """
    ``gen`` = the ``site_generation`` block.

    Tiers example::

        tiers:
          - name: Tier1
            count: 3
            num_cpus:         {dist: randint, low: 128, high: 256}
            cpu_speed_gflops: {dist: uniform, low: 22.0, high: 28.0}
            disk_capacity_tb: {dist: uniform, low: 50.0, high: 200.0}
          - name: Tier2
            count: 12
            ...
    """
    sites = []
    counters: Dict[str, int] = {}
    for tier in gen["tiers"]:
        tname = tier["name"]
        count = int(tier["count"])
        counters[tname] = counters.get(tname, 0)
        for _ in range(count):
            idx = counters[tname]
            counters[tname] += 1
            cpus = int(_sample(tier["num_cpus"], rng, clip=1))
            speed = float(_sample(tier["cpu_speed_gflops"], rng, clip=1.0))
            disk_tb = float(_sample(tier["disk_capacity_tb"], rng, clip=0.1))
            sites.append({
                "id": f"{tname}_{idx:03d}",
                "tier": tname,
                "num_cpus": max(1, cpus),
                "cpu_speed_gflops": round(speed, 2),
                "disk_capacity": f"{disk_tb:.2f} TB",
            })
    return sites


# ---------------------------------------------------------------------------
# Network generation
# ---------------------------------------------------------------------------

def _generate_network(gen: dict, sites: List[dict], rng: np.random.Generator) -> dict:
    """
    ``gen`` = the ``network_generation`` block.

    Supports two models:

    **tier_matrix** (default) — mean bandwidth between every pair of tier
    combinations, plus log-normal noise::

        network_generation:
          model: tier_matrix
          lan_multiplier: 10
          noise_sigma: 0.25          # log-space std-dev
          tier_matrix:               # mean WAN MB/s between tier combos
            Tier1-Tier1: 200
            Tier1-Tier2: 120
            Tier1-Tier3:  60
            Tier2-Tier2:  80
            Tier2-Tier3:  40
            Tier3-Tier3:  20

    **distance** — random 2-D layout; BW decays with Euclidean distance::

        network_generation:
          model: distance
          lan_multiplier: 10
          wan_base_mbps: 200.0
          wan_min_mbps: 10.0
          distance_scale: 1.0        # characteristic length scale in [0,1]² grid
    """
    n = len(sites)
    model = gen.get("model", "tier_matrix")
    bw = np.zeros((n, n))

    if model == "tier_matrix":
        tm = gen["tier_matrix"]
        noise_sigma = float(gen.get("noise_sigma", 0.2))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ti, tj = sites[i]["tier"], sites[j]["tier"]
                key1 = f"{ti}-{tj}"
                key2 = f"{tj}-{ti}"
                mean_bw = float(tm.get(key1, tm.get(key2, 30)))
                # log-normal noise: noise ~ exp(N(0, sigma))
                noise = float(rng.lognormal(0.0, noise_sigma))
                bw[i, j] = mean_bw * noise

    elif model == "distance":
        base = float(gen.get("wan_base_mbps", 200.0))
        min_bw = float(gen.get("wan_min_mbps", 10.0))
        scale = float(gen.get("distance_scale", 0.5))
        coords = rng.uniform(0, 1, size=(n, 2))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist = float(np.linalg.norm(coords[i] - coords[j]))
                bw[i, j] = min_bw + base * np.exp(-dist / scale)

    else:
        raise ValueError(f"Unknown network model: {model!r}")

    lan_multiplier = float(gen.get("lan_multiplier", 10.0))
    return {
        "bandwidth_unit": "MB/s",
        "lan_multiplier": lan_multiplier,
        "wan_bandwidth_matrix": bw.tolist(),
    }


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

def _generate_files(gen: dict, sites: List[dict], rng: np.random.Generator) -> List[dict]:
    """
    ``gen`` = the ``file_generation`` block.

    Example::

        file_generation:
          count: 500
          size_gb:            {dist: lognormal, mean: 2.0, sigma: 1.0}
          replication_factor: {dist: randint, low: 1, high: 5}
    """
    count = int(gen["count"])
    site_ids = [s["id"] for s in sites]
    files = []
    for i in range(count):
        size_gb = float(_sample(gen["size_gb"], rng, clip=0.001))
        rep = int(_sample(gen.get("replication_factor", 2), rng, clip=1))
        rep = max(1, min(rep, len(site_ids)))
        locs = rng.choice(site_ids, size=rep, replace=False).tolist()
        files.append({
            "id": f"file_{i:05d}",
            "size": f"{size_gb:.4f} GB",
            "initial_locations": locs,
        })
    return files


# ---------------------------------------------------------------------------
# Task (DAG workflow) generation
# ---------------------------------------------------------------------------

# Baked-in defaults for the popularity-phase mechanism.  These live here
# rather than in the YAML because the phase-shift is a property of the
# environment's underlying access dynamics, not a scheduler-tunable knob.
_DEFAULT_POPULARITY: Dict[str, Any] = {
    "enabled": True,
    "num_phases": 5,
    "hot_set_size": 30,
    "hot_weight": 0.85,
    "churn_fraction": 0.4,
    "evergreen_fraction": 0.2,
    "zipf_alpha": 1.1,
    "hot_replica_min": 3,
    "hot_replica_per_phase": 1.5,
    "hot_replica_max": 8,
}


def _resolve_popularity_cfg(cfg: dict) -> Dict[str, Any]:
    """Merge baked-in defaults with any override the caller supplied.

    We accept an override under either ``popularity_dynamics`` or the
    innocuously-named ``_pop`` key so that unit tests can disable the
    mechanism explicitly (``enabled: false``) without turning it into a
    documented feature.  Missing keys fall through to the defaults.
    """
    override = cfg.get("popularity_dynamics") or cfg.get("_pop") or {}
    merged = dict(_DEFAULT_POPULARITY)
    merged.update(override)
    return merged


def _compute_popularity_phases(
    file_ids: List[str],
    pop_cfg: dict,
    rng: np.random.Generator,
) -> Tuple[List[List[str]], List[np.ndarray], Dict[str, int]]:
    """
    Precompute the popularity phases used to bias job input-file selection.

    ATLAS analysis campaigns rotate: a handful of DAOD datasets dominate
    accesses for a few days, then a new physics topic (new-analysis push,
    conference deadline, calibration release) shifts the "hot" set.  We model
    this by cutting the submission window into ``num_phases`` slabs; each slab
    has its own hot set, and a ``churn_fraction`` of members rotates between
    consecutive slabs while a small evergreen core persists throughout.

    Returned artefacts (kept strictly inside the generator — no runtime API):
      * ``hot_sets``    : file IDs that are hot in each phase (ordered by weight)
      * ``phase_probs`` : per-phase sampling weight over ALL files (sums to 1)
      * ``ever_hot``    : file_id → number of phases in which it appears hot
    """
    num_phases = max(1, int(pop_cfg.get("num_phases", 4)))
    hot_size = int(pop_cfg.get("hot_set_size", max(1, len(file_ids) // 10)))
    hot_size = max(1, min(hot_size, len(file_ids)))
    hot_weight = float(pop_cfg.get("hot_weight", 0.85))
    churn = float(pop_cfg.get("churn_fraction", 0.35))
    zipf_alpha = float(pop_cfg.get("zipf_alpha", 1.1))
    evergreen_frac = float(pop_cfg.get("evergreen_fraction", 0.2))

    # Zipf-shaped weights within a hot set (hottest file first)
    ranks = np.arange(1, hot_size + 1, dtype=float)
    zipf = 1.0 / (ranks ** zipf_alpha)
    zipf /= zipf.sum()

    n_files = len(file_ids)
    n_evergreen = int(round(evergreen_frac * hot_size))
    n_replace = int(round(churn * (hot_size - n_evergreen)))
    n_replace = max(0, min(n_replace, hot_size - n_evergreen))

    # Evergreen core: same across all phases (mimics stable calibration files
    # / standard-candle datasets that every analysis touches).
    perm = rng.permutation(n_files)
    evergreen_idx = perm[:n_evergreen].tolist()
    pool = perm[n_evergreen:].tolist()   # non-evergreen candidates

    hot_sets: List[List[str]] = []
    phase_probs: List[np.ndarray] = []
    ever_hot: Dict[str, int] = {file_ids[i]: 0 for i in range(n_files)}

    # Seed phase-0 hot set
    burst_size = hot_size - n_evergreen
    burst_idx = list(rng.choice(pool, size=burst_size, replace=False)) if burst_size > 0 else []
    for p in range(num_phases):
        current = list(evergreen_idx) + list(burst_idx)
        # Reorder so evergreen gets the top zipf weights (they're always popular)
        # then the burst files follow, shuffled per phase for variety.
        burst_perm = rng.permutation(len(burst_idx)) if burst_idx else np.array([], dtype=int)
        ordered_burst = [burst_idx[i] for i in burst_perm]
        ordered = list(evergreen_idx) + ordered_burst
        ids_hot = [file_ids[i] for i in ordered]
        hot_sets.append(ids_hot)

        # Build the per-phase sampling distribution over ALL files
        w = np.zeros(n_files, dtype=float)
        w[ordered] = zipf * hot_weight
        cold_mask = np.ones(n_files, dtype=bool)
        cold_mask[ordered] = False
        n_cold = int(cold_mask.sum())
        if n_cold > 0 and hot_weight < 1.0:
            w[cold_mask] = (1.0 - hot_weight) / n_cold
        if w.sum() > 0:
            w /= w.sum()
        phase_probs.append(w)

        for fid in ids_hot:
            ever_hot[fid] = ever_hot.get(fid, 0) + 1

        # Prepare the next phase's burst set: replace n_replace members with
        # freshly-drawn files from outside the current hot set.
        if p < num_phases - 1 and burst_idx:
            keep_n = len(burst_idx) - n_replace
            keep = list(rng.choice(burst_idx, size=keep_n, replace=False)) if keep_n > 0 else []
            outside = [i for i in pool if i not in set(evergreen_idx) and i not in set(keep)]
            if n_replace > 0 and outside:
                fresh = list(rng.choice(outside, size=min(n_replace, len(outside)), replace=False))
            else:
                fresh = []
            burst_idx = keep + fresh

    return hot_sets, phase_probs, ever_hot


def _boost_hot_replication(
    files: List[dict],
    sites: List[dict],
    ever_hot: Dict[str, int],
    pop_cfg: dict,
    rng: np.random.Generator,
) -> None:
    """
    Grow the ``initial_locations`` list of files that are hot in one or more
    phases.  Popular DAODs at ATLAS live at 3–8 T1/T2 sites simultaneously
    because ops-side rules pin them there.  This function reproduces that
    pattern deterministically.

    The boost is size-aware: bigger files get fewer extra replicas so we do
    not blow up aggregate storage.  Purely rewrites ``initial_locations``
    in-place; no runtime hook is exposed.
    """
    if not any(ever_hot.values()):
        return

    base = int(pop_cfg.get("hot_replica_min", 3))
    per_phase = float(pop_cfg.get("hot_replica_per_phase", 1.5))
    cap = int(pop_cfg.get("hot_replica_max", 8))

    # Prefer higher-tier sites (Tier1 > Tier2 > Tier3) for hot replicas.
    tier_rank = {"Tier1": 0, "Tier2": 1, "Tier3": 2}
    site_order = sorted(
        range(len(sites)),
        key=lambda i: (tier_rank.get(sites[i].get("tier", "Tier3"), 3), i),
    )
    ordered_site_ids = [sites[i]["id"] for i in site_order]

    for f in files:
        n_phases_hot = ever_hot.get(f["id"], 0)
        if n_phases_hot <= 0:
            continue
        target = min(cap, base + int(round(per_phase * n_phases_hot)))
        current = list(f.get("initial_locations", []))
        current_set = set(current)
        # Prefer higher-tier sites first, then jitter with a light shuffle so
        # replicas do not all cluster on the same three Tier-1s.
        candidates = [sid for sid in ordered_site_ids if sid not in current_set]
        head = candidates[: max(1, len(candidates) // 3)]
        tail = candidates[len(head):]
        rng.shuffle(tail)
        candidates = head + tail
        needed = max(0, target - len(current))
        for sid in candidates[:needed]:
            current.append(sid)
        f["initial_locations"] = current


def _generate_tasks(
    gen: dict, files: List[dict], rng: np.random.Generator,
    phase_probs: Optional[List[np.ndarray]] = None,
    phase_boundaries: Optional[List[float]] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Generate tasks (multi-stage pipelines) and the jobs they contain.

    Each task is a linear pipeline of stages.  Stage-0 jobs read external
    data files.  Each subsequent stage reads the combined outputs of the
    previous stage, creating explicit parent-child DAG edges.

    Example config block::

        task_generation:
          num_tasks: 130
          stages_per_task:    {dist: randint, low: 2, high: 3}
          jobs_per_stage:     {dist: randint, low: 1, high: 2}
          workload_tflops:    {dist: lognormal, mean: 5.0, sigma: 1.2}
          num_input_files:    {dist: randint,   low: 1, high: 3}
          output_file_size_gb:{dist: uniform,   low: 0.1, high: 2.0}
          inter_arrival_s:    {dist: lognormal, mean: 5.0, sigma: 0.7}
          num_cpus_required:  {dist: choice, values: [1,2,4], weights: [0.80,0.15,0.05]}

    Returns
    -------
    tasks : List[dict]  — {id, job_ids}
    jobs  : List[dict]  — job dicts compatible with HEPSimulator.from_config()
    """
    num_tasks = int(gen["num_tasks"])
    stages_spec = gen.get("stages_per_task", 2)
    jobs_per_stage_spec = gen.get("jobs_per_stage", 1)
    workload_spec = gen["workload_tflops"]
    output_size_spec = gen.get("output_file_size_gb", 0.5)
    inter_arrival_spec = gen.get("inter_arrival_s", 5.0)
    ncpus_spec = gen.get("num_cpus_required", 1)
    num_input_files_spec = gen.get("num_input_files", 2)

    raw_file_ids = [f["id"] for f in files]
    tasks_out: List[dict] = []
    jobs_out: List[dict] = []
    job_counter = 0
    t = 0.0  # task-level arrival time (submit_time for stage-0 jobs)

    use_phases = phase_probs is not None and phase_boundaries is not None

    for task_idx in range(num_tasks):
        n_stages = max(1, int(_sample(stages_spec, rng, clip=1)))
        task_job_ids: List[str] = []
        prev_stage: List[dict] = []   # job dicts from the previous stage

        # Which popularity phase covers this task's submission time?
        if use_phases:
            phase_idx = 0
            for b in phase_boundaries:
                if t >= b:
                    phase_idx += 1
            phase_idx = min(phase_idx, len(phase_probs) - 1)
            probs = phase_probs[phase_idx]
        else:
            probs = None

        for stage_idx in range(n_stages):
            n_jobs = max(1, int(_sample(jobs_per_stage_spec, rng, clip=1)))
            stage: List[dict] = []

            for _ in range(n_jobs):
                job_id = f"job_{job_counter:05d}"
                out_fid = f"out_{job_id}"
                job_counter += 1

                wl_tflops = float(_sample(workload_spec, rng, clip=0.01))
                out_gb = float(_sample(output_size_spec, rng, clip=0.001))
                ncpus = max(1, int(_sample(ncpus_spec, rng, clip=1)))

                if stage_idx == 0:
                    # Root stage: read from the pre-existing data files
                    n_in = max(1, min(int(_sample(num_input_files_spec, rng, clip=1)),
                                     len(raw_file_ids)))
                    if probs is not None:
                        in_files = rng.choice(
                            raw_file_ids, size=n_in, replace=False, p=probs
                        ).tolist()
                    else:
                        in_files = rng.choice(raw_file_ids, size=n_in, replace=False).tolist()
                    parent_ids: List[str] = []
                else:
                    # Non-root: inputs are the combined outputs of all prev-stage jobs
                    in_files = [pj["output_files"][0] for pj in prev_stage]
                    parent_ids = [pj["id"] for pj in prev_stage]

                job_dict = {
                    "id": job_id,
                    "workload": f"{wl_tflops:.4f} TFLOPs",
                    "input_files": in_files,
                    "output_files": [out_fid],
                    "output_file_size": f"{out_gb:.4f} GB",
                    "submit_time": round(t, 3),
                    "num_cpus_required": ncpus,
                    "task_id": f"task_{task_idx:04d}",
                    "parent_job_ids": parent_ids,
                }
                stage.append(job_dict)
                task_job_ids.append(job_id)
                jobs_out.append(job_dict)

            prev_stage = stage

        tasks_out.append({"id": f"task_{task_idx:04d}", "job_ids": task_job_ids})

        inter = float(_sample(inter_arrival_spec, rng, clip=0.0))
        t += max(0.0, inter)

    return tasks_out, jobs_out


# ---------------------------------------------------------------------------
# Job generation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Site warm-cache initialisation
# ---------------------------------------------------------------------------

def _parse_bytes_gen(s) -> float:
    """Parse a size string like '2.5 TB' or '500 GB' into bytes."""
    s = str(s).strip().upper()
    for unit, mult in [("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3), ("B", 1.0)]:
        if s.endswith(unit):
            return float(s[: -len(unit)].strip()) * mult
    return float(s)


def _initialize_sites(
    init_cfg: dict,
    sites: List[dict],
    files: List[dict],
    rng: np.random.Generator,
) -> None:
    """
    Pre-populate each site with files so the scheduler faces a realistic
    warm-cache start rather than empty disks.

    ``init_cfg`` = the ``site_initialization`` block, e.g.::

        site_initialization:
          seed_files_per_tier:
            Tier1: 60
            Tier2: 20
            Tier3: 5

    Extra replicas are added until each site reaches its target file count
    OR its disk capacity is full — whichever comes first.
    Operates deterministically given the simulation seed.
    Mutates each file dict's ``initial_locations`` list in-place.
    """
    target_counts: Dict[str, int] = init_cfg.get("seed_files_per_tier", {})
    if not target_counts:
        return

    file_ids: List[str] = [f["id"] for f in files]
    file_map: Dict[str, dict] = {f["id"]: f for f in files}

    # Pre-parse file sizes (bytes) for capacity checking
    file_bytes: Dict[str, float] = {f["id"]: _parse_bytes_gen(f["size"]) for f in files}

    # Parse each site's disk capacity (bytes)
    site_capacity: Dict[str, float] = {
        s["id"]: _parse_bytes_gen(s["disk_capacity"]) for s in sites
    }

    # Current site → files already placed there by file_generation
    site_cache: Dict[str, Set[str]] = {s["id"]: set() for s in sites}
    for f in files:
        for loc in f.get("initial_locations", []):
            if loc in site_cache:
                site_cache[loc].add(f["id"])

    # Track bytes already used per site
    site_used: Dict[str, float] = {
        sid: sum(file_bytes.get(fid, 0) for fid in fids)
        for sid, fids in site_cache.items()
    }

    for site in sites:
        sid = site["id"]
        tier = site.get("tier", "Tier3")
        target = int(target_counts.get(tier, 0))
        already = site_cache[sid]
        needed = target - len(already)
        if needed <= 0:
            continue

        cap = site_capacity.get(sid, 2e12)
        candidates = [fid for fid in file_ids if fid not in already]

        # Shuffle deterministically, then fill until target count or disk full
        order = rng.permutation(len(candidates))
        added = 0
        for idx in order:
            if added >= needed:
                break
            fid = candidates[int(idx)]
            fsize = file_bytes.get(fid, 0)
            if site_used[sid] + fsize > cap:
                continue  # skip files that don't fit
            already.add(fid)
            site_used[sid] += fsize
            locs = file_map[fid].setdefault("initial_locations", [])
            if sid not in locs:
                locs.append(sid)
            added += 1


# ---------------------------------------------------------------------------
# Job generation
# ---------------------------------------------------------------------------

def _generate_jobs(gen: dict, files: List[dict], rng: np.random.Generator) -> List[dict]:
    """
    ``gen`` = the ``job_generation`` block.

    Example::

        job_generation:
          count: 200
          workload_tflops:    {dist: lognormal, mean: 5.0, sigma: 1.2}
          num_input_files:    {dist: randint, low: 1, high: 4}
          output_file_size_gb:{dist: uniform, low: 0.1, high: 2.0}
          inter_arrival_s:    {dist: lognormal, mean: 2.0, sigma: 0.5}
          num_cpus_required:  {dist: choice, values: [1, 2, 4], weights: [0.80, 0.15, 0.05]}
    """
    count = int(gen["count"])
    file_ids = [f["id"] for f in files]
    jobs = []
    t = 0.0
    for i in range(count):
        wl_tflops = float(_sample(gen["workload_tflops"], rng, clip=0.01))
        n_in = int(_sample(gen.get("num_input_files", 2), rng, clip=1))
        n_in = max(1, min(n_in, len(file_ids)))
        in_files = rng.choice(file_ids, size=n_in, replace=False).tolist()
        out_gb = float(_sample(gen.get("output_file_size_gb", 0.5), rng, clip=0.001))
        ncpus = int(_sample(gen.get("num_cpus_required", 1), rng, clip=1))
        inter = float(_sample(gen.get("inter_arrival_s", 1.0), rng, clip=0.0))
        jobs.append({
            "id": f"job_{i:05d}",
            "workload": f"{wl_tflops:.4f} TFLOPs",
            "input_files": in_files,
            "output_files": [f"out_{i:05d}"],
            "output_file_size": f"{out_gb:.4f} GB",
            "submit_time": round(t, 3),
            "num_cpus_required": max(1, ncpus),
        })
        t += max(0.0, inter)
    return jobs


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def expand_generated_config(cfg: dict) -> dict:
    """
    If ``cfg`` contains any ``*_generation`` keys, expand them into full
    ``sites``, ``network``, ``data_files``, and ``jobs`` lists and return
    the merged config dict.  Keys already present (e.g. a hand-written
    ``sites`` list alongside ``job_generation``) are kept as-is.
    """
    seed = cfg.get("seed", 0)
    rng = np.random.default_rng(seed)

    result = dict(cfg)  # shallow copy; we overwrite generated sections

    # ---- sites -----------------------------------------------------------
    if "site_generation" in cfg:
        result["sites"] = _generate_sites(cfg["site_generation"], rng)

    sites = result.get("sites", [])

    # ---- network ---------------------------------------------------------
    if "network_generation" in cfg:
        result["network"] = _generate_network(cfg["network_generation"], sites, rng)

    # ---- data files ------------------------------------------------------
    if "file_generation" in cfg:
        result["data_files"] = _generate_files(cfg["file_generation"], sites, rng)

    data_files = result.get("data_files", [])

    # ---- popularity dynamics (hidden) -----------------------------------
    # Compute phase-shifted popularity BEFORE site initialisation and task
    # generation so that (a) initial replica placement can concentrate on
    # currently-hot files, and (b) each task's input files are drawn from
    # the popularity distribution of its arrival-time phase.  The mechanism
    # is intentionally not surfaced anywhere in the simulator's runtime
    # API — schedulers can only infer popularity from observed accesses.
    # Defaults live in `_DEFAULT_POPULARITY` and are always applied when a
    # task-generation block is present; the config file need not (and by
    # design does not) advertise it.
    pop_cfg = _resolve_popularity_cfg(cfg)
    phase_probs: Optional[List[np.ndarray]] = None
    phase_boundaries: Optional[List[float]] = None
    if pop_cfg.get("enabled", False) and data_files and (
        "task_generation" in cfg or "job_generation" in cfg
    ):
        file_ids = [f["id"] for f in data_files]
        hot_sets, phase_probs, ever_hot = _compute_popularity_phases(
            file_ids, pop_cfg, rng
        )
        _boost_hot_replication(data_files, sites, ever_hot, pop_cfg, rng)

        # Estimate the submission-window length and cut it into phases.
        # We use the task_generation inter-arrival mean × num_tasks as a
        # cheap proxy — it is deterministic given the seed.
        tg = cfg.get("task_generation", {}) or cfg.get("job_generation", {}) or {}
        num_tasks_est = int(tg.get("num_tasks", tg.get("count", 1)))
        inter_spec = tg.get("inter_arrival_s", 5.0)
        if isinstance(inter_spec, (int, float)):
            mean_inter = float(inter_spec)
        else:
            mean_inter = float(inter_spec.get("mean", inter_spec.get("low", 5.0)))
        window = max(1.0, num_tasks_est * mean_inter)
        n = len(phase_probs)
        phase_boundaries = [window * (k / n) for k in range(1, n)]

    # ---- site warm-cache initialisation ----------------------------------
    if "site_initialization" in cfg:
        _initialize_sites(cfg["site_initialization"], sites, data_files, rng)

    # ---- tasks (DAG workflows) ------------------------------------------
    if "task_generation" in cfg:
        tasks, jobs = _generate_tasks(
            cfg["task_generation"], data_files, rng,
            phase_probs=phase_probs, phase_boundaries=phase_boundaries,
        )
        result["tasks"] = tasks
        result["jobs"] = jobs
    elif "job_generation" in cfg:
        result["jobs"] = _generate_jobs(cfg["job_generation"], data_files, rng)

    return result
