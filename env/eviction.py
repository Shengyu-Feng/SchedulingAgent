"""
eviction.py — storage lifecycle tracking and eviction policy.

FileMeta  : per-file, per-site metadata (last_access_time, lifecycle state).
EvictionWeights : tunable scoring coefficients.
EvictionPolicy  : selects which files to evict to free a target number of bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from .core import DataFile, FileLifecycleState

if TYPE_CHECKING:
    from .data_node import DataNode


# ---------------------------------------------------------------------------
# Per-file metadata (one instance per site per file)
# ---------------------------------------------------------------------------

@dataclass
class FileMeta:
    creation_time: float
    last_access_time: float
    lifecycle: FileLifecycleState = FileLifecycleState.HOT


# ---------------------------------------------------------------------------
# Eviction scoring weights
# ---------------------------------------------------------------------------

@dataclass
class EvictionWeights:
    # Normalize raw age (seconds) by this value before adding to score.
    # Default: 1 hour → age of 1 h contributes +1 to score.
    age_s: float = 3600.0

    # Bonus per extra replica that exists elsewhere.
    # A file with 3 copies contributes +2 * extra_replicas to score.
    extra_replicas: float = 0.5

    # Penalty per queued job (anywhere) that needs this file.
    downstream: float = 3.0

    # Multiply by DataFile.recomputation_cost.
    recompute: float = 1.5

    # Added to score when the file is NOT tape-archived (strongly protect).
    no_tape_penalty: float = 1000.0

    # Age thresholds for lifecycle transitions
    warm_threshold_s: float = 3600.0     # 1 h  → HOT  → WARM
    cold_threshold_s: float = 86400.0    # 24 h → WARM → COLD


# ---------------------------------------------------------------------------
# Eviction policy
# ---------------------------------------------------------------------------

class EvictionPolicy:
    """
    Selects the minimum set of files to evict from a DataNode to satisfy
    a space requirement.

    The scheduler agent does NOT call this directly — the simulator triggers
    reclamation automatically whenever a write would exceed disk capacity.
    Override ``score`` to implement a custom eviction strategy.
    """

    def __init__(self, weights: Optional[EvictionWeights] = None) -> None:
        self.weights = weights or EvictionWeights()

    # ------------------------------------------------------------------
    def score(
        self,
        file_id: str,
        file: DataFile,
        meta: FileMeta,
        current_time: float,
        replica_count: int,
        downstream_count: int,
    ) -> float:
        """
        Higher score → evict sooner.

        Components
        ----------
        + age            : older files score higher (normalised by age_s)
        + extra_replicas : files with many copies are cheaper to lose locally
        - downstream     : files needed by queued jobs score lower (protect)
        - recompute      : expensive-to-recompute files score lower
        - no_tape        : files without tape backup get a heavy penalty
        """
        w = self.weights
        age = (current_time - meta.last_access_time) / w.age_s
        extras = max(0, replica_count - 1) * w.extra_replicas
        dep = downstream_count * w.downstream
        recompute = file.recomputation_cost * w.recompute
        tape_pen = 0.0 if file.is_tape_archived else w.no_tape_penalty

        return age + extras - dep - recompute - tape_pen

    # ------------------------------------------------------------------
    def select_candidates(
        self,
        node: "DataNode",
        needed_bytes: float,
        current_time: float,
        protected_ids: Set[str],
        global_files: Dict[str, DataFile],
        replica_counts: Dict[str, int],
        downstream_counts: Dict[str, int],
    ) -> List[str]:
        """
        Return an ordered list of file_ids to evict (highest-score first)
        sufficient to free at least ``needed_bytes``.

        Files are excluded from eviction if they are:
        - in ``protected_ids`` (currently staged / being transferred)
        - ``DataFile.is_raw == True`` (raw detector data)
        - the last replica AND not tape-archived (would be permanent data loss)
        """
        w = self.weights
        candidates = []
        for fid, f_meta in node.get_all_meta().items():
            f = global_files.get(fid)
            if f is None:
                continue
            # Hard-protection rules
            if fid in protected_ids:
                continue
            if f.is_raw:
                continue
            rc = replica_counts.get(fid, 1)
            if rc <= 1 and not f.is_tape_archived:
                continue  # last copy, no tape — evicting would be data loss

            s = self.score(
                fid, f, f_meta, current_time, rc,
                downstream_counts.get(fid, 0),
            )
            candidates.append((s, fid, global_files[fid].size_bytes))

        # Sort descending by score (evict worst candidates first)
        candidates.sort(key=lambda x: x[0], reverse=True)

        chosen: List[str] = []
        freed = 0.0
        for _, fid, size in candidates:
            if freed >= needed_bytes:
                break
            chosen.append(fid)
            freed += size
        return chosen

    # ------------------------------------------------------------------
    def update_lifecycles(self, node: "DataNode", current_time: float) -> None:
        """Transition HOT→WARM→COLD based on time since last access."""
        w = self.weights
        for fid, meta in node.get_all_meta().items():
            age = current_time - meta.last_access_time
            if age < w.warm_threshold_s:
                meta.lifecycle = FileLifecycleState.HOT
            elif age < w.cold_threshold_s:
                meta.lifecycle = FileLifecycleState.WARM
            else:
                meta.lifecycle = FileLifecycleState.COLD
