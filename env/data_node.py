"""DataNode: disk storage attached to a computing site."""
from __future__ import annotations

from typing import Dict, Optional
from .core import DataFile, FileLifecycleState
from .eviction import FileMeta


class DataNode:
    def __init__(
        self,
        node_id: str,
        disk_capacity_bytes: float,
        tape_bandwidth_bytes_s: float = 500e6,   # 500 MB/s default
    ):
        self.node_id = node_id
        self.disk_capacity_bytes = disk_capacity_bytes
        self.tape_bandwidth_bytes_s = tape_bandwidth_bytes_s

        self.files: Dict[str, DataFile] = {}
        self._used_bytes: float = 0.0
        self._meta: Dict[str, FileMeta] = {}   # file_id → per-site metadata

        # Every successful evict() is counted here, whoever called it — the
        # simulator's reclamation path or an eviction policy.  site_stats()
        # reads this, so eviction can never be invisible in the telemetry.
        self.eviction_count: int = 0

    # ------------------------------------------------------------------ #
    @property
    def used_bytes(self) -> float:
        return self._used_bytes

    @property
    def free_bytes(self) -> float:
        return self.disk_capacity_bytes - self._used_bytes

    # ------------------------------------------------------------------ #
    def has_file(self, file_id: str) -> bool:
        return file_id in self.files

    def get_all_files(self) -> Dict[str, DataFile]:
        """Return a live (not copied) view of every file currently resident
        on this site's disk, keyed by file_id."""
        return self.files

    def add_file(self, file: DataFile, current_time: float = 0.0) -> bool:
        """
        Store file on disk if there is room.  Returns True on success,
        False if capacity is exceeded (caller must evict first).
        Does NOT perform eviction — that is the simulator's responsibility.
        """
        if file.file_id in self.files:
            # Already present; refresh access time
            self.touch(file.file_id, current_time)
            return True
        if self._used_bytes + file.size_bytes > self.disk_capacity_bytes:
            return False
        self.files[file.file_id] = file
        self._used_bytes += file.size_bytes
        self._meta[file.file_id] = FileMeta(
            creation_time=current_time,
            last_access_time=current_time,
            lifecycle=FileLifecycleState.HOT,
        )
        return True

    def touch(self, file_id: str, current_time: float) -> None:
        """Record an access to file_id; transition lifecycle to HOT."""
        if file_id in self._meta:
            self._meta[file_id].last_access_time = current_time
            self._meta[file_id].lifecycle = FileLifecycleState.HOT

    def evict(self, file_id: str) -> Optional[DataFile]:
        """
        Remove file_id from disk.  Returns the DataFile if it was present,
        None otherwise.  The file may still exist on tape or at other sites.
        """
        if file_id not in self.files:
            return None
        f = self.files.pop(file_id)
        self._used_bytes -= f.size_bytes
        self._meta.pop(file_id, None)
        self.eviction_count += 1
        return f

    # Keep legacy name for any callers that used remove_file
    def remove_file(self, file_id: str) -> bool:
        return self.evict(file_id) is not None

    # ------------------------------------------------------------------ #
    def get_meta(self, file_id: str) -> Optional[FileMeta]:
        return self._meta.get(file_id)

    def get_all_meta(self) -> Dict[str, FileMeta]:
        """Return a live (not copied) view of the metadata dict."""
        return self._meta

    def lifecycle_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {s.value: 0 for s in FileLifecycleState}
        for m in self._meta.values():
            counts[m.lifecycle.value] += 1
        return counts

    # ------------------------------------------------------------------ #
    def get_state(self) -> dict:
        return {
            "node_id": self.node_id,
            "disk_capacity_bytes": self.disk_capacity_bytes,
            "used_bytes": self._used_bytes,
            "free_bytes": self.free_bytes,
            "num_files": len(self.files),
            "lifecycle_counts": self.lifecycle_counts(),
        }
