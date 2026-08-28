"""Where a dataset lives, and what that costs us.

Qt-free so it can be unit-tested without a display or a real mount.

Datasets open either from local NVMe or from the lab server over CIFS
(``//bach/Fieldlab`` on ``/mnt/lab``). The two behave nothing alike for the
access pattern this app uses, and the difference is not bandwidth — it is what
happens when several threads read at once.

Measured on 20260715A/data007-010 (1 GbE, page cache evicted, 48 STA cells
read through ``LazySTADict``):

===========  ==============  ================  ===============
threads      wall (server)   median (server)   median (local)
===========  ==============  ================  ===============
1            0.81 s          3.2 ms            1.8 ms
4            0.77 s          51.6 ms           2.7 ms
8            0.76 s          118.8 ms          5.5 ms
===========  ==============  ================  ===============

Wall time is *flat* across thread counts on the server while per-read latency
grows very nearly linearly with them. The link is the serialization point, so
concurrency buys no throughput and only queues each caller behind the others.
That is actively harmful here: the STA read path guards itself with an 8 s
timeout meant to catch a corrupt byte offset, and inflating every read by 37x
walks normal reads toward a deadline designed for pathology. On local disk the
same concurrency is free, so the fan-out stays.

Hence :func:`io_workers` — one reader on a network path, the caller's default
on local disk.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Filesystems whose reads cross a network. Anything not listed is treated as
# local, which is the safe default: a wrong "local" verdict costs some latency,
# a wrong "network" verdict needlessly serializes a fast disk.
NETWORK_FSTYPES = frozenset({
    "cifs", "smb2", "smb3", "smbfs",
    "nfs", "nfs4",
    "afs", "9p", "ncpfs",
    "ceph", "glusterfs", "lustre", "beegfs",
    "fuse.sshfs", "fuse.davfs", "fuse.rclone",
})


def _mount_table():
    """``[(mountpoint, fstype)]`` from /proc/mounts, or [] where unreadable."""
    entries = []
    try:
        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                # Mountpoints are octal-escaped in /proc/mounts (\040 for space).
                mountpoint = parts[1].encode().decode("unicode_escape")
                entries.append((mountpoint, parts[2]))
    except OSError:
        logger.debug("could not read /proc/mounts", exc_info=True)
    return entries


def filesystem_type(path):
    """Filesystem type backing *path*, or None if it cannot be determined.

    Resolves the longest matching mountpoint, so ``/mnt/lab/x`` picks the
    ``/mnt/lab`` entry rather than ``/``.
    """
    try:
        target = Path(path).resolve()
    except (OSError, ValueError):
        return None

    best_type = None
    best_len = -1
    for mountpoint, fstype in _mount_table():
        try:
            mp = Path(mountpoint).resolve()
        except (OSError, ValueError):
            continue
        if target == mp or mp in target.parents:
            if len(str(mp)) > best_len:
                best_len = len(str(mp))
                best_type = fstype
    return best_type


def is_network_path(path):
    """True when *path* is served over a network filesystem."""
    fstype = filesystem_type(path)
    if fstype is None:
        return False
    return fstype in NETWORK_FSTYPES or fstype.startswith("fuse.")


def io_workers(path, local_default=4):
    """Thread count for reads under *path*.

    One on a network mount — see the module docstring; extra threads there add
    latency without adding throughput. ``local_default`` everywhere else.
    """
    if is_network_path(path):
        return 1
    return max(1, int(local_default))
