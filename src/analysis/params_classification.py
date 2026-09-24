"""Read and write the classification stored in a Vision .params file.

Vision keeps each cell's class in the .params file, column ``classID``, as a
path with an ``All`` root: ``"All/ON/brisk transient"``. A cell with no class
has ``"All"``. Vision shows this column in its classification tree, so writing
it is how a classification made in Encore reaches Vision.

File layout (vision7 ``io/ParametersFile.java``, all big-endian):

    int32 nColumns, int32 nRows, int32 maxRows
    nColumns x (string name, string type)       string = int32 length + bytes
    seek table: nRows x nColumns int32 offsets  (Vision reserves maxRows rows)
    cells, row-major, each one tag:
        uint16 header = tagID << 6 | length       (length 0x3F: + uint16,
                                                    length 0x3E: + uint32)
        tag 3 Double: float64; tag 4 DoubleArray: int32 n + n float64;
        tag 5 String: int32 length + bytes

Vision reads the cells in sequence from the first seek offset. It does not use
the rest of the seek table. Encore's ParametersFileReader uses every offset.
A file is accepted here only when both readings agree.

Vision opens the file read-write and saves it in place, with no temporary
file and no truncation. An interrupted save leaves a mix of old and new bytes.
This module never edits the file in place. It writes a new file next to it,
reads that file back, keeps the previous version as ``<name>.params.bak``, and
then renames the new file over the old one.
"""

from __future__ import annotations

import logging
import math
import os
import stat
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT_CLASS = "All"
ID_COLUMN = "ID"
CLASS_COLUMN = "classID"
BACKUP_SUFFIX = ".bak"

TAG_DOUBLE = 3
TAG_DOUBLE_ARRAY = 4
TAG_STRING = 5
TYPE_TAGS = {"Double": TAG_DOUBLE, "DoubleArray": TAG_DOUBLE_ARRAY, "String": TAG_STRING}

_INT32_MAX = 2**31 - 1


class ParamsFileError(Exception):
    """The file does not parse the way Vision parses it."""


class ParamsChangedError(Exception):
    """The file changed on disk after Encore read it."""


def file_stamp(path) -> Tuple[int, int]:
    """(mtime in ns, size) — changes when any program saves the file."""
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


@dataclass
class ParamsTable:
    """One strictly parsed .params file, with the raw bytes of every cell."""

    path: Path
    raw: bytes
    stamp: Tuple[int, int]
    n_cols: int
    n_rows: int
    max_rows: int
    names: List[str]
    types: List[str]
    header_end: int
    data_start: int
    spans: List[Tuple[int, int]] = field(repr=False)

    def column(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise ParamsFileError(f"no {name!r} column in {self.path.name}") from None

    def cell(self, row: int, col: int) -> bytes:
        a, b = self.spans[row * self.n_cols + col]
        return self.raw[a:b]

    def ids(self) -> List[int]:
        col = self.column(ID_COLUMN)
        return [_decode_id(self.cell(r, col), self.path) for r in range(self.n_rows)]

    def classes(self) -> Dict[int, str]:
        """Vision id → classID string, in file row order."""
        col = self.column(CLASS_COLUMN)
        return {vid: decode_string_tag(self.cell(r, col))
                for r, vid in enumerate(self.ids())}


# ── tag encoding (vision7 io/tags/NeuroOutputStream.writeTagHeader) ────────

def _tag_header(tag_id: int, length: int) -> bytes:
    th = tag_id << 6
    if length < 0x3E:
        return struct.pack(">H", th | length)
    if length <= 0xFFFF:
        return struct.pack(">HH", th | 0x3F, length)
    return struct.pack(">HI", th | 0x3E, length)


def encode_string_tag(text: str) -> bytes:
    body = text.encode("utf-8")
    body = struct.pack(">i", len(body)) + body
    return _tag_header(TAG_STRING, len(body)) + body


def decode_string_tag(cell: bytes) -> str:
    tag_id, _, p = _read_tag_header(cell, 0)
    if tag_id != TAG_STRING:
        raise ParamsFileError(f"expected a String tag, found tag {tag_id}")
    n = struct.unpack_from(">i", cell, p)[0]
    return cell[p + 4:p + 4 + n].decode("utf-8")


def _decode_id(cell: bytes, path) -> int:
    tag_id, _, p = _read_tag_header(cell, 0)
    if tag_id != TAG_DOUBLE:
        raise ParamsFileError(f"{Path(path).name}: ID cell is tag {tag_id}, not Double")
    value = struct.unpack_from(">d", cell, p)[0]
    if not math.isfinite(value) or value != int(value):
        raise ParamsFileError(f"{Path(path).name}: ID {value!r} is not a whole number")
    return int(value)


def _read_tag_header(buf: bytes, p: int) -> Tuple[int, int, int]:
    th = struct.unpack_from(">H", buf, p)[0]
    p += 2
    tag_id, length = th >> 6, th & 0x3F
    if length == 0x3F:
        length = struct.unpack_from(">H", buf, p)[0]
        p += 2
    elif length == 0x3E:
        length = struct.unpack_from(">I", buf, p)[0]
        p += 4
    return tag_id, length, p


def _body_length(buf: bytes, p: int, tag_id: int) -> int:
    """Body size as Vision's typed readers consume it (they ignore the header length)."""
    if tag_id == TAG_DOUBLE:
        return 8
    n = struct.unpack_from(">i", buf, p)[0]
    if n < 0:
        raise ParamsFileError(f"negative element count {n}")
    return 4 + (8 * n if tag_id == TAG_DOUBLE_ARRAY else n)


# ── reading ────────────────────────────────────────────────────────────────

def read_table(path) -> ParamsTable:
    """Parse ``path`` exactly as Vision does, and check the seek table too.

    Raises ParamsFileError if the file is damaged in any way either reader
    would notice, and ParamsChangedError if it changed while being read.
    """
    path = Path(path)
    before = file_stamp(path)
    raw = path.read_bytes()
    if file_stamp(path) != before:
        raise ParamsChangedError(f"{path.name} changed while it was being read")
    try:
        return _parse(path, raw, before)
    except (struct.error, UnicodeDecodeError, IndexError) as exc:
        raise ParamsFileError(f"{path.name}: {type(exc).__name__}: {exc}") from None


def _parse(path: Path, raw: bytes, stamp) -> ParamsTable:
    n_cols, n_rows, max_rows = struct.unpack_from(">iii", raw, 0)
    if not (0 < n_cols < 100_000 and 0 <= n_rows <= max_rows):
        raise ParamsFileError(
            f"{path.name}: bad header (columns={n_cols}, rows={n_rows}, max rows={max_rows})")
    p = 12
    names, types = [], []
    for _ in range(n_cols):
        for out in (names, types):
            n = struct.unpack_from(">i", raw, p)[0]
            if not 0 <= n <= len(raw) - p - 4:
                raise ParamsFileError(f"{path.name}: bad column-name length {n}")
            out.append(raw[p + 4:p + 4 + n].decode("utf-8"))
            p += 4 + n
    header_end = p
    unknown = sorted(set(types) - set(TYPE_TAGS))
    if unknown:
        raise ParamsFileError(f"{path.name}: unknown column types {unknown}")

    n_cells = n_rows * n_cols
    if header_end + 4 * n_cells > len(raw):
        raise ParamsFileError(f"{path.name}: file ends inside the seek table")
    seeks = struct.unpack_from(f">{n_cells}i", raw, header_end)
    data_start = seeks[0] if n_cells else header_end + 4 * n_cols * max_rows
    if n_cells and data_start < header_end + 4 * n_cells:
        raise ParamsFileError(f"{path.name}: cell data overlaps the seek table")

    spans = []
    p = data_start
    for k in range(n_cells):
        if seeks[k] != p:
            raise ParamsFileError(
                f"{path.name}: row {k // n_cols}, column {names[k % n_cols]!r}: the seek "
                "table does not match the cell data (the file looks half-saved)")
        # The header's length field is not checked: Vision and Encore both
        # read by type, and many good files (e.g. 2024 kilosort40 runs) carry
        # a DoubleArray length that does not match the element count.
        tag_id, _, body = _read_tag_header(raw, p)
        want = TYPE_TAGS[types[k % n_cols]]
        if tag_id != want:
            raise ParamsFileError(
                f"{path.name}: row {k // n_cols}, column {names[k % n_cols]!r}: tag "
                f"{tag_id}, expected {want} ({types[k % n_cols]})")
        end = body + _body_length(raw, body, tag_id)
        if end > len(raw):
            raise ParamsFileError(f"{path.name}: file ends inside row {k // n_cols}")
        spans.append((p, end))
        p = end

    table = ParamsTable(path, raw, stamp, n_cols, n_rows, max_rows, names, types,
                        header_end, data_start, spans)
    if TYPE_TAGS[types[table.column(ID_COLUMN)]] != TAG_DOUBLE:
        raise ParamsFileError(f"{path.name}: the ID column is not Double")
    if TYPE_TAGS[types[table.column(CLASS_COLUMN)]] != TAG_STRING:
        raise ParamsFileError(f"{path.name}: the classID column is not String")
    ids = table.ids()
    if len(set(ids)) != len(ids):
        raise ParamsFileError(f"{path.name}: duplicate cell IDs")
    return table


def read_classes(path) -> Dict[int, str]:
    """Vision id → classID for every row of ``path``."""
    return read_table(path).classes()


# ── class strings ──────────────────────────────────────────────────────────

def class_path(groups) -> str:
    """``["ON", "brisk transient"]`` → ``"All/ON/brisk transient"``; ``[]`` → ``"All"``."""
    parts = [str(g).strip() for g in groups]
    if any(not g or "/" in g or "\n" in g for g in parts):
        raise ValueError(f"group names must be non-empty and contain no '/': {groups!r}")
    return "/".join([ROOT_CLASS, *parts])


def class_groups(class_id: str) -> List[str]:
    """``"All/ON/brisk transient"`` → ``["ON", "brisk transient"]``; ``"All"`` → ``[]``."""
    parts = [p for p in str(class_id).strip().split("/") if p.strip()]
    if parts and parts[0] == ROOT_CLASS:
        parts = parts[1:]
    return parts


# ── diff ───────────────────────────────────────────────────────────────────

@dataclass
class ClassDiff:
    """How a new classification differs from the one in the file."""

    changed: List[int]           # rows whose classID will change
    unclassified: List[int]      # of those, rows that lose a class (→ "All")
    not_in_file: List[int]       # Encore cells with no row in the file (not saved)
    not_in_encore: List[int]     # file rows Encore did not send (left as they are)
    n_file_rows: int = 0

    @property
    def n_changed(self) -> int:
        return len(self.changed)


def diff_classes(old: Dict[int, str], new: Dict[int, str]) -> ClassDiff:
    changed = sorted(v for v, c in new.items() if v in old and old[v] != c)
    return ClassDiff(
        changed=changed,
        unclassified=[v for v in changed
                      if new[v] == ROOT_CLASS and old[v] != ROOT_CLASS],
        not_in_file=sorted(set(new) - set(old)),
        not_in_encore=sorted(set(old) - set(new)),
        n_file_rows=len(old),
    )



# ── writing ────────────────────────────────────────────────────────────────

def rebuild(table: ParamsTable, new_classes: Dict[int, str]) -> Tuple[bytes, List[int]]:
    """The file bytes with the classID cells of ``new_classes`` replaced.

    Every other byte before the cells is copied, the seek table is recomputed,
    and every other cell is copied unchanged. Trailing bytes after the last
    cell (left by Vision's in-place saves) are dropped.
    """
    class_col = table.column(CLASS_COLUMN)
    old = table.classes()
    cells, changed = [], []
    for row, vid in enumerate(table.ids()):
        for col in range(table.n_cols):
            if col == class_col and vid in new_classes and new_classes[vid] != old[vid]:
                cells.append(encode_string_tag(new_classes[vid]))
                changed.append(vid)
            else:
                cells.append(table.cell(row, col))

    seeks, pos = [], table.data_start
    for c in cells:
        seeks.append(pos)
        pos += len(c)
    if pos > _INT32_MAX:
        raise ParamsFileError(f"{table.path.name}: the new file would exceed 2 GB")

    n_cells = table.n_rows * table.n_cols
    out = b"".join([
        table.raw[:table.header_end],
        struct.pack(f">{n_cells}i", *seeks),
        table.raw[table.header_end + 4 * n_cells:table.data_start],
        *cells,
    ])
    return out, changed


@dataclass
class SaveReport:
    path: Path
    changed: List[int]
    backup: Optional[Path]
    stamp: Tuple[int, int]


def backup_path(path) -> Path:
    path = Path(path)
    return path.with_name(path.name + BACKUP_SUFFIX)


def write_classes(path, new_classes: Dict[int, str],
                  expected_stamp: Optional[Tuple[int, int]] = None,
                  keep_backup: bool = True) -> SaveReport:
    """Set the classID of each Vision id in ``new_classes``; leave all else as is.

    ``expected_stamp`` is the file_stamp() of the version the user reviewed. If
    the file has changed since (for example Vision saved it), nothing is
    written and ParamsChangedError is raised. Ids with no row are ignored.
    """
    path = Path(path)
    for vid, cls in new_classes.items():
        if class_groups(cls) and cls != class_path(class_groups(cls)):
            raise ValueError(f"cell {vid}: {cls!r} is not a Vision class path")
        if not cls.startswith(ROOT_CLASS):
            raise ValueError(f"cell {vid}: {cls!r} does not start with {ROOT_CLASS!r}")

    table = read_table(path)
    if expected_stamp is not None and table.stamp != tuple(expected_stamp):
        raise ParamsChangedError(
            f"{path.name} changed on disk after Encore read it (saved from Vision?)")
    new_raw, changed = rebuild(table, new_classes)
    if not changed:
        return SaveReport(path, [], None, table.stamp)

    expected = {**table.classes(), **{v: new_classes[v] for v in changed}}
    mode = stat.S_IMODE(os.stat(path).st_mode)
    tmp = _write_temp(path, new_raw, mode)
    try:
        check = read_table(tmp)
        if check.raw != new_raw or check.classes() != expected:
            raise ParamsFileError(f"{path.name}: the new file did not read back as written")

        backup = None
        if keep_backup:
            backup = backup_path(path)
            os.replace(_write_temp(path, table.raw, mode), backup)

        if file_stamp(path) != table.stamp:
            raise ParamsChangedError(f"{path.name} changed on disk while Encore was saving")
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    logger.info("Wrote %d classIDs to %s (previous version: %s)", len(changed), path, backup)
    return SaveReport(path, changed, backup, file_stamp(path))


def _write_temp(path: Path, data: bytes, mode: int) -> str:
    """Write ``data`` to a new temporary file beside ``path``, with ``mode``.

    mkstemp creates the file 0600; Vision opens .params read-write, so the
    mode of the original is copied (a no-op on mounts that fix the mode).
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp, mode)
        except OSError:
            logger.debug("could not set the mode of %s", tmp, exc_info=True)
    except BaseException:
        os.unlink(tmp)
        raise
    return tmp
