"""Vision .params classification read/write (params_classification.py).

Fixtures are written with an encoder local to this file (vision7
NeuroOutputStream rules), not with the module under test.
"""

import os
import shutil
import stat
import struct
import subprocess

import numpy as np
import pytest

from src.analysis import params_classification as pc
from src.analysis import vision_sort_check as vsc
from src.analysis import visionloader as vl

COLS = [("ID", "Double"), ("classID", "String"), ("x0", "Double"),
        ("TimeCourse", "DoubleArray"), ("note", "String")]


def _hdr(tag_id, n):
    th = tag_id << 6
    if n < 0x3E:
        return struct.pack(">H", th | n)
    if n <= 0xFFFF:
        return struct.pack(">HH", th | 0x3F, n)
    return struct.pack(">HI", th | 0x3E, n)


def _cell(kind, v, lie_about_length=0):
    if kind == "Double":
        body, tid = struct.pack(">d", v), 3
    elif kind == "DoubleArray":
        body, tid = struct.pack(">i", len(v)) + struct.pack(f">{len(v)}d", *v), 4
    else:
        b = v.encode()
        body, tid = struct.pack(">i", len(b)) + b, 5
    return _hdr(tid, len(body) + lie_about_length) + body


def write_params(path, rows, cols=COLS, max_rows=20, trailing=b"", lie_col=None):
    head = struct.pack(">iii", len(cols), len(rows), max_rows)
    for name, kind in cols:
        head += struct.pack(">i", len(name)) + name.encode()
        head += struct.pack(">i", len(kind)) + kind.encode()
    cells = [_cell(kind, v, 160 if name == lie_col else 0)
             for r in rows for (name, kind), v in zip(cols, r)]
    pos, seeks = len(head) + 4 * len(cols) * max_rows, []
    for c in cells:
        seeks.append(pos)
        pos += len(c)
    table = struct.pack(f">{len(seeks)}i", *seeks)
    reserved = b"\x00" * (4 * len(cols) * max_rows - len(table))
    path.write_bytes(head + table + reserved + b"".join(cells) + trailing)
    return path


def rows_example():
    return [
        [2.0, "All/ON/brisk transient", 10.5, [0.1, -0.2, 0.3], "a"],
        [3.0, "All", 11.0, [], "b"],
        [7.0, "All/OFF/nc12", 12.0, [1.0] * 30, "c"],
    ]


@pytest.fixture
def params(tmp_path):
    return write_params(tmp_path / "data000.params", rows_example())


# ── reading ────────────────────────────────────────────────────────────────

def test_reads_ids_and_classes(params):
    t = pc.read_table(params)
    assert t.ids() == [2, 3, 7]
    assert t.classes() == {2: "All/ON/brisk transient", 3: "All", 7: "All/OFF/nc12"}


def test_double_array_with_a_wrong_length_field_is_accepted_and_kept(tmp_path):
    # 2024 kilosort40 .params files carry this; Vision reads by type.
    p = write_params(tmp_path / "d.params", rows_example(), lie_col="TimeCourse")
    before = p.read_bytes()
    t = pc.read_table(p)
    pc.write_classes(p, {3: "All/ON"})
    after = pc.read_table(p)
    col = t.column("TimeCourse")
    assert all(after.cell(r, col) == t.cell(r, col) for r in range(3))
    assert before != p.read_bytes()


def test_half_saved_file_is_refused(params):
    raw = bytearray(params.read_bytes())
    t = pc.read_table(params)
    k = 1 * t.n_cols + 2                       # row 1, x0: point the seek elsewhere
    struct.pack_into(">i", raw, t.header_end + 4 * k, t.spans[k][0] + 3)
    params.write_bytes(bytes(raw))
    with pytest.raises(pc.ParamsFileError, match="half-saved"):
        pc.read_table(params)


def test_zeroed_bytes_in_the_data_are_refused(params):
    raw = bytearray(params.read_bytes())
    t = pc.read_table(params)
    a = t.spans[2 * t.n_cols][0]               # row 2, ID cell
    raw[a:a + 4] = b"\x00" * 4
    params.write_bytes(bytes(raw))
    with pytest.raises(pc.ParamsFileError, match="tag 0"):
        pc.read_table(params)


def test_duplicate_ids_are_refused(tmp_path):
    rows = rows_example()
    rows[1][0] = 2.0
    with pytest.raises(pc.ParamsFileError, match="duplicate"):
        pc.read_table(write_params(tmp_path / "d.params", rows))


# ── writing ────────────────────────────────────────────────────────────────

def test_write_changes_only_the_classid_cells(params):
    old = pc.read_table(params)
    rep = pc.write_classes(params, {3: "All/ON/new type", 7: "All/OFF/nc12", 99: "All/X"})
    new = pc.read_table(params)
    assert rep.changed == [3]                  # 7 unchanged, 99 has no row
    assert new.classes() == {2: "All/ON/brisk transient", 3: "All/ON/new type",
                             7: "All/OFF/nc12"}
    cls = new.column("classID")
    for r in range(3):
        for c in range(new.n_cols):
            if not (r == 1 and c == cls):
                assert new.cell(r, c) == old.cell(r, c)
    assert new.raw[:new.header_end] == old.raw[:old.header_end]
    assert new.data_start == old.data_start


def test_backup_holds_the_previous_version_with_the_same_mode(params):
    os.chmod(params, 0o664)
    original = params.read_bytes()
    rep = pc.write_classes(params, {2: "All"})
    assert rep.backup == pc.backup_path(params)
    assert rep.backup.read_bytes() == original
    assert stat.S_IMODE(os.stat(params).st_mode) == 0o664
    assert stat.S_IMODE(os.stat(rep.backup).st_mode) == 0o664
    assert not list(params.parent.glob("*.tmp"))


def test_nothing_to_change_writes_nothing(params):
    stamp = pc.file_stamp(params)
    rep = pc.write_classes(params, {2: "All/ON/brisk transient"})
    assert rep.changed == [] and rep.backup is None
    assert pc.file_stamp(params) == stamp
    assert not pc.backup_path(params).exists()


def test_a_file_changed_since_review_is_not_written(params):
    stamp = pc.file_stamp(params)
    pc.write_classes(params, {2: "All/OFF"}, keep_backup=False)   # "Vision saved it"
    saved = params.read_bytes()
    with pytest.raises(pc.ParamsChangedError):
        pc.write_classes(params, {3: "All/ON"}, expected_stamp=stamp)
    assert params.read_bytes() == saved


@pytest.mark.parametrize("n", [1, 57, 58, 59, 300, 70000])
def test_string_tag_lengths_round_trip(params, n):
    cls = "All/" + "x" * (n - 4)
    pc.write_classes(params, {7: cls})
    assert pc.read_classes(params)[7] == cls
    assert pc.encode_string_tag(cls)[:2] == _hdr(5, 4 + len(cls))[:2]


def test_trailing_bytes_left_by_vision_are_dropped(tmp_path):
    p = write_params(tmp_path / "d.params", rows_example(), trailing=b"\x07" * 50)
    pc.write_classes(p, {2: "All"})
    t = pc.read_table(p)
    assert t.spans[-1][1] == len(t.raw)


@pytest.mark.parametrize("bad", ["All/ON/", "Allx", "All//ON", "ON/brisk", ""])
def test_malformed_class_strings_are_refused(params, bad):
    with pytest.raises(ValueError):
        pc.write_classes(params, {2: bad})


def test_encore_reader_reads_the_rewritten_file(params):
    pc.write_classes(params, {2: "All/ON/new type"})
    with vl.ParametersFileReader(str(params.parent), "data000") as r:
        vcd = r.update_visioncelldata_obj(vl.VisionCellDataTable())
    assert vcd.get_cell_type_for_cell(2) == "ON new type"
    assert vcd.get_data_for_cell(7, "x0") == 12.0


def _vision_jar():
    jar = os.environ.get("VISION_JAR", os.path.expanduser(
        "~/Documents/Development/MEA-fieldlab/src/vision7_symphony/Vision.jar"))
    javac = shutil.which("javac") or os.path.expanduser("~/miniconda3/bin/javac")
    if not (os.path.isfile(jar) and os.path.isfile(javac)):
        return None, None
    return jar, os.path.dirname(javac)


def test_vision_itself_reads_the_rewritten_file(tmp_path):
    jar, bindir = _vision_jar()
    if jar is None:
        pytest.skip("Vision.jar or a JDK is not on this machine")
    p = write_params(tmp_path / "data000.params", rows_example())
    pc.write_classes(p, {3: "All/ON/" + "y" * 80, 7: "All"})
    (tmp_path / "C.java").write_text(
        "import edu.ucsc.neurobiology.vision.io.ParametersFile;\n"
        "public class C { public static void main(String[] a) throws Exception {\n"
        "  ParametersFile p = new ParametersFile(a[0]);\n"
        "  for (int id : p.getIDList()) System.out.println(id + \"\\t\" + "
        "p.getClassIDs().get(id) + \"\\t\" + p.getDoubleCell(id, \"x0\"));\n"
        "  p.close(false); } }\n")
    subprocess.run([os.path.join(bindir, "javac"), "-cp", jar, "-d", str(tmp_path),
                    str(tmp_path / "C.java")], check=True)
    out = subprocess.run([os.path.join(bindir, "java"), "-Djava.awt.headless=true", "-cp",
                          f"{jar}{os.pathsep}{tmp_path}", "C", str(p)],
                         check=True, capture_output=True, text=True).stdout.splitlines()
    assert out == ["2\tAll/ON/brisk transient\t10.5",
                   "3\tAll/ON/" + "y" * 80 + "\t11.0",
                   "7\tAll\t12.0"]


# ── class strings and diff ─────────────────────────────────────────────────

def test_class_path_and_groups_are_inverse():
    assert pc.class_path([]) == "All"
    assert pc.class_path(["ON", "brisk transient"]) == "All/ON/brisk transient"
    assert pc.class_groups("All/ON/brisk transient/") == ["ON", "brisk transient"]
    assert pc.class_groups("All") == []
    with pytest.raises(ValueError):
        pc.class_path(["a/b"])


def test_diff_classes():
    old = {1: "All/ON", 2: "All/OFF", 3: "All", 4: "All/ON"}
    new = {1: "All/ON", 2: "All", 3: "All/ON", 5: "All/OFF"}
    d = pc.diff_classes(old, new)
    assert d.changed == [2, 3]
    assert d.unclassified == [2]
    assert d.not_in_file == [5]
    assert d.not_in_encore == [4]
    assert d.n_file_rows == 4


# ── Vision files from another sort (vision_sort_check.py) ─────────────────

def _cells(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-900, 900, size=(n, 2))
    rf = pos @ np.array([[0.02, 0.001], [-0.001, -0.02]]) + [25.0, 20.0]
    rf += rng.normal(0, 0.3, size=rf.shape)
    return ({i + 1: tuple(r) for i, r in enumerate(rf)},
            {i + 1: tuple(p) for i, p in enumerate(pos)})


def test_matched_files_pass_the_sort_check():
    rf, pos = _cells(200)
    c = vsc.check_pairing(rf, pos)
    assert c.decided and not c.mismatch and c.r2_robust > 0.9


def test_files_from_another_sort_fail_the_sort_check():
    rf, pos = _cells(200)
    ids = list(pos)
    shuffled = dict(zip(ids, [pos[i] for i in np.random.default_rng(1).permutation(ids)]))
    c = vsc.check_pairing(rf, shuffled)
    assert c.decided and c.mismatch


def test_too_few_cells_or_unfitted_rfs_give_no_verdict():
    rf, pos = _cells(200)
    few = {k: rf[k] for k in list(rf)[:20]}
    assert not vsc.check_pairing(few, pos).decided
    zeros = {k: (0.0, 0.0) for k in rf}
    assert not vsc.check_pairing(zeros, pos).decided
