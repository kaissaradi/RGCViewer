"""
vision.py - Pure-Python reader/writer for Litke/Chichilnisky Vision files.

Single-file, NumPy-only replacement for the compiled `visionloader` /
`visionwriter` packages. No Cython, no pybind11, no build step.

Every byte-decode that the old C/Cython extensions did is a big-endian
reinterpret, which NumPy does with typed dtypes ('>i4', '>f4', '>f8').
Because the old extensions were *scalar* loops, the vectorized versions
here are generally faster, not slower.

Public API is a drop-in for the subset used by Encore / data_manager:

    GlobalsFileReader      .get_electrode_map() -> (positions[N,2], disconnected_set)
    EIReader               .get_all_eis_by_cell_id(), .get_ei_for_cell_id(),
                           .get_electrode_map(), .get_disconnected_electrodes()
    ParametersFileReader   .update_visioncelldata_obj(vcd), .get_all_field_names()
    STAReader              .cell_id_to_byte_offset, .get_sta_for_cell_id(),
                           .get_all_stas_by_cell_id(), .chunked_load_all_stas()
    NeuronsReader          .get_spike_sample_nums_for_neuron/all_real_neurons(),
                           .get_identifier_electrode(s)..., .get_TTL_times(),
                           .sample_freq, .n_samples
    VisionCellDataTable, VisionFieldNames, STAContainer, EIContainer, STAFit

External deps preserved (these hold Litke array geometry, not byte layout):
    import electrode_map as elmap
    from bin2py import FakeArrayID, PyBinHeader

Writers for .neurons / .globals / .ei / .sta live at the bottom.
"""

import logging
import os
import struct
from collections import namedtuple
from typing import Dict, Tuple, List, Optional, Any, Union, Set

import numpy as np

logger = logging.getLogger(__name__)

# Seek-table cache for .ei files. Guarded like the geometry deps below so this
# reader still works standalone; without it every open just rescans, which is
# what it did before the cache existed.
try:
    from . import ei_index_cache
except Exception:  # pragma: no cover - reader used outside the package
    class _NoIndexCache:
        @staticmethod
        def load(_ei_path):
            return None

        @staticmethod
        def save(_ei_path, _offsets, _nspikes):
            return False

    ei_index_cache = _NoIndexCache()

# --- external geometry deps (unchanged; not byte decoders) ------------------
try:
    import src.analysis.electrode_map as elmap
except Exception:  # pragma: no cover - allow reader use without geometry pkg
    elmap = None
try:
    from bin2py import FakeArrayID, PyBinHeader
except Exception:  # pragma: no cover
    FakeArrayID = None
    PyBinHeader = None

# Vision's sentinel array id for arbitrary/reconfigurable electrode geometry
# (e.g. the 512 flat MEA). Coordinates for these are stored in the .globals
# file itself, so no external geometry package is needed. Used as a fallback
# when bin2py.FakeArrayID is not importable.
BOARD_ID_RECONFIGURABLE = 9999

N_BYTES_BOOLEAN = 1
N_BYTES_16BIT = 2
N_BYTES_32BIT = 4
N_BYTES_64BIT = 8

# big-endian numpy dtypes: this is the entire trick
BE_I4 = np.dtype('>i4')
BE_U4 = np.dtype('>u4')
BE_F4 = np.dtype('>f4')
BE_F8 = np.dtype('>f8')


# ============================================================================
# Field names / containers  (byte-identical to the originals)
# ============================================================================

STAFit = namedtuple('STAFit', ['center_x', 'center_y', 'std_x', 'std_y', 'rot'])

EIContainer = namedtuple('EIContainer',
                         ['ei', 'ei_error', 'nl_points', 'nr_points',
                          'n_samples_total', 'n_spikes'])

STAContainer = namedtuple('STAContainer',
                          ['stixel_size', 'refresh_time', 'sta_offset',
                           'red', 'red_error', 'green', 'green_error',
                           'blue', 'blue_error'])


class VisionFieldNames:
    CLASSID_FIELDNAME = 'classID'
    CELLID_FIELDNAME = 'ID'
    EI_FIELDNAME = 'EI'
    STA_FIELDNAME = 'STAraw'
    SPIKE_TIMES_FIELDNAME = 'SpikeTimes'
    ELEC_SPIKE_TIMES_FIELDNAME = 'ElecSpikeTimes'
    CHANNEL_NOISE_FIELDNAME = 'ChannelNoise'
    MODEL_FIELD_NAME = 'Model'
    STA_FITX_STD_FIELDNAME = 'SigmaX'
    STA_FITY_STD_FIELDNAME = 'SigmaY'
    STA_XCENTER_FIELDNAME = 'x0'
    STA_YCENTER_FIELDNAME = 'y0'
    STA_ROT_FIELDNAME = 'Theta'
    CONTAMINATION_FIELDNAME = 'contamination'
    ACF_FIELDNAME = 'acfMean'
    ACF_NUMPAIRS_FIELDNAME = 'Auto'


class VisionCellDataTable:
    """Data table keyed by cell id, matching the original API surface used
    by data_manager (get_cell_ids, set/get_electrode_map, etc.)."""

    def __init__(self) -> None:
        self.main_datatable = {}          # type: Dict[int, Dict[str, Any]]
        self.all_field_names = set()      # type: Set[str]
        self.electrode_map = None
        self.disconnected_electrodes = set()
        self.ttl_times = None
        self.n_samples = None
        self.runtimemovie_params = None
        self.electrode_spike_times = None

    # --- lookups ---
    def get_cell_ids(self) -> List[int]:
        return list(self.main_datatable.keys())

    def get_all_field_names(self) -> List[str]:
        return list(self.all_field_names)

    def get_all_present_cell_types(self) -> List[str]:
        return list({d[VisionFieldNames.CLASSID_FIELDNAME]
                     for d in self.main_datatable.values()})

    def get_all_cells_of_type(self, class_name: str) -> List[int]:
        return [cid for cid, d in self.main_datatable.items()
                if d[VisionFieldNames.CLASSID_FIELDNAME] == class_name]

    def get_all_cells_similar_to_type(self, class_name: str) -> List[int]:
        class_name = class_name.lower()
        if class_name and class_name[-1] == 's':
            class_name = class_name[:-1]
        return [cid for cid, d in self.main_datatable.items()
                if class_name in d[VisionFieldNames.CLASSID_FIELDNAME].lower()]

    def get_cell_type_for_cell(self, cell_id: int) -> str:
        assert cell_id in self.main_datatable, \
            "Cell id {0} not found in dataset".format(cell_id)
        return self.main_datatable[cell_id][VisionFieldNames.CLASSID_FIELDNAME]

    def get_all_data_for_cell(self, cell_id: int) -> Dict:
        return self.main_datatable[cell_id]

    def get_data_for_cell(self, cell_id: int, field_name: str) -> Any:
        return self.main_datatable[cell_id][field_name]

    def get_cell_type_for_cell_id(self, cell_id: int) -> str:
        return self.get_cell_type_for_cell(cell_id)

    # --- electrode map / misc setters ---
    def set_electrode_map(self, electrode_map: np.ndarray) -> None:
        self.electrode_map = electrode_map

    def get_electrode_map(self) -> np.ndarray:
        return self.electrode_map

    def set_disconnected_electrodes(self, s: Set[int]) -> None:
        self.disconnected_electrodes = s

    def get_disconnected_electrodes(self) -> Set[int]:
        return self.disconnected_electrodes

    def set_runtimemovie_params(self, rtmp) -> None:
        self.runtimemovie_params = rtmp

    def get_runtimemovie_params(self):
        return self.runtimemovie_params

    def get_n_samples(self) -> int:
        assert self.n_samples is not None, "Load .neurons for n_samples"
        return self.n_samples

    def get_ttl_times(self) -> np.ndarray:
        assert self.ttl_times is not None, "No TTL times, load .neurons"
        return self.ttl_times

    # --- mutation ---
    def update_data_for_cell_id_and_field_name(self, cell_id, field_name, data):
        self.all_field_names.add(field_name)
        self.main_datatable.setdefault(cell_id, {})[field_name] = data

    def add_ei_from_loaded_ei_dict(self, ei_by_cell_id, restrict_to_existing_cells=True):
        self.all_field_names.add(VisionFieldNames.EI_FIELDNAME)
        for cid, ei in ei_by_cell_id.items():
            if restrict_to_existing_cells and cid in self.main_datatable:
                self.main_datatable[cid][VisionFieldNames.EI_FIELDNAME] = ei
            elif not restrict_to_existing_cells and cid not in self.main_datatable:
                self.main_datatable[cid] = {VisionFieldNames.EI_FIELDNAME: ei}

    def add_sta_from_loaded_sta_dict(self, sta_by_cell_id):
        self.all_field_names.add(VisionFieldNames.STA_FIELDNAME)
        for cid, sta in sta_by_cell_id.items():
            self.main_datatable.setdefault(cid, {})[VisionFieldNames.STA_FIELDNAME] = sta

    def add_spike_times_from_loaded_spike_times_dict(self, spikes_by_cell_id,
                                                     ttl_times, n_samples_dataset):
        self.all_field_names.add(VisionFieldNames.SPIKE_TIMES_FIELDNAME)
        for cid, st in spikes_by_cell_id.items():
            self.main_datatable.setdefault(cid, {})[VisionFieldNames.SPIKE_TIMES_FIELDNAME] = st
        self.ttl_times = ttl_times
        self.n_samples = n_samples_dataset

    # --- STA fit convenience (matches MATLAB/Vision coordinate hack) ---
    def get_sta_for_cell(self, cell_id: int) -> np.ndarray:
        return self.get_data_for_cell(cell_id, VisionFieldNames.STA_FIELDNAME)

    def get_stafit_for_cell(self, cell_id: int) -> STAFit:
        x = self.get_data_for_cell(cell_id, VisionFieldNames.STA_XCENTER_FIELDNAME)
        y = self.get_data_for_cell(cell_id, VisionFieldNames.STA_YCENTER_FIELDNAME)
        sx = self.get_data_for_cell(cell_id, VisionFieldNames.STA_FITX_STD_FIELDNAME)
        sy = self.get_data_for_cell(cell_id, VisionFieldNames.STA_FITY_STD_FIELDNAME)
        rot = self.get_data_for_cell(cell_id, VisionFieldNames.STA_ROT_FIELDNAME)
        if self.runtimemovie_params is None:
            return STAFit(x, y, sx, sy, rot)
        return STAFit(x + 0.5, self.runtimemovie_params.height - y + 0.5, sx, sy, rot)


# ============================================================================
# ChunkFile base (globals). 8-byte file id, then repeated (tag,?,size) headers.
# ============================================================================

class ChunkFileReader:
    def __init__(self, cf_path: str) -> None:
        self.fp = open(cf_path, 'rb')

    def _iter_to_tag(self, tag: int):
        self.fp.seek(N_BYTES_64BIT, 0)  # skip 64-bit file id
        while True:
            hdr = self.fp.read(N_BYTES_32BIT * 3)
            if len(hdr) != N_BYTES_32BIT * 3:
                raise IOError("tag {0} not found".format(tag))
            curr_tag, _, curr_size = struct.unpack('>III', hdr)
            if curr_tag == tag:
                return curr_size
            self.fp.seek(curr_size, 1)

    def get_chunk_bytearray(self, tag: int) -> Tuple[int, bytes]:
        size = self._iter_to_tag(tag)
        return size, self.fp.read(size)

    def has_tag(self, tag: int) -> bool:
        try:
            self._iter_to_tag(tag)
            return True
        except IOError:
            return False

    def read_chunk_int_dtype(self, tag: int) -> int:
        size, content = self.get_chunk_bytearray(tag)
        assert size == 4, "Content is not a single 32-bit integer"
        return struct.unpack('>i', content)[0]

    def read_chunk_float64_dtype(self, tag: int) -> float:
        size, content = self.get_chunk_bytearray(tag)
        assert size == N_BYTES_64BIT, "Content is not a single 64-bit float"
        return struct.unpack('>d', content)[0]

    def read_chunk_pair_float_array_dtype(self, tag: int) -> np.ndarray:
        # replaces vcext.unpack_alternating_32bit_float_from_bytearray:
        # payload is N x-coords (f32 BE) followed by N y-coords (f32 BE)
        size, content = self.get_chunk_bytearray(tag)
        assert size % (N_BYTES_32BIT * 2) == 0, \
            "Size not divisible by 8; not pairs of 32-bit float"
        n = size >> 3
        flat = np.frombuffer(content, dtype=BE_F4)          # (2n,)
        out = np.empty((n, 2), dtype=np.float32)
        out[:, 0] = flat[:n]
        out[:, 1] = flat[n:2 * n]
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fp.close()

    def close(self):
        self.fp.close()


class GlobalsFileReader(ChunkFileReader):
    ARRAYID_TAG = 6
    RDH512_TAG = 5
    ICP_TAG = 0
    ELECTRODEMAP_TAG = 7
    ELECTRODEPITCH_TAG = 8

    def __init__(self, analysis_folder_path, dataset_name, globals_extension='globals'):
        assert os.path.isdir(analysis_folder_path)
        path = os.path.join(analysis_folder_path,
                            "{0}.{1}".format(dataset_name, globals_extension))
        assert os.path.isfile(path)
        super().__init__(path)

    def get_electrode_map(self) -> Tuple[np.ndarray, Set[int]]:
        has_array_id = self.has_tag(self.ARRAYID_TAG)
        has_rdh512 = self.has_tag(self.RDH512_TAG)
        has_icp = self.has_tag(self.ICP_TAG)

        if not (has_array_id or has_rdh512 or has_icp):
            raise AssertionError(
                "No array id / RDH512 / ICP in globals file")

        array_id = -1
        if has_array_id:
            array_id = self.read_chunk_int_dtype(self.ARRAYID_TAG) & 0xFFFF
        elif has_rdh512:
            _, hdr = self.get_chunk_bytearray(self.RDH512_TAG)
            array_id = PyBinHeader.construct_from_bytearray(hdr).array_id
        elif has_icp:
            _, icp = self.get_chunk_bytearray(self.ICP_TAG)
            array_id = ImageCalibrationParamsReader.construct_from_bytearray(icp).array_id

        # Reconfigurable arrays (e.g. the 512 flat MEA, array_id 9999) store the
        # electrode coordinates directly in the .globals file, so no external
        # geometry package is needed. Prefer in-file coords whenever present.
        reconfig_id = (FakeArrayID.BOARD_ID_RECONFIGURABLE
                       if FakeArrayID is not None else BOARD_ID_RECONFIGURABLE)

        if array_id == reconfig_id or self.has_tag(self.ELECTRODEMAP_TAG):
            if not self.has_tag(self.ELECTRODEMAP_TAG):
                raise AssertionError("Reconfigurable map not in globals file")
            coords_with_ttl = self.read_chunk_pair_float_array_dtype(self.ELECTRODEMAP_TAG)
            disconnected = (elmap.get_disconnected_electrode_set_by_array_id(reconfig_id)
                            if elmap is not None else set())
            return coords_with_ttl[1:, :], disconnected

        # Standard Litke array: coordinates are NOT in the file; need the table.
        if elmap is None:
            raise ImportError(
                "array_id {0} is a standard Litke array whose coordinates are not "
                "stored in the .globals file; the 'electrode_map' package is required "
                "but is not importable.".format(array_id))
        return (elmap.get_litke_array_coordinates_by_array_id(array_id),
                elmap.get_disconnected_electrode_set_by_array_id(array_id))


class ImageCalibrationParamsReader:
    """Minimal stub kept only for array_id extraction from the ICP chunk."""
    def __init__(self, array_id):
        self.array_id = array_id

    @staticmethod
    def construct_from_bytearray(b: bytes) -> 'ImageCalibrationParamsReader':
        bs = memoryview(b)
        off = N_BYTES_64BIT * 2 + N_BYTES_64BIT * 2 + N_BYTES_BOOLEAN * 2 + N_BYTES_64BIT
        array_id = struct.unpack('>i', bs[off:off + N_BYTES_32BIT])[0]
        return ImageCalibrationParamsReader(array_id)


# ============================================================================
# EI reader.  Replaces vcext.unpack_ei_from_array with one reshape.
# ============================================================================

# Vision neuron ids are small positive ints. A cid above this is almost
# always float payload bytes interpreted as >I — the signature of a
# wrong record stride (globals map 512 vs .ei payload 519).
_MAX_PLAUSIBLE_VISION_CELL_ID = 100_000


def _ei_record_nbytes(n_payload_channels: int, n_samples: int) -> int:
    """One cell: 8-byte (id, nspikes) header + interleaved (value, error) f32."""
    return (N_BYTES_32BIT * 2
            + 2 * N_BYTES_32BIT * n_samples * n_payload_channels)


def _resolve_ei_payload_channels(file_size: int, header_size: int,
                                 n_samples: int, array_id: int,
                                 n_map_electrodes: int) -> int:
    """Channel count *written in the .ei*, including the TTL row.

    Payload width and plot coordinates are not the same thing. kilosort4
    converters have been seen to write a 512-electrode .globals next to a
    519-wide .ei (header array_id 1551). Using the globals map for the
    stride then walks the file through the waveform and treats floats as
    cell ids.

    Prefer the .ei header's array_id when that width divides the file;
    otherwise the globals map; otherwise the two standard Litke widths.
    """
    candidates = []
    if elmap is not None:
        try:
            n_from_id = elmap.get_litke_array_coordinates_by_array_id(
                array_id).shape[0] + 1
            candidates.append(n_from_id)
        except Exception:
            pass
    if n_map_electrodes > 0:
        candidates.append(int(n_map_electrodes) + 1)
    candidates.extend((520, 513))

    body = file_size - header_size
    seen = set()
    dividing = []
    for n_ch in candidates:
        if n_ch in seen or n_ch <= 0:
            continue
        seen.add(n_ch)
        rec = _ei_record_nbytes(n_ch, n_samples)
        if rec > 0 and body >= rec and body % rec == 0:
            dividing.append(n_ch)

    if dividing:
        return dividing[0]
    return max(int(n_map_electrodes) + 1, 1)


class EIReader:
    def __init__(self, analysis_folder_path, dataset_name,
                 ei_extension='ei', globals_extension='globals'):
        assert os.path.isdir(analysis_folder_path)
        ei_path = os.path.join(analysis_folder_path,
                               "{0}.{1}".format(dataset_name, ei_extension))
        assert os.path.isfile(ei_path), "EI file {0} missing".format(ei_path)

        num_bytes_in_file = os.path.getsize(ei_path)
        self.ei_fp = open(ei_path, 'rb')

        with GlobalsFileReader(analysis_folder_path, dataset_name,
                               globals_extension=globals_extension) as g:
            self.electrode_map, self.disconnected_electrodes = g.get_electrode_map()

        self.nl_points, self.nr_points, self.array_id = \
            struct.unpack('>III', self.ei_fp.read(N_BYTES_32BIT * 3))
        self.n_samples_per_ei = self.nl_points + self.nr_points + 1
        self.header_size = self.ei_fp.tell()

        # Shape comes from the .ei file, not from the globals map.
        self.n_payload_channels = _resolve_ei_payload_channels(
            num_bytes_in_file, self.header_size, self.n_samples_per_ei,
            self.array_id, self.electrode_map.shape[0],
        )
        self.n_electrodes = self.n_payload_channels - 1

        # Plot coordinates must match the payload. kilosort4 converters
        # write a 512-row .globals next to a 519-wide .ei; using that map
        # makes the EI panel refuse to draw (ei=519, positions=512).
        if (self.electrode_map.shape[0] != self.n_electrodes
                and elmap is not None):
            try:
                coords = elmap.get_litke_array_coordinates_by_array_id(
                    self.array_id)
                if coords.shape[0] == self.n_electrodes:
                    logger.info(
                        "EI payload is %d channels but .globals map is %d; "
                        "using Litke coordinates for array_id %d",
                        self.n_electrodes,
                        self.electrode_map.shape[0],
                        self.array_id,
                    )
                    self.electrode_map = coords
            except Exception:
                logger.debug(
                    "Could not replace mismatched electrode map for array_id %s",
                    self.array_id,
                    exc_info=True,
                )

        # per electrode per sample: (value, error) f32 pair; includes TTL
        self.num_bytes_per_ei = (2 * N_BYTES_32BIT * self.n_samples_per_ei
                                 * self.n_payload_channels)

        # The seek table, from cache when one matches this exact .ei file.
        #
        # Building it means one 8-byte read per record, and records are a whole
        # EI apart (~825 KB), so it is ~700 scattered reads over 600 MB. Free on
        # a local disk; over CIFS each is an SMB round trip at ~9 ms and the
        # scan alone was 6.5 s of a 19.5 s open. See ei_index_cache — the table
        # is ~25 KB and depends on nothing but this file.
        cached = ei_index_cache.load(ei_path)
        if cached is not None:
            self.cell_id_to_offset, self.cell_id_to_nspikes = cached
            logger.debug("EIReader: seek table for %s came from cache (%d cells)",
                         ei_path, len(self.cell_id_to_offset))
            return

        self.cell_id_to_offset = {}
        self.cell_id_to_nspikes = {}
        step = self.num_bytes_per_ei + N_BYTES_32BIT * 2
        off = self.header_size
        n_dropped = 0
        while off + N_BYTES_32BIT * 2 <= num_bytes_in_file:
            self.ei_fp.seek(off, 0)
            cid, nspikes = struct.unpack('>II', self.ei_fp.read(N_BYTES_32BIT * 2))
            if 0 < cid <= _MAX_PLAUSIBLE_VISION_CELL_ID:
                self.cell_id_to_nspikes[cid] = nspikes
                self.cell_id_to_offset[cid] = off + N_BYTES_32BIT * 2
            else:
                n_dropped += 1
            off += step
        if n_dropped:
            logger.warning(
                "EIReader: dropped %d implausible cell ids from %s "
                "(wrong stride or corrupt file); kept %d",
                n_dropped, ei_path, len(self.cell_id_to_offset),
            )

        # Only a clean scan is worth keeping. A file that produced dropped ids
        # was misread — caching that would make one bad guess permanent.
        if not n_dropped:
            ei_index_cache.save(
                ei_path, self.cell_id_to_offset, self.cell_id_to_nspikes
            )

    def get_ei_for_cell_id(self, cell_id: int) -> EIContainer:
        assert cell_id in self.cell_id_to_offset, \
            "cell id {0} not in EI file".format(cell_id)
        self.ei_fp.seek(self.cell_id_to_offset[cell_id], 0)
        raw = self.ei_fp.read(self.num_bytes_per_ei)

        # interleaved (value, error) f32 BE -> (n_elec_with_ttl, n_samples, 2)
        arr = np.frombuffer(raw, dtype=BE_F4).reshape(
            self.n_payload_channels, self.n_samples_per_ei, 2)
        full_ei = np.ascontiguousarray(arr[:, :, 0], dtype=np.float32)
        ei_err = np.ascontiguousarray(arr[:, :, 1], dtype=np.float32)

        # drop TTL channel (row 0), meaningless in the EI
        return EIContainer(full_ei[1:, :], ei_err[1:, :],
                           self.nl_points, self.nr_points,
                           self.n_samples_per_ei,
                           self.cell_id_to_nspikes[cell_id])

    def get_all_eis_by_cell_id(self) -> Dict[int, EIContainer]:
        return {cid: self.get_ei_for_cell_id(cid) for cid in self.cell_id_to_offset}

    def get_electrode_map(self) -> np.ndarray:
        return self.electrode_map

    def get_disconnected_electrodes(self) -> Set[int]:
        return self.disconnected_electrodes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.ei_fp.close()

    def close(self):
        self.ei_fp.close()


# ============================================================================
# Parameters reader.  Replaces vcext.unpack_64bit_float_from_bytearray.
# ============================================================================

class ParametersFileReader:
    PARAMS_TAG_CONST_DOUBLE = 3
    PARAMS_TAG_CONST_DOUBLE_ARRAY = 4
    PARAMS_TAG_CONST_STRING = 5

    def __init__(self, analysis_folder_path, dataset_name, parameters_extension='params'):
        assert os.path.isdir(analysis_folder_path)
        path = os.path.join(analysis_folder_path,
                            "{0}.{1}".format(dataset_name, parameters_extension))
        assert os.path.isfile(path), "Params file {0} missing".format(path)
        self.params_fp = open(path, 'rb')

        self.n_cols, self.n_rows, self.n_rows_max = \
            struct.unpack(">III", self.params_fp.read(N_BYTES_32BIT * 3))

        self.column_names, self.column_types = [], []
        for _ in range(self.n_cols):
            ln = struct.unpack(">I", self.params_fp.read(N_BYTES_32BIT))[0]
            self.column_names.append(self.params_fp.read(ln).decode('utf-8'))
            ln = struct.unpack(">I", self.params_fp.read(N_BYTES_32BIT))[0]
            self.column_types.append(self.params_fp.read(ln).decode('utf-8'))

        # seek table: (col,row) -> absolute offset
        n = self.n_rows * self.n_cols
        raw_offsets = np.frombuffer(
            self.params_fp.read(N_BYTES_32BIT * n), dtype=BE_U4).astype(np.int64)
        base = int(raw_offsets[0])  # offset of (0,0)
        rel = raw_offsets - base    # relative to data section start

        self.params_fp.seek(base, 0)
        data = self.params_fp.read()

        self.col_row_to_arbitrary_data = {}
        k = 0
        for j in range(self.n_rows):
            for i in range(self.n_cols):
                self.col_row_to_arbitrary_data[(i, j)] = \
                    self._read_field(data, int(rel[k]))
                k += 1

    def get_all_field_names(self) -> List[str]:
        return self.column_names

    def _read_field(self, buf: bytes, idx: int) -> Any:
        temp = struct.unpack('>H', buf[idx:idx + N_BYTES_16BIT])[0]
        idx += N_BYTES_16BIT
        if temp == 0xFFFF:
            return None
        tag_id = temp >> 6
        length = temp & 0x3F
        if length == 0x3F:
            length = struct.unpack('>H', buf[idx:idx + N_BYTES_16BIT])[0]
            idx += N_BYTES_16BIT
        elif length == 0x3E:
            length = struct.unpack('>I', buf[idx:idx + N_BYTES_32BIT])[0]
            idx += N_BYTES_32BIT

        if tag_id == self.PARAMS_TAG_CONST_DOUBLE:
            return struct.unpack('>d', buf[idx:idx + N_BYTES_64BIT])[0]
        elif tag_id == self.PARAMS_TAG_CONST_DOUBLE_ARRAY:
            n = struct.unpack('>I', buf[idx:idx + N_BYTES_32BIT])[0]
            idx += N_BYTES_32BIT
            return np.frombuffer(buf[idx:idx + 8 * n], dtype=BE_F8).astype(np.float64)
        elif tag_id == self.PARAMS_TAG_CONST_STRING:
            n = struct.unpack('>I', buf[idx:idx + N_BYTES_32BIT])[0]
            idx += N_BYTES_32BIT
            return buf[idx:idx + n].decode('utf-8')
        raise AssertionError("tag_id {0} in params not valid".format(tag_id))

    def update_visioncelldata_obj(self, vcd: VisionCellDataTable) -> VisionCellDataTable:
        cell_id_col = class_id_col = -1
        for i, name in enumerate(self.column_names):
            if name == VisionFieldNames.CELLID_FIELDNAME:
                cell_id_col = i
            elif name == VisionFieldNames.CLASSID_FIELDNAME:
                class_id_col = i
        assert cell_id_col != -1, "No cell ids in params file"
        assert class_id_col != -1, "No cell class ids in params file"

        for j in range(self.n_rows):
            cell_id = int(round(self.col_row_to_arbitrary_data[(cell_id_col, j)]))
            vcd.update_data_for_cell_id_and_field_name(
                cell_id, VisionFieldNames.CELLID_FIELDNAME, cell_id)

            cell_type = self.col_row_to_arbitrary_data[(class_id_col, j)]
            if cell_type.startswith('All/'):
                cell_type = cell_type.replace('All/', '', 1)
            if cell_type == 'All':
                cell_type = 'Unclassified'
            cell_type = cell_type.replace('/', ' ').replace('-', ' ')
            vcd.update_data_for_cell_id_and_field_name(
                cell_id, VisionFieldNames.CLASSID_FIELDNAME, cell_type)

            for i, name in enumerate(self.column_names):
                if i not in (cell_id_col, class_id_col):
                    vcd.update_data_for_cell_id_and_field_name(
                        cell_id, name, self.col_row_to_arbitrary_data[(i, j)])
        return vcd

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.params_fp.close()

    def close(self):
        self.params_fp.close()


# ============================================================================
# STA reader.  Replaces vcppext.unpack_rgb_sta with a structured-dtype view.
# Layout per STA: 12-byte top header (f64 refresh, i32 depth) then, per frame,
# 16-byte header (i32 w, i32 h, f64 stixel) + w*h stixels of 6 f32
# (rv, re, gv, ge, bv, be), all big-endian.
# ============================================================================

class STAReader:
    FILE_HEADER_LENGTH_BYTES = 164
    NEURON_ID_UNUSED = -2147483648
    NUM_BYTES_PER_STIXEL = N_BYTES_32BIT * 6

    def __init__(self, analysis_folder_path, dataset_name,
                 sta_extension='sta', read_chunk_size=100):
        assert os.path.isdir(analysis_folder_path)
        path = os.path.join(analysis_folder_path,
                            "{0}.{1}".format(dataset_name, sta_extension))
        assert os.path.isfile(path), "{0} is not a file".format(path)
        self.sta_fp = open(path, 'rb')

        (self.version, self.num_entries, self.height, self.width, self.depth) = \
            struct.unpack(">iiiii", self.sta_fp.read(N_BYTES_32BIT * 5))
        (self.stixel_size, self.refresh_time, self.sta_offset) = \
            struct.unpack(">ddi", self.sta_fp.read(N_BYTES_64BIT * 2 + N_BYTES_32BIT))

        self.sta_fp.seek(self.FILE_HEADER_LENGTH_BYTES, 0)
        self.cell_id_to_byte_offset = {}
        for _ in range(self.num_entries):
            cid, offset = struct.unpack(">iq", self.sta_fp.read(N_BYTES_32BIT + N_BYTES_64BIT))
            if cid != self.NEURON_ID_UNUSED:
                self.cell_id_to_byte_offset[cid] = offset

        self.n_bytes_per_sta = (self.NUM_BYTES_PER_STIXEL * self.depth * self.height * self.width
                                + self.depth * (N_BYTES_32BIT * 2 + N_BYTES_64BIT)
                                + N_BYTES_64BIT + N_BYTES_32BIT)

        b2c = {v: k for k, v in self.cell_id_to_byte_offset.items()}
        self.sorted_jt_byte_offsets = sorted(b2c.keys())
        self.cell_id_in_file_order = [b2c[v] for v in self.sorted_jt_byte_offsets]
        self.n_cells_per_chunk = read_chunk_size

        # structured dtype: one full STA payload after the 12-byte top header
        frame_px = self.width * self.height
        self._frame_dtype = np.dtype([
            ('hdr', 'V16'),                       # i32 w, i32 h, f64 stixel
            ('px', BE_F4, (frame_px, 6)),         # rv,re,gv,ge,bv,be per stixel
        ])

    def _unpack_single_sta_from_buffer(self, sta_bytearray: bytes) -> STAContainer:
        # top header is 12 bytes (f64 refresh + i32 depth); frames follow
        frames = np.frombuffer(sta_bytearray, dtype=self._frame_dtype,
                               count=self.depth, offset=12)
        px = frames['px']                                   # (depth, w*h, 6)
        px = px.reshape(self.depth, self.width, self.height, 6)
        # original returns (width, height, depth) per channel
        chans = [np.ascontiguousarray(np.transpose(px[:, :, :, c], (1, 2, 0)),
                                      dtype=np.float32) for c in range(6)]
        rv, re_, gv, ge, bv, be = chans
        return STAContainer(self.stixel_size, self.refresh_time, self.sta_offset,
                            rv, re_, gv, ge, bv, be)

    def get_sta_for_cell_id(self, cell_id: int) -> STAContainer:
        assert cell_id in self.cell_id_to_byte_offset, \
            "Cell id {0} does not have STA".format(cell_id)
        self.sta_fp.seek(self.cell_id_to_byte_offset[cell_id], 0)
        return self._unpack_single_sta_from_buffer(self.sta_fp.read(self.n_bytes_per_sta))

    def green_peak_to_rms(self, cell_id: int) -> float:
        """Peak-to-RMS of the green plane only.

        The quality column only needs this scalar. Unpacking the other five
        STA planes (red/blue plus three error maps) is wasted work during
        the all-cell sweep.
        """
        assert cell_id in self.cell_id_to_byte_offset, \
            "Cell id {0} does not have STA".format(cell_id)
        self.sta_fp.seek(self.cell_id_to_byte_offset[cell_id], 0)
        raw = self.sta_fp.read(self.n_bytes_per_sta)
        frames = np.frombuffer(
            raw, dtype=self._frame_dtype, count=self.depth, offset=12
        )
        gv = np.asarray(frames["px"][..., 2], dtype=np.float32)
        if gv.size == 0:
            return float("nan")
        gv = gv - gv.mean()
        rms = float(np.sqrt((gv * gv).mean()))
        if rms <= 0 or not np.isfinite(rms):
            return float("nan")
        return float(np.abs(gv).max() / rms)

    def get_all_stas_by_cell_id(self) -> Dict[int, STAContainer]:
        return {cid: self.get_sta_for_cell_id(cid) for cid in self.cell_id_to_byte_offset}

    def chunked_load_all_stas(self) -> Dict[int, STAContainer]:
        out = {}
        n = len(self.cell_id_in_file_order)
        for i in range(0, n, self.n_cells_per_chunk):
            top = min(i + self.n_cells_per_chunk, n)
            offs = np.array(self.sorted_jt_byte_offsets[i:top])
            rel = list(offs - offs[0])
            ids = self.cell_id_in_file_order[i:top]
            n_read = (offs[-1] - offs[0]) + self.n_bytes_per_sta
            self.sta_fp.seek(offs[0], 0)
            chunk = self.sta_fp.read(int(n_read))
            for cid, r in zip(ids, rel):
                out[cid] = self._unpack_single_sta_from_buffer(chunk[r:r + self.n_bytes_per_sta])
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.sta_fp.close()

    def close(self):
        self.sta_fp.close()


# ============================================================================
# Neurons reader.  Replaces vcext.unpack_32bit_integers_from_bytearray.
# ============================================================================

class NeuronsReader:
    HEADER_LENGTH_BYTES = (4 * N_BYTES_32BIT) + N_BYTES_64BIT + 128
    TTL_ID = -1
    IS_EMPTY = -2147483648

    def __init__(self, analysis_folder_path, dataset_name, neuron_extension='neurons'):
        assert os.path.isdir(analysis_folder_path)
        path = os.path.join(analysis_folder_path,
                            "{0}.{1}".format(dataset_name, neuron_extension))
        assert os.path.isfile(path), "{0} is not a file".format(path)
        self.neuron_fp = open(path, 'rb')

        (self.version, self.num_slots_in_file, self.n_samples, self.sample_freq) = \
            struct.unpack(">iiii", self.neuron_fp.read(N_BYTES_32BIT * 4))

        self.map_neuron_id_to_memory_offset = {}
        self.map_neuron_id_to_initial_electrode = {}
        self.neuron_fp.seek(self.HEADER_LENGTH_BYTES, 0)

        ttl_id, should_be_zero, seek_off = \
            struct.unpack(">iiq", self.neuron_fp.read(N_BYTES_32BIT * 2 + N_BYTES_64BIT))
        assert ttl_id == self.TTL_ID and should_be_zero == 0
        self.map_neuron_id_to_memory_offset[self.TTL_ID] = seek_off

        for _ in range(1, self.num_slots_in_file):
            nid, elec, seek_off = \
                struct.unpack(">iiq", self.neuron_fp.read(N_BYTES_32BIT * 2 + N_BYTES_64BIT))
            if elec == self.IS_EMPTY or elec < 0:
                continue
            self.map_neuron_id_to_memory_offset[nid] = seek_off
            self.map_neuron_id_to_initial_electrode[nid] = elec

    def get_spike_sample_nums_for_neuron(self, neuron_id: int) -> np.ndarray:
        assert neuron_id in self.map_neuron_id_to_memory_offset, \
            "Neuron id {0} not in neurons file".format(neuron_id)
        self.neuron_fp.seek(self.map_neuron_id_to_memory_offset[neuron_id], 0)
        n = struct.unpack('>i', self.neuron_fp.read(N_BYTES_32BIT))[0]
        raw = self.neuron_fp.read(N_BYTES_32BIT * n)
        return np.frombuffer(raw, dtype=BE_I4).astype(np.int32)

    def get_TTL_times(self) -> np.ndarray:
        return self.get_spike_sample_nums_for_neuron(self.TTL_ID)

    def get_identifier_electrode_for_neuron(self, neuron_id: int) -> int:
        return self.map_neuron_id_to_initial_electrode[neuron_id]

    def get_identifier_electrodes_for_all_real_neurons(self) -> Dict[int, int]:
        return self.map_neuron_id_to_initial_electrode

    def get_spike_sample_nums_for_all_real_neurons(self) -> Dict[int, np.ndarray]:
        return {nid: self.get_spike_sample_nums_for_neuron(nid)
                for nid in self.map_neuron_id_to_memory_offset if nid != self.TTL_ID}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.neuron_fp.close()

    def close(self):
        self.neuron_fp.close()


# ============================================================================
# WRITERS
# ============================================================================

def _pack_be_i4(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=BE_I4).tobytes()


def _pack_be_f4(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=BE_F4).tobytes()


class NeuronsFileWriter:
    """Writes a .neurons file. ttl_times + {cell_id: spike_samples}."""
    TTL_ID = -1
    IS_EMPTY = -2147483648
    HEADER_LEN = (4 * N_BYTES_32BIT) + N_BYTES_64BIT + 128

    def __init__(self, folder, dataset_name, n_samples, sample_freq,
                 neuron_extension='neurons', version=32):
        self.path = os.path.join(folder, "{0}.{1}".format(dataset_name, neuron_extension))
        self.n_samples = int(n_samples)
        self.sample_freq = int(sample_freq)
        self.version = int(version)

    def write(self, ttl_times: np.ndarray,
              spike_times_by_cell_id: Dict[int, np.ndarray],
              seed_electrode_by_cell_id: Optional[Dict[int, int]] = None) -> None:
        seed_electrode_by_cell_id = seed_electrode_by_cell_id or {}
        cell_ids = list(spike_times_by_cell_id.keys())
        n_slots = len(cell_ids) + 1  # +1 for TTL

        seek_entry_size = N_BYTES_32BIT * 2 + N_BYTES_64BIT
        data_start = self.HEADER_LEN + n_slots * seek_entry_size

        # compute data offsets: each block = i32 count + count*i32
        offsets = {}
        cursor = data_start
        offsets[self.TTL_ID] = cursor
        cursor += N_BYTES_32BIT + N_BYTES_32BIT * len(ttl_times)
        for cid in cell_ids:
            offsets[cid] = cursor
            cursor += N_BYTES_32BIT + N_BYTES_32BIT * len(spike_times_by_cell_id[cid])

        with open(self.path, 'wb') as fp:
            fp.write(struct.pack(">iiii", self.version, n_slots,
                                 self.n_samples, self.sample_freq))
            fp.write(b'\x00' * N_BYTES_64BIT)   # 8-byte unused field
            fp.write(b'\x00' * 128)
            # seek table: TTL first (electrode field must be 0)
            fp.write(struct.pack(">iiq", self.TTL_ID, 0, offsets[self.TTL_ID]))
            for cid in cell_ids:
                elec = int(seed_electrode_by_cell_id.get(cid, 1))
                fp.write(struct.pack(">iiq", int(cid), elec, offsets[cid]))
            # data blocks
            fp.write(struct.pack(">i", len(ttl_times)))
            fp.write(_pack_be_i4(np.asarray(ttl_times)))
            for cid in cell_ids:
                st = np.asarray(spike_times_by_cell_id[cid])
                fp.write(struct.pack(">i", len(st)))
                fp.write(_pack_be_i4(st))


class GlobalsFileWriter:
    """Writes a minimal .globals with an array-id tag (and optional reconfig map)."""
    ARRAYID_TAG = 6
    ELECTRODEMAP_TAG = 7
    FILE_ID = 0xABCDEF0123456789  # arbitrary 64-bit file id

    def __init__(self, folder, dataset_name, globals_extension='globals'):
        self.path = os.path.join(folder, "{0}.{1}".format(dataset_name, globals_extension))

    @staticmethod
    def _chunk(tag: int, payload: bytes) -> bytes:
        # (tag:u32, reserved:u32=0, size:u32) + payload
        return struct.pack(">III", tag, 0, len(payload)) + payload

    def write(self, array_id: int,
              electrode_coords_with_ttl: Optional[np.ndarray] = None) -> None:
        with open(self.path, 'wb') as fp:
            fp.write(struct.pack(">Q", self.FILE_ID))
            fp.write(self._chunk(self.ARRAYID_TAG, struct.pack(">i", int(array_id))))
            if electrode_coords_with_ttl is not None:
                c = np.ascontiguousarray(electrode_coords_with_ttl, dtype=np.float64)
                n = c.shape[0]
                payload = (_pack_be_f4(c[:, 0]) + _pack_be_f4(c[:, 1]))
                fp.write(self._chunk(self.ELECTRODEMAP_TAG, payload))


class EIFileWriter:
    """Writes a .ei file from {cell_id: (ei[n_elec_with_ttl, n_samp],
    ei_error[...], n_spikes)}. Rows must INCLUDE the TTL row at index 0."""

    def __init__(self, folder, dataset_name, nl_points, nr_points, array_id,
                 ei_extension='ei'):
        self.path = os.path.join(folder, "{0}.{1}".format(dataset_name, ei_extension))
        self.nl_points = int(nl_points)
        self.nr_points = int(nr_points)
        self.array_id = int(array_id)
        self.n_samples = self.nl_points + self.nr_points + 1

    def write(self, ei_by_cell_id: Dict[int, Tuple[np.ndarray, np.ndarray, int]]) -> None:
        with open(self.path, 'wb') as fp:
            fp.write(struct.pack(">III", self.nl_points, self.nr_points, self.array_id))
            for cid, (ei, ei_err, n_spikes) in ei_by_cell_id.items():
                fp.write(struct.pack(">II", int(cid), int(n_spikes)))
                # interleave (value, error) per elec per sample
                stacked = np.stack([np.asarray(ei, dtype=np.float32),
                                    np.asarray(ei_err, dtype=np.float32)], axis=-1)
                fp.write(_pack_be_f4(stacked))


class STAFileWriter:
    """Writes a .sta file from {cell_id: STAContainer}. Channels are
    (width, height, depth) as returned by STAReader."""
    FILE_HEADER_LENGTH_BYTES = 164
    NUM_BYTES_PER_STIXEL = N_BYTES_32BIT * 6

    def __init__(self, folder, dataset_name, width, height, depth,
                 stixel_size, refresh_time, sta_offset, sta_extension='sta',
                 version=32):
        self.path = os.path.join(folder, "{0}.{1}".format(dataset_name, sta_extension))
        self.width = int(width)
        self.height = int(height)
        self.depth = int(depth)
        self.stixel_size = float(stixel_size)
        self.refresh_time = float(refresh_time)
        self.sta_offset = int(sta_offset)
        self.version = int(version)

    def _n_bytes_per_sta(self) -> int:
        return (self.NUM_BYTES_PER_STIXEL * self.depth * self.height * self.width
                + self.depth * (N_BYTES_32BIT * 2 + N_BYTES_64BIT)
                + N_BYTES_64BIT + N_BYTES_32BIT)

    def _pack_one(self, sta: STAContainer) -> bytes:
        # (width,height,depth) -> (depth,width,height) then stack 6 channels
        chans = [np.transpose(getattr(sta, name), (2, 0, 1))
                 for name in ('red', 'red_error', 'green', 'green_error',
                              'blue', 'blue_error')]
        px = np.stack(chans, axis=-1)            # (depth, w, h, 6)
        px = px.reshape(self.depth, self.width * self.height, 6)

        out = bytearray()
        out += struct.pack(">di", self.refresh_time, self.depth)  # 12-byte top header
        for f in range(self.depth):
            out += struct.pack(">iid", self.width, self.height, self.stixel_size)
            out += _pack_be_f4(px[f])
        return bytes(out)

    def write(self, sta_by_cell_id: Dict[int, STAContainer]) -> None:
        cell_ids = list(sta_by_cell_id.keys())
        num_entries = len(cell_ids)
        n_bytes_per_sta = self._n_bytes_per_sta()

        seek_entry_size = N_BYTES_32BIT + N_BYTES_64BIT
        data_start = self.FILE_HEADER_LENGTH_BYTES + num_entries * seek_entry_size

        with open(self.path, 'wb') as fp:
            # header: version,num_entries,height,width,depth then dd i
            fp.write(struct.pack(">iiiii", self.version, num_entries,
                                 self.height, self.width, self.depth))
            fp.write(struct.pack(">ddi", self.stixel_size, self.refresh_time,
                                 self.sta_offset))
            # pad to FILE_HEADER_LENGTH_BYTES
            fp.write(b'\x00' * (self.FILE_HEADER_LENGTH_BYTES - fp.tell()))
            # seek table
            cursor = data_start
            for cid in cell_ids:
                fp.write(struct.pack(">iq", int(cid), cursor))
                cursor += n_bytes_per_sta
            # data
            for cid in cell_ids:
                fp.write(self._pack_one(sta_by_cell_id[cid]))