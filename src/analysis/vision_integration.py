from pathlib import Path
import logging
import threading
logger = logging.getLogger(__name__)

# Guard import of visionloader so the app can run without it installed
try:
    import src.analysis.visionloader as vl
    VISION_LOADER_AVAILABLE = True
except Exception:
    vl = None
    VISION_LOADER_AVAILABLE = False
    logger.info("'visionloader' not available; vision integration disabled")

MAX_STA_CACHE_CELLS = 500

# Far smaller than the STA budget on purpose — see LazyEIDict.__init__.
MAX_EI_CACHE_CELLS = 32


class LazySTADict:
    """
    A dictionary-like wrapper that keeps the .sta file pointer open 
    and reads STA movies directly from disk on-demand. 
    Includes a lightweight FIFO cache for smooth UI scrolling.
    """
    def __init__(self, vision_dir, dataset_name):
        self.vision_dir = vision_dir
        self.dataset_name = dataset_name
        self.reader = vl.STAReader(str(vision_dir), dataset_name)
        # Filter out non-integer or obviously bad keys (NaN cell IDs from corrupt rows).
        raw_keys = list(self.reader.cell_id_to_byte_offset.keys())
        self.keys_list = []
        _skipped = 0
        for k in raw_keys:
            try:
                int_k = int(k)
                if int_k > 0:          # Vision IDs are always positive
                    self.keys_list.append(int_k)
                else:
                    _skipped += 1
            except (TypeError, ValueError):
                _skipped += 1
        if _skipped:
            logger.warning(
                "LazySTADict: skipped %d invalid cell IDs (NaN/non-integer) from %s",
                _skipped, vision_dir,
            )
        self._keys_set = set(self.keys_list)
        
        # --- NEW: LRU Cache to fix UI Scrolling lag ---
        self._cache = {}
        self._cache_keys = []
        self._max_cache = min(MAX_STA_CACHE_CELLS, max(200, len(self.keys_list)))
        self._cache_lock = threading.Lock()
        
        # Thread-local storage for readers to avoid file handle race conditions
        self._local = threading.local()
        self._creator_thread = threading.current_thread()
        self._all_readers = [self.reader]
        self._readers_lock = threading.Lock()
        
    def __contains__(self, key):
        keys_set = getattr(self, '_keys_set', None)
        if keys_set is None:
            self._keys_set = set(getattr(self, 'keys_list', []))
            keys_set = self._keys_set
        try:
            return int(key) in keys_set
        except (TypeError, ValueError):
            return False
    
    def get(self, key, default=None):
        """Emulate dict.get() using our internal cache and SSD reader."""
        if key in self:
            return self[key]  # This safely triggers __getitem__ and the RAM cache
        return default
        
    def _thread_local_reader(self):
        """One STAReader per thread so workers do not share a file handle."""
        local = getattr(self, '_local', None)
        if local is None:
            self._local = threading.local()
            local = self._local

        local_reader = getattr(local, 'reader', None)
        if local_reader is not None:
            return local_reader

        creator_thread = getattr(self, '_creator_thread', None)
        if creator_thread is not None and threading.current_thread() is creator_thread:
            local_reader = self.reader
        else:
            vision_dir = getattr(self, 'vision_dir', None)
            dataset_name = getattr(self, 'dataset_name', None)
            if vision_dir is not None and dataset_name is not None and VISION_LOADER_AVAILABLE:
                local_reader = vl.STAReader(str(vision_dir), dataset_name)
                readers_lock = getattr(self, '_readers_lock', None)
                if readers_lock is not None:
                    with readers_lock:
                        self._all_readers.append(local_reader)
            else:
                local_reader = self.reader
        local.reader = local_reader
        return local_reader

    def sta_peak_to_rms(self, key):
        """Green-channel peak-to-RMS. Does not unpack or cache the movie."""
        try:
            key = int(key)
        except (TypeError, ValueError):
            return float("nan")
        if key not in self:
            return float("nan")
        reader = self._thread_local_reader()
        if reader is None or not hasattr(reader, "green_peak_to_rms"):
            return float("nan")
        try:
            return float(reader.green_peak_to_rms(key))
        except Exception as e:
            logger.warning("green_peak_to_rms failed for cell %s: %s", key, e)
            return float("nan")

    def __getitem__(self, key):
        try:
            key = int(key)
        except (TypeError, ValueError):
            return None
        # Missing cells are the common case (many KS units have no STA).
        # Do not open a reader / thread pool just to assert.
        if key not in self:
            return None

        # 1. Fast path: check RAM cache under lock (instant dict lookup).
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]

        # 2. Get thread-local reader, or create one for this thread
        local_reader = self._thread_local_reader()

        # 3. Read is outside cache lock, and thread-safe because each thread has its own reader.
        # Guard against BOTH exceptions (corrupt data) AND hangs (blocked seek on a bad
        # byte offset). We run the actual read on a fresh thread and impose a hard timeout.
        # Returning None is safe — all callers check hasattr(sta_data, 'red').
        import concurrent.futures as _cf
        _READ_TIMEOUT_S = 8
        _reader_ref = local_reader  # capture before entering executor

        def _do_read():
            return _reader_ref.get_sta_for_cell_id(key)

        # IMPORTANT: do NOT use `with ThreadPoolExecutor() as pool:` here.
        # Exiting that context manager calls shutdown(wait=True), which blocks
        # until the submitted thread finishes — so a genuinely stuck read
        # (e.g. blocked seek on a corrupt byte offset) would silently undo
        # the timeout below by hanging on the way out of the `with` block.
        # During serial warm-up (ensure_physics_cache, max_workers=1) this
        # is exactly what turned a per-cell 8s budget into an unbounded
        # stall. Shut down with wait=False and deliberately leak the stuck
        # worker thread instead; it exits on its own whenever the
        # underlying read eventually returns (or never, if the process
        # exits first), but it no longer blocks this caller.
        _pool = _cf.ThreadPoolExecutor(max_workers=1)
        _future = _pool.submit(_do_read)
        try:
            sta_data = _future.result(timeout=_READ_TIMEOUT_S)
            _pool.shutdown(wait=False)
        except _cf.TimeoutError:
            logger.warning(
                "get_sta_for_cell_id timed out after %ds for cell %s "
                "(likely corrupt byte offset in .sta file); returning None.",
                _READ_TIMEOUT_S, key,
            )
            _pool.shutdown(wait=False)
            return None
        except Exception as e:
            logger.warning("get_sta_for_cell_id failed for cell %s: %s", key, e)
            _pool.shutdown(wait=False)
            return None

        # 4. Write result under lock with double-check pattern
        with self._cache_lock:
            if key not in self._cache:
                self._cache[key] = sta_data
                self._cache_keys.append(key)
                if len(self._cache_keys) > self._max_cache:
                    oldest = self._cache_keys.pop(0)
                    self._cache.pop(oldest, None)

        return sta_data
        
    def __iter__(self):
        return iter(self.keys_list)
        
    def __len__(self):
        return len(self.keys_list)
        
    def keys(self):
        return self.keys_list
        
    def close(self):
        """Close every open reader. Safe to call more than once."""
        readers = getattr(self, '_all_readers', [])
        if not readers and hasattr(self, 'reader'):
            readers = [self.reader]
        self._all_readers = []
        for r in readers:
            try:
                r.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class LazyEIDict:
    """Read-on-demand mapping of Vision cell id → EIContainer.

    The .ei file is the largest in a preparation — on the 512 array one cell is
    ``513 electrodes x ~201 samples x 2 floats x 4 bytes``, about 825 KB, so a
    900-cell run is roughly 740 MB. Reading all of it at load time cost that
    much RAM and that much I/O on *every* load of the same run, including the
    tenth. Half of it was never wanted either: ``ei_error`` is read by one
    panel, for the one cell on screen.

    ``EIReader`` already builds a complete ``cell_id_to_offset`` seek table in
    its constructor and already exposes a per-cell random read, so this needs
    no reader change — only the choice not to call
    ``get_all_eis_by_cell_id()``. Construction now costs one pass of 8-byte
    header reads instead of the whole file.

    Mirrors LazySTADict, including its thread-local readers: the physics
    warm-up and the panels read EIs from several threads, and one file pointer
    cannot serve concurrent seeks.

    Unlike LazySTADict there is no per-read timeout. That guard exists because
    .sta byte offsets come from a stored seek table that can point anywhere in
    a corrupt file; EI offsets are derived from a fixed stride, so a bad file
    fails the reshape immediately rather than blocking on a wild seek.
    """

    def __init__(self, vision_dir, dataset_name):
        self.vision_dir = vision_dir
        self.dataset_name = dataset_name
        self.reader = vl.EIReader(str(vision_dir), dataset_name)

        # Same defence as LazySTADict: a corrupt row can yield a NaN or
        # non-integer cell id, and Vision ids are always positive.
        self.keys_list = []
        _skipped = 0
        for k in self.reader.cell_id_to_offset:
            try:
                int_k = int(k)
            except (TypeError, ValueError):
                _skipped += 1
                continue
            if int_k > 0:
                self.keys_list.append(int_k)
            else:
                _skipped += 1
        if _skipped:
            logger.warning(
                "LazyEIDict: skipped %d invalid cell IDs (NaN/non-integer) from %s",
                _skipped, vision_dir,
            )
        self._keys_set = set(self.keys_list)

        # Deliberately small. An EI container is ~825 KB, so the 500 cells
        # LazySTADict keeps would be 412 MB — the very cost this class exists
        # to remove. 32 cells is ~26 MB and still covers clicking through a
        # group of cells without re-reading.
        self._cache = {}
        self._cache_keys = []
        self._max_cache = MAX_EI_CACHE_CELLS
        self._cache_lock = threading.Lock()

        self._local = threading.local()
        self._creator_thread = threading.current_thread()
        self._all_readers = [self.reader]
        self._readers_lock = threading.Lock()

    # --- geometry, needed by load_ei_data and the EI panels ----------------

    def get_electrode_map(self):
        return self.reader.get_electrode_map()

    # --- mapping protocol ---------------------------------------------------

    def __contains__(self, key):
        try:
            return int(key) in self._keys_set
        except (TypeError, ValueError):
            return False

    def __len__(self):
        return len(self.keys_list)

    def __iter__(self):
        return iter(self.keys_list)

    def keys(self):
        return list(self.keys_list)

    def values(self):
        return (self[k] for k in self.keys_list)

    def items(self):
        """Stream every cell.

        A generator, not a dict: the callers that walk all cells (Cell Tracer,
        and the cold EI-correlation pass) would otherwise rebuild in RAM
        exactly the eager dict this class replaces.
        """
        return ((k, self[k]) for k in self.keys_list)

    def get(self, key, default=None):
        if key not in self:
            return default
        value = self[key]
        return default if value is None else value

    def __getitem__(self, key):
        key = int(key)

        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]

        reader = self._reader_for_this_thread()
        try:
            ei = reader.get_ei_for_cell_id(key)
        except Exception:
            logger.warning("Could not read the EI for cell %s", key, exc_info=True)
            return None

        with self._cache_lock:
            if key not in self._cache:
                self._cache[key] = ei
                self._cache_keys.append(key)
                if len(self._cache_keys) > self._max_cache:
                    oldest = self._cache_keys.pop(0)
                    self._cache.pop(oldest, None)

        return ei

    def _reader_for_this_thread(self):
        """One open EIReader per calling thread; the creator reuses its own."""
        local = getattr(self, "_local", None)
        if local is None:
            self._local = threading.local()
            local = self._local

        reader = getattr(local, "reader", None)
        if reader is not None:
            return reader

        if threading.current_thread() is getattr(self, "_creator_thread", None):
            reader = self.reader
        else:
            try:
                reader = vl.EIReader(str(self.vision_dir), self.dataset_name)
                with self._readers_lock:
                    self._all_readers.append(reader)
            except Exception:
                # Out of file handles, or the file went away with the mount.
                # The shared reader still works; concurrent seeks on it can
                # only garble a read, and __getitem__ turns that into None.
                logger.warning(
                    "Could not open a per-thread EI reader for %s; "
                    "falling back to the shared one.",
                    self.vision_dir,
                    exc_info=True,
                )
                reader = self.reader

        local.reader = reader
        return reader

    # --- teardown -----------------------------------------------------------

    def close(self):
        """Close every open reader and drop the cached containers."""
        with self._readers_lock:
            readers, self._all_readers = self._all_readers, []
        for r in readers:
            try:
                r.close()
            except Exception:
                pass
        with self._cache_lock:
            self._cache = {}
            self._cache_keys = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def load_vision_data(vision_dir: Path, dataset_name: str):
    """
    Loads all relevant data from Vision files (.ei, .sta, .params, .neurons).
    """
    logger.debug(f"Loading Vision files from: {vision_dir}")

    if not VISION_LOADER_AVAILABLE:
        logger.warning(
            "visionloader is not available; skipping vision load for %s",
            vision_dir,
        )
        return {'ei': None, 'sta': None, 'params': None, 'neurons': None}

    ei_data = None
    sta_data = None
    params_data = None
    neurons_data = None

    try:
        ei_data = load_ei_data(vision_dir, dataset_name)
    except Exception as e:
        logger.warning(f"Could not load EI data: {e}")

    try:
        sta_data = load_sta_data(vision_dir, dataset_name)
    except Exception as e:
        logger.warning(f"Could not load STA data: {e}")

    try:
        neurons_data = load_neurons_data(vision_dir, dataset_name)
    except Exception as e:
        logger.warning(f"Could not load Neurons data: {e}")

    # NOTE: params is loaded LAST and under a hard timeout. Unlike the other
    # Vision readers, ParametersFileReader.__init__ eagerly parses the ENTIRE
    # file synchronously (full seek table + every field, with no validation
    # of the header-derived row/col counts). A truncated or malformed
    # .params file can make that constructor take a very long time (or
    # effectively hang) instead of raising a catchable exception. Loading it
    # last, off-thread, with a timeout ensures a bad params file can never
    # prevent EI/STA/neurons data (which are already loaded above) from
    # being returned to the caller.
    params_data = _load_params_with_timeout(vision_dir, dataset_name)

    logger.info(
        "load_vision_data complete: sta=%s, params=%s, ei=%s",
        "present" if sta_data is not None else "None",
        "present" if params_data is not None else "None",
        "present" if ei_data is not None else "None",
    )

    return {
        'ei': ei_data,
        'sta': sta_data,
        'params': params_data,
        'neurons': neurons_data
    }


def load_ei_data(vision_dir: Path, dataset_name: str):
    """Opens the .ei file for on-demand reads.

    Returns a LazyEIDict rather than a plain dict. The reader stays open for
    the life of the dataset, so DataManager.close() has to close it — the file
    handle is not released by dropping the reference alone.
    """
    if not VISION_LOADER_AVAILABLE:
        return None

    try:
        lazy_eis = LazyEIDict(vision_dir, dataset_name)
        logger.info("Initialized Lazy EI Reader for %d cells", len(lazy_eis))
        return {'ei_data': lazy_eis, 'electrode_map': lazy_eis.get_electrode_map()}
    except FileNotFoundError:
        logger.warning(f"EI file not found in {vision_dir}; skipping EI")
        return None
    except AssertionError as e:
        logger.warning(f"EI file missing/invalid ({e}); skipping EI")
        return None
    except Exception:
        logger.exception("Unexpected error loading EI data")
        return None


def load_sta_data(vision_dir: Path, dataset_name: str):
    """
    Initializes a Lazy Reader for STA data instead of loading into RAM.
    """
    if not VISION_LOADER_AVAILABLE:
        return None

    try:
        # We return the lazy dictionary wrapper instead of executing chunked_load_all_stas()
        lazy_sta_dict = LazySTADict(vision_dir, dataset_name)
        logger.info(f"Initialized Lazy STA Reader for {len(lazy_sta_dict)} cells")
        return lazy_sta_dict
    except FileNotFoundError:
        logger.warning(f"STA file not found in {vision_dir}; skipping STA")
        return None
    except AssertionError as e:
        logger.warning("STA file missing/invalid (%s); skipping STA", e)
        return None
    except Exception:
        logger.exception("Unexpected error initializing Lazy STA Reader")
        return None


_PARAMS_LOAD_TIMEOUT_S = 20


def _load_params_with_timeout(vision_dir: Path, dataset_name: str):
    """
    Runs load_params_data() on a worker thread with a hard timeout.

    ParametersFileReader's constructor parses the whole .params file
    synchronously and trusts the header-derived row/col counts without
    validating them against the file size. A corrupt/truncated/mismatched
    params file can therefore make construction take an extremely long
    time instead of raising. Wrapping it the same way LazySTADict guards
    individual STA reads keeps a bad params file from blocking EI/STA/
    neurons data from ever reaching the caller.
    """
    if not VISION_LOADER_AVAILABLE:
        return None

    import concurrent.futures as _cf

    # IMPORTANT: do NOT use `with ThreadPoolExecutor() as pool:` here.
    # Exiting that context manager calls shutdown(wait=True), which blocks
    # until the submitted thread finishes — so if the read genuinely hangs,
    # leaving the `with` block (even via the TimeoutError path) hangs right
    # back. We shut down with wait=False instead and deliberately leak the
    # stuck worker thread; it will exit on its own whenever the underlying
    # file read finally returns (or never, if the process exits first).
    _pool = _cf.ThreadPoolExecutor(max_workers=1)
    future = _pool.submit(load_params_data, vision_dir, dataset_name)
    try:
        result = future.result(timeout=_PARAMS_LOAD_TIMEOUT_S)
        _pool.shutdown(wait=False)
        return result
    except _cf.TimeoutError:
        logger.warning(
            "load_params_data timed out after %ds for %s/%s "
            "(likely a corrupt/oversized .params file); "
            "continuing without params data.",
            _PARAMS_LOAD_TIMEOUT_S, vision_dir, dataset_name,
        )
        _pool.shutdown(wait=False)
        return None
    except Exception as e:
        logger.warning(f"Could not load Params data: {e}")
        _pool.shutdown(wait=False)
        return None


def load_params_data(vision_dir: Path, dataset_name: str):
    """Loads receptive field parameters from a .params file."""
    if not VISION_LOADER_AVAILABLE:
        return None

    try:
        vcd = vl.VisionCellDataTable()
        with vl.ParametersFileReader(str(vision_dir), dataset_name) as pfr:
            pfr.update_visioncelldata_obj(vcd)
            logger.info(f"Loaded params for {len(vcd.get_cell_ids())} cells")
            return vcd
    except FileNotFoundError:
        logger.warning("Params file not found in %s; skipping params", vision_dir)
        return None
    except AssertionError as e:
        logger.warning("Params file missing/invalid (%s); skipping params", e)
        return None
    except Exception:
        logger.exception("Unexpected error loading params data")
        return None


def load_neurons_data(vision_dir: Path, dataset_name: str):
    """Loads spike times, seed electrodes, and sampling rate from a .neurons file."""
    if not VISION_LOADER_AVAILABLE:
        return None

    try:
        with vl.NeuronsReader(str(vision_dir), dataset_name) as nr:
            spikes_by_id = nr.get_spike_sample_nums_for_all_real_neurons()
            seed_electrodes = nr.get_identifier_electrodes_for_all_real_neurons()
            sampling_rate = nr.sample_freq
            logger.info(f"Loaded .neurons for {len(spikes_by_id)} cells")
            return {
                'spikes_by_id': spikes_by_id, 
                'seed_electrodes': seed_electrodes,
                'sampling_rate': sampling_rate
            }
    except FileNotFoundError:
        logger.error(f"Neurons file not found in {vision_dir}")
        return None
    except Exception:
        logger.exception("Unexpected error loading Neurons data")
        return None
