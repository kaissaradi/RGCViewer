"""One raw read at a time, latest request wins (no 2 s wait-and-terminate on a cell change)."""

from types import SimpleNamespace


def test_busy_reads_queue_the_latest_and_start_it_when_done(qtbot, monkeypatch):
    from src.gui.main_window import MainWindow
    w = MainWindow()
    try:
        started = []
        monkeypatch.setattr(w, "_start_feature_worker", lambda cid: started.append(cid))
        monkeypatch.setattr(w, "_feature_thread_alive", lambda: True)
        w.data_manager = SimpleNamespace(dat_path="x.bin", get_lightweight_features=lambda c: None)
        selected = {"id": 3}
        monkeypatch.setattr(w, "_get_selected_cluster_id", lambda: selected["id"])
        w._feature_busy = True
        for cid in (1, 2, 3):                       # three quick selections while a read runs
            w._pending_cluster_id = cid
            if w._feature_busy and w._feature_thread_alive():
                w._feature_pending = cid
        assert started == [] and w._feature_pending == 3
        w._feature_worker_done()
        assert started == [3]
        # a finished read whose pending cell is no longer selected starts nothing
        started.clear()
        w._feature_pending, selected["id"] = 7, 8
        w._feature_worker_done()
        assert started == []
    finally:
        w.data_manager = None
        w.close()
        w.deleteLater()
