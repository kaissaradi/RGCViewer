import logging

from qtpy.QtCore import QThread

from src.gui.workers.workers import StandardPlotsWorker


class FailingStandardPlotDataManager:
    def __init__(self):
        self.requested_clusters = []

    def load_persisted_caches(self):
        pass

    def get_standard_plot_data(self, cluster_id):
        self.requested_clusters.append(cluster_id)
        raise RuntimeError("missing raw data")


def test_standard_plots_worker_marks_cluster_finished_after_compute_error(monkeypatch, caplog):
    dm = FailingStandardPlotDataManager()
    worker = StandardPlotsWorker(dm)
    finished_clusters = []
    errors = []

    worker.finished_cluster.connect(finished_clusters.append)
    worker.error.connect(errors.append)
    worker.add_to_queue(42)

    def stop_worker(_ms):
        worker.stop()

    monkeypatch.setattr(QThread, "msleep", stop_worker)

    with caplog.at_level(logging.ERROR, logger="src.gui.workers.workers"):
        worker.run()

    assert dm.requested_clusters == [42]
    assert finished_clusters == [42]
    assert errors == ["Background precompute failed for cluster 42: missing raw data"]
    assert "Failed to compute standard plots for cluster 42" in caplog.text
