import logging

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QStackedLayout, QVBoxLayout, QWidget

from ..theme import resolve_theme_colors

logger = logging.getLogger(__name__)

pg.setConfigOptions(antialias=True)


class ChirpPanel(QWidget):
    """
    Displays the chirp-stimulus PSTH for the selected cluster.

    Data is entirely precomputed offline by chirp_analysis.py and loaded
    once into DataManager.chirp_data via load_chirp_data() (called from
    KilosortLoadWorker.run()). This panel only performs a read-only row
    lookup per cluster via DataManager.get_chirp_data_for_cluster() — no
    background worker is spawned per cluster.

    Panels never reference other panels (AGENTS.md §2) — reads DataManager
    only.
    """

    # Regions drawn back-to-front, low z-order, non-interactive.
    PHASE_COLORS = {
        "phase_step_on": (120, 120, 120, 40),
        "phase_step_off": (120, 120, 120, 40),
        "phase_freq_sweep": (80, 140, 220, 35),
        "phase_contrast": (220, 140, 80, 35),
    }

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        colors = resolve_theme_colors(self.main_window.get_current_colors())

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedLayout()
        outer_layout.addLayout(self.stack)

        # ---------------------------------------------------------
        # Page 0: PSTH view
        # ---------------------------------------------------------
        data_page = QWidget()
        data_layout = QVBoxLayout(data_page)
        data_layout.setContentsMargins(4, 4, 4, 4)

        header = QHBoxLayout()
        title = QLabel(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; "
            f"letter-spacing:0.06em;'>CHIRP PSTH</span>"
        )
        header.addWidget(title)
        header.addStretch()
        self.qi_label = QLabel("")
        header.addWidget(self.qi_label)
        data_layout.addLayout(header)

        self.psth_plot = pg.PlotWidget()
        self.psth_plot.setLabel("bottom", "Time (s)")
        self.psth_plot.setLabel("left", "Firing Rate (Hz)")
        self._style_plot(self.psth_plot, colors)

        self._psth_curve = self.psth_plot.plot(
            [], [], pen=pg.mkPen(colors.get("plot_fr", "y"), width=2)
        )

        # Persistent, non-movable phase-boundary regions (created once,
        # repositioned per cluster — same "persistent items" pattern as
        # StandardPlotsPanel's _acg_line / _isi_curve).
        self._phase_regions = {}
        for key, rgba in self.PHASE_COLORS.items():
            region = pg.LinearRegionItem(
                values=(0, 0),
                brush=pg.mkBrush(*rgba),
                movable=False,
            )
            region.setZValue(-10)
            for line in region.lines:
                line.setPen(pg.mkPen(None))
            self.psth_plot.addItem(region)
            self._phase_regions[key] = region

        data_layout.addWidget(self.psth_plot)
        self.stack.addWidget(data_page)

        # ---------------------------------------------------------
        # Page 1: placeholder (no chirp data available / cell not in file)
        # ---------------------------------------------------------
        placeholder_page = QWidget()
        placeholder_layout = QVBoxLayout(placeholder_page)
        self.placeholder_label = QLabel("No chirp data")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_layout.addWidget(self.placeholder_label)
        self.stack.addWidget(placeholder_page)

        self.stack.setCurrentIndex(1)

    def _style_plot(self, plot_widget, colors=None):
        if colors is None:
            colors = resolve_theme_colors(self.main_window.get_current_colors())
        else:
            colors = resolve_theme_colors(colors)

        plot_item = plot_widget.getPlotItem()
        plot_widget.setBackground(colors["bg_panel"])
        plot_item.getAxis("bottom").setPen(pg.mkPen(colors["border_default"]))
        plot_item.getAxis("left").setPen(pg.mkPen(colors["border_default"]))
        plot_item.getAxis("bottom").setTextPen(pg.mkPen(colors["text_secondary"]))
        plot_item.getAxis("left").setTextPen(pg.mkPen(colors["text_secondary"]))
        plot_item.showAxis("top", False)
        plot_item.showAxis("right", False)
        plot_item.showGrid(x=True, y=True, alpha=0.08)
        plot_item.setContentsMargins(8, 8, 8, 8)

    def restyle_plots(self, colors):
        """Called by MainWindow.toggle_theme() for every panel with this method."""
        colors = resolve_theme_colors(colors)
        self._style_plot(self.psth_plot, colors)
        self._psth_curve.setPen(pg.mkPen(colors.get("plot_fr", "y"), width=2))

    def update_all(self, cluster_id):
        """
        Redraw the PSTH for cluster_id.

        Tier 2 ONLY. Never call from MainWindow.update_cluster_views()'s
        Tier 1 block — AGENTS.md §1 Law 2 forbids any panel.update_all()
        call there regardless of how cheap the underlying data access is.
        The underlying lookup itself (DataManager.get_chirp_data_for_cluster)
        is a plain dict/row read — no lock, no disk I/O — but the rebuild of
        this panel's plot items still must happen in _process_selection() /
        _draw_plots() / on_tab_changed(), matching StandardPlotsPanel.
        """
        if cluster_id is None:
            return

        dm = self.main_window.data_manager
        if dm is None:
            return

        data = dm.get_chirp_data_for_cluster(cluster_id)

        if data is None:
            self.stack.setCurrentIndex(1)
            if not getattr(dm, "chirp_available", False):
                self.placeholder_label.setText("No chirp data")
            else:
                self.placeholder_label.setText(
                    f"No chirp response for cluster {cluster_id}"
                )
            self._psth_curve.setData([], [])
            return

        self.stack.setCurrentIndex(0)

        psth = np.asarray(data["psth_mean"])
        bin_size_ms = data["bin_size_ms"]
        n_bins = psth.shape[0]
        t = (np.arange(n_bins) * bin_size_ms) / 1000.0  # seconds

        self._psth_curve.setData(t, psth)

        for key, region in self._phase_regions.items():
            start_bin, end_bin = data[key]
            start_s = (start_bin * bin_size_ms) / 1000.0
            end_s = (end_bin * bin_size_ms) / 1000.0
            region.setRegion((start_s, end_s))

        qi = data.get("quality_index")
        if qi is not None and not np.isnan(qi):
            self.qi_label.setText(f"QI: {qi:.2f}")
        else:
            self.qi_label.setText("QI: n/a")

        self.psth_plot.enableAutoRange()

    def cleanup(self):
        """Explicitly cleanup pyqtgraph resources to prevent memory leaks."""
        if self.psth_plot:
            self.psth_plot.clear()
            if hasattr(self.psth_plot, "close"):
                self.psth_plot.close()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)
