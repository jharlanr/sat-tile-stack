"""
Standalone GUI labeling tool for satellite timestacks.

Pops out a matplotlib window for scrubbing through frames and assigning labels.
No notebook or external GUI framework required — just matplotlib.

Features:
- Scroll wheel to scrub through frames (mouse over the image)
- Per-class probability sliders (25% resolution)
- Label is auto-computed as argmax of probabilities
- Progress bar and pie chart
- Notes field
- Keyboard shortcuts (left/right arrows, enter to submit)

Usage:
    >>> from sat_tile_stack.labeler import Labeler
    >>> labeler = Labeler(
    ...     nc_dir="data/processed/",
    ...     labels_csv="labels/my_labels.csv",
    ... )
    >>> labeler.start()  # pops out a window
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

from sat_tile_stack.visualize import _load_data, _render_frame


# Default 5-class drainage scheme
DEFAULT_CLASSES = ["ND", "HF", "MD", "LD", "CD"]
DEFAULT_CLASS_LABELS = {
    "ND": "No Drainage",
    "HF": "Hydrofracture",
    "MD": "Moulin Drainage",
    "LD": "Lateral Drainage",
    "CD": "Crevasse Drainage",
}


class Labeler:
    """
    Standalone labeling GUI using matplotlib.

    Parameters
    ----------
    nc_dir : str or Path
        Directory containing .nc timestack files.
    labels_csv : str or Path
        Path to the labels CSV file. Created if it doesn't exist.
    id_col : str
        Column name for feature IDs (default: 'lake_id').
    label_col : str
        Column name for the argmax class label (default: 'label').
    class_names : list of str
        Class names (default: ['ND', 'HF', 'MD', 'LD', 'CD']).
    bands : tuple of str
        Three band names for RGB display (default: ('B04', 'B03', 'B02')).
    scale : str
        Scaling method (default: 'divide').
    scale_kwargs : dict, optional
        Extra kwargs for scaling.
    var : str
        Variable name in NetCDF files (default: 'reflectance').
    mask_band : str or None
        Band name for mask contour overlay (default: None).
    prob_step : float
        Step size for probability sliders (default: 0.25).
    """

    def __init__(
        self,
        nc_dir,
        labels_csv,
        id_col="lake_id",
        label_col="label",
        class_names=None,
        bands=("B04", "B03", "B02"),
        scale="divide",
        scale_kwargs=None,
        var="reflectance",
        mask_band=None,
        prob_step=0.25,
    ):
        self.nc_dir = Path(nc_dir)
        self.labels_csv = Path(labels_csv)
        self.id_col = id_col
        self.label_col = label_col
        self.class_names = class_names or DEFAULT_CLASSES
        self.bands = bands
        self.scale = scale
        self.scale_kwargs = scale_kwargs or {}
        self.var = var
        self.mask_band = mask_band
        self.prob_step = prob_step

        # Probability column names: p_ND, p_HF, etc.
        self.prob_cols = [f"p_{cn}" for cn in self.class_names]

        # Load or create labels CSV
        if self.labels_csv.exists():
            self.labels_df = pd.read_csv(self.labels_csv)
        else:
            cols = [id_col, label_col] + self.prob_cols + ["notes"]
            self.labels_df = pd.DataFrame(columns=cols)

        # Get all NC files
        self.nc_files = sorted(self.nc_dir.glob("*.nc"))
        if not self.nc_files:
            raise ValueError(f"No .nc files found in {self.nc_dir}")

        self.all_ids = [f.stem for f in self.nc_files]
        self._refresh_unlabeled()

        # State
        self._sample_idx = 0
        self._frame_idx = 0
        self._current_da = None
        self._current_id = None
        self._frame_indices = []
        self._notes = ""

    def _refresh_unlabeled(self):
        if self.id_col in self.labels_df.columns and self.label_col in self.labels_df.columns:
            labeled = set(
                self.labels_df.dropna(subset=[self.label_col])[self.id_col].astype(str)
            )
        else:
            labeled = set()
        self.unlabeled_ids = [fid for fid in self.all_ids if fid not in labeled]
        self.labeled_ids = [fid for fid in self.all_ids if fid in labeled]

    def _get_label_counts(self):
        if self.label_col not in self.labels_df.columns:
            return {}
        valid = self.labels_df.dropna(subset=[self.label_col])
        return dict(Counter(valid[self.label_col].astype(str)))

    def _load_sample(self, sample_idx):
        if sample_idx >= len(self.unlabeled_ids):
            self._current_id = None
            self._current_da = None
            return
        self._sample_idx = sample_idx
        lake_id = self.unlabeled_ids[sample_idx]
        self._current_id = lake_id
        nc_path = self.nc_dir / f"{lake_id}.nc"
        da, _ = _load_data(nc_path, var=self.var)
        self._current_da = da
        self._frame_indices = list(range(len(da.time)))
        self._frame_idx = 0

    def _render_current_frame(self):
        if self._current_da is None:
            return np.zeros((128, 128, 3))
        i = self._frame_indices[self._frame_idx]
        tslice = self._current_da.isel(time=i).load()
        band_data = tslice.sel(band=list(self.bands))
        if np.isnan(band_data.values).all():
            return None
        return _render_frame(tslice, list(self.bands), self.scale, self.scale_kwargs)

    def _get_current_mask(self):
        if self._current_da is None or self.mask_band is None:
            return None
        if self.mask_band not in self._current_da.band.values:
            return None
        i = self._frame_indices[self._frame_idx]
        mask = self._current_da.isel(time=i).sel(band=self.mask_band).values
        if np.isnan(mask).all():
            return None
        return mask

    def _get_current_date(self):
        if self._current_da is None:
            return ""
        i = self._frame_indices[self._frame_idx]
        return np.datetime_as_string(self._current_da.time.values[i], unit="D")

    def _save_label(self, lake_id, probs, notes=""):
        """Save label as argmax of probabilities."""
        argmax_idx = int(np.argmax(probs))
        label_value = self.class_names[argmax_idx]

        row = {self.id_col: lake_id, self.label_col: label_value, "notes": notes}
        for col, p in zip(self.prob_cols, probs):
            row[col] = p

        if lake_id in self.labels_df[self.id_col].values:
            idx = self.labels_df[self.labels_df[self.id_col] == lake_id].index[0]
            for k, v in row.items():
                self.labels_df.at[idx, k] = v
        else:
            self.labels_df = pd.concat(
                [self.labels_df, pd.DataFrame([row])], ignore_index=True
            )
        self.labels_df.to_csv(self.labels_csv, index=False)
        self._refresh_unlabeled()
        return label_value

    def start(self):
        """Launch the labeling GUI."""
        if not self.unlabeled_ids:
            print("All samples are labeled!")
            return

        self._load_sample(0)

        n_classes = len(self.class_names)

        # --- Figure layout ---
        fig = plt.figure(figsize=(11, 8.5))
        fig.canvas.manager.set_window_title("sat-tile-stack Labeler")
        fig.patch.set_facecolor("#1a1a1a")

        # Main image
        ax_img = fig.add_axes([0.03, 0.25, 0.58, 0.70])
        ax_img.set_facecolor("black")
        ax_img.axis("off")

        # Frame slider (just below image)
        ax_slider = fig.add_axes([0.03, 0.20, 0.58, 0.03])

        n_frames = len(self._frame_indices)
        frame_slider = Slider(
            ax_slider, "", 0, max(n_frames - 1, 1),
            valinit=0, valstep=1, color="steelblue",
        )

        # Progress bar (small, top right)
        ax_prog = fig.add_axes([0.65, 0.90, 0.32, 0.04])

        # Pie chart (small, right)
        ax_pie = fig.add_axes([0.68, 0.58, 0.30, 0.30])

        # Probability sliders (right side, stacked)
        prob_sliders = []
        prob_labels = []
        slider_height = 0.03
        slider_gap = 0.005
        slider_top = 0.52

        for i, cn in enumerate(self.class_names):
            y = slider_top - i * (slider_height + slider_gap + 0.015)

            # Label above slider
            ax_label = fig.add_axes([0.65, y + slider_height, 0.32, 0.015])
            ax_label.axis("off")
            txt = ax_label.text(0, 0.5, f"{cn}:", color="white", fontsize=9,
                               va="center", ha="left")
            prob_labels.append(txt)

            # Slider
            ax_sl = fig.add_axes([0.65, y, 0.25, slider_height])
            sl = Slider(
                ax_sl, "", 0, 1.0,
                valinit=0.0, valstep=self.prob_step, color="#4488cc",
            )
            prob_sliders.append(sl)

        # Argmax label display
        ax_argmax = fig.add_axes([0.65, slider_top - n_classes * (slider_height + slider_gap + 0.015) - 0.01, 0.32, 0.03])
        ax_argmax.axis("off")
        argmax_text = ax_argmax.text(
            0.5, 0.5, "Label: —", color="#33ff33", fontsize=12,
            ha="center", va="center", transform=ax_argmax.transAxes,
            fontweight="bold",
        )

        # Notes text box
        ax_notes_lbl = fig.add_axes([0.03, 0.13, 0.06, 0.03])
        ax_notes_lbl.axis("off")
        ax_notes_lbl.text(0, 0.5, "Notes:", color="white", fontsize=9,
                         va="center", ha="right")
        ax_notes = fig.add_axes([0.10, 0.13, 0.51, 0.04])
        text_box = TextBox(ax_notes, "", initial="")

        # Buttons
        ax_back = fig.add_axes([0.03, 0.04, 0.12, 0.05])
        ax_skip = fig.add_axes([0.18, 0.04, 0.12, 0.05])
        ax_submit = fig.add_axes([0.33, 0.04, 0.18, 0.05])

        btn_back = Button(ax_back, "Back", color="#555555", hovercolor="#777777")
        btn_skip = Button(ax_skip, "Skip", color="#aa7700", hovercolor="#cc9900")
        btn_submit = Button(ax_submit, "Submit & Next", color="#228833", hovercolor="#33aa44")

        for btn in [btn_back, btn_skip, btn_submit]:
            btn.label.set_color("white")
            btn.label.set_fontsize(10)

        # Status text
        ax_status = fig.add_axes([0.55, 0.04, 0.42, 0.05])
        ax_status.axis("off")
        status_text = ax_status.text(
            0.5, 0.5, "", color="white", fontsize=9,
            ha="center", va="center", transform=ax_status.transAxes,
        )

        # --- Drawing functions ---

        def get_probs():
            return [sl.val for sl in prob_sliders]

        def update_argmax():
            probs = get_probs()
            if max(probs) == 0:
                argmax_text.set_text("Label: —")
            else:
                idx = int(np.argmax(probs))
                argmax_text.set_text(f"Label: {self.class_names[idx]}")
            fig.canvas.draw_idle()

        def draw_frame():
            ax_img.clear()
            ax_img.set_facecolor("black")
            ax_img.axis("off")

            if self._current_da is None:
                ax_img.text(0.5, 0.5, "All done!", ha="center", va="center",
                           fontsize=16, color="white", transform=ax_img.transAxes)
                fig.canvas.draw_idle()
                return

            rgb = self._render_current_frame()
            date = self._get_current_date()
            frame_num = self._frame_idx + 1
            total = len(self._frame_indices)

            if rgb is None:
                ax_img.text(0.5, 0.5, "No data", ha="center", va="center",
                           fontsize=14, color="gray", transform=ax_img.transAxes)
            else:
                ax_img.imshow(rgb)
                mask = self._get_current_mask()
                if mask is not None:
                    ax_img.contour(mask, levels=[0.5], colors="red", linewidths=1)

            ax_img.set_title(
                f"{self._current_id}  |  {date}  |  {frame_num}/{total}",
                color="white", fontsize=11,
            )
            fig.canvas.draw_idle()

        def draw_progress():
            ax_prog.clear()
            ax_prog.set_facecolor("#1a1a1a")
            n_total = len(self.all_ids)
            n_labeled = len(self.labeled_ids)
            n_remaining = len(self.unlabeled_ids)
            ax_prog.barh([0], [n_labeled], color="steelblue", height=0.6)
            ax_prog.barh([0], [n_remaining], left=[n_labeled], color="#444444", height=0.6)
            ax_prog.set_xlim(0, n_total)
            ax_prog.set_yticks([])
            ax_prog.set_xticks([])
            ax_prog.text(0.5, 0.5, f"{n_labeled}/{n_total}",
                        ha="center", va="center", transform=ax_prog.transAxes,
                        color="white", fontsize=8)
            for spine in ax_prog.spines.values():
                spine.set_visible(False)

        def draw_pie():
            ax_pie.clear()
            ax_pie.set_facecolor("#1a1a1a")
            counts = self._get_label_counts()
            if not counts:
                ax_pie.text(0.5, 0.5, "No labels\nyet", ha="center", va="center",
                           fontsize=9, color="gray", transform=ax_pie.transAxes)
                return
            labels, sizes = [], []
            for cn in self.class_names:
                count = counts.get(cn, 0)
                if count > 0:
                    labels.append(f"{cn} ({count})")
                    sizes.append(count)
            if sizes:
                ax_pie.pie(sizes, labels=labels,
                          textprops={"fontsize": 8, "color": "white"}, startangle=90)

        def update_all():
            draw_frame()
            draw_progress()
            draw_pie()
            remaining = len(self.unlabeled_ids) - self._sample_idx
            status_text.set_text(f"{remaining} remaining")
            fig.canvas.draw_idle()

        def reset_prob_sliders():
            for sl in prob_sliders:
                sl.set_val(0)
            update_argmax()

        def load_and_display(sample_idx):
            self._load_sample(sample_idx)
            n_frames = len(self._frame_indices)
            frame_slider.valmax = max(n_frames - 1, 1)
            frame_slider.set_val(0)
            ax_slider.set_xlim(0, frame_slider.valmax)
            text_box.set_val("")
            self._notes = ""
            reset_prob_sliders()
            update_all()

        # --- Callbacks ---

        def on_frame_slider(val):
            self._frame_idx = int(val)
            draw_frame()

        def _clamp_probs(changed_idx):
            """Ensure probabilities sum to <= 1.0 by clamping the changed slider."""
            total = sum(sl.val for sl in prob_sliders)
            if total > 1.0:
                excess = total - 1.0
                # Pull back the slider that was just moved
                new_val = max(0, prob_sliders[changed_idx].val - excess)
                # Round to nearest step to avoid floating point drift
                new_val = round(new_val / self.prob_step) * self.prob_step
                prob_sliders[changed_idx].set_val(new_val)

        def _make_prob_callback(idx):
            """Create a callback for a specific probability slider."""
            def on_prob_slider(val):
                _clamp_probs(idx)
                # Show remaining budget
                total = sum(sl.val for sl in prob_sliders)
                remaining = 1.0 - total
                if abs(remaining) < 0.01:
                    status_text.set_text("")
                else:
                    status_text.set_text(f"{remaining:.0%} remaining")
                update_argmax()
                fig.canvas.draw_idle()
            return on_prob_slider

        def on_notes(text):
            self._notes = text

        def on_submit(event):
            if self._current_id is None:
                return
            probs = get_probs()
            total = sum(probs)
            if abs(total - 1.0) > 0.01:
                status_text.set_text(f"Probabilities must sum to 1.0 (currently {total:.2f})")
                fig.canvas.draw_idle()
                return
            label = self._save_label(self._current_id, probs, self._notes)
            status_text.set_text(f"Saved: {self._current_id} = {label} ({probs})")
            self._refresh_unlabeled()
            if self._sample_idx >= len(self.unlabeled_ids):
                self._sample_idx = max(0, len(self.unlabeled_ids) - 1)
            if self.unlabeled_ids:
                load_and_display(self._sample_idx)
            else:
                self._current_da = None
                self._current_id = None
                update_all()

        def on_skip(event):
            next_idx = min(self._sample_idx + 1, len(self.unlabeled_ids) - 1)
            load_and_display(next_idx)

        def on_back(event):
            prev_idx = max(0, self._sample_idx - 1)
            load_and_display(prev_idx)

        def on_scroll(event):
            """Scroll wheel over image scrubs frames."""
            if event.inaxes != ax_img:
                return
            if event.button == "up":
                new_val = min(self._frame_idx + 1, len(self._frame_indices) - 1)
            elif event.button == "down":
                new_val = max(0, self._frame_idx - 1)
            else:
                return
            frame_slider.set_val(new_val)

        def on_key(event):
            if event.key == "right":
                new_val = min(self._frame_idx + 1, len(self._frame_indices) - 1)
                frame_slider.set_val(new_val)
            elif event.key == "left":
                new_val = max(0, self._frame_idx - 1)
                frame_slider.set_val(new_val)
            elif event.key == "enter":
                on_submit(event)

        # Connect callbacks
        frame_slider.on_changed(on_frame_slider)
        for i, sl in enumerate(prob_sliders):
            sl.on_changed(_make_prob_callback(i))
        text_box.on_submit(on_notes)
        text_box.on_text_change(on_notes)
        btn_submit.on_clicked(on_submit)
        btn_skip.on_clicked(on_skip)
        btn_back.on_clicked(on_back)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        fig.canvas.mpl_connect("key_press_event", on_key)

        # Initial display
        update_all()
        plt.show()
