"""
Streamlit labeling GUI for satellite timestacks.

Usage:
    streamlit run labeling/app.py -- --nc_dir path/to/stacks

Compact single-viewport layout — no page scrolling required.
Scroll wheel over image scrubs through frames.
"""

import sys
import io
import base64
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sat_tile_stack.visualize import _load_data, _render_frame


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CLASSES = ["ND", "HF", "MD", "LD", "CD"]
CLASS_DESCRIPTIONS = {
    "ND": "No Drainage",
    "HF": "Hydrofracture",
    "MD": "Moulin Drainage",
    "LD": "Lateral Drainage",
    "CD": "Crevasse Drainage",
}
PROB_STEP = 0.25


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, default=None)
    parser.add_argument("--var", type=str, default="reflectance")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    args, _ = parser.parse_known_args()
    if args.labels_csv is None:
        args.labels_csv = str(Path(__file__).parent / "labels" / "labels.csv")
    return args


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_nc_file(nc_path, var):
    da, _ = _load_data(nc_path, var=var)
    return da.load()


@st.cache_data
def get_nc_file_list(nc_dir):
    return sorted(Path(nc_dir).glob("*.nc"))


def load_labels(labels_csv, classes):
    prob_cols = [f"p_{cn}" for cn in classes]
    if Path(labels_csv).exists():
        return pd.read_csv(labels_csv)
    else:
        Path(labels_csv).parent.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame(columns=["lake_id", "label"] + prob_cols + ["notes"])


def save_labels(df, labels_csv):
    df.to_csv(labels_csv, index=False)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_rgb(da, time_idx, bands, scale="divide"):
    """Render frame as RGB, return None if all-NaN or bands missing."""
    available = [str(b) for b in da.band.values]
    for b in bands:
        if b not in available:
            return None
    tslice = da.isel(time=time_idx)
    band_data = tslice.sel(band=list(bands))
    if np.isnan(band_data.values).all():
        return None
    return _render_frame(tslice, list(bands), scale, {})


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120,
                facecolor="black", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def render_main_image(da, frame_idx, bands, scale="divide"):
    """Render the main image as a base64 PNG."""
    rgb = render_rgb(da, frame_idx, bands, scale)

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    if rgb is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=14, color="gray", transform=ax.transAxes)
    else:
        ax.imshow(rgb)
        # Mask overlay
        available = [str(b) for b in da.band.values]
        if "mask" in available:
            mask = da.isel(time=frame_idx).sel(band="mask").values
            if not np.isnan(mask).all():
                ax.contour(mask, levels=[0.5], colors="red", linewidths=1)

    ax.axis("off")
    date = np.datetime_as_string(da.time.values[frame_idx], unit="D")
    ax.set_title(date, color="white", fontsize=11)
    plt.tight_layout(pad=0.3)

    return fig_to_base64(fig)


def render_thumbnail(da, frame_idx, bands, title, scale="divide"):
    """Render a small thumbnail as base64 PNG."""
    available = [str(b) for b in da.band.values]

    fig, ax = plt.subplots(figsize=(1.8, 1.8))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("black")

    if len(bands) == 1 and bands[0] in available:
        data = da.isel(time=frame_idx).sel(band=bands[0]).values
        ax.imshow(np.nan_to_num(data, nan=0), cmap="viridis")
    else:
        rgb = render_rgb(da, frame_idx, bands, scale)
        if rgb is not None:
            ax.imshow(rgb)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    fontsize=8, color="gray", transform=ax.transAxes)

    ax.set_title(title, color="white", fontsize=7)
    ax.axis("off")
    plt.tight_layout(pad=0.2)

    return fig_to_base64(fig)


def get_band_combos(available_bands):
    """Get available band combos."""
    combos = {}
    options = [
        ("RGB", ["B04", "B03", "B02"]),
        ("SWIR/NIR/R", ["B11", "B08", "B04"]),
        ("NIR/R/G", ["B08", "B04", "B03"]),
        ("Cloud Mask", ["cloudmask"]),
        ("SCL", ["SCL"]),
        ("Mask", ["mask"]),
    ]
    for name, bands in options:
        if all(b in available_bands for b in bands):
            combos[name] = bands
    return combos


# ---------------------------------------------------------------------------
# Scroll-to-scrub component
# ---------------------------------------------------------------------------

def scrollable_image(img_base64, frame_idx, max_frame, height=450):
    """Display image with scroll-to-scrub via JavaScript."""
    html = f"""
    <div id="scroll-container" style="text-align:center; background:black;
         cursor:grab; user-select:none; outline:none;" tabindex="0">
        <img src="data:image/png;base64,{img_base64}"
             style="max-height:{height}px; max-width:100%;" />
    </div>
    <script>
        const container = document.getElementById('scroll-container');
        container.addEventListener('wheel', function(e) {{
            e.preventDefault();
            const delta = e.deltaY > 0 ? 1 : -1;
            const current = {frame_idx};
            const maxF = {max_frame};
            const newVal = Math.max(0, Math.min(maxF, current + delta));
            if (newVal !== current) {{
                // Send to Streamlit
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: newVal
                }}, '*');
            }}
        }});
    </script>
    """
    result = components.html(html, height=height + 10, scrolling=False)
    return result


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    st.set_page_config(
        page_title="Labeler",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Minimal CSS to reduce padding
    st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        .stSlider { padding-top: 0; padding-bottom: 0; }
        header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # Load data
    nc_files = get_nc_file_list(args.nc_dir)
    if not nc_files:
        st.error(f"No .nc files in {args.nc_dir}")
        return

    all_ids = [f.stem for f in nc_files]
    labels_df = load_labels(args.labels_csv, args.classes)

    if "label" in labels_df.columns:
        labeled_ids = set(labels_df.dropna(subset=["label"])["lake_id"].astype(str))
    else:
        labeled_ids = set()
    unlabeled_ids = [fid for fid in all_ids if fid not in labeled_ids]

    if not unlabeled_ids:
        st.success("All samples labeled!")
        return

    # Sample index
    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = 0
    sample_idx = st.session_state.sample_idx
    if sample_idx >= len(unlabeled_ids):
        sample_idx = len(unlabeled_ids) - 1
    current_id = unlabeled_ids[sample_idx]

    # Load sample
    nc_path = Path(args.nc_dir) / f"{current_id}.nc"
    da = load_nc_file(str(nc_path), args.var)
    n_frames = len(da.time)
    available_bands = [str(b) for b in da.band.values]
    band_combos = get_band_combos(available_bands)

    # Default bands
    if "RGB" in band_combos:
        default_bands = band_combos["RGB"]
    else:
        default_bands = list(band_combos.values())[0] if band_combos else available_bands[:3]

    # =====================================================================
    # LAYOUT: two columns, everything in one viewport
    # =====================================================================

    left_col, right_col = st.columns([3, 1.3])

    # --- LEFT: image + slider + buttons ---
    with left_col:
        # Header
        n_labeled = len(labeled_ids)
        n_total = len(all_ids)
        remaining = len(unlabeled_ids) - sample_idx
        st.caption(f"**{current_id}** — {sample_idx+1}/{len(unlabeled_ids)} unlabeled — {n_labeled}/{n_total} total labeled — {remaining} remaining")

        # Frame slider
        frame_idx = st.slider("Frame", 0, n_frames - 1, 0, key="frame_idx",
                              label_visibility="collapsed")
        date_str = np.datetime_as_string(da.time.values[frame_idx], unit="D")
        st.caption(f"{date_str}  |  frame {frame_idx+1}/{n_frames}")

        # Main image
        img_b64 = render_main_image(da, frame_idx, default_bands)
        st.image(f"data:image/png;base64,{img_b64}", use_container_width=True)

        # Buttons + notes in one row
        c1, c2, c3, c4 = st.columns([1, 1, 2, 4])
        with c1:
            if st.button("⬅ Back", disabled=sample_idx == 0, use_container_width=True):
                st.session_state.sample_idx = max(0, sample_idx - 1)
                st.rerun()
        with c2:
            if st.button("Skip ➡", use_container_width=True):
                st.session_state.sample_idx = min(sample_idx + 1, len(unlabeled_ids) - 1)
                st.rerun()
        with c4:
            notes = st.text_input("Notes", value="", key="notes", label_visibility="collapsed",
                                  placeholder="Notes...")

    # --- RIGHT: probabilities + pie + thumbnails ---
    with right_col:
        # Probability sliders
        st.caption("**Class probabilities**")
        probs = []
        for cn in args.classes:
            desc = CLASS_DESCRIPTIONS.get(cn, cn)
            p = st.select_slider(
                cn, options=[0.0, 0.25, 0.50, 0.75, 1.0],
                value=0.0, key=f"prob_{cn}",
            )
            probs.append(p)

        total = sum(probs)
        can_submit = abs(total - 1.0) < 0.01 and max(probs) > 0

        if can_submit:
            argmax_label = args.classes[int(np.argmax(probs))]
            st.success(f"**{argmax_label}**")
        elif total > 0:
            st.warning(f"Sum: {total:.2f} (need 1.0)")

        # Submit button
        if st.button("✓ Submit & Next", type="primary", disabled=not can_submit,
                     use_container_width=True):
            argmax_label = args.classes[int(np.argmax(probs))]
            row = {"lake_id": current_id, "label": argmax_label, "notes": notes}
            for cn, p in zip(args.classes, probs):
                row[f"p_{cn}"] = p

            if current_id in labels_df["lake_id"].values:
                idx = labels_df[labels_df["lake_id"] == current_id].index[0]
                for k, v in row.items():
                    labels_df.at[idx, k] = v
            else:
                labels_df = pd.concat(
                    [labels_df, pd.DataFrame([row])], ignore_index=True
                )
            save_labels(labels_df, args.labels_csv)
            st.session_state.sample_idx = min(sample_idx + 1, len(unlabeled_ids) - 1)
            st.rerun()

        # Small pie chart
        if n_labeled > 0:
            counts = Counter(labels_df.dropna(subset=["label"])["label"].astype(str))
            fig_pie, ax_pie = plt.subplots(figsize=(2, 2))
            fig_pie.patch.set_facecolor("#0e1117")
            pie_labels = []
            pie_sizes = []
            for cn in args.classes:
                c = counts.get(cn, 0)
                if c > 0:
                    pie_labels.append(f"{cn}({c})")
                    pie_sizes.append(c)
            if pie_sizes:
                ax_pie.pie(pie_sizes, labels=pie_labels,
                          textprops={"fontsize": 7, "color": "white"}, startangle=90)
            plt.tight_layout(pad=0)
            st.pyplot(fig_pie)
            plt.close(fig_pie)

        # Thumbnails for other band combos
        if len(band_combos) > 1:
            st.caption("**Other views**")
            for name, bands in band_combos.items():
                if bands == default_bands:
                    continue
                thumb_b64 = render_thumbnail(da, frame_idx, bands, name)
                st.image(f"data:image/png;base64,{thumb_b64}", width=140)


if __name__ == "__main__":
    main()
