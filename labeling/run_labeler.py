"""
Launch the labeler GUI as a standalone script.

Usage:
    cd sat-tile-stack
    python labeling/run_labeler.py --nc_dir path/to/nc/files

Labels are saved to labeling/labels/ by default.
"""
import sys
import argparse
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labeler import Labeler


def main():
    parser = argparse.ArgumentParser(description="Launch the labeling GUI")
    parser.add_argument("--nc_dir", type=str, required=True,
                        help="Directory containing .nc timestack files")
    parser.add_argument("--labels_csv", type=str, default="labeling/labels/labels.csv",
                        help="Path to labels CSV (created if it doesn't exist)")
    parser.add_argument("--classes", nargs="+", default=["ND", "HF", "MD", "LD", "CD"],
                        help="Class names (default: ND HF MD LD CD)")
    parser.add_argument("--bands", nargs=3, default=["B04", "B03", "B02"],
                        help="Three band names for RGB display")
    parser.add_argument("--scale", type=str, default="divide",
                        help="Scaling method (divide, percentile, minmax, none)")
    parser.add_argument("--var", type=str, default="reflectance",
                        help="Variable name in NetCDF files")
    args = parser.parse_args()

    nc_dir = Path(args.nc_dir)
    nc_files = sorted(nc_dir.glob("*.nc"))
    if not nc_files:
        print(f"No .nc files found in {nc_dir}")
        print("Build test stacks first with notebooks/test_labeler.ipynb")
        sys.exit(1)

    print(f"Found {len(nc_files)} stacks in {nc_dir}")

    labeler = Labeler(
        nc_dir=args.nc_dir,
        labels_csv=args.labels_csv,
        id_col="lake_id",
        label_col="label",
        class_names=args.classes,
        bands=tuple(args.bands),
        scale=args.scale,
        var=args.var,
    )
    labeler.start()


if __name__ == "__main__":
    main()
