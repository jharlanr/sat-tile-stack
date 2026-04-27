"""
Convert existing 2019 CW labels from the old format (label_rines, p_nd/p_ed/p_ld/p_cd)
to the new GUI format (label, p_ND/p_HF/p_MD/p_LD/p_CD).

The key transformation: splits p_ed into p_HF and p_MD based on the edm_edf column:
  - edm_edf='m' (moulin)       → p_ed goes to p_MD
  - edm_edf='f' (hydrofracture) → p_ed goes to p_HF
  - edm_edf='?' (unknown)       → p_ed goes to p_HF (assumption)

Usage:
    python labeling/convert_2019_labels.py \
        --input labeling/labels/labels_2019_volumes_CW.csv \
        --output labeling/CW_2019/labels_CW_2019.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ["ND", "HF", "MD", "LD", "CD"]
PROB_COLS = ["p_ND", "p_HF", "p_MD", "p_LD", "p_CD"]


def convert(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["new_id", "label_rines"])

    out = pd.DataFrame()
    out["lake_id"] = df["new_id"].values
    out["p_ND"] = df["p_nd"].fillna(0).values
    out["p_HF"] = 0.0
    out["p_MD"] = 0.0
    out["flagged"] = False
    out["p_LD"] = df["p_ld"].fillna(0).values
    out["p_CD"] = df["p_cd"].fillna(0).values

    # Split p_ed into p_HF / p_MD based on edm_edf (only for ED lakes)
    for i, (_, row) in enumerate(df.iterrows()):
        label = row.get("label_rines", -1)
        p_ed = row.get("p_ed", 0)
        if pd.isna(p_ed):
            p_ed = 0

        if label == 1:  # ED lake — split based on edm_edf
            edm = str(row.get("edm_edf", "?")).strip().lower()
            if edm == "m":
                out.loc[i, "p_MD"] = p_ed
            elif edm == "f":
                out.loc[i, "p_HF"] = p_ed
            else:  # '?', 'nan', or unknown → assume HF, flag for revisit
                out.loc[i, "p_HF"] = p_ed
                out.loc[i, "flagged"] = True
        elif p_ed > 0:
            # Non-ED lake but has some p_ed probability — put it all in p_HF
            out.loc[i, "p_HF"] = p_ed

    # Label = argmax of probabilities
    out["label"] = out[PROB_COLS].apply(
        lambda row: CLASSES[np.argmax(row.values)], axis=1
    )

    out["notes"] = df["notes"].fillna("").values if "notes" in df.columns else ""

    # Reorder columns
    out = out[["lake_id", "label"] + PROB_COLS + ["notes", "flagged"]]

    # Save
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    # Summary
    print(f"Converted {len(out)} labels")
    print(f"  Input:  {input_csv}")
    print(f"  Output: {output_csv}")
    print(f"\nLabel distribution:")
    print(out["label"].value_counts().to_string())
    print(f"\nedm_edf breakdown for ED lakes:")
    ed_mask = df["label_rines"] == 1
    if "edm_edf" in df.columns:
        print(df.loc[ed_mask, "edm_edf"].value_counts().to_string())
    print(f"\nSample output:")
    print(out.head(5).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 2019 labels to GUI format")
    parser.add_argument("--input", type=str,
                        default="labeling/labels/labels_2019_volumes_CW.csv",
                        help="Input CSV (old format)")
    parser.add_argument("--output", type=str,
                        default="labeling/CW_2019/labels_CW_2019.csv",
                        help="Output CSV (GUI format)")
    args = parser.parse_args()

    convert(args.input, args.output)
