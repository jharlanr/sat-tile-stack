# labeling/ — local data workspace

This directory holds the labeling-related data files (labels, lake lists,
inventories, generated stacks). **Everything in here except this README is
gitignored** — the contents only live in your working copy.

## Layout

| Path | Contents |
|---|---|
| `CW_{YEAR}/labels_CW_{YEAR}.csv` | Canonical 5-class labels (Josh's set) |
| `CW_{YEAR}/stacks/` | `.nc` timestacks, one per lake. Usually only present on Sherlock / OAK; mounted via `/Volumes/groups/cyaolai/...` on macOS |
| `irr/` | Inter-rater reliability assets — sample reference + blind lake lists for additional labelers; receives their `irr_labels_{NAME}.csv` outputs |
| `dunmire/` | Dunmire et al. (2025) lake inventories (`labels_{YEAR}_volumes.geojson`) |
| `lookups/` | Persistent-ID lookup tables (region × year → lake_id mappings) |
| `labels/` | Older volume CSVs (pre-5-class schema) — superseded by `CW_{YEAR}/`; kept for traceability |
| `test_stacks/` | A handful of `.nc` files (`lake_A.nc`, `lake_B.nc`, `lake_C.nc`) for smoke-testing the labeling GUI |

## Related code

- **GUI:** [`sat_tile_stack/labeling/server.py`](../sat_tile_stack/labeling/server.py) — the Flask labeling server (installed as the `lakelabel` console script when the package is installed with the `[labeling]` extra).
- **Collaborator setup:** [`docs/LABELING_FOR_COLLABORATORS.md`](../docs/LABELING_FOR_COLLABORATORS.md) — full setup instructions for IRR labelers.
- **Schema migration:** [`engine/labeling/convert_2019_labels.py`](../engine/labeling/convert_2019_labels.py) — one-shot converter from the old `p_nd/p_ed/p_ld/p_cd` schema to the current 5-class `p_ND/p_HF/p_MD/p_LD/p_CD` schema. Already run against the canonical 2019 labels; kept for reproducibility.
