# Labeling guide for IRR collaborators

This is a step-by-step setup guide for labeling 200 supraglacial lakes for the
ESSD inter-rater reliability (IRR) study. It assumes nothing about your Python
setup beyond having a working terminal and a browser.

You will end up with a single CSV file of your labels that you send back when
you're done. The labeling tool saves your work after every lake — you can quit
at any time and resume later, and you don't need to label everything in one
sitting.

---

## What you need from Josh

Before you start, Josh will give you (or point you at):

1. **A labeler ID** — your assigned identifier for this study (e.g. `labeler_2`
   or `labeler_3`). Use it everywhere `<LABELER_ID>` appears below.
2. **Two blind lake lists** — `irr_lakes_<LABELER_ID>_CW_2018.csv` and
   `irr_lakes_<LABELER_ID>_CW_2019.csv`, 100 lakes each (just a `lake_id`
   column, randomized per labeler).
3. **An OAK mount on your machine** — the `.nc` stack files live on Stanford's
   OAK at:
   - `/Volumes/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks/CW_2018/`
   - `/Volumes/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks/CW_2019/`
4. **A short calibration walkthrough** — Josh will go over one canonical
   example per class (ND/HF/MD/LD/CD) before you start so we're aligned on
   the taxonomy.

---

## One-time setup

### 1. Confirm OAK access

The stack files (~30 GB total) live on OAK and are read on demand — you don't
need to copy them locally, but the mount has to be working. Try:

```bash
ls /Volumes/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks/CW_2019/ | head
```

You should see a list of `CW2019_*.nc` files.

If you don't see the stacks directory, mount OAK first (Stanford SRCC docs:
<https://www.sherlock.stanford.edu/docs/storage/data-transfer/#smb>). Off-campus
will need VPN.

### 2. Set up a Python environment

You need Python 3.9 or newer. If you don't already have a managed env, install
[Miniforge](https://github.com/conda-forge/miniforge) and create one:

```bash
conda create -n labeling python=3.11
conda activate labeling
```

### 3. Install the labeling tool

```bash
pip install "sat-tile-stack[labeling] @ git+https://github.com/jharlanr/sat-tile-stack"
```

This installs the package and a `lakelabel` command. Verify it worked:

```bash
lakelabel --help
```

You should see usage output listing flags like `--nc_dir`, `--lake_list`,
`--labels_csv`.

### 4. Make a working directory for your labels

Pick anywhere convenient:

```bash
mkdir ~/irr_labeling
cd ~/irr_labeling
```

Drop both `irr_lakes_<LABELER_ID>_CW_2018.csv` and
`irr_lakes_<LABELER_ID>_CW_2019.csv` (the files Josh sent you) into this
folder. Your two output files will land here too — one per year:
`irr_labels_<LABELER_ID>_CW_2018.csv` and `irr_labels_<LABELER_ID>_CW_2019.csv`.

---

## Labeling

The 200 lakes are split across two years (100 from 2018, 100 from 2019).
You'll run the GUI **separately per year** — each year has its own input
lake list and its own output labels CSV. The two years stay fully
independent. Within a year, the tool auto-skips lakes you've already
labeled, so you can quit and resume freely across sessions.

### Start the 2019 lakes

```bash
lakelabel \
    --nc_dir /Volumes/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks/CW_2019 \
    --labels_csv irr_labels_<LABELER_ID>_CW_2019.csv \
    --lake_list irr_lakes_<LABELER_ID>_CW_2019.csv
```

Your browser should open `http://localhost:5050` automatically. If it doesn't,
open that URL manually.

You'll see:

- An imagery viewer (large center panel) — scroll the mouse wheel over the
  image to scrub through frames.
- Five probability sliders (ND/HF/MD/LD/CD) — drag to allocate probability
  mass. The argmax becomes your label. You can put fractional weight on
  multiple classes if you're uncertain (e.g., 0.75 HF + 0.25 MD).
- A Notes field — free text, optional.
- A Flag button — mark lakes you want to revisit before finalizing.
- A Submit button — saves and advances to the next unlabeled lake.

**Save behavior:** every Submit immediately writes to the year-specific labels
CSV (`irr_labels_<LABELER_ID>_CW_2019.csv` while you're on 2019 lakes) in the
directory where you launched the command. There is no "save all" step.
Closing the terminal or browser is safe.

### Switch to the 2018 lakes (later, same or different session)

When you're done with 2019 (or want to mix it up), stop the server (`Ctrl-C`
in the terminal) and relaunch with the 2018 stacks and the 2018 output CSV:

```bash
lakelabel \
    --nc_dir /Volumes/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks/CW_2018 \
    --labels_csv irr_labels_<LABELER_ID>_CW_2018.csv \
    --lake_list irr_lakes_<LABELER_ID>_CW_2018.csv
```

Both the input lake list and the output labels file switch to the 2018
versions. The two years are fully independent — there's no shared CSV
between them, so you can't accidentally cross years.

### Resuming after a break

Just rerun the same `lakelabel ...` command for whichever year you were on.
The tool reads the year's labels CSV on startup and starts you at the next
unlabeled lake. No special "resume" step.

---

## Pacing expectations

- ~200 lakes total, ~30–60 seconds per lake once you're calibrated → ~2–3
  hours total active time. You can spread that over a week.
- The first lake of each session takes ~50 seconds to load (OAK is slow);
  after that the next 2–3 lakes prefetch in the background while you work,
  so subsequent loads are usually instant.
- If a lake takes more than ~90 s to display, something is wrong with the
  OAK mount — quit and ping Josh.

---

## When you're done

Send **both** `irr_labels_<LABELER_ID>_CW_2018.csv` and
`irr_labels_<LABELER_ID>_CW_2019.csv` back to Josh. Each file should have
100 rows.

Sanity check before sending:

```bash
wc -l irr_labels_<LABELER_ID>_CW_2018.csv irr_labels_<LABELER_ID>_CW_2019.csv
# each should be 101 (100 rows + header)

head -3 irr_labels_<LABELER_ID>_CW_2019.csv
# should show lake_id, label, p_ND, p_HF, p_MD, p_LD, p_CD, notes, flagged
```

That's it.

---

## Troubleshooting

**`lakelabel: command not found`**
The pip install didn't put the script on your PATH. Activate the conda env
where you installed it (`conda activate labeling`), or run
`python -m sat_tile_stack.labeling --help` instead.

**`ERROR: NC directory does not exist`**
The OAK path is wrong, the mount isn't active, or you're off-VPN. Try
`ls <the path>` first to confirm.

**`ERROR: No .nc files in <dir> match the lake list`**
`--nc_dir` and `--lake_list` are out of sync — likely you pointed `--nc_dir` at
the wrong year's stacks. Switch to the matching year.

**Browser opened but page is blank or won't load**
Wait ~5 seconds and refresh — Flask sometimes takes a moment to come up.
If still blank, check the terminal for errors and forward them to Josh.

**Anything else** → email Josh with the full terminal output.
