# retina-connectomics

Computational pipeline for analyzing 3D neuron morphology and orientation relationships in mouse retina EM data (EyeWire II).

Given a set of EM segment IDs, the pipeline reconstructs neuron skeletons, warps them into a common retinal coordinate frame aligned to the starburst amacrine cell (SAC) sheets, and computes pairwise orientation angles between cell populations using PCA on skeleton node positions.

## Goal

Determine how neurons are oriented relative to each other in 3D space — which cells are co-aligned, oblique, or near-orthogonal — across a population of retinal neurons.

## Pipeline overview

```
annotations_flat.csv + state.json          (Neuroglancer exports)
    │
    ▼
infer_annotations.py          parse synapse annotations → connectivity report
    │                                                    → transition matrix
    └── --out-segment-ids ──► segment_ids.txt           → unique segment IDs
                                    │
                                    ▼
                               flatone                   mesh download → skeletonize → warp to SAC frame
                                    │
                                    ├── outputs/<seg_id>/mesh.obj
                                    ├── outputs/<seg_id>/skeleton.swc / .npz
                                    └── outputs/<seg_id>/skeleton_warped.swc / .npz
                                    │
                          ┌─────────┴──────────────────────────────────┐
                          │                                            │
                          ▼                                            ▼
              orientation_relations.py                    hull_sac_analysis.py
              PCA → principal axis per cell               XY convex hull of post cell
                          │                               identify pre-SAC segments in hull
                          │                                            │
              ├── per_cell_orientation.csv             ├── hull_sac_results.csv
              ├── pairwise_orientation_angles.csv       ├── hull_sac_results.json
              ├── orientation_angle_matrix.csv          └── hull_sac_plot.png
              └── orientation_summary.json
```

## Installation

### Prerequisites

**macOS:**
```bash
brew install suite-sparse git-lfs
```

**Debian / Ubuntu / WSL 2:**
```bash
sudo apt-get install libsuitesparse-dev git-lfs
```

> SuiteSparse is required for skeletonization. Native Windows is not supported — use WSL 2.

### Python environment

```bash
git lfs install
git clone <this-repo>
cd retina-connectomics

python -m venv .venv
source .venv/bin/activate
pip install -e flatone/
```

### Authentication

Get a CAVEclient token from the EyeWire II DAF portal and register it once:

```bash
flatone add-token <TOKEN>
```

## Usage

### Step 1 — Extract segment IDs from annotations

Parse your Neuroglancer annotation export and extract all unique segment IDs:

```bash
python src/infer_annotations.py \
  --csv path/to/annotations_flat.csv \
  --state path/to/state.json \
  --out-segment-ids outputs/segment_ids.txt \
  --out-md outputs/inference_report.md \
  --out-transition-matrix-csv outputs/transition_matrix.csv
```

`outputs/segment_ids.txt` is a plain text file (one numeric ID per line) that feeds directly into Step 3.

You can also use any hand-curated ID list (e.g. `data/segment_ids/off_bk_seg_ids.txt`) in place of the annotations-derived one — the format is the same.

### Step 2 — Process neurons with flatone

Run the full pipeline (mesh download → skeletonize → warp) for one segment ID:

```bash
flatone <SEGMENT_ID> --output-dir outputs/cells
```

Process a batch of IDs:

```bash
while read seg_id; do
  flatone "$seg_id" --output-dir outputs/cells --no-verbose
done < outputs/segment_ids.txt
```

Or let `orientation_relations.py` drive flatone automatically for any missing IDs (Step 3).

**Key flags:**

| Flag | Description |
|------|-------------|
| `--output-dir PATH` | Root folder for outputs (default: `./output`) |
| `--mapping j1\|j2` | Conformal map version (default: `j2`) |
| `--warp-mesh` | Also warp the raw mesh geometry (slow, optional) |
| `--overwrite` | Recompute all steps |
| `--overwrite-skeleton` | Recompute skeleton only |
| `--overwrite-profile` | Recompute stratification profile only |
| `--no-verbose` | Suppress progress output |
| `--soma-threshold FLOAT` | Percentile for soma detection (default: 99.9) |

### Step 3 — Compute orientation relationships

```bash
python src/orientation_relations.py \
  --segment-ids outputs/segment_ids.txt \
  --output-dir outputs/cells \
  --skeleton-kind auto \
  --run-flatone-missing \
  --out-per-cell-csv outputs/orientation/per_cell_orientation.csv \
  --out-pairwise-csv outputs/orientation/pairwise_orientation_angles.csv \
  --out-matrix-csv outputs/orientation/orientation_angle_matrix.csv \
  --out-summary-json outputs/orientation/orientation_summary.json
```

`--run-flatone-missing` automatically runs flatone for any IDs that do not yet have a skeleton, so Steps 2 and 3 can be collapsed into a single command once you have your ID list.

**Output files:**

| File | Contents |
|------|----------|
| `per_cell_orientation.csv` | Principal axis (3D + XY projection), azimuth, elevation, elongation per cell |
| `pairwise_orientation_angles.csv` | Undirected angle between every cell pair in 3D and XY, with relation label |
| `orientation_angle_matrix.csv` | N×N matrix of 3D pairwise angles |
| `orientation_summary.json` | Run statistics: processed, missing, failed IDs |

**Angle classification:**

| Range | Label |
|-------|-------|
| < 15° | `aligned` |
| 15° – 44° | `oblique` |
| ≥ 45° | `near-orthogonal` |

**Skeleton kind (`--skeleton-kind`):**

- `auto` (default) — use `skeleton_warped.npz` if present, fall back to `skeleton.npz`
- `warped` — require warped skeleton (in SAC-aligned coordinates)
- `raw` — always use raw skeleton (in original EM nanometre space)

Warped skeletons are recommended for orientation analysis because they place all cells in the same retinal coordinate frame.


### Step 4 — Hull analysis: find pre-SAC segments inside a post cell's XY footprint

For a given post-synaptic cell, compute its convex hull in the XY plane and identify
which pre-SAC segments have their centroid inside it.

```bash
python src/hull_sac_analysis.py \
  --post-seg-id <POST_SEGMENT_ID> \
  --csv path/to/annotations_flat.csv \
  --output-dir outputs/cells \
  --skeleton-kind auto \
  --out-csv outputs/hull/hull_sac_results.csv \
  --out-json outputs/hull/hull_sac_results.json \
  --out-plot outputs/hull/hull_sac_plot.png
```

**How it works:**

1. Loads the post cell's `skeleton_warped.npz` and computes its 2D convex hull in XY.
2. Scans the annotations CSV for rows where the post cell is in `segment_2` and the description starts with `SAC->`.  The segment in `segment_1` is the pre-SAC ID.
3. For each pre-SAC ID, gets a representative XY position — skeleton centroid if an NPZ exists, otherwise the mean of the synapse annotation point coordinates in the CSV.
4. Tests each position against the hull and outputs `inside` / `outside` / `no_position`.

**Key flags:**

| Flag | Description |
|------|-------------|
| `--post-seg-id` | Segment ID of the post-synaptic cell (required) |
| `--csv` | Neuroglancer annotations CSV (required) |
| `--output-dir` | Flatone output root for skeleton NPZ lookup |
| `--sac-label` | Cell-type prefix to match in descriptions (default: `SAC`) |
| `--pre-col` | CSV column for the pre-synaptic segment (default: `segment_1`) |
| `--post-col` | CSV column for the post-synaptic segment (default: `segment_2`) |
| `--out-plot` | Save XY hull + SAC positions as a PNG |

If your CSV uses the reverse column convention pass `--pre-col segment_2 --post-col segment_1`.

### Other utilities

```bash
# Export SWC files for a batch of IDs into a single folder
python src/download_swcs.py \
  --segment-ids-file data/segment_ids/off_bk_seg_ids.txt \
  --export-dir outputs/swcs \
  --flatone-output-dir outputs/cells \
  --warped

# Check EyeWire II token validity
python src/check_eyewire_token.py

# Validate segment IDs against an annotations CSV
python src/validate_segment_id.py --csv annotations_flat.csv 720575940587958525

# Interactive 3D viewer
flatone view3d --output-dir outputs/cells [--warped]

# End-to-end pipeline validation
bash scripts/prove_flatone_pipeline.sh <SEGMENT_ID> outputs/proof
```

## Repository structure

```
retina-connectomics/
├── flatone/                            installable CLI package
│   ├── flatone/cli.py                  mesh → skeleton → warp pipeline
│   └── pyproject.toml
├── src/                                analysis scripts
│   ├── orientation_relations.py        PCA-based orientation analysis  ← main analysis
│   ├── hull_sac_analysis.py            XY hull of post cell + pre-SAC containment test
│   ├── infer_annotations.py            Neuroglancer annotation inference + segment ID export
│   ├── swc_orientations.py             lightweight SWC-only orientation alternative
│   ├── download_swcs.py                SWC batch export
│   ├── check_eyewire_token.py          token validation
│   └── validate_segment_id.py         segment ID checker
├── scripts/
│   └── prove_flatone_pipeline.sh       end-to-end validation
├── data/
│   └── segment_ids/                    input ID lists
│       └── off_bk_seg_ids.txt          281 OFF/BK bipolar segment IDs
└── valid_segment_ids.txt               762 validated segment IDs
```

Generated outputs go to `outputs/` (gitignored).

## Cell types

| Code | Cell type |
|------|-----------|
| BK | OFF bipolar (Kolb) |
| CB | Cone bipolar |
| GC | Ganglion cell |
| AC | Amacrine cell |
| A2 | AII amacrine |
| WAC | Wide-field amacrine |

## Dependencies

| Package | Purpose |
|---------|---------|
| `caveclient` | CAVE API authentication and data access |
| `cloud-volume` | EM mesh download from CloudVolume |
| `skeliner` | Mesh-to-skeleton conversion |
| `pywarper` | Warp skeletons/meshes to SAC coordinate frame |
| `numpy` | PCA and linear algebra |
| `matplotlib` | Visualization and stratification profiles |

## Notes

- Python 3.10+ required.
- The SAC reference planes in `strat_profile.png`: z = 0 µm is the ON SAC layer, z = 12 µm is the OFF SAC layer (pywarper coordinate convention).
- `swc_orientations.py` is a lightweight standalone alternative to `orientation_relations.py` that reads SWC files directly without requiring NPZ files or flatone integration.
- On macOS, `brew install suite-sparse` must be run before `pip install -e flatone/`.
