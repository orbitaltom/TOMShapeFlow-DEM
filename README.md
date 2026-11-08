# TOMShapeFlow-DEM: particle shape and angle of repose

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
<!-- DOI placeholder: replace XXXXXXX after the first Zenodo release. -->

Discrete-element (DEM) study of how grain shape (roundness and aspect ratio) sets the angle of repose of a
granular material, for handling lunar regolith in ISRU systems. The question behind it is how much a preprocessing
(rounding) step would change the bulk behaviour. The study uses the Chrono DEM-Engine (DEME) on one GPU.

Twelve grain shapes are made from overlapping spheres: Wadell roundness 0.13, 0.45, 0.80 and 1.0, each at aspect
ratio (b/a = c/a) 0.6, 0.8 and 1.0. The grains are 2 mm. Each shape is tested in the same rough-floor slump test
at Earth gravity, two seeds per shape. Three shapes were also tested at lunar gravity. Rolling resistance is
calibrated on a reference angular shape (L0) to 40°. The lunar highland simulant LHS-1 measures 39.6°.

## Key results

### Angle of repose, Earth gravity (mean of 2 seeds ± half the seed range)

| Roundness R \ aspect A | 0.6 | 0.8 | 1.0 |
|---|---|---|---|
| **0.13** | 39.38 ± 1.25° | 36.92 ± 0.86° | 35.45 ± 1.32° |
| **0.45** | 36.21 ± 0.08° | 35.02 ± 0.20° | 33.37 ± 0.30° |
| **0.80** | 37.31 ± 0.64° | 33.25 ± 0.48° | 33.50 ± 0.26° |
| **1.0** | 37.14 ± 0.81° | 30.72 ± 1.14° | 30.63 ± 0.95° |

Rows and columns are the target values; measured R and A for each shape are in
`results/repose_grid/grid_results.csv` and `shapes/repose_grid/shapes_summary.csv`.

### Design surface

The design surface is fitted over the 12 shapes, with R = measured roundness and A = measured (b/a + c/a)/2:

**angle ≈ 46.0 − 4.3·R − 10.7·A (degrees), ±1.6° (leave-one-out RMSE)**

- **Fit quality:** in-sample RMSE 1.18°.
- **Worst leave-one-out miss:** R = 1.0 / A = 0.6, which the fit predicts 2.95° too low.
- **Alternative forms:** adding an R² term, or using R·(1−A) alone, gave larger leave-one-out errors (1.75° and 2.85°).
- **Caveat:** the form was chosen as the lowest of the three, so the ±1.6° is slightly optimistic.
- **Data:** `results/repose_grid/design_fit.json`.
- **Blind test:** before its runs finished, the fit to the other 11 shapes predicted 38.5° for R 0.45 / A 0.6. The measured value was 36.21°, a miss of 2.3°.

### Lunar gravity (1.62 m/s²), 2 seeds each

Lunar minus Earth is within ±1.2° for all three shapes, so no gravity effect is resolved:

| Shape | Lunar | Earth | Lunar − Earth (±1 sd) |
|---|---|---|---|
| R 1.0, A 0.8 | 31.80° | 30.72° | +1.08 ± 1.14° |
| R 0.45, A 0.8 | 35.21° | 35.02° | +0.19 ± 1.14° |
| R 0.13, A 0.8 | 35.67° | 36.92° | −1.25 ± 1.14° |

### Calibration anchor

The L0 reference shape (R 0.132, b/a 0.765), at the rolling resistance used for the grid, gives **39.28 ± 0.44°**
(2 seeds). The target was 40°, and LHS-1 measures 39.6°.

### Time-step convergence

On L0, with 3 seeds at each time step:

| Time step | Mean ± SE |
|---|---|
| 5e-6 s | 41.54 ± 0.67° |
| 2.5e-6 s | 41.47 ± 0.98° |

The two agree within one SE. Most runs use 5e-6 s. The shape with the most spheres per grain (R 0.45 / A 0.6) and
the second lunar seed of two shapes used 2.5e-6 s. At 5e-6 s that shape exhausted the 12 GB of GPU memory; at
2.5e-6 s its adaptive contact margin is smaller, so it fits.

The full method record is `results/methods_log.md`: dated decisions, calibration, checks and every deviation.

## Method in brief

| Parameter | Value |
|---|---|
| Solver | DEME 3.0.19, Hertz-Mindlin contacts, E = 1e7 Pa, grain density 3100 kg/m³ |
| Friction | grain-grain µ 0.6; grain-wall (6061 aluminium) µ 0.35 |
| Rolling resistance | Crr 0.22, grain-grain and grain-wall |
| Time step | 5e-6 s |
| Column | 0.08 m wide × 0.04 m tall, 8749 grains, in a 20 mm slice with frictionless side walls |
| Floor | one frozen layer of the same grains |

Test sequence:
1. The column is filled in stages and settled.
2. The gate opens and the heap slumps until the grains stop (at least 1 s).
3. The angle is the slope of the heap profile over 15–85 % of its height, measured from the top of the frozen layer.

For lunar runs, all settle times and speed thresholds are scaled by √(9.81/1.62).

## How to run

Tested setup:

| Component | Version |
|---|---|
| OS | Windows 11 with WSL2 (Ubuntu 24.04) |
| GPU | NVIDIA RTX 3500 Ada, 12 GB |
| CUDA | 12 |
| Python | 3.12 |
| DEME | 3.0.19 |

```bash
python3 -m venv ~/deme-env && source ~/deme-env/bin/activate
pip install "deme[cuda12]==3.0.19"
pip install -r requirements.txt

# 1. Grain templates (already included in shapes/repose_grid/; regenerate only if needed, CPU only)
python scripts/08_repose_grid_shapes.py

# 2. One slump run: shape, seed, gravity; --frames saves 0.02 s frames for rendering; --dt overrides the time step
python scripts/09_repose_grid_run.py --shape R045_A08 --seed 1 --frames
python scripts/09_repose_grid_run.py --shape R045_A08 --seed 2 --gravity lunar --dt 2.5e-6

# 3. Results table, design surface, design chart
python scripts/12_grid_table.py        # -> results/repose_grid/grid_results.csv
python scripts/11_design_fit.py        # -> results/repose_grid/design_fit.json
python scripts/11_design_chart.py      # -> results/repose_grid/figures/: design_chart.png, shape_strip.png, design_table.csv

# 4. Figures and videos (videos and stills need the saved frames, see "Data bundle")
python scripts/10_repose_grid_media.py plots      # contour map and angle vs roundness
python scripts/10_repose_grid_media.py heaps      # heap stills and the 12-panel montage
python scripts/10_repose_grid_media.py gallery    # template gallery
python scripts/10_repose_grid_media.py video R045_A08
```

Run times on the tested GPU:
- **Typical:** 6 min (spheres) to about 4 h (angular shapes, 30–50 spheres per grain) per Earth run.
- **Lunar runs:** about 2.5 times longer.

`scripts/03_repose_calibration.py` is the calibration driver used to set rolling resistance on L0, for example
`python scripts/03_repose_calibration.py --level L0 --crr 0.22`. The L0 templates are in `shapes/templates/`, so
the calibration and the L0 anchor (`python scripts/09_repose_grid_run.py --shape L0 --seed 1`) can be rerun.
`config.json` holds every parameter. Its chute and hopper sections belong to later phases of the study and are not
used by the repose scripts.

Results go to `results/` by default. Set `SHAPEFLOW_RESULTS=/some/dir` to write elsewhere.

## Reading a template

Each grain shape has 8 templates (`shapes/repose_grid/templates/<shape>_t<k>.csv`). Shape names are
`R<roundness x100>_A<aspect x10>`; for example, `R045_A08` is roundness 0.45, aspect 0.8. Each file lists the
spheres of one rigid grain:

```
x,y,z,r
6.765445905e-04,2.303055504e-04,2.877114289e-04,4.236745558e-04
...
```

Each row is one sphere: centre `x,y,z` and radius `r`, in metres. Coordinates are in the grain's principal-axis
frame, with the centre of mass at the origin.

`templates_index.json` gives each template's descriptors:

| Field | Meaning |
|---|---|
| `mass_kg` | grain mass |
| `moi_kg_m2` | principal moments of inertia, kg·m² (DEME `LoadClumpType` takes the mass, the principal moments, the radii and the centres) |
| `d_v_m` | volume-equivalent diameter, m |
| `corner_radius_ratio` | measured Wadell roundness |
| `aspect_b_a`, `aspect_c_a` | measured aspect ratios |
| `sphericity` | sphericity |
| `n_spheres` | spheres in the template |
| `bump97_pct_d` | 97th-percentile surface bump as % of d |

Volumes span d_v = 1.8–2.2 mm across the 8 templates. `shapes/repose_grid/shapes_summary.csv` has the per-shape
means, and `template_gallery.png` shows one template of each shape.

## Repository layout

```
shapeflow/            simulation package (repose test, solver setup, shape generation)
scripts/              08 templates, 09 one run, 10 figures/videos, 11 design fit and chart, 12 results table,
                      03 calibration driver
shapes/repose_grid/   grain templates (CSV), templates_index.json, shapes_summary.csv, template_gallery.png
shapes/templates/     L0 reference-shape templates (calibration anchor), templates_index.json
results/repose_grid/  grid_results.csv, design_fit.json, figures/ (design chart, shape strip, design table),
                      contour and angle-vs-roundness plots, media/heaps_montage.png, runs/<run>/summary.json + heap_profile.csv
results/methods_log.md
tools/                citation_metadata.py: writes CITATION.cff and .zenodo.json from one definition
```

Run folders are named `<shape>_<gravity>[_dt2.5e-06]_s<seed>`. A `_dt2.5e-06` suffix means that run used the
2.5e-6 s time step.

## Data bundle (separate Zenodo upload)

Data archive (CC BY 4.0): Zenodo DOI to follow.

These files are not in git:

| File | Contents | Size |
|---|---|---|
| `repose_grid_frames.zip` | slump frames, every 0.02 s: positions, orientations and velocities of every grain, seed-1 runs | 241.9 MB |
| `repose_grid_heaps.zip` | raw final grain states (`heap.csv`) and rms-speed time series of all 32 runs | 8.0 MB |
| `repose_grid_media.zip` | slump videos (speed-coloured, 0.5x) and heap stills | 45.7 MB |
| `repose_grid_logs.zip` | run logs, GPU-memory/step-rate watchdog logs, solver-call timing, queue record | 0.1 MB |

Unzip them into `results/repose_grid/` (for example `unzip repose_grid_frames.zip -d results/repose_grid/`); the
paths inside the zips are relative to that folder. The frames are needed to re-render videos and stills.

## How to cite

Please cite the archived release (DOI above) and see `CITATION.cff`:

> Torres, L. (2026). *TOMShapeFlow-DEM: particle shape and angle of repose, a DEM study for lunar regolith
> handling.* Torres Orbital Mining Inc. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

## License

- Code (`shapeflow/`, `scripts/`): MIT, see `LICENSE`.
- Shape templates and results (`shapes/`, `results/`, data bundle): CC BY 4.0, see `LICENSE-DATA`.
- The citation files (`CITATION.cff`, `.zenodo.json`) describe the software release and carry MIT. The data archive
  is a separate Zenodo dataset under CC BY 4.0.
