# SeamGPT (Mesh Normalization, Quantization, and Error Analysis)

GitHub repo link : https://github.com/ishitcode/3D_Mixar

This repository contains scripts and outputs for the mesh preprocessing assignment.

Contents
- `mesh_processing.ipynb` — Jupyter notebook with the full pipeline (load, normalize, quantize, dequantize, evaluate). 
- `mesh_processing_summary_robust.py` — summary script that aggregates results from `/mnt/data/mesh_outputs`.
- `task1_load_inspect.py` — Task 1: load and print mesh statistics for `8samples/branch.obj`.
- `task2_normalize_quantize.py` — Task 2: normalize (Min–Max and Unit Sphere), quantize (1024 bins), and save outputs.
- `task3_error_analysis.py` — Task 3: dequantize, denormalize, compute MSE/MAE, save plots and colored reconstructions.
- `seam_tokenizer.py` — Prototype seam tokenization encode/decode utilities.
- `output/` — folder with generated `.ply`, `.npy`, and plot PNGs.
- `final_report.pdf` — brief PDF summarizing results (generated programmatically).

Requirements

Install dependencies (Windows PowerShell):

```powershell
py -m pip install --upgrade pip
py -m pip install trimesh numpy matplotlib
```

If you want rendered 3D images using trimesh offscreen renderer, also install `pyglet`:

```powershell
py -m pip install pyglet
```

How to run
1. Inspect Task 1 (statistics):

```powershell
py d:\3D_mixar\task1_load_inspect.py
```

2. Run Task 2 (normalize & quantize):

```powershell
py d:\3D_mixar\task2_normalize_quantize.py
```

3. Run Task 3 (dequantize, denormalize, error analysis):

```powershell
py d:\3D_mixar\task3_error_analysis.py
```

4. Generate final PDF (if you want to regenerate):

```powershell
py d:\3D_mixar\generate_report.py
```

Notes / Observations

- Min–Max normalization preserved the mesh structure much better for `branch.obj` (Overall MSE ≈ 2.34e-06) than Unit-Sphere normalization (Overall MSE ≈ 1.98e-01).
- Unit-Sphere re-centers and rescales isotropically, which can amplify quantization error on axes with small original ranges.
- Per-axis plots and colored PLYs in `output/` show where errors concentrate; adaptive per-axis quantization could reduce errors further.

# Mesh Normalization, Quantization, and Error Analysis

This workspace contains a reproducible pipeline to:

- Inspect 3D meshes (Task 1)
- Normalize and quantize vertex coordinates (Task 2)
- Dequantize, denormalize and compute reconstruction errors/visualizations (Task 3)
- (Bonus) A seam-tokenizer prototype for encoding seam polylines

Files of interest
- `mesh_processing.ipynb` — notebook with an interactive pipeline and notes
- `task1_load_inspect.py` / `inspect_all_samples.py` — Task 1: mesh statistics and dataset inspection
- `task2_normalize_quantize.py` / `batch_task2_all.py` — Task 2: normalization + quantization
- `task3_error_analysis.py` / `batch_task3_all.py` — Task 3: reconstruction, MSE/MAE, plots
- `seam_tokenizer.py` — seam tokenization encode/decode prototype (self-test included)
- `organize_output.py` — reorganizes `output/` into `task1|task2|task3/<sample>/` and `output/summary/`
- `generate_report.py` — programmatically generates `final_report.pdf` (full or seam-only)
- `output/` — generated artifacts (per-sample PLYs, .npy quantized arrays, PNG plots, CSV summaries)

Step 0 — (optional) create & activate a venv

Recommended so package installs are isolated. Run from D:\3D_mixar.

```powershell
cd D:\3D_mixar
py -3 -m venv .venv
# activate
.\.venv\Scripts\Activate.ps1
```

Step 1 — Install Python packages (recommended)

Installs the packages used in the project. Offscreen rendering (`pyglet`) is optional.

```powershell
py -3 -m pip install --upgrade pip
py -3 -m pip install numpy trimesh matplotlib scipy scikit-learn

# Optional: for nicer offscreen 3D screenshots
py -3 -m pip install pyglet

# Optional: Open3D for advanced mesh rendering/IO
py -3 -m pip install open3d
```

If any import errors occur (e.g., ImportError when running scripts), pip install the missing module shown in the error.

Step 2 — Run Task 1: Inspect meshes (per-file stats)

This script inspects meshes and writes `output/inspections.csv`.

```powershell
py -3 .\inspect_all_samples.py
```

Check output:

```powershell
Get-ChildItem .\output\summary\inspections.csv
Get-Content .\output\summary\inspections.csv -TotalCount 20
```

Step 3 — Run Task 2: Normalize & Quantize (single or batch)

You may run per-file script or the batch driver.

```powershell

py -3 .\batch_task2_all.py
```

Step 4 — Run Task 3: Dequantize, Denormalize & Error Analysis

Run per-file or batch version.

```powershell

py -3 .\batch_task3_all.py
```

Step 5 — (Optional) Run seam tokenizer self-test

We added `seam_tokenizer.py`. This runs a built-in demo to validate the tokenizer.

```powershell
py -3 .\seam_tokenizer.py
```

Notes: This only tests tokenizer logic. Seam extraction (edge chains from meshes) is not yet integrated automatically; if you want I can add `seam_extractor.py` and integrate it into Task 2/3.

Step 6 — Organize outputs (move per-sample files into task folders)

The `organize_output.py` script creates `output/task1|task2|task3/<sample>/` and `output/summary/` and moves files accordingly.

```powershell
py -3 .\organize_output.py
```

Step 7 — Regenerate the final PDF report

The report scanner will look for summary CSVs and seam JSON files under `output/task2/<sample>/` (if any).

```powershell
py -3 .\generate_report.py
```

Step 8 — Package for submission (full or trimmed)

Full archive (everything in workspace) — adjust paths if you only want select folders:

```powershell
Compress-Archive -Path .\* -DestinationPath D:\3D_mixar\submission_package_full.zip -Force
```

Trimmed archive example: include scripts, `output/summary/`, and `final_report.pdf`, but exclude large per-sample PLYs and `.npy` if you want to shrink size:

```powershell
# create a temporary staging folder
New-Item -ItemType Directory -Path .\staging -Force
Copy-Item .\*.py .\staging\ -Force
Copy-Item .\README.md .\staging\ -Force
Copy-Item .\final_report.pdf .\staging\ -Force
Copy-Item .\output\summary\* .\staging\output_summary\ -Recurse -Force
Compress-Archive -Path .\staging\* -DestinationPath D:\3D_mixar\submission_package_trimmed.zip -Force
Remove-Item -Recurse -Force .\staging
```

Quick verification commands

List summary CSVs, seam JSONs, and the final PDF:

```powershell
Get-ChildItem .\output\summary\*.csv
Get-ChildItem .\output\task2\*\seams*.json -Recurse -ErrorAction SilentlyContinue
Test-Path .\final_report.pdf
```

Troubleshooting & tips

- If a script reports "No module named X", pip install that module into the active environment.
- If offscreen rendering fails even after installing pyglet, use fallback previews (the scripts already fall back to 2D previews).
- Seam tokens: `seam_tokenizer.py` is present and tested. To include seam tokens in the report you must produce per-sample JSON files (recommended name `seams.json`) inside `output/task2/<sample>/` with the seam token structure. I can implement `seam_extractor.py` and integrate it if you want full automatic seam outputs.
- If the pipeline errors on large meshes, try increasing memory or processing one sample at a time.

Suggested end-to-end one-shot (assuming packages are installed)

All together, from D:\\3D_mixar:

```powershell
cd D:\3D_mixar
py -3 .\inspect_all_samples.py
py -3 .\batch_task2_all.py
py -3 .\batch_task3_all.py
py -3 .\organize_output.py
py -3 .\generate_report.py
Start-Process .\final_report.pdf
```

Report and outputs
- `d:\3D_mixar\final_report.pdf` — full report including images and CSV tables (regenerated by `generate_report.py`)
- `d:\3D_mixar\final_report_seam.pdf` — seam-only report (generated with `--seam-only`)
- `d:\3D_mixar\output\summary\` — summary CSVs: `inspections.csv`, `task2_summary.csv`, `task3_summary.csv`
- `d:\3D_mixar\output\task2\<sample>\` and `...\task3\<sample>\` — per-sample artifacts (PNGs, PLYs, `.npy` quantized arrays)

Seam tokenizer and integration
- `seam_tokenizer.py` provides an encode/decode prototype and a self-test. It does not automatically extract seams from meshes yet.
- To include seam-token outputs in the report you can either:
	- Run an explicit seam extraction script (not yet added) that writes `seams.json` into `output/task2/<sample>/`, or
	- Manually place `seams.json` files in `output/task2/<sample>/` (format: list of seams; each seam can be a token list or an object with `ints` and `inv` mappings). The report generator will detect and summarize them.

Tips & troubleshooting
- If you see "No module named ..." when running a script, install the missing package into the active Python environment.
- If offscreen rendering fails, scripts fall back to 2D previews; installing `pyglet` often fixes it.
- If you want smaller ZIPs for submission, use `organize_output.py` then compress only `output/summary/`, scripts, and `final_report.pdf`.

Packaging for submission (example)

```powershell
# create a small staging area with required artifacts
New-Item -ItemType Directory -Path .\staging -Force
Copy-Item .\*.py .\staging\ -Force
Copy-Item .\README.md .\staging\ -Force
Copy-Item .\final_report.pdf .\staging\ -Force
Copy-Item .\output\summary\* .\staging\output_summary\ -Recurse -Force
Compress-Archive -Path .\staging\* -DestinationPath D:\3D_mixar\submission_package_trimmed.zip -Force
Remove-Item -Recurse -Force .\staging
```

Further work / extensions
- Add `seam_extractor.py` (dihedral + boundary edge detector) and integrate it in `batch_task2_all.py` to produce `seams.json` automatically.
- Implement adaptive per-axis quantization or local density-based quantization to reduce reconstruction error in high-detail areas.
- Improve report layout (tables, multi-image layouts) or render high-quality PLY screenshots via Open3D/offscreen renderers.

Contact / next steps
If you'd like I can:
- Implement automatic seam extraction and integrate tokenizer into the pipeline (dihedral+boundary recommended, specify a threshold or choose auto-tune),
- Or produce a trimmed submission ZIP with only specific artifacts.

---