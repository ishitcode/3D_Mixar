"""
Organize files in d:\3D_mixar\output into per-task and per-sample subfolders.
Usage: py -3 organize_output.py
"""
import shutil
from pathlib import Path

ROOT = Path(r"d:\3D_mixar")
OUT = ROOT / 'output'

samples = ['branch','cylinder','explosive','fence','girl','person','table','talwar']
tasks = ['task1','task2','task3','bonus']

# create task folders and per-sample subfolders
for t in tasks:
    (OUT / t).mkdir(exist_ok=True)
    for s in samples:
        (OUT / t / s).mkdir(exist_ok=True)

# summary folder
(OUT / 'summary').mkdir(exist_ok=True)

# helper to move with overwrite
def move(src: Path, dst: Path):
    dst_parent = dst.parent
    dst_parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            dst.unlink()
        except Exception:
            pass
    shutil.move(str(src), str(dst))

# classify files
for p in list(OUT.iterdir()):
    if p.is_dir():
        # skip task folders we created
        if p.name in tasks + ['summary']:
            continue
        # else continue (we'll not recurse)
        continue
    name = p.name
    lname = name.lower()

    # summary files -> summary
    summary_names = ['inspections.csv','task2_summary.csv','task3_summary.csv','failed_files.json','task2_failed.json','task3_failed.json','final_report.pdf','submission_package.zip','mse_mae_per_axis.png','per_axis_error_histograms.png','quantized_minmax.npy','quantized_sphere.npy','quantized_minmax.ply','quantized_sphere.ply','normalized_minmax.ply','normalized_sphere.ply','reconstructed_minmax.ply','reconstructed_sphere.ply','recon_min_colored.ply','recon_sph_colored.ply','preview_normalized_minmax.png','preview_normalized_sphere.png','preview_quantized_minmax.png','preview_quantized_sphere.png']
    if lname in summary_names or (lname.startswith('preview_') and lname.endswith('.png')):
        try:
            move(p, OUT / 'summary' / name)
            print(f"Moved summary: {name} -> output/summary/")
        except Exception as e:
            print(f"Failed to move summary {name}: {e}")
        continue

    placed = False
    for s in samples:
        if lname.startswith(s + '_') or lname.startswith(s + '-') or lname == (s + '.ply'):
            # per-sample file
            # Determine task by keywords
            if ('normalized' in lname) or ('quantized' in lname) or ('preview_quant' in lname) or ('_normalized_' in lname):
                dest = OUT / 'task2' / s / name
            elif ('recon' in lname) or ('per_axis' in lname) or ('mse_mae' in lname) or ('_recon_' in lname) or ('_recon' in lname):
                dest = OUT / 'task3' / s / name
            elif ('_mse_mae_bar' in lname) or lname.endswith('mse_mae_bar.png'):
                dest = OUT / 'task3' / s / name
            else:
                # fallback -> task1
                dest = OUT / 'task1' / s / name
            try:
                move(p, dest)
                print(f"Moved: {name} -> {dest.relative_to(ROOT)}")
            except Exception as e:
                print(f"Failed to move {name}: {e}")
            placed = True
            break
    if not placed:
        # unknown file, put into summary
        try:
            move(p, OUT / 'summary' / name)
            print(f"Moved unknown: {name} -> output/summary/")
        except Exception as e:
            print(f"Failed to move unknown {name}: {e}")

print('Organization complete.')
