#!/usr/bin/env python3
"""
inspect_all_samples.py

Inspects all .obj files in d:\3D_mixar\8samples and writes a CSV summary to output/inspections.csv
Also records any failed files to output/failed_files.json
"""
from pathlib import Path
import csv
import json
import sys

samples_dir = Path(r"d:\3D_mixar\8samples")
output_dir = Path(r"d:\3D_mixar\output")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / 'inspections.csv'
failed_path = output_dir / 'failed_files.json'

try:
    import trimesh
    import numpy as np
except Exception as e:
    print("Missing required packages: trimesh, numpy. Install with: py -m pip install trimesh numpy")
    raise

obj_files = sorted(samples_dir.glob('*.obj'))
if not obj_files:
    print(f"No .obj files found in {samples_dir}")
    sys.exit(0)

rows = []
failed = []
for p in obj_files:
    try:
        mesh = trimesh.load(p, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        n = verts.shape[0]
        if n == 0:
            raise ValueError('no vertices')
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        means = verts.mean(axis=0)
        stds = verts.std(axis=0)
        row = {
            'file': str(p.name),
            'vertex_count': int(n),
            'face_count': int(faces.shape[0]) if faces is not None else 0,
            'x_min': float(mins[0]), 'x_max': float(maxs[0]), 'x_mean': float(means[0]), 'x_std': float(stds[0]),
            'y_min': float(mins[1]), 'y_max': float(maxs[1]), 'y_mean': float(means[1]), 'y_std': float(stds[1]),
            'z_min': float(mins[2]), 'z_max': float(maxs[2]), 'z_mean': float(means[2]), 'z_std': float(stds[2])
        }
        rows.append(row)
        print(f"Processed: {p.name} (verts={n}, faces={faces.shape[0]})")
    except Exception as e:
        print(f"Failed: {p.name} -> {e}")
        failed.append({'file': str(p.name), 'error': str(e)})

# write CSV
fieldnames = ['file','vertex_count','face_count',
              'x_min','x_max','x_mean','x_std',
              'y_min','y_max','y_mean','y_std',
              'z_min','z_max','z_mean','z_std']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

with open(failed_path, 'w') as f:
    json.dump(failed, f, indent=2)

print(f"\nWrote CSV summary to: {csv_path}")
print(f"Wrote failed files to: {failed_path}")
