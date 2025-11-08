#!/usr/bin/env python3
"""
batch_task2_all.py

Run normalization (Min-Max and Unit Sphere), quantization (1024 bins), and save outputs
for every .obj in d:\3D_mixar\8samples. Produce output/task2_summary.csv summarizing per-file MSEs
(computed by dequantize->denormalize->compare to original). This complements Task 2 across the dataset.
"""
from pathlib import Path
import numpy as np
import csv
import sys

samples_dir = Path(r"d:\3D_mixar\8samples")
output_dir = Path(r"d:\3D_mixar\output")
output_dir.mkdir(exist_ok=True)
summary_csv = output_dir / 'task2_summary.csv'
failed = []
rows = []

try:
    import trimesh
except Exception as e:
    print("Please install trimesh and numpy: py -m pip install trimesh numpy")
    raise

n_bins = 1024

def normalize_minmax(vertices):
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    denom = vmax - vmin
    denom[denom == 0] = 1.0
    normalized = (vertices - vmin) / denom
    params = {'min': vmin, 'max': vmax}
    return normalized, params

def normalize_unit_sphere(vertices):
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    max_radius = np.max(np.linalg.norm(centered, axis=1))
    if max_radius == 0:
        max_radius = 1.0
    normalized = centered / max_radius
    params = {'centroid': centroid, 'radius': max_radius}
    return normalized, params

def quantize(normalized_vertices, n_bins=n_bins):
    q = np.floor(normalized_vertices * (n_bins - 1)).astype(int)
    q = np.clip(q, 0, n_bins - 1)
    return q

def dequantize(q, n_bins=n_bins):
    return q.astype(float) / (n_bins - 1)

def save_mesh(vertices, faces, filename):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(str(filename))

for p in sorted(samples_dir.glob('*.obj')):
    try:
        mesh = trimesh.load(p, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        name = p.stem
        print(f"Processing {p.name} (verts={len(verts)})")

        # Min-Max
        nm, nm_params = normalize_minmax(verts)
        save_mesh(nm, faces, output_dir / f"{name}_normalized_minmax.ply")
        q_nm = quantize(nm, n_bins)
        np.save(output_dir / f"{name}_quantized_minmax.npy", q_nm)
        dq_nm = dequantize(q_nm, n_bins)
        save_mesh(dq_nm, faces, output_dir / f"{name}_quantized_minmax.ply")

        # Unit Sphere
        ns, ns_params = normalize_unit_sphere(verts)
        save_mesh(ns, faces, output_dir / f"{name}_normalized_sphere.ply")
        q_ns = quantize(ns, n_bins)
        np.save(output_dir / f"{name}_quantized_sphere.npy", q_ns)
        dq_ns = dequantize(q_ns, n_bins)
        save_mesh(dq_ns, faces, output_dir / f"{name}_quantized_sphere.ply")

        # Dequantize and denormalize to compute MSE (reconstruction in original space)
        recon_nm = dq_nm * (nm_params['max'] - nm_params['min']) + nm_params['min']
        recon_ns = dq_ns * ns_params['radius'] + ns_params['centroid']

        mse_nm = float(np.mean(np.sum((verts - recon_nm)**2, axis=1)))
        mse_ns = float(np.mean(np.sum((verts - recon_ns)**2, axis=1)))

        rows.append({
            'file': p.name,
            'vertex_count': int(len(verts)),
            'minmax_mse': mse_nm,
            'unitsphere_mse': mse_ns,
            'minmax_norm': str(output_dir / f"{name}_normalized_minmax.ply"),
            'sphere_norm': str(output_dir / f"{name}_normalized_sphere.ply"),
            'minmax_quant': str(output_dir / f"{name}_quantized_minmax.npy"),
            'sphere_quant': str(output_dir / f"{name}_quantized_sphere.npy")
        })

        print(f"  Saved normalized & quantized for {name}: mse_minmax={mse_nm:.6e}, mse_sphere={mse_ns:.6e}")
    except Exception as e:
        print(f"Failed processing {p.name}: {e}")
        failed.append({'file': p.name, 'error': str(e)})

# write CSV
fieldnames = ['file','vertex_count','minmax_mse','unitsphere_mse','minmax_norm','sphere_norm','minmax_quant','sphere_quant']
with open(summary_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# write failed
with open(output_dir / 'task2_failed.json', 'w') as f:
    import json
    json.dump(failed, f, indent=2)

print(f"Wrote summary CSV: {summary_csv}")
print(f"Wrote failures: {output_dir / 'task2_failed.json'}")
