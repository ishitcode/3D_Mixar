#!/usr/bin/env python3
"""
batch_task3_all.py

For every .obj in d:\3D_mixar\8samples, dequantize and denormalize the quantized outputs
(from task2) and compute per-axis MSE and MAE. Save per-file plots and colored PLYs.
Writes: output/task3_summary.csv and per-file PNG/PLY outputs in output/.
"""
from pathlib import Path
import numpy as np
import csv
import sys

samples_dir = Path(r"d:\3D_mixar\8samples")
output_dir = Path(r"d:\3D_mixar\output")
output_dir.mkdir(exist_ok=True)
summary_csv = output_dir / 'task3_summary.csv'

try:
    import trimesh
    import matplotlib.pyplot as plt
except Exception as e:
    print("Please install trimesh and matplotlib: py -m pip install trimesh matplotlib")
    raise

n_bins = 1024

def dequantize(q, n_bins=n_bins):
    return q.astype(float) / (n_bins - 1)

rows = []
failed = []
for p in sorted(samples_dir.glob('*.obj')):
    try:
        mesh = trimesh.load(p, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        name = p.stem
        print(f"Processing {p.name} (verts={len(verts)})")

        # Paths to quantized arrays (saved by batch_task2_all)
        q_min_path = output_dir / f"{name}_quantized_minmax.npy"
        q_sph_path = output_dir / f"{name}_quantized_sphere.npy"

        # If quantized files missing, recompute quantization from normalization
        if not q_min_path.exists() or not q_sph_path.exists():
            # recompute normalization and quantization
            vmin = verts.min(axis=0)
            vmax = verts.max(axis=0)
            denom = vmax - vmin
            denom[denom == 0] = 1.0
            norm_min = (verts - vmin) / denom
            centroid = verts.mean(axis=0)
            centered = verts - centroid
            rmax = np.max(np.linalg.norm(centered, axis=1))
            if rmax == 0:
                rmax = 1.0
            norm_sph = centered / rmax
            q_min = np.clip(np.floor(norm_min * (n_bins - 1)).astype(int), 0, n_bins-1)
            q_sph = np.clip(np.floor(norm_sph * (n_bins - 1)).astype(int), 0, n_bins-1)
            np.save(q_min_path, q_min)
            np.save(q_sph_path, q_sph)
        else:
            q_min = np.load(q_min_path)
            q_sph = np.load(q_sph_path)

        # Dequantize
        dq_min = dequantize(q_min)
        dq_sph = dequantize(q_sph)

        # Recompute normalization params for denormalization
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        centroid = verts.mean(axis=0)
        centered = verts - centroid
        rmax = np.max(np.linalg.norm(centered, axis=1))
        if rmax == 0:
            rmax = 1.0

        recon_min = dq_min * (vmax - vmin) + vmin
        recon_sph = dq_sph * rmax + centroid

        # Compute per-axis MSE and MAE
        mse_min_axis = np.mean((verts - recon_min)**2, axis=0)
        mae_min_axis = np.mean(np.abs(verts - recon_min), axis=0)
        mse_sph_axis = np.mean((verts - recon_sph)**2, axis=0)
        mae_sph_axis = np.mean(np.abs(verts - recon_sph), axis=0)
        mse_min_overall = float(np.mean(np.sum((verts - recon_min)**2, axis=1)))
        mae_min_overall = float(np.mean(np.sum(np.abs(verts - recon_min), axis=1)))
        mse_sph_overall = float(np.mean(np.sum((verts - recon_sph)**2, axis=1)))
        mae_sph_overall = float(np.mean(np.sum(np.abs(verts - recon_sph), axis=1)))

        # Save reconstructed meshes
        recon_min_ply = output_dir / f"{name}_recon_minmax.ply"
        recon_sph_ply = output_dir / f"{name}_recon_sphere.ply"
        trimesh.Trimesh(vertices=recon_min, faces=faces, process=False).export(recon_min_ply)
        trimesh.Trimesh(vertices=recon_sph, faces=faces, process=False).export(recon_sph_ply)

        # Save per-axis histograms and bar plots
        errors_min = verts - recon_min
        errors_sph = verts - recon_sph
        # histograms
        hist_png = output_dir / f"{name}_per_axis_errors.png"
        fig, axs = plt.subplots(2,3, figsize=(12,6))
        axes = ['x','y','z']
        for i in range(3):
            axs[0,i].hist(errors_min[:,i], bins=80, color='tab:blue')
            axs[0,i].set_title(f'MinMax {axes[i]}-axis')
            axs[1,i].hist(errors_sph[:,i], bins=80, color='tab:orange')
            axs[1,i].set_title(f'UnitSphere {axes[i]}-axis')
        plt.tight_layout()
        fig.savefig(hist_png)
        plt.close(fig)

        # MSE/MAE bar
        mse_mae_png = output_dir / f"{name}_mse_mae_bar.png"
        x = np.arange(3)
        width = 0.35
        fig2, ax2 = plt.subplots(1,2, figsize=(10,4))
        ax2[0].bar(x - width/2, mse_min_axis, width, label='MinMax')
        ax2[0].bar(x + width/2, mse_sph_axis, width, label='UnitSphere')
        ax2[0].set_xticks(x); ax2[0].set_xticklabels(axes); ax2[0].set_title('MSE per axis')
        ax2[0].legend()
        ax2[1].bar(x - width/2, mae_min_axis, width, label='MinMax')
        ax2[1].bar(x + width/2, mae_sph_axis, width, label='UnitSphere')
        ax2[1].set_xticks(x); ax2[1].set_xticklabels(axes); ax2[1].set_title('MAE per axis')
        ax2[1].legend()
        plt.tight_layout()
        fig2.savefig(mse_mae_png)
        plt.close(fig2)

        # Colored reconstructions by error magnitude
        err_mag_min = np.linalg.norm(verts - recon_min, axis=1)
        err_mag_sph = np.linalg.norm(verts - recon_sph, axis=1)
        def to_colors(mag):
            m = mag.copy()
            if m.max() > 0:
                m = (m / m.max()) * 255.0
            cols = np.stack([m, 50 + 0*m, 255 - m], axis=1).astype(np.uint8)
            return cols
        cols_min = to_colors(err_mag_min)
        cols_sph = to_colors(err_mag_sph)
        recon_min_col = output_dir / f"{name}_recon_min_colored.ply"
        recon_sph_col = output_dir / f"{name}_recon_sph_colored.ply"
        trimesh.Trimesh(vertices=recon_min, faces=faces, vertex_colors=cols_min, process=False).export(recon_min_col)
        trimesh.Trimesh(vertices=recon_sph, faces=faces, vertex_colors=cols_sph, process=False).export(recon_sph_col)

        rows.append({
            'file': p.name,
            'vertex_count': int(len(verts)),
            'mse_min_x': float(mse_min_axis[0]), 'mse_min_y': float(mse_min_axis[1]), 'mse_min_z': float(mse_min_axis[2]),
            'mae_min_x': float(mae_min_axis[0]), 'mae_min_y': float(mae_min_axis[1]), 'mae_min_z': float(mae_min_axis[2]),
            'mse_sph_x': float(mse_sph_axis[0]), 'mse_sph_y': float(mse_sph_axis[1]), 'mse_sph_z': float(mse_sph_axis[2]),
            'mae_sph_x': float(mae_sph_axis[0]), 'mae_sph_y': float(mae_sph_axis[1]), 'mae_sph_z': float(mae_sph_axis[2]),
            'mse_min_overall': mse_min_overall,
            'mse_sph_overall': mse_sph_overall,
            'mae_min_overall': mae_min_overall,
            'mae_sph_overall': mae_sph_overall,
            'recon_min_ply': str(recon_min_ply), 'recon_sph_ply': str(recon_sph_ply),
            'hist_png': str(hist_png), 'mse_mae_png': str(mse_mae_png),
            'recon_min_col': str(recon_min_col), 'recon_sph_col': str(recon_sph_col)
        })

        print(f"  Saved outputs for {name}")

    except Exception as e:
        print(f"Failed processing {p.name}: {e}")
        failed.append({'file': p.name, 'error': str(e)})

# write CSV
fieldnames = list(rows[0].keys()) if rows else ['file']
with open(summary_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

with open(output_dir / 'task3_failed.json', 'w') as f:
    import json
    json.dump(failed, f, indent=2)

print(f"Wrote task3 summary CSV: {summary_csv}")
print(f"Wrote failures: {output_dir / 'task3_failed.json'}")
