#!/usr/bin/env python3
"""
generate_report.py
Create a short PDF report summarizing Task 1-3 results and include plots saved in output/.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import csv
import json
from pathlib import PurePath
import itertools
import argparse

output = Path(r"d:\3D_mixar\output")
report_path = Path(r"d:\3D_mixar\final_report.pdf")

# CLI: allow a seam-only report
parser = argparse.ArgumentParser(description='Generate final_report.pdf; use --seam-only to produce seam-only PDF')
parser.add_argument('--seam-only', action='store_true', help='produce a seam-only report (writes final_report_seam.pdf)')
args = parser.parse_args()
SEAM_ONLY = args.seam_only
if SEAM_ONLY:
    report_path = Path(r"d:\3D_mixar\final_report_seam.pdf")

title = "Mesh Normalization, Quantization & Error Analysis - Final Report"

def read_csv_table(path):
    if not path.exists():
        return []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)

inspections = read_csv_table(output / 'inspections.csv')
task2 = read_csv_table(output / 'task2_summary.csv')
task3 = read_csv_table(output / 'task3_summary.csv')

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def agg_task2(task2_rows):
    if not task2_rows:
        return None
    minmax = [safe_float(r.get('minmax_mse')) for r in task2_rows if r.get('minmax_mse')]
    sphere = [safe_float(r.get('unitsphere_mse')) for r in task2_rows if r.get('unitsphere_mse')]
    return {
        'n': len(task2_rows),
        'avg_minmax_mse': np.mean(minmax) if minmax else None,
        'avg_unitsphere_mse': np.mean(sphere) if sphere else None
    }

agg2 = agg_task2(task2)

with PdfPages(report_path) as pdf:
    # Title + Summary
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.axis('off')
    plt.text(0.5, 0.9, title, ha='center', va='center', fontsize=16, weight='bold')
    y = 0.78
    plt.text(0.1, y, 'Dataset: d:/3D_mixar/8samples (8 OBJ files)', fontsize=11)
    y -= 0.04
    if agg2:
        plt.text(0.1, y, f"Processed meshes: {agg2['n']}", fontsize=10)
        y -= 0.03
        plt.text(0.1, y, f"Average Min-Max MSE: {agg2['avg_minmax_mse']:.6e}", fontsize=10)
        y -= 0.03
        plt.text(0.1, y, f"Average Unit-Sphere MSE: {agg2['avg_unitsphere_mse']:.6e}", fontsize=10)
        y -= 0.03
        better = 'Min-Max' if (agg2['avg_minmax_mse'] or float('inf')) < (agg2['avg_unitsphere_mse'] or float('inf')) else 'Unit-Sphere'
        plt.text(0.1, y, f"Preferred method (lower avg MSE): {better}", fontsize=10)
        y -= 0.04
    # Short observations
    lines = [
        "Observations:",
        "- Min–Max normalization consistently produced lower reconstruction MSE across these samples.",
        "- Unit-Sphere normalization often amplifies quantization error on axes with small original ranges.",
        "- Per-sample plots (included) show axis-specific error distributions and hotspots.",
        "- Colored reconstructions (vertex colors) highlight high-error regions for further inspection.",
    ]
    for ln in lines:
        plt.text(0.1, y, ln, fontsize=9)
        y -= 0.03
    pdf.savefig(fig)
    plt.close(fig)

    # Inspections table (first page)
    if inspections:
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.5, 0.95, 'Per-file Inspections (vertex / face counts & axis stats)', ha='center', fontsize=12, weight='bold')
        y = 0.9
        for r in inspections:
            try:
                line = f"{r['file']}: verts={r['vertex_count']}, faces={r.get('face_count', '')} | x:[{float(r['x_min']):.4f},{float(r['x_max']):.4f}] y:[{float(r['y_min']):.4f},{float(r['y_max']):.4f}] z:[{float(r['z_min']):.4f},{float(r['z_max']):.4f}]"
            except Exception:
                line = f"{r.get('file','')}: data parsing error"
            plt.text(0.05, y, line, fontsize=8)
            y -= 0.035
            if y < 0.05:
                pdf.savefig(fig); plt.close(fig)

    # --- Include ALL images under output/ (recursively) ---
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    all_imgs = sorted([p for p in output.rglob('*') if p.suffix.lower() in img_extensions])
    if all_imgs:
        # Add a header page
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.5, 0.95, 'All Output Images (recursive)', ha='center', fontsize=14, weight='bold')
        y = 0.9
        plt.text(0.1, y, f'Total images found: {len(all_imgs)}', fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # Add each image on its own page; group small images 2-up per page when they are narrow
        for img_path in all_imgs:
            try:
                img = plt.imread(str(img_path))
            except Exception:
                continue
            fig = plt.figure(figsize=(8.27,11.69))
            plt.imshow(img)
            plt.axis('off')
            # caption with relative path
            rel = img_path.relative_to(output)
            plt.title(str(rel), fontsize=8)
            pdf.savefig(fig); plt.close(fig)

    # --- Include CSV tables (inspections, task2_summary, task3_summary) as text pages ---
    csv_files = [output / 'inspections.csv', output / 'task2_summary.csv', output / 'task3_summary.csv']
    for csvf in csv_files:
        if csvf.exists():
            try:
                with open(csvf, newline='') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
            except Exception:
                rows = []
            if not rows:
                continue
            # paginate rows into pages showing ~30 rows per page
            page_size = 30
            headers = rows[0]
            body = rows[1:]
            chunks = [body[i:i+page_size] for i in range(0, len(body), page_size)]
            for ci, chunk in enumerate(chunks):
                fig = plt.figure(figsize=(8.27,11.69))
                plt.axis('off')
                title_text = f"{csvf.name} (page {ci+1}/{len(chunks)})"
                plt.text(0.5, 0.95, title_text, ha='center', fontsize=12, weight='bold')
                y = 0.9
                # header
                header_line = ' | '.join(headers)
                plt.text(0.01, y, header_line, fontsize=8, weight='bold')
                y -= 0.04
                for r in chunk:
                    row_line = ' | '.join([str(c) for c in r])
                    plt.text(0.01, y, row_line, fontsize=7)
                    y -= 0.03
                    if y < 0.05:
                        break
                pdf.savefig(fig); plt.close(fig)

    # --- Add explicit Task 1/2/3 sections with per-sample metrics and sample images ---
    def add_task_section(task_label: str, summary_csv_path: Path, task_dir_name: str):
        csvp = summary_csv_path
        if not csvp.exists():
            return
        try:
            with open(csvp, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            rows = []
        if not rows:
            return
        # Header
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.5, 0.95, f'{task_label} Summary', ha='center', fontsize=14, weight='bold')
        plt.text(0.1, 0.9, f'Source CSV: {csvp.relative_to(output)}', fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        # Per-sample pages
        task_root = output / task_dir_name
        for r in rows:
            sample_name = r.get('file') or r.get('sample') or r.get('name')
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.axis('off')
            plt.text(0.5, 0.95, f"{task_label} - {sample_name}", ha='center', fontsize=12, weight='bold')
            y = 0.9
            # metrics: print all fields in the row
            for k, v in r.items():
                plt.text(0.05, y, f"{k}: {v}", fontsize=9)
                y -= 0.03
                if y < 0.35:
                    break

            # try to include an image from the sample folder
            img_inserted = False
            try:
                if task_root.exists() and sample_name:
                    sample_dir = task_root / sample_name
                    if sample_dir.exists():
                        # prefer per-axis or mse_mae or preview images
                        patterns = ['*per_axis*.png', '*mse_mae*.png', '*preview*.png', '*.png']
                        for pat in patterns:
                            imgs = sorted(sample_dir.glob(pat))
                            if imgs:
                                imgp = imgs[0]
                                img = plt.imread(str(imgp))
                                # place image in lower half of page
                                ax = fig.add_axes([0.05, 0.05, 0.9, 0.45])
                                ax.imshow(img)
                                ax.axis('off')
                                plt.text(0.05, 0.4, f"Image: {imgp.relative_to(output)}", fontsize=7)
                                img_inserted = True
                                break
            except Exception:
                img_inserted = False

            if not img_inserted:
                plt.text(0.05, 0.35, '(no sample image found in task folder)', fontsize=8)

            pdf.savefig(fig); plt.close(fig)

    # call sections
    add_task_section('Task 1 - Inspection', output / 'summary' / 'inspections.csv', 'task1')
    add_task_section('Task 2 - Normalize & Quantize', output / 'summary' / 'task2_summary.csv', 'task2')
    add_task_section('Task 3 - Reconstruction & Errors', output / 'summary' / 'task3_summary.csv', 'task3')

    # Task2 summary table
    if task2:
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.5, 0.95, 'Task 2: Min-Max vs Unit-Sphere MSE per file', ha='center', fontsize=12, weight='bold')
        y = 0.9
        for r in task2:
            mm = safe_float(r.get('minmax_mse'))
            us = safe_float(r.get('unitsphere_mse'))
            mm_s = f"{mm:.6e}" if mm is not None else 'N/A'
            us_s = f"{us:.6e}" if us is not None else 'N/A'
            line = f"{r['file']}: minmax_mse={mm_s} | unitsphere_mse={us_s}"
            plt.text(0.05, y, line, fontsize=9)
            y -= 0.035
            if y < 0.05:
                pdf.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69)); plt.axis('off')
                y = 0.95
        pdf.savefig(fig); plt.close(fig)

    # Include a few representative plots if present
    sample_plots = ['per_axis_error_histograms.png', 'mse_mae_per_axis.png', 'preview_normalized_minmax.png']
    for fname in sample_plots:
        p = output / fname
        if p.exists():
            img = plt.imread(str(p))
            fig = plt.figure(figsize=(8.27,11.69))
            plt.imshow(img); plt.axis('off')
            pdf.savefig(fig); plt.close(fig)

        # Seam tokenizer summary: look for per-sample seam token JSON files under task2
        seam_summary = []
        task2_dir = output / 'task2'
        if task2_dir.exists():
            for sample_dir in sorted(task2_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                # look for JSON files that contain seam/token data
                seam_files = list(sample_dir.glob('*seam*.json')) + list(sample_dir.glob('seams.json'))
                if not seam_files:
                    # also try a couple common names
                    seam_files = list(sample_dir.glob('*tokens*.json'))
                sample_stats = {'sample': sample_dir.name, 'n_files': len(seam_files), 'n_seams': 0, 'avg_seam_vertices': None, 'filepath': None}
                total_vertices = 0
                total_seams = 0
                for sf in seam_files:
                    try:
                        data = json.loads(sf.read_text())
                    except Exception:
                        continue
                    # normalize possible formats
                    seams_list = None
                    if isinstance(data, dict):
                        # common keys: 'seams' -> list, or single mapping
                        if 'seams' in data and isinstance(data['seams'], list):
                            seams_list = data['seams']
                        elif 'items' in data and isinstance(data['items'], list):
                            seams_list = data['items']
                        else:
                            # maybe the file is a list under another key or contains token lists per entry
                            # try treating dict values as seam entries
                            possible = [v for v in data.values() if isinstance(v, list)]
                            if possible:
                                seams_list = possible[0]
                    elif isinstance(data, list):
                        seams_list = data
                    if seams_list is None:
                        continue
                    for s in seams_list:
                        # s can be a list of token strings or dict with 'tokens' or 'ints'/'inv'
                        tokens = None
                        if isinstance(s, list):
                            tokens = s
                        elif isinstance(s, dict):
                            if 'tokens' in s and isinstance(s['tokens'], list):
                                tokens = s['tokens']
                            elif 'ints' in s and 'inv' in s and isinstance(s['inv'], dict):
                                # reconstruct tokens from inv mapping
                                inv = {int(k): v for k, v in s['inv'].items()} if isinstance(s['inv'].keys(), (list,)) else {int(k): v for k, v in s['inv'].items()}
                                try:
                                    tokens = [inv[int(i)] for i in s.get('ints', [])]
                                except Exception:
                                    tokens = None
                        if tokens is None:
                            continue
                        # count seam vertices: count of 'D' tokens + 1 or count of V tokens
                        n_vertices = 1 + sum(1 for t in tokens if isinstance(t, str) and t.startswith('D'))
                        total_seams += 1
                        total_vertices += n_vertices
                    sample_stats['filepath'] = str(sf.relative_to(output))
                    sample_stats['n_seams'] += len(seams_list)
                if total_seams > 0:
                    sample_stats['n_seams'] = total_seams
                    sample_stats['avg_seam_vertices'] = total_vertices / total_seams if total_seams else None
                seam_summary.append(sample_stats)

        # Add seam summary page
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.5, 0.95, 'Seam Tokenizer Summary', ha='center', fontsize=14, weight='bold')
        y = 0.9
        if not seam_summary:
            plt.text(0.1, y, 'No seam token files found under output/task2/. If you wish to include seam tokens, run the seam extraction + tokenizer integration to produce JSON files named e.g. "seams.json" in each sample folder.', fontsize=10)
        else:
            plt.text(0.1, y, 'Per-sample seam token stats (if present):', fontsize=11)
            y -= 0.04
            for s in seam_summary:
                line = f"{s['sample']}: files={s['n_files']}, seams={s.get('n_seams',0)}, avg_vertices={s.get('avg_seam_vertices') if s.get('avg_seam_vertices') is not None else 'N/A'}"
                plt.text(0.05, y, line, fontsize=9)
                y -= 0.035
                if y < 0.05:
                    pdf.savefig(fig); plt.close(fig)
                    fig = plt.figure(figsize=(8.27, 11.69)); plt.axis('off')
                    y = 0.95
        pdf.savefig(fig); plt.close(fig)

    # Add final conclusions page
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.axis('off')
    plt.text(0.5, 0.85, 'Conclusions & Recommendations', ha='center', fontsize=14, weight='bold')
    y = 0.78
    concl = [
        '1) For this dataset (8 samples), Min–Max normalization + 1024-bin quantization consistently gave the lowest reconstruction error.',
        '2) Unit-Sphere normalization may be useful when rotation-invariance is prioritized, but it increases quantization loss when axis ranges differ.',
        '3) Per-axis/Per-vertex error visualizations show where errors cluster; adaptive per-axis bin sizes or local adaptive quantization can reduce errors further.',
        '4) For model input pipelines, prefer per-axis normalization when preserving geometric fidelity is critical; consider PCA-based alignment if rotation invariance is required.',
        '5) Recommended next steps: implement adaptive quantization by local density and add automated per-mesh screenshots for the report.'
    ]
    for c in concl:
        plt.text(0.1, y, c, fontsize=10)
        y -= 0.05
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved report to: {report_path}")
