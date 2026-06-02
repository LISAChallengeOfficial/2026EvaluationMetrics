#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:49:46 2025

"""
### For Basal Ganglia 
import os
import argparse
import pandas as pd
import SimpleITK as sitk
import numpy as np
import json
import difflib
from surface_distance.metrics import (
    compute_surface_distances,
    compute_average_surface_distance,
    compute_dice_coefficient,
    compute_robust_hausdorff,
)

def load_image(path):
    return sitk.ReadImage(path)

def compute_metrics(gt_array, pred_array, voxel_size):
    def score(gt, pred):
        dsc = round(compute_dice_coefficient(gt, pred), 3)
        surface_dist = compute_surface_distances(pred, gt, spacing_mm=voxel_size)
        hd = round(compute_robust_hausdorff(surface_dist, 100), 3)
        hd95 = round(compute_robust_hausdorff(surface_dist, 95), 3)
        assd = round(np.mean(compute_average_surface_distance(surface_dist)), 3)
        rve = round(abs(np.sum(pred) - np.sum(gt)) / (np.sum(gt) + 1e-5), 3)
        return [dsc, hd, hd95, assd, rve]

    label_metrics = {}
    for label in range(1, 5):  # labels: 1 to 4
        label_metrics[label] = score(gt_array == label, pred_array == label)
    return label_metrics

def find_best_match(target_filename, candidate_filenames):
    """Find the closest matching filename from a list."""
    matches = difflib.get_close_matches(target_filename, candidate_filenames, n=1, cutoff=0.5)
    return matches[0] if matches else None

def run_scoring(gt_dir, pred_dir, output_path):
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".nii.gz")]
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")]
    results = []
    voxel_size = [1, 1, 1]  # Adjust if needed

    for gt_file in gt_files:
        subject_id = gt_file.replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, gt_file)

        # Find best matching prediction file
        matched_pred_file = find_best_match(gt_file, pred_files)
        if not matched_pred_file:
            print(f"Skipping {subject_id}: no matching prediction file found.")
            continue

        pred_path = os.path.join(pred_dir, matched_pred_file)
        print(f"Matched: {gt_file}  <-->  {matched_pred_file}")

        gt_image = load_image(gt_path)
        pred_image = load_image(pred_path)
        pred_image.CopyInformation(gt_image)

        gt_array = sitk.GetArrayFromImage(gt_image)
        pred_array = sitk.GetArrayFromImage(pred_image)

        metrics = compute_metrics(gt_array, pred_array, voxel_size)

        row = {"scan_id": subject_id}
        all_vals = []
        for label, region in zip(range(1, 5), ["Caudate_L", "Caudate_R", "Lentiform_L", "Lentiform_R"]):
            dsc, hd, hd95, assd, rve = metrics[label]
            row.update({
                f"DSC_{region}": dsc,
                f"HD_{region}": hd,
                f"HD95_{region}": hd95,
                f"ASSD_{region}": assd,
                f"RVE_{region}": rve
            })
            all_vals.append(metrics[label])

        # Compute averages
        avg_vals = np.mean(np.array(all_vals), axis=0).round(3)
        row.update({
            "DSC_Avg": avg_vals[0],
            "HD_Avg": avg_vals[1],
            "HD95_Avg": avg_vals[2],
            "ASSD_Avg": avg_vals[3],
            "RVE_Avg": avg_vals[4],
        })

        results.append(row)

    if not results:
        print("No valid predictions found. No results to score.")
        with open("result_annotations.json", "w", encoding="utf-8") as f:
            json.dump({"status": "NO_VALID_RESULTS"}, f, indent=4, ensure_ascii=False)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        return

    df = pd.DataFrame(results).sort_values(by="scan_id")
    df.to_csv("all_scores_seg.csv", index=False)

    numeric_df = df.select_dtypes(include=[np.number])
    avg = numeric_df.mean().round(3)
    std = numeric_df.std().round(3)

    summary = {"status": "SCORED"}
    for col in numeric_df.columns:
        summary[col] = f"{avg[col]:.2f}±{std[col]:.2f}"

    with open("result_annotations.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(json.dumps(summary, indent=4, ensure_ascii=False))
    print("Saved evaluation results.")

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation script for basal ganglia segmentation.")
    parser.add_argument("-g", "--gt_dir", default="./masks_bg", help="Path to the ground truth masks folder.")
    parser.add_argument("-p", "--pred_dir", default="./segment_bg", help="Path to the prediction folder.")
    parser.add_argument("-o", "--output", default="results.json", help="Path to the output JSON file.")
    return parser.parse_args()

def main():
    args = get_args()
    run_scoring(args.gt_dir, args.pred_dir, args.output)

if __name__ == "__main__":
    main()