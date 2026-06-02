# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:01:24 2024
Updated: Aligns subject IDs and supports multi-label QA prediction
"""

import argparse
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--predictions_file", type=str, default="predicted_output.csv",
        help="Path to the predictions CSV file"
    )
    parser.add_argument(
        "-g", "--goldstandard_file", type=str, default="LISA_LF_QC_goldstandard.csv",
        help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="results.json",
        help="Path to the output JSON file"
    )
    return parser.parse_args()

class Metrics:
    def score_qc(self, gt, pred):
        categories = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]

        f1_micro, f2_micro, precision_micro, recall_micro, accuracy_micro = self.calculate_metrics(gt, pred, categories, 'micro')
        f1_macro, f2_macro, precision_macro, recall_macro, accuracy_macro = self.calculate_metrics(gt, pred, categories, 'macro')
        f1_weighted, f2_weighted, precision_weighted, recall_weighted, accuracy_weighted = self.calculate_metrics(gt, pred, categories, 'weighted')

        return {
            "F1_micro": f1_micro, "F2_micro": f2_micro, "Precision_micro": precision_micro, "Recall_micro": recall_micro, "Accuracy_micro": accuracy_micro,
            "F1_macro": f1_macro, "F2_macro": f2_macro, "Precision_macro": precision_macro, "Recall_macro": recall_macro, "Accuracy_macro": accuracy_macro,
            "F1_weighted": f1_weighted, "F2_weighted": f2_weighted, "Precision_weighted": precision_weighted, "Recall_weighted": recall_weighted, "Accuracy_weighted": accuracy_weighted
        }

    def calculate_metrics(self, gt, pred, categories, average):
        f1_scores = []
        f2_scores = []
        precision_scores = []
        recall_scores = []
        accuracy_scores = []

        for category in categories:
            f1_scores.append(f1_score(gt[category], pred[category], average=average, zero_division=0))
            f2_scores.append(fbeta_score(gt[category], pred[category], beta=2, average=average, zero_division=0))
            precision_scores.append(precision_score(gt[category], pred[category], average=average, zero_division=0))
            recall_scores.append(recall_score(gt[category], pred[category], average=average, zero_division=0))
            accuracy_scores.append(accuracy_score(gt[category], pred[category]))

        return (
            round(sum(f1_scores) / len(f1_scores), 3),
            round(sum(f2_scores) / len(f2_scores), 3),
            round(sum(precision_scores) / len(precision_scores), 3),
            round(sum(recall_scores) / len(recall_scores), 3),
            round(sum(accuracy_scores) / len(accuracy_scores), 3)
        )

def score_qc(gt, pred):
    metrics = Metrics()
    return metrics.score_qc(gt, pred)

def main():
    args = get_args()

    categories = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]
    pred_columns = [f"Pred_label_{cat}" for cat in categories]

    # Load prediction CSV and set index to Subject ID
    df_pred = pd.read_csv(args.predictions_file)
    df_pred = df_pred.set_index("Subject ID")

    # Rename prediction columns to match ground truth format
    df_pred = df_pred[pred_columns]
    df_pred.columns = categories

    # Load ground truth CSV and fix index to match predictions
    df_gt = pd.read_csv(args.goldstandard_file)
    df_gt = df_gt.rename(columns={"Unnamed: 0": "Subject ID"}) if "Unnamed: 0" in df_gt.columns else df_gt
    df_gt["Subject ID"] = df_gt["Subject ID"].str.replace(".nii.gz", "", regex=False)
    df_gt = df_gt.set_index("Subject ID")
    df_gt = df_gt[categories]

    # Align rows based on index
    df_pred, df_gt = df_pred.align(df_gt, join="inner", axis=0)

    # Compute metrics
    results = score_qc(df_gt, df_pred)

    # Save results
    pd.DataFrame([results]).to_csv("all_scores_qc.csv", index=False)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
