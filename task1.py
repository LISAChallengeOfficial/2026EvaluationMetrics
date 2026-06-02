# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:01:24 2024

@author: rrouhi, szanjal
"""

## DESCRIPTION ##
# This script evaluates classification performance for multiple image artifact categories:
# - Noise, Zipper, Positioning, Banding, Motion, Contrast, Distortion
# It computes F1, F2, Precision, Recall, and Accuracy using micro, macro, and weighted averaging.
# Input: CSVs with predictions and gold standard labels (columns = artifact categories, rows = samples)
# Output: CSV and JSON with all metrics

import os
import argparse
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

def get_args():
    """Set up command-line interface and get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions_file", type=str, default="predicted_output_all.csv")
    parser.add_argument("-g", "--goldstandard_file", type=str, default="LISA_LF_QC_goldstandard.csv")
    parser.add_argument("-o", "--output", type=str, default="results.json")
    return parser.parse_args()

class Metrics:
    def score_qc(self, gt, pred, categories):
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

        f1_value = round(sum(f1_scores) / len(f1_scores), 3)
        f2_value = round(sum(f2_scores) / len(f2_scores), 3)
        precision_value = round(sum(precision_scores) / len(precision_scores), 3)
        recall_value = round(sum(recall_scores) / len(recall_scores), 3)
        accuracy_value = round(sum(accuracy_scores) / len(accuracy_scores), 3)

        return f1_value, f2_value, precision_value, recall_value, accuracy_value

def score_qc(gt, pred, categories):
    """Compute and return scores for each scan."""
    metrics = Metrics()
    return metrics.score_qc(gt, pred, categories)

def main():
    """Main function."""
    args = get_args()
    df_pred = pd.read_csv(args.predictions_file)
    df_gt = pd.read_csv(args.goldstandard_file)

    # Evaluate all relevant artifact categories
    focus_columns = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]
    df_gt = df_gt[focus_columns]
    df_pred = df_pred[focus_columns]

    results = score_qc(df_gt, df_pred, focus_columns)

    # Save the metrics to CSV and JSON
    results_df = pd.DataFrame([results])
    results_df.to_csv('all_scores_qc.csv', index=False)

    with open(args.output, 'w') as json_file:
        json.dump(results, json_file, indent=4)

if __name__ == "__main__": 
    main()
