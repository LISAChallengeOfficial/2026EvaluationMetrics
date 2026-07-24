
## DESCRIPTION ##
# This script evaluates Task 1b image enhancement performance using a fully unpaired protocol.
# It compares a directory of enhanced images against a directory of artifact-free reference images.
#
# The reference bank can be restricted using a QC spreadsheet. In that mode, the spreadsheet
# must contain a filename column and a cleanliness column (for example, "Total"), where lower
# values indicate cleaner images. By default, only images with Total <= 0 are retained in the
# reference bank, which corresponds to the strict artifact-free subset.
#
# Core metrics:
# - FID (distribution-to-distribution)
# - LPIPS (distribution-level perceptual distance estimated from random cross-set pairs)
# - PSNR (distribution-level PSNR estimated from random cross-set pairs)
# Optional metrics:
# - BRISQUE (no-reference quality score, lower is better)
# - CLIP-IQA (no-reference quality score, higher is better)
# - FRD (Fréchet Radiomics Distance, lower is better)
#
# Input:
# - predictions_dir: directory of enhanced images
# - reference_dir: directory of candidate reference images
# - reference_csv: optional QC spreadsheet used to define the clean reference bank
#
# Supported formats include common medical image files (e.g., .nii, .nii.gz, .mha, .mhd, .nrrd)
# and common 2D image files (e.g., .png, .jpg, .jpeg, .tif, .tiff, .bmp).
# Output: CSV and JSON with all metrics.

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

SUPPORTED_SUFFIXES = (
    ".nii", ".nii.gz", ".mha", ".mhd", ".nrrd",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
)
MEDICAL_SUFFIXES = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")
TWO_D_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Submitted inputs arrive as .zip files (same as Task2). Unzip them with the
# shared helper before any path collection, mirroring task2_new.py.
import utils

PRED_PARENT_DIR = "enhanced_images"
REF_PARENT_DIR = "artifact_free_reference"
INPUT_PARENT_DIR = "input_images"

METADATA_MISMATCH_MESSAGE = (
    "A metadata mismatch between one or more prediction files and input files has been found. "
    "Please ensure all metadata between input and predictions are identical"
)

LIVEBOARD_METRIC_FIELDS = [
    "FID_mean",
    "FID_min",
    "FID_max",
    "LPIPS_mean",
    "LPIPS_min",
    "LPIPS_max",
    "PSNR_mean",
    "PSNR_min",
    "PSNR_max",
    "BRISQUE_enhanced_mean",
    "BRISQUE_enhanced_min",
    "BRISQUE_enhanced_max",
    "BRISQUE_delta_case_mean",
    "BRISQUE_delta_case_min",
    "BRISQUE_delta_case_max",
    "CLIPIQA_enhanced_case_mean",
    "CLIPIQA_enhanced_case_min",
    "CLIPIQA_enhanced_case_max",
    "CLIPIQA_delta_case_mean",
    "CLIPIQA_delta_case_min",
    "CLIPIQA_delta_case_max",
    "FRD"
]


def normalize_cli_args(argv: Sequence[str]) -> List[str]:
    """Normalize common copy/paste dash variants before argparse parses flags."""
    normalized: List[str] = []
    for arg in argv:
        if arg.startswith("\u2014\u2014") or arg.startswith("\u2013\u2013"):
            normalized.append("--" + arg[2:])
        elif arg.startswith("\u2014") or arg.startswith("\u2013"):
            normalized.append("--" + arg[1:])
        else:
            normalized.append(arg)
    return normalized


def get_args():
    """Set up command-line interface and get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, default="")
    parser.add_argument("-p", "--predictions_dir", type=str, default="enhanced_images")
    parser.add_argument("-r", "--reference_dir", type=str, default="artifact_free_reference")
    parser.add_argument("-o", "--output", type=str, default="results.json")
    parser.add_argument("--summary_output_csv", type=str, default="all_scores_task1b.csv")
    parser.add_argument("--per_case_output_csv", type=str, default="per_case_scores_task1b.csv")
    parser.add_argument(
        "--liveboard_metrics_as_strings",
        action="store_true",
        help=(
            "Serialize the exact liveboard summary metric fields as strings instead of numbers. "
            "Use this only if the liveboard ingestion schema requires string-valued JSON fields."
        ),
    )
    parser.add_argument("--reference_csv", type=str, default="")
    parser.add_argument("--reference_filename_column", type=str, default="filename")
    parser.add_argument("--reference_total_column", type=str, default="Total")
    parser.add_argument("--reference_clean_max_total", type=float, default=0.0)
    parser.add_argument("--keep_unlisted_reference_images", action="store_true")
    parser.add_argument("--slice_axis", type=int, default=0)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--fid_resize", type=int, default=299)
    parser.add_argument("--max_slices_per_volume", type=int, default=64)
    parser.add_argument("--max_total_slices", type=int, default=512)
    parser.add_argument("--num_random_pairs", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_slice_std", type=float, default=1e-4)
    parser.add_argument("--min_nonzero_fraction", type=float, default=1e-3)
    parser.add_argument("--normalization_percentiles", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute_brisque", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute_clipiqa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute_frd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--radimagenet_weights", type=str, default="")
    return parser.parse_args(normalize_cli_args(sys.argv[1:]))


class Metrics:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        self.rng = np.random.default_rng(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        self.lpips_backend = "unknown"
        self.fid_backend = "unknown"

    def score_task1b(
        self,
        enhanced_paths: Sequence[str],
        reference_paths: Sequence[str],
        reference_qc_summary: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        enhanced_slices, enhanced_case_count = self.build_slice_pool(enhanced_paths)
        reference_slices, reference_case_count = self.build_slice_pool(reference_paths)

        if len(enhanced_slices) == 0:
            raise ValueError("No usable slices were found in predictions_dir.")
        if len(reference_slices) == 0:
            raise ValueError("No usable slices were found in reference_dir after QC filtering.")

        results = {
            "Num_enhanced_cases": int(enhanced_case_count),
            "Num_reference_cases": int(reference_case_count),
            "Num_enhanced_slices": int(len(enhanced_slices)),
            "Num_reference_slices": int(len(reference_slices)),
            "Num_random_pairs": int(min(self.args.num_random_pairs, len(enhanced_slices) * len(reference_slices))),
        }

        if reference_qc_summary is not None:
            results.update(reference_qc_summary)

        fid_value, fid_backend = self.fid(enhanced_slices, reference_slices)
        results["FID"] = round(fid_value, 3)
        results["FID_backend"] = fid_backend

        lpips_value, lpips_array = self.lpips_unpaired(enhanced_slices, reference_slices)
        results["LPIPS_unpaired"] = round(lpips_value, 3)
        results["LPIPS_backend"] = self.lpips_backend
        results["LPIPS_pair_values"] = lpips_array

        psnr_value = self.psnr_unpaired(enhanced_slices, reference_slices)
        results["PSNR_unpaired"] = round(psnr_value, 3)

        if self.args.compute_brisque:

            brisque_enhanced = self.brisque_qc(enhanced_slices)
            brisque_reference = self.brisque_qc(reference_slices)
            results["BRISQUE_enhanced"] = round(brisque_enhanced, 3)
            results["BRISQUE_reference"] = round(brisque_reference, 3)
            results["BRISQUE_delta"] = round(brisque_enhanced - brisque_reference, 3)

        if self.args.compute_clipiqa:
            clipiqa_enhanced = self.clipiqa(enhanced_slices)
            clipiqa_reference = self.clipiqa(reference_slices)
            results["CLIPIQA_enhanced"] = round(clipiqa_enhanced, 3)
            results["CLIPIQA_reference"] = round(clipiqa_reference, 3)
            results["CLIPIQA_delta"] = round(clipiqa_enhanced - clipiqa_reference, 3)

        if self.args.compute_frd:
            frd_value = self.frd(enhanced_paths, reference_paths)
            results["FRD"] = round(frd_value, 3)

        return results

    def build_slice_pool(self, image_paths: Sequence[str]) -> Tuple[List[np.ndarray], int]:
        all_slices: List[np.ndarray] = []
        used_cases = 0

        for image_path in image_paths:
            image = self.read_image(image_path)
            image = self.normalize_image(image)
            image_slices = self.extract_slices(image)

            if len(image_slices) == 0:
                continue

            used_cases += 1
            all_slices.extend(image_slices)

        if len(all_slices) > self.args.max_total_slices:
            keep_indices = np.linspace(0, len(all_slices) - 1, self.args.max_total_slices, dtype=int)
            all_slices = [all_slices[idx] for idx in keep_indices]

        return all_slices, used_cases

    def read_image(self, image_path: str) -> np.ndarray:
        image_path_lower = image_path.lower()

        if image_path_lower.endswith(MEDICAL_SUFFIXES):
            if sitk is None:
                raise ImportError("SimpleITK is required to read medical image files.")
            image = sitk.ReadImage(image_path)
            array = sitk.GetArrayFromImage(image)
        elif image_path_lower.endswith(TWO_D_SUFFIXES):
            array = np.asarray(Image.open(image_path).convert("F"), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported image format: {image_path}")

        array = np.asarray(array, dtype=np.float32)
        array = np.squeeze(array)

        if array.ndim == 0:
            raise ValueError(f"Image has no spatial dimensions: {image_path}")

        if array.ndim > 3:
            while array.ndim > 3:
                array = array[0]

        return array

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        finite_mask = np.isfinite(image)

        if not np.any(finite_mask):
            return np.zeros_like(image, dtype=np.float32)

        values = image[finite_mask]
        low, high = np.percentile(values, self.args.normalization_percentiles)

        if high <= low:
            low = float(np.min(values))
            high = float(np.max(values))

        if high <= low:
            return np.zeros_like(image, dtype=np.float32)

        image = np.clip(image, low, high)
        image = (image - low) / (high - low)
        image[~finite_mask] = 0.0
        return image.astype(np.float32)

    def extract_slices(self, image: np.ndarray) -> List[np.ndarray]:
        if image.ndim == 2:
            candidate_slices = [image]
        elif image.ndim == 3:
            axis = int(np.clip(self.args.slice_axis, 0, 2))
            moved = np.moveaxis(image, axis, 0)
            candidate_slices = [moved[idx] for idx in range(moved.shape[0])]
        else:
            raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")

        informative_slices: List[np.ndarray] = []
        for slice_2d in candidate_slices:
            if not self.is_informative_slice(slice_2d):
                continue
            informative_slices.append(slice_2d.astype(np.float32))

        if len(informative_slices) > self.args.max_slices_per_volume:
            keep_indices = np.linspace(0, len(informative_slices) - 1, self.args.max_slices_per_volume, dtype=int)
            informative_slices = [informative_slices[idx] for idx in keep_indices]

        return informative_slices

    def is_informative_slice(self, slice_2d: np.ndarray) -> bool:
        if slice_2d.size == 0:
            return False

        slice_std = float(np.std(slice_2d))
        nonzero_fraction = float(np.mean(slice_2d > 0))

        return slice_std >= self.args.min_slice_std and nonzero_fraction >= self.args.min_nonzero_fraction

    def prepare_tensor(
        self,
        slice_2d: np.ndarray,
        size: int,
        channels: int = 3,
        dtype: torch.dtype = torch.float32,
        scale_to_uint8: bool = False,
    ) -> torch.Tensor:
        tensor = torch.from_numpy(slice_2d).float().unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        tensor = tensor.clamp(0.0, 1.0)

        if channels == 3:
            tensor = tensor.repeat(1, 3, 1, 1)

        if scale_to_uint8:
            tensor = (tensor * 255.0).round().to(torch.uint8)
        else:
            tensor = tensor.to(dtype=dtype)

        return tensor

    def sample_cross_set_indices(self, num_a: int, num_b: int) -> Tuple[np.ndarray, np.ndarray]:
        max_possible_pairs = num_a * num_b
        num_pairs = min(self.args.num_random_pairs, max_possible_pairs)

        idx_a = self.rng.integers(0, num_a, size=num_pairs)
        idx_b = self.rng.integers(0, num_b, size=num_pairs)
        return idx_a, idx_b

    def psnr_unpaired(self, enhanced_slices: Sequence[np.ndarray], reference_slices: Sequence[np.ndarray]) -> float:
        idx_enhanced, idx_reference = self.sample_cross_set_indices(len(enhanced_slices), len(reference_slices))
        mse_values: List[float] = []

        for enhanced_idx, reference_idx in zip(idx_enhanced, idx_reference):
            enhanced_tensor = self.prepare_tensor(enhanced_slices[enhanced_idx], size=self.args.resize, channels=1)
            reference_tensor = self.prepare_tensor(reference_slices[reference_idx], size=self.args.resize, channels=1)
            mse = torch.mean((enhanced_tensor - reference_tensor) ** 2).item()
            mse_values.append(mse)

        mean_mse = float(np.mean(mse_values)) if len(mse_values) > 0 else float("inf")

        if mean_mse <= 0:
            return float("inf")

        return float(10.0 * math.log10(1.0 / mean_mse))

    def lpips_unpaired(self, enhanced_slices: Sequence[np.ndarray], reference_slices: Sequence[np.ndarray]) -> Tuple[float, List[float]]:
        """Estimate LPIPS from random enhanced/reference slice pairs.

        Returns the mean and the individual pair values. These pair values are
        not per-image values; case-level scoring is handled separately by
        score_task1b_case_level().
        """
        metric = None
        lpips_backend = None

        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            try:
                metric = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex",
                    normalize=True,
                    reduction="none",
                ).to(self.device)
                lpips_backend = "torchmetrics_alex_none"
            except TypeError:
                metric = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex",
                    normalize=True,
                ).to(self.device)
                lpips_backend = "torchmetrics_alex_mean"
        except ImportError:
            try:
                from piq import LPIPS
                try:
                    metric = LPIPS(reduction="none").to(self.device)
                    lpips_backend = "piq_vgg16_none"
                except TypeError:
                    metric = LPIPS(reduction="mean").to(self.device)
                    lpips_backend = "piq_vgg16_mean"
            except ImportError as exc:
                raise ImportError("LPIPS requires either torchmetrics or piq.") from exc

        metric.eval()

        idx_enhanced, idx_reference = self.sample_cross_set_indices(len(enhanced_slices), len(reference_slices))
        lpips_values: List[float] = []

        with torch.no_grad():
            for start_idx in range(0, len(idx_enhanced), self.args.batch_size):
                end_idx = min(start_idx + self.args.batch_size, len(idx_enhanced))

                enhanced_batch = torch.cat([
                    self.prepare_tensor(enhanced_slices[idx], size=self.args.resize, channels=3)
                    for idx in idx_enhanced[start_idx:end_idx]
                ], dim=0).to(self.device)

                reference_batch = torch.cat([
                    self.prepare_tensor(reference_slices[idx], size=self.args.resize, channels=3)
                    for idx in idx_reference[start_idx:end_idx]
                ], dim=0).to(self.device)

                batch_value = metric(enhanced_batch, reference_batch)
                if batch_value.ndim == 0:
                    # Some backends only return the batch mean. Keep the mean as
                    # a valid contribution, but do not label it as per-image.
                    lpips_values.append(float(batch_value.item()))
                else:
                    lpips_values.extend(batch_value.detach().cpu().reshape(-1).numpy().astype(float).tolist())

        self.lpips_backend = lpips_backend
        return float(np.mean(lpips_values)), lpips_values

    def fid(self, enhanced_slices: Sequence[np.ndarray], reference_slices: Sequence[np.ndarray]) -> Tuple[float, str]:
        """Compute FID between enhanced and reference slice pools.

        Uses correctly batched torchmetrics FID when available. The tensors are
        float images in [0, 1], so normalize=True is required for torchmetrics.
        Falls back to ResNet/RadImageNet/pooled features if torchmetrics is not
        available or fails at runtime.
        """
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance

            metric = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
            metric.eval()

            with torch.no_grad():
                for start in range(0, len(reference_slices), self.args.batch_size):
                    end = min(start + self.args.batch_size, len(reference_slices))
                    batch = torch.cat([
                        self.prepare_tensor(slice_2d, size=self.args.fid_resize, channels=3, scale_to_uint8=False)
                        for slice_2d in reference_slices[start:end]
                    ], dim=0).to(self.device)
                    metric.update(batch, real=True)

                for start in range(0, len(enhanced_slices), self.args.batch_size):
                    end = min(start + self.args.batch_size, len(enhanced_slices))
                    batch = torch.cat([
                        self.prepare_tensor(slice_2d, size=self.args.fid_resize, channels=3, scale_to_uint8=False)
                        for slice_2d in enhanced_slices[start:end]
                    ], dim=0).to(self.device)
                    metric.update(batch, real=False)

                fid_value = float(metric.compute().item())
            self.fid_backend = "torchmetrics_inception"
            return fid_value, self.fid_backend

        except Exception:
            fid_value = self.fid_fallback(enhanced_slices, reference_slices)
            self.fid_backend = "fallback_resnet"
            return fid_value, self.fid_backend

    def fid_fallback(self, enhanced_slices: Sequence[np.ndarray], reference_slices: Sequence[np.ndarray]) -> float:
        feature_extractor = self.get_fallback_feature_extractor()

        if feature_extractor is None:
            reference_features = self.extract_pooled_features(reference_slices)
            enhanced_features = self.extract_pooled_features(enhanced_slices)
        else:
            feature_extractor = feature_extractor.to(self.device)
            feature_extractor.eval()
            reference_features = self.extract_features(reference_slices, feature_extractor, self.args.resize)
            enhanced_features = self.extract_features(enhanced_slices, feature_extractor, self.args.resize)

        mu_reference, sigma_reference = self.feature_mean_and_covariance(reference_features)
        mu_enhanced, sigma_enhanced = self.feature_mean_and_covariance(enhanced_features)

        return self.frechet_distance(mu_reference, sigma_reference, mu_enhanced, sigma_enhanced)

    def feature_mean_and_covariance(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return a covariance matrix with a stable shape even for one-sample cases."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            features = np.atleast_2d(features)

        mean = np.mean(features, axis=0)
        feature_dim = features.shape[1]

        if features.shape[0] <= 1:
            covariance = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        else:
            covariance = np.cov(features, rowvar=False)
            covariance = np.atleast_2d(covariance)

        return mean, covariance

    def get_fallback_feature_extractor(self) -> Optional[nn.Module]:
        if self.args.radimagenet_weights:
            try:
                from radimagenet_models.models.resnet import radimagenet_resnet50

                model = radimagenet_resnet50()
                state_dict = torch.load(self.args.radimagenet_weights, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict, strict=False)
                if hasattr(model, "fc"):
                    model.fc = nn.Identity()
                return model
            except Exception:
                pass

        try:
            from torchvision.models import ResNet50_Weights, resnet50
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            model.fc = nn.Identity()
            return model
        except Exception:
            return None

    def extract_features(self, slices: Sequence[np.ndarray], model: nn.Module, size: int) -> np.ndarray:
        all_features: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(slices), self.args.batch_size):
                end = min(start + self.args.batch_size, len(slices))
                batch = torch.cat([
                    self.prepare_tensor(slice_2d, size=size, channels=3)
                    for slice_2d in slices[start:end]
                ], dim=0).to(self.device)

                features = model(batch)
                features = features.view(features.size(0), -1)
                all_features.append(features.detach().cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def extract_pooled_features(self, slices: Sequence[np.ndarray]) -> np.ndarray:
        all_features: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(slices), self.args.batch_size):
                end = min(start + self.args.batch_size, len(slices))
                batch = torch.cat([
                    self.prepare_tensor(slice_2d, size=64, channels=3)
                    for slice_2d in slices[start:end]
                ], dim=0)
                pooled = F.adaptive_avg_pool2d(batch, output_size=(8, 8))
                features = pooled.view(pooled.size(0), -1)
                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def frechet_distance(
        self,
        mu_a: np.ndarray,
        sigma_a: np.ndarray,
        mu_b: np.ndarray,
        sigma_b: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        mu_a = np.atleast_1d(mu_a)
        mu_b = np.atleast_1d(mu_b)
        sigma_a = np.atleast_2d(sigma_a)
        sigma_b = np.atleast_2d(sigma_b)

        if sigma_a.shape != sigma_b.shape:
            raise ValueError("Covariance matrices must have the same shape for FID computation.")

        diff = mu_a - mu_b
        # NOTE: FID needs only trace(sqrtm(A @ B)), not the full matrix square root.
        # scipy.linalg.sqrtm on a 2048x2048 covariance is extremely slow (Schur
        # decomposition, minutes, no output) and was the scoring hang. The trace of
        # the matrix square root equals the sum of the square roots of the eigenvalues
        # of (A @ B), which is mathematically equivalent and orders of magnitude faster.
        product = (sigma_a + eps * np.eye(sigma_a.shape[0])) @ (sigma_b + eps * np.eye(sigma_b.shape[0]))
        eigenvalues = linalg.eigvals(product)
        trace_covmean = float(np.sum(np.sqrt(np.clip(eigenvalues.real, 0.0, None))))

        fid_value = diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2.0 * trace_covmean
        return float(max(fid_value, 0.0))

    def brisque_qc(self, slices: Sequence[np.ndarray]) -> float:
        try:
            from piq import BRISQUELoss
        except ImportError as exc:
            raise ImportError("piq is required to compute BRISQUE.") from exc

        metric = BRISQUELoss(data_range=1.0, reduction="mean").to(self.device)
        metric.eval()
        values: List[float] = []
        skipped = 0

        # Per-slice evaluation. piq's BRISQUE asserts inside _aggd_parameters on
        # degenerate (near-flat / low-texture) slices: it requires the pairwise
        # products of neighbouring MSCN coefficients to contain both positive and
        # negative values. Edge background slices (air outside the head) are
        # legitimate and unavoidable in 3D MRI volumes, so we skip them instead of
        # letting the whole submission fail. Scoring one slice at a time also
        # prevents a single degenerate slice from taking down an entire
        # reduction="mean" batch.
        with torch.no_grad():
            for slice_2d in slices:
                # Cheap pre-filter: a constant / near-constant slice has no
                # negative MSCN products and would trip the assert anyway.
                if float(np.std(np.asarray(slice_2d, dtype=np.float32))) < 1e-6:
                    skipped += 1
                    continue
                tensor = self.prepare_tensor(
                    slice_2d, size=self.args.resize, channels=3
                ).to(self.device)
                try:
                    value = float(metric(tensor).item())
                except (AssertionError, RuntimeError, ValueError):
                    # piq could not compute AGGD parameters on this slice.
                    skipped += 1
                    continue
                if math.isfinite(value):
                    values.append(value)
                else:
                    skipped += 1

        if skipped:
            print(
                f"[brisque_qc] skipped {skipped}/{len(slices)} degenerate slice(s)",
                file=sys.stderr,
            )

        if not values:
            # Every slice was degenerate; emit NaN and let clean_number handle it
            # downstream rather than raising and killing results.json.
            return float("nan")

        return float(np.mean(values))

    def clipiqa(self, slices: Sequence[np.ndarray]) -> float:
        try:
            from piq import CLIPIQA
        except ImportError as exc:
            raise ImportError("piq is required to compute CLIP-IQA.") from exc

        metric = CLIPIQA(data_range=1.0).to(self.device)
        metric.eval()
        values: List[float] = []

        with torch.no_grad():
            for start in range(0, len(slices), self.args.batch_size):
                end = min(start + self.args.batch_size, len(slices))
                batch = torch.cat([
                    self.prepare_tensor(slice_2d, size=self.args.resize, channels=3)
                    for slice_2d in slices[start:end]
                ], dim=0).to(self.device)
                batch_scores = metric(batch)
                values.extend(batch_scores.detach().cpu().reshape(-1).numpy().astype(float).tolist())

        return float(np.mean(values))

    def frd(self, prediction_paths: Sequence[str], reference_paths: Sequence[str]) -> float:
        try:
            from frd_score import compute_frd
        except ImportError as exc:
            raise ImportError("frd-score is required to compute FRD.") from exc

        with tempfile.TemporaryDirectory(prefix="frd_pred_") as pred_dir, tempfile.TemporaryDirectory(prefix="frd_ref_") as ref_dir:
            self.stage_files_for_frd(prediction_paths, pred_dir)
            self.stage_files_for_frd(reference_paths, ref_dir)
            # num_workers=1 is REQUIRED: frd_score defaults to cpu_count()-2 workers
            # and spawns a multiprocessing.Pool that fork-bombs under the scoring
            # container (30+ procs all blocked on queue.get -> futex, never finishes).
            # Serial FRD is slightly slower but deterministic and cannot deadlock.
            return float(compute_frd([pred_dir, ref_dir], num_workers=1))

    def stage_files_for_frd(self, image_paths: Sequence[str], destination_dir: str) -> None:
        for idx, source_path in enumerate(image_paths):
            source = Path(source_path)
            target = Path(destination_dir) / f"{idx:05d}_{source.name}"
            try:
                os.symlink(source.resolve(), target)
            except Exception:
                shutil.copy2(source, target)


def collect_image_paths(root_dir: str, suffixes: Iterable[str] = SUPPORTED_SUFFIXES) -> List[str]:
    """Collect all supported image files recursively from a directory."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    image_paths: List[str] = []
    for suffix in suffixes:
        image_paths.extend([str(path) for path in root.rglob(f"*{suffix}")])

    image_paths = sorted(set(image_paths))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No supported image files found in {root_dir}")

    return image_paths


def resolve_column_name(columns: Sequence[str], requested_name: str) -> str:
    """Resolve a column name with case-insensitive matching."""
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    key = requested_name.strip().lower()
    if key not in lookup:
        raise KeyError(f"Column '{requested_name}' was not found. Available columns: {list(columns)}")
    return lookup[key]


def normalize_filename_for_lookup(name: str) -> str:
    """Normalize a file name for spreadsheet/path matching."""
    return Path(str(name)).name.strip().lower()


def filter_reference_paths_by_qc(reference_paths: Sequence[str], args) -> Tuple[List[str], Dict[str, float]]:
    """Filter the reference bank using a QC spreadsheet and the Total cleanliness score."""
    if not args.reference_csv:
        return list(reference_paths), {
            "Reference_QC_filter_applied": False,
            "Reference_clean_max_total": None,
            "Num_reference_candidates_before_qc": int(len(reference_paths)),
            "Num_reference_qc_matched": int(len(reference_paths)),
            "Num_reference_qc_missing": 0,
            "Num_reference_cases_after_qc_filter": int(len(reference_paths)),
        }

    qc_df = pd.read_csv(args.reference_csv)
    filename_column = resolve_column_name(qc_df.columns, args.reference_filename_column)
    total_column = resolve_column_name(qc_df.columns, args.reference_total_column)

    qc_subset = qc_df[[filename_column, total_column]].copy()
    qc_subset[filename_column] = qc_subset[filename_column].astype(str)
    qc_subset["_lookup_name"] = qc_subset[filename_column].map(normalize_filename_for_lookup)
    qc_subset[total_column] = pd.to_numeric(qc_subset[total_column], errors="coerce")
    qc_subset = qc_subset.dropna(subset=[total_column])
    qc_subset = qc_subset.drop_duplicates(subset=["_lookup_name"], keep="first")

    qc_lookup = dict(zip(qc_subset["_lookup_name"], qc_subset[total_column]))

    filtered_paths: List[str] = []
    matched_totals: List[float] = []
    missing_count = 0

    for path in reference_paths:
        lookup_name = normalize_filename_for_lookup(path)
        if lookup_name not in qc_lookup:
            if args.keep_unlisted_reference_images:
                filtered_paths.append(path)
            else:
                missing_count += 1
            continue

        total_value = float(qc_lookup[lookup_name])
        if total_value <= args.reference_clean_max_total:
            filtered_paths.append(path)
            matched_totals.append(total_value)

    summary = {
        "Reference_QC_filter_applied": True,
        "Reference_QC_csv": str(args.reference_csv),
        "Reference_QC_filename_column": filename_column,
        "Reference_QC_total_column": total_column,
        "Reference_clean_max_total": float(args.reference_clean_max_total),
        "Num_reference_candidates_before_qc": int(len(reference_paths)),
        "Num_reference_qc_rows": int(len(qc_subset)),
        "Num_reference_qc_matched": int(len(reference_paths) - missing_count),
        "Num_reference_qc_missing": int(missing_count),
        "Num_reference_cases_after_qc_filter": int(len(filtered_paths)),
    }

    if len(matched_totals) > 0:
        summary["Reference_selected_total_mean"] = round(float(np.mean(matched_totals)), 3)
        summary["Reference_selected_total_min"] = round(float(np.min(matched_totals)), 3)
        summary["Reference_selected_total_max"] = round(float(np.max(matched_totals)), 3)

    if len(filtered_paths) == 0:
        raise ValueError(
            "No reference images remained after applying the QC spreadsheet filter. "
            "Check filename matching and the reference_clean_max_total threshold."
        )

    return filtered_paths, summary

def extract_case_id(image_path: str) -> str:
    name = Path(image_path).name
    subj = re.search(r"LISA_(?:VALIDATION|TESTING)_(\d+)", name, flags=re.IGNORECASE)
    ori = re.search(r"(?:^|[_-])(axi|cor|sag)(?:[_-]|\.|$)", name, flags=re.IGNORECASE)
    if subj and ori:
        return f"LISA_{subj.group(1)}_{ori.group(1).upper()}"
    match = re.search(r"(LISAHF\d+)", name, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem



def clean_number(value, digits: int = 3):
    """Round finite numeric values and convert non-finite values to None for JSON safety."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(value):
        return None
    return round(value, digits)


def sanitize_for_json(value):
    """Recursively replace NaN/Inf with None before json.dump(..., allow_nan=False)."""
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return clean_number(value)
    return value


def add_case_distribution_stats(summary: Dict[str, object], case_rows: Sequence[Dict[str, object]], metric_name: str) -> None:
    """Add mean/median/min/max stats for a per-case metric when finite values exist."""
    values: List[float] = collect_finite_case_metric_values(case_rows, metric_name)

    if not values:
        summary[f"{metric_name}_case_mean"] = None
        summary[f"{metric_name}_case_median"] = None
        summary[f"{metric_name}_case_min"] = None
        summary[f"{metric_name}_case_max"] = None
        return

    array = np.asarray(values, dtype=float)
    summary[f"{metric_name}_case_mean"] = clean_number(np.mean(array))
    summary[f"{metric_name}_case_median"] = clean_number(np.median(array))
    summary[f"{metric_name}_case_min"] = clean_number(np.min(array))
    summary[f"{metric_name}_case_max"] = clean_number(np.max(array))


def collect_finite_case_metric_values(case_rows: Sequence[Dict[str, object]], metric_name: str) -> List[float]:
    """Collect finite numeric values from a per-case metric column."""
    values: List[float] = []
    for row in case_rows:
        value = row.get(metric_name)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            values.append(numeric_value)
    return values


def format_liveboard_value(value, args):
    """Format exact liveboard metric fields as numbers or strings."""
    value = clean_number(value)
    if getattr(args, "liveboard_metrics_as_strings", False):
        return "" if value is None else str(value)
    return value


def add_liveboard_summary_fields(
    summary: Dict[str, object],
    case_rows: Sequence[Dict[str, object]],
    args,
) -> None:
    """Add the exact summary metric fields expected by the liveboard.

    The liveboard fields are aliases of the per-case metric distributions.
    They are intentionally added in addition to the more descriptive internal
    names so that older analysis code and the liveboard can both read the
    same JSON output.
    """

    def assign(metric_name: str, output_names: Tuple[str, str, str]) -> None:
        values = collect_finite_case_metric_values(case_rows, metric_name)
        mean_name, min_name, max_name = output_names

        if not values:
            summary[mean_name] = format_liveboard_value(None, args)
            summary[min_name] = format_liveboard_value(None, args)
            summary[max_name] = format_liveboard_value(None, args)
            return

        array = np.asarray(values, dtype=float)
        summary[mean_name] = format_liveboard_value(np.mean(array), args)
        summary[min_name] = format_liveboard_value(np.min(array), args)
        summary[max_name] = format_liveboard_value(np.max(array), args)

    assign("FID_to_reference", ("FID_mean", "FID_min", "FID_max"))
    assign("LPIPS_to_reference", ("LPIPS_mean", "LPIPS_min", "LPIPS_max"))
    assign("PSNR_to_reference", ("PSNR_mean", "PSNR_min", "PSNR_max"))
    assign("BRISQUE_enhanced", ("BRISQUE_enhanced_mean", "BRISQUE_enhanced_min", "BRISQUE_enhanced_max"))
    assign("BRISQUE_delta", ("BRISQUE_delta_case_mean", "BRISQUE_delta_case_min", "BRISQUE_delta_case_max"))
    assign("CLIPIQA_enhanced", ("CLIPIQA_enhanced_case_mean", "CLIPIQA_enhanced_case_min", "CLIPIQA_enhanced_case_max"))
    assign("CLIPIQA_delta", ("CLIPIQA_delta_case_mean", "CLIPIQA_delta_case_min", "CLIPIQA_delta_case_max"))
    assign("FRD_to_reference", ("FRD_mean", "FRD_min", "FRD_max"))


def get_liveboard_payload(summary: Dict[str, object]) -> Dict[str, object]:
    """Return only the exact liveboard metric fields from the summary."""
    return {field: summary.get(field) for field in LIVEBOARD_METRIC_FIELDS}


def build_case_path_lookup(image_paths: Sequence[str], label: str) -> Dict[str, str]:
    """Map image paths by case ID and fail if duplicate case IDs are found."""
    lookup: Dict[str, str] = {}
    duplicates: List[str] = []

    for image_path in image_paths:
        case_id = extract_case_id(image_path)
        if case_id in lookup:
            duplicates.append(case_id)
        else:
            lookup[case_id] = image_path

    if duplicates:
        raise ValueError(METADATA_MISMATCH_MESSAGE)

    return lookup


def _tuples_equal(a: Sequence[float], b: Sequence[float], atol: float = 1e-6) -> bool:
    """Compare numeric SimpleITK metadata tuples with a small tolerance."""
    if len(a) != len(b):
        return False
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=0.0, atol=atol))


def medical_image_metadata_matches(input_path: str, prediction_path: str) -> bool:
    """Return True when input and prediction share the same acquisition geometry.

    Only geometry is compared (dimension, size, components, spacing, origin,
    direction). The full SimpleITK metadata dictionary is intentionally NOT
    compared: enhancement pipelines (nibabel/SimpleITK rewrite) routinely add or
    drop harmless header fields without changing geometry, and requiring the
    dictionaries to be identical falsely rejects valid submissions. Geometry
    equality is sufficient to prevent resampling / geometric tampering.
    """
    if sitk is None:
        raise ImportError("SimpleITK is required to validate medical image metadata.")

    input_image = sitk.ReadImage(input_path)
    prediction_image = sitk.ReadImage(prediction_path)

    if input_image.GetDimension() != prediction_image.GetDimension():
        return False
    if input_image.GetSize() != prediction_image.GetSize():
        return False
    if input_image.GetNumberOfComponentsPerPixel() != prediction_image.GetNumberOfComponentsPerPixel():
        return False
    if not _tuples_equal(input_image.GetSpacing(), prediction_image.GetSpacing()):
        return False
    if not _tuples_equal(input_image.GetOrigin(), prediction_image.GetOrigin()):
        return False
    if not _tuples_equal(input_image.GetDirection(), prediction_image.GetDirection()):
        return False

    return True


def validate_prediction_metadata_against_inputs(
    input_paths: Sequence[str],
    prediction_paths: Sequence[str],
) -> None:
    """Validate that each prediction has an input with identical metadata.

    Files are paired by the LISA case ID extracted from the filename, so an
    input named LISAHF12345.nii.gz can be matched to a prediction named
    LISAHF12345qualityimprovement.nii.gz.
    """
    input_lookup = build_case_path_lookup(input_paths, "input")
    prediction_lookup = build_case_path_lookup(prediction_paths, "prediction")

    if set(input_lookup) != set(prediction_lookup):
        raise ValueError(METADATA_MISMATCH_MESSAGE)

    for case_id, prediction_path in prediction_lookup.items():
        input_path = input_lookup[case_id]
        if not medical_image_metadata_matches(input_path, prediction_path):
            raise ValueError(METADATA_MISMATCH_MESSAGE)


def score_task1b_case_level(
    enhanced_paths: Sequence[str],
    reference_paths: Sequence[str],
    args,
    reference_qc_summary: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Compute aggregate and true per-case Task 1b scores.

    Each enhanced case is independently compared with the same clean reference
    slice bank. This produces one row per submitted case instead of only a
    distribution-level score across the full submission.
    """
    metrics = Metrics(args)

    reference_slices, reference_case_count = metrics.build_slice_pool(reference_paths)
    if len(reference_slices) == 0:
        raise ValueError("No usable slices were found in reference_dir after QC filtering.")

    enhanced_slices, enhanced_case_count = metrics.build_slice_pool(enhanced_paths)
    if len(enhanced_slices) == 0:
        raise ValueError("No usable slices were found in predictions_dir.")

    summary: Dict[str, object] = {
        "Num_enhanced_cases": int(enhanced_case_count),
        "Num_reference_cases": int(reference_case_count),
        "Num_enhanced_slices": int(len(enhanced_slices)),
        "Num_reference_slices": int(len(reference_slices)),
        "Num_random_pairs": int(min(args.num_random_pairs, len(enhanced_slices) * len(reference_slices))),
        "Num_random_pairs_per_case_requested": int(args.num_random_pairs),
        "Case_score_protocol": "Each enhanced case is compared independently to the shared clean reference slice bank.",
    }

    if reference_qc_summary is not None:
        summary.update(reference_qc_summary)

    fid_value, fid_backend = metrics.fid(enhanced_slices, reference_slices)
    summary["FID"] = clean_number(fid_value)
    summary["FID_backend"] = fid_backend

    lpips_value, lpips_pair_values = metrics.lpips_unpaired(enhanced_slices, reference_slices)
    summary["LPIPS_unpaired"] = clean_number(lpips_value)
    summary["LPIPS_backend"] = metrics.lpips_backend
    summary["LPIPS_pair_values"] = [clean_number(value, digits=6) for value in lpips_pair_values]

    psnr_value = metrics.psnr_unpaired(enhanced_slices, reference_slices)
    summary["PSNR_unpaired"] = clean_number(psnr_value)

    reference_brisque = None
    reference_clipiqa = None

    if args.compute_brisque:
        reference_brisque = metrics.brisque_qc(reference_slices)
        brisque_enhanced = metrics.brisque_qc(enhanced_slices)
        summary["BRISQUE_enhanced"] = clean_number(brisque_enhanced)
        summary["BRISQUE_reference"] = clean_number(reference_brisque)
        summary["BRISQUE_delta"] = clean_number(brisque_enhanced - reference_brisque)

    if args.compute_clipiqa:
        reference_clipiqa = metrics.clipiqa(reference_slices)
        clipiqa_enhanced = metrics.clipiqa(enhanced_slices)
        summary["CLIPIQA_enhanced"] = clean_number(clipiqa_enhanced)
        summary["CLIPIQA_reference"] = clean_number(reference_clipiqa)
        summary["CLIPIQA_delta"] = clean_number(clipiqa_enhanced - reference_clipiqa)

    if args.compute_frd:
        frd_value = metrics.frd(enhanced_paths, reference_paths)
        summary["FRD"] = clean_number(frd_value)

    case_rows: List[Dict[str, object]] = []

    for enhanced_path in enhanced_paths:
        case_id = extract_case_id(enhanced_path)
        case_slices, used_case_count = metrics.build_slice_pool([enhanced_path])

        row: Dict[str, object] = {
            "Case_ID": case_id,
            "Enhanced_filename": Path(enhanced_path).name,
            "Enhanced_path": str(enhanced_path),
            "Status": "ok" if used_case_count == 1 and len(case_slices) > 0 else "failed_no_usable_slices",
            "Num_enhanced_slices": int(len(case_slices)),
            "Num_reference_slices": int(len(reference_slices)),
            "Num_random_pairs": int(min(args.num_random_pairs, len(case_slices) * len(reference_slices))),
        }

        if len(case_slices) == 0:
            row.update({
                "FID_to_reference": None,
                "FID_backend": None,
                "LPIPS_to_reference": None,
                "LPIPS_backend": None,
                "PSNR_to_reference": None,
            })
            case_rows.append(row)
            continue

        case_fid, case_fid_backend = metrics.fid(case_slices, reference_slices)
        case_lpips, case_lpips_values = metrics.lpips_unpaired(case_slices, reference_slices)
        case_psnr = metrics.psnr_unpaired(case_slices, reference_slices)

        row.update({
            "FID_to_reference": clean_number(case_fid),
            "FID_backend": case_fid_backend,
            "LPIPS_to_reference": clean_number(case_lpips),
            "LPIPS_backend": metrics.lpips_backend,
            "LPIPS_pair_mean": clean_number(np.mean(case_lpips_values)) if len(case_lpips_values) > 0 else None,
            "LPIPS_pair_std": clean_number(np.std(case_lpips_values)) if len(case_lpips_values) > 0 else None,
            "LPIPS_pair_min": clean_number(np.min(case_lpips_values)) if len(case_lpips_values) > 0 else None,
            "LPIPS_pair_max": clean_number(np.max(case_lpips_values)) if len(case_lpips_values) > 0 else None,
            "PSNR_to_reference": clean_number(case_psnr),
        })

        if args.compute_brisque:
            case_brisque = metrics.brisque_qc(case_slices)
            row["BRISQUE_enhanced"] = clean_number(case_brisque)
            row["BRISQUE_reference"] = clean_number(reference_brisque)
            row["BRISQUE_delta"] = clean_number(case_brisque - reference_brisque)

        if args.compute_clipiqa:
            case_clipiqa = metrics.clipiqa(case_slices)
            row["CLIPIQA_enhanced"] = clean_number(case_clipiqa)
            row["CLIPIQA_reference"] = clean_number(reference_clipiqa)
            row["CLIPIQA_delta"] = clean_number(case_clipiqa - reference_clipiqa)



        case_rows.append(row)

    for metric_name in [
        "FID_to_reference",
        "LPIPS_to_reference",
        "PSNR_to_reference",
        "BRISQUE_enhanced",
        "BRISQUE_delta",
        "CLIPIQA_enhanced",
        "CLIPIQA_delta",
        "FRD_to_reference",
    ]:
        add_case_distribution_stats(summary, case_rows, metric_name)

    add_liveboard_summary_fields(summary, case_rows, args)

    summary["Num_case_rows"] = int(len(case_rows))
    summary["Num_case_rows_ok"] = int(sum(1 for row in case_rows if row.get("Status") == "ok"))
    summary["Num_case_rows_failed"] = int(sum(1 for row in case_rows if row.get("Status") != "ok"))

    return summary, case_rows

def score_task1b(
    enhanced_paths: Sequence[str],
    reference_paths: Sequence[str],
    args,
    reference_qc_summary: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute and return scores for Task 1b."""
    metrics = Metrics(args)
    return metrics.score_task1b(enhanced_paths, reference_paths, reference_qc_summary=reference_qc_summary)


def main():
    """Main function."""
    args = get_args()

    predictions_dir = utils.inspect_zip(args.predictions_dir, path=PRED_PARENT_DIR)
    enhanced_paths = collect_image_paths(predictions_dir)

    if args.input_dir:
        input_dir = utils.inspect_zip(args.input_dir, path=INPUT_PARENT_DIR)
        input_paths = collect_image_paths(input_dir, suffixes=MEDICAL_SUFFIXES)
        try:
            validate_prediction_metadata_against_inputs(input_paths, enhanced_paths)
        except ValueError as exc:
            if str(exc) == METADATA_MISMATCH_MESSAGE:
                print(METADATA_MISMATCH_MESSAGE, file=sys.stderr)
                raise SystemExit(1) from None
            raise

    reference_dir = utils.inspect_zip(args.reference_dir, path=REF_PARENT_DIR)
    reference_paths = collect_image_paths(reference_dir)
    reference_paths, reference_qc_summary = filter_reference_paths_by_qc(reference_paths, args)

    summary, case_rows = score_task1b_case_level(
        enhanced_paths,
        reference_paths,
        args,
        reference_qc_summary=reference_qc_summary,
    )

    pd.DataFrame([summary]).to_csv(args.summary_output_csv, index=False)
    pd.DataFrame(case_rows).to_csv(args.per_case_output_csv, index=False)

    payload = {
        "submission_status": "SCORED",
        **get_liveboard_payload(summary),
        "FRD": clean_number(summary.get("FRD")),
    }

    with open(args.output, "w", encoding="utf-8") as json_file:
        json.dump(sanitize_for_json(payload), json_file, indent=4, allow_nan=False)


if __name__ == "__main__":
    main()
