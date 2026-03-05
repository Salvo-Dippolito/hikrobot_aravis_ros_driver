#!/usr/bin/env python3

import yaml
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
import random

# -----------------------------
# Configuration
# -----------------------------
YAML_PATH = "/home/percro_drone/ros_ws/src/aravis_camera_ros_driver/calib_images/corners.yaml"
NUM_FOLDS = 8
RANDOM_SEED = 42
PER_REGION_WARNING_THRESHOLD = 1.5  # px
FIX_K3 = True                         # Set True if lens distortion is very low
USE_COVERAGE_WEIGHTING = False        # Optional coverage-weighted calibration

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------------
# Data Structures
# -----------------------------
class ImageEntry:
    def __init__(self, meta):
        self.filename = meta["filename"]
        self.image_size = tuple(meta["image_size"])
        self.sharpness = meta["sharpness"]
        self.coverage = meta["coverage"]
        self.region_id = meta["region_id"]
        self.pose_class = meta["pose_class"]
        self.scale_class = meta["scale_class"]
        self.tilt_h = meta["tilt_h"]
        self.tilt_v = meta["tilt_v"]
        self.corners_count = meta["corners_count"]

        corners = np.array(meta["corners"], dtype=np.float32)
        assert corners.shape == (self.corners_count, 2)
        self.image_points = corners

# -----------------------------
# YAML Parsing
# -----------------------------
def load_dataset(yaml_path):
    with open(yaml_path, "r") as f:
        lines = f.readlines()
    # Remove OpenCV YAML directive if present
    if lines[0].startswith("%YAML"):
        lines = lines[1:]
    if lines[0].strip() == "---":
        lines = lines[1:]

    data = yaml.safe_load("".join(lines))
    board = data["board"]
    rows = int(board["rows"])
    cols = int(board["cols"])
    square_size = float(board["square_size"])
    images = [ImageEntry(m) for m in data["images"]]
    return rows, cols, square_size, images

# -----------------------------
# Object Points Construction
# -----------------------------
def build_object_points(rows, cols, square_size):
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= square_size
    return objp

# -----------------------------
# Calibration
# -----------------------------
def calibrate_full_dataset_simple(images, objp):
    object_points = []
    image_points = []
    image_size = images[0].image_size

    for img in images:
        if img.corners_count != objp.shape[0]:
            continue
        if USE_COVERAGE_WEIGHTING:
            weight = max(1, int(np.ceil(img.coverage * 10)))
            for _ in range(weight):
                object_points.append(objp)
                image_points.append(img.image_points)
        else:
            object_points.append(objp)
            image_points.append(img.image_points)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if FIX_K3:
        flags |= cv2.CALIB_FIX_K3

    K_init = np.array([[1000, 0, image_size[0]/2],
                       [0, 1000, image_size[1]/2],
                       [0, 0, 1]], dtype=np.float64)
    dist_init = np.zeros((5,1), dtype=np.float64)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        K_init,
        dist_init,
        flags=flags
    )
    return ret, K, dist, rvecs, tvecs

def solve_extrinsics(objp, img_points, K, dist):
    success, rvec, tvec = cv2.solvePnP(objp, img_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError("PnP failed")
    return rvec, tvec

def reprojection_rmse(objp, img_points, rvec, tvec, K, dist):
    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    projected = projected.reshape(-1,2)
    err = projected - img_points
    return np.sqrt(np.mean(np.sum(err**2, axis=1)))

# -----------------------------
# Folds
# -----------------------------
def build_folds(images, num_folds):
    by_region = defaultdict(list)
    for idx, img in enumerate(images):
        by_region[img.region_id].append(idx)
    for region in by_region:
        random.shuffle(by_region[region])
    folds = [[] for _ in range(num_folds)]
    for region, indices in by_region.items():
        for i, idx in enumerate(indices):
            folds[i % num_folds].append(idx)
    return folds

# -----------------------------
# Weighted Fold Construction
# -----------------------------

def build_weighted_folds(images, num_folds, weight_attr="coverage"):
    """
    Builds k folds of indices for cross-validation.
    Images with higher weight_attr values are distributed to balance folds.
    
    :param images: list of ImageEntry
    :param num_folds: int
    :param weight_attr: str, name of the attribute to weight by (e.g., "coverage" or "corners_count")
    :return: list of lists of indices, folds[num_folds][indices]
    """
    # Create list of (index, weight) tuples
    weighted_indices = [(i, getattr(img, weight_attr)) for i, img in enumerate(images)]
    
    # Sort by weight descending
    weighted_indices.sort(key=lambda x: x[1], reverse=True)
    
    folds = [[] for _ in range(num_folds)]
    fold_weights = [0.0 for _ in range(num_folds)]
    
    # Greedy assignment: assign heaviest image to fold with lowest current total weight
    for idx, weight in weighted_indices:
        min_fold = fold_weights.index(min(fold_weights))
        folds[min_fold].append(idx)
        fold_weights[min_fold] += weight
    
    return folds


def print_fold_stats(folds, images):
    print("\nFold statistics:\n")
    for k, fold in enumerate(folds):
        regions = defaultdict(int)
        poses = defaultdict(int)
        scales = defaultdict(int)
        for idx in fold:
            img = images[idx]
            regions[img.region_id] += 1
            poses[img.pose_class] += 1
            scales[img.scale_class] += 1
        print(f"Fold {k}: {len(fold)} images")
        print(f"  Regions: {dict(regions)}")
        print(f"  Poses:   {dict(poses)}")
        print(f"  Scales:  {dict(scales)}\n")

# -----------------------------
# Per-Fold Calibration
# -----------------------------
def calibrate_one_fold(fold_id, folds, images, objp):
    train_indices = []
    val_indices = folds[fold_id]
    for k, fold in enumerate(folds):
        if k != fold_id:
            train_indices.extend(fold)

    train_images = [images[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]

    rms_train, K, dist, _, _ = calibrate_full_dataset_simple(train_images, objp)

    val_errors = []
    per_region_errors = defaultdict(list)
    for img in val_images:
        rvec, tvec = solve_extrinsics(objp, img.image_points, K, dist)
        rmse = reprojection_rmse(objp, img.image_points, rvec, tvec, K, dist)
        val_errors.append(rmse)
        per_region_errors[img.region_id].append(rmse)

    val_rms_per_region = {r: float(np.mean(v)) for r,v in per_region_errors.items()}
    for r, rms in val_rms_per_region.items():
        if rms > PER_REGION_WARNING_THRESHOLD:
            print(f"WARNING: Fold {fold_id}, region {r} has high RMS = {rms:.3f}")

    return {
        "fold": fold_id,
        "K": K,
        "dist": dist,
        "train_rms": rms_train,
        "val_rms": float(np.mean(val_errors)),
        "val_rms_per_region": val_rms_per_region
    }

def evaluate_whole_dataset(K, dist, images, objp):
    errors = []
    per_region_errors = defaultdict(list)
    for img in images:
        rvec, tvec = solve_extrinsics(objp, img.image_points, K, dist)
        rmse = reprojection_rmse(objp, img.image_points, rvec, tvec, K, dist)
        errors.append(rmse)
        per_region_errors[img.region_id].append(rmse)
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))
    mean_per_region = {r: float(np.mean(v)) for r,v in per_region_errors.items()}
    return mean_err, std_err, mean_per_region

# -----------------------------
# Main
# -----------------------------
def main():
    rows, cols, square_size, images = load_dataset(YAML_PATH)
    print(f"Loaded {len(images)} images")
    print(f"Board: {rows} x {cols}, square size = {square_size}")

    objp = build_object_points(rows, cols, square_size)

    # Initial full-dataset calibration
    rms_init, K_init, dist_init, _, _ = calibrate_full_dataset_simple(images, objp)
    print("\nInitial full-dataset calibration:")
    print(f"  RMS: {rms_init:.4f}")
    print("  K:\n", K_init)
    print("  dist:\n", dist_init.ravel())

    # Build folds
    # folds = build_folds(images, NUM_FOLDS)
    folds = build_weighted_folds(images, NUM_FOLDS, weight_attr="coverage")
    print_fold_stats(folds, images)

    # Per-fold calibration
    results = []
    for k in range(NUM_FOLDS):
        res = calibrate_one_fold(k, folds, images, objp)
        results.append(res)
        print(f"Fold {k}: train RMS = {res['train_rms']:.4f}, val RMS = {res['val_rms']:.4f}, per-region: {res['val_rms_per_region']}")

    # Evaluate whole dataset per fold
    print("\nWhole-dataset evaluation per fold:")
    for res in results:
        mean_err, std_err, mean_per_region = evaluate_whole_dataset(res["K"], res["dist"], images, objp)
        print(f"Fold {res['fold']}: mean RMS = {mean_err:.4f}, std RMS = {std_err:.4f}, per-region: {mean_per_region}")

    # Select best fold
    best_fold = min(results, key=lambda r: r["val_rms"])
    print(f"\nSelected best fold: {best_fold['fold']} with val RMS = {best_fold['val_rms']:.4f}")

    # Recalibrate on all data using best fold as initial guess
    obj_points_all = [objp for _ in images]
    img_points_all = [img.image_points for img in images]
    flags = cv2.CALIB_USE_INTRINSIC_GUESS 
    if FIX_K3:
        flags |= cv2.CALIB_FIX_K3
    rms_final, K_final, dist_final, _, _ = cv2.calibrateCamera(
        obj_points_all,
        img_points_all,
        images[0].image_size,
        best_fold["K"],
        best_fold["dist"],
        flags=flags
    )
    print(f"\nFinal full-dataset RMS: {rms_final:.4f}")
    print("Final K:\n", K_final)
    print("Final dist:\n", dist_final.ravel())

if __name__ == "__main__":
    main()
