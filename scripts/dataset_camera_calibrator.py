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
PER_REGION_WARNING_THRESHOLD = 1.5  # px, warn if any region exceeds this RMS

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
# Initial Full-Dataset Calibration
# -----------------------------

def calibrate_full_dataset(images, objp):
    object_points = []
    image_points = []

    for img in images:
        if img.corners_count != objp.shape[0]:
            continue
        object_points.append(objp)
        image_points.append(img.image_points)

    image_size = images[0].image_size

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=0
    )

    return ret, K, dist

# ----------------------------- 
# Error Metrics Utilities       
# -----------------------------

def reprojection_rmse(objp, img_points, rvec, tvec, K, dist):
    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    err = projected - img_points
    rmse = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    return rmse

def solve_extrinsics(objp, img_points, K, dist):
    success, rvec, tvec = cv2.solvePnP(
        objp,
        img_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("PnP failed")
    return rvec, tvec

# -----------------------------
# Per-Fold Calibration
# -----------------------------

def calibrate_one_fold(fold_id, folds, images, objp, image_size):
    train_indices = []
    val_indices = folds[fold_id]

    for k, fold in enumerate(folds):
        if k != fold_id:
            train_indices.extend(fold)

    obj_points_train = []
    img_points_train = []

    for idx in train_indices:
        img = images[idx]
        if img.corners_count != objp.shape[0]:
            continue
        obj_points_train.append(objp)
        img_points_train.append(img.image_points)

    rms_train, K, dist, _, _ = cv2.calibrateCamera(
        obj_points_train,
        img_points_train,
        image_size,
        None,
        None,
        flags=0
    )

    val_errors = []
    per_region_errors = defaultdict(list)

    for idx in val_indices:
        img = images[idx]
        rvec, tvec = solve_extrinsics(objp, img.image_points, K, dist)
        rmse = reprojection_rmse(objp, img.image_points, rvec, tvec, K, dist)
        val_errors.append(rmse)
        per_region_errors[img.region_id].append(rmse)

    # Compute per-region means
    val_rms_per_region = {r: float(np.mean(v)) for r, v in per_region_errors.items()}

    # Warn if any region is unusually high
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
    mean_per_region = {r: float(np.mean(v)) for r, v in per_region_errors.items()}

    return mean_err, std_err, mean_per_region

# -----------------------------
# Fold Construction
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
            fold_id = i % num_folds
            folds[fold_id].append(idx)

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
# Main
# -----------------------------

def main():
    rows, cols, square_size, images = load_dataset(YAML_PATH)
    print(f"Loaded {len(images)} images")
    print(f"Board: {rows} x {cols}, square size = {square_size}")

    objp = build_object_points(rows, cols, square_size)

    # Initial calibration on full dataset
    rms, K_init, dist_init = calibrate_full_dataset(images, objp)
    print("\nInitial calibration on full dataset:")
    print(f"  RMS reprojection error: {rms:.4f}")
    print("  Camera matrix K:\n", K_init)
    print("  Distortion coefficients:\n", dist_init.ravel())

    # Build folds
    folds = build_folds(images, NUM_FOLDS)
    print_fold_stats(folds, images)

    print("\nRunning per-fold calibration...\n")
    results = []
    image_size = images[0].image_size

    for k in range(NUM_FOLDS):
        res = calibrate_one_fold(k, folds, images, objp, image_size)
        results.append(res)
        print(
            f"Fold {k}: train RMS = {res['train_rms']:.4f}, val RMS = {res['val_rms']:.4f}, "
            f"per-region: {res['val_rms_per_region']}"
        )

    print("\nWhole-dataset evaluation per fold:")
    for res in results:
        mean_err, std_err, mean_per_region = evaluate_whole_dataset(res["K"], res["dist"], images, objp)
        print(f"Fold {res['fold']}: mean RMS = {mean_err:.4f}, std RMS = {std_err:.4f}, per-region: {mean_per_region}")

    # Select best fold (lowest validation RMS)
    best_fold = min(results, key=lambda r: r["val_rms"])
    print(f"\nSelected best fold: {best_fold['fold']} with val RMS = {best_fold['val_rms']:.4f}")

    # Recalibrate on all data using best fold as initial guess
    obj_points_all = [objp for _ in images]
    img_points_all = [img.image_points for img in images]

    K_init = best_fold["K"]
    dist_init = best_fold["dist"]

    rms_final, K_final, dist_final, _, _ = cv2.calibrateCamera(
        obj_points_all,
        img_points_all,
        images[0].image_size,
        K_init,
        dist_init,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS 
    )

    print(f"\nFinal full-dataset RMS: {rms_final:.4f}")
    print("Final camera matrix K:\n", K_final)
    print("Final distortion coefficients:\n", dist_final.ravel())

if __name__ == "__main__":
    main()
