#!/usr/bin/env python3
"""
calib_kfold.py

Robust camera calibration with k-fold cross-validation + filtering + optional bootstrap.
Saves best result to YAML (OpenCV/ROS camera_info style).

Dependencies:
    pip install numpy opencv-python pyyaml

Usage examples in the header of the assistant response.
"""
import argparse
import glob
import os
import random
import time
from collections import namedtuple

import cv2
import numpy as np
import yaml

# --------------------
# Utilities
# --------------------
def compute_laplacian_var(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def compute_coverage(corners, img_shape):
    # corners: Nx2 array-like
    xs = corners[:, 0]
    ys = corners[:, 1]
    cov_x = (xs.max() - xs.min()) / float(img_shape[1])
    cov_y = (ys.max() - ys.min()) / float(img_shape[0])
    return min(cov_x, cov_y)

def make_objp(cols, rows, square_size):
    # inner corners: cols x rows
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp

def load_npz_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    # Accept common keys: objpoints/imgpoints or obj_points/img_points
    objpoints = data.get('objpoints') or data.get('obj_points') or data.get('obj_pts')
    imgpoints = data.get('imgpoints') or data.get('img_points') or data.get('img_pts')
    image_shapes = data.get('image_shapes') if 'image_shapes' in data else None
    if objpoints is None or imgpoints is None:
        raise ValueError("npz must contain 'objpoints' and 'imgpoints' arrays")
    # Ensure lists
    objpoints = list(objpoints)
    imgpoints = [np.asarray(ip, dtype=np.float32) for ip in imgpoints]
    return objpoints, imgpoints, image_shapes

def reprojection_error_for_matrix(objpoints_list, imgpoints_list, K, D):
    total_err = 0.0
    total_pts = 0
    per_image = []
    for objp, imgp in zip(objpoints_list, imgpoints_list):
        # SolvePnP to get rvec,tvec
        if len(objp) == 0:
            per_image.append(np.nan); continue
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            per_image.append(np.inf); continue
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(imgp - proj, axis=1)
        per_image.append(err.mean())
        total_err += err.sum()
        total_pts += len(err)
    mean_err = total_err / total_pts if total_pts > 0 else np.inf
    return mean_err, per_image

# --------------------
# Core pipeline
# --------------------
Result = namedtuple('Result', ['K', 'D', 'rvecs', 'tvecs', 'train_idx', 'val_idx', 'val_error'])

def calibrate_on_indices(objpoints_all, imgpoints_all, image_shape, train_idx, flags=0, fix_principal_point=False, use_intrinsic_guess=False, Kinit=None, Dinit=None):
    obj_tr = [objpoints_all[i] for i in train_idx]
    img_tr = [imgpoints_all[i] for i in train_idx]
    if len(img_tr) == 0:
        return None
    if use_intrinsic_guess and Kinit is None:
        raise ValueError("use_intrinsic_guess requires Kinit/Dinit")
    # calibrate
    if use_intrinsic_guess:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        K = Kinit.copy()
        D = Dinit.copy()
    else:
        K = None
        D = None
    # optionally fix principal point
    if fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    # common: don't let solver overfit too many radial params by default
    # Caller can set flags externally
    ret, K_res, D_res, rvecs, tvecs = cv2.calibrateCamera(obj_tr, img_tr, image_shape, K, D, flags=flags)
    return ret, K_res, D_res, rvecs, tvecs

def kfold_calibration(objpoints, imgpoints, image_shape, k=5, shuffle=True, seed=42, flags=0):
    N = len(imgpoints)
    idxs = np.arange(N)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idxs)
    folds = []
    base = 0
    # create roughly equal folds
    sizes = [N // k + (1 if i < (N % k) else 0) for i in range(k)]
    cur = 0
    for s in sizes:
        folds.append(idxs[cur:cur+s])
        cur += s
    results = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        # calibrate on train_idx
        out = calibrate_on_indices(objpoints, imgpoints, image_shape, train_idx, flags=flags)
        if out is None:
            continue
        ret, K, D, rvecs, tvecs = out
        # Evaluate on val set: use solvePnP per image on val set, compute mean reproj error
        val_objs = [objpoints[j] for j in val_idx]
        val_imgs = [imgpoints[j] for j in val_idx]
        mean_err, per_image = reprojection_error_for_matrix(val_objs, val_imgs, K, D)
        results.append(Result(K=K, D=D, rvecs=rvecs, tvecs=tvecs, train_idx=train_idx, val_idx=val_idx, val_error=mean_err))
        print(f"[fold {i}] val mean reproj: {mean_err:.4f} px  (train size {len(train_idx)}, val size {len(val_idx)})")
    return results

# --------------------
# CLI / orchestrator
# --------------------
def save_camera_yaml(filename, K, D, width, height):
    # Save in a simple YAML camera_info style
    data = {
        'image_width': int(width),
        'image_height': int(height),
        'camera_matrix': {
            'rows': 3, 'cols': 3,
            'data': K.flatten().tolist()
        },
        'distortion_coefficients': {
            'rows': 1, 'cols': int(len(D)),
            'data': D.flatten().tolist()
        }
    }
    with open(filename, 'w') as f:
        yaml.safe_dump(data, f)
    print("Saved YAML:", filename)

def main():
    p = argparse.ArgumentParser(description="K-fold camera calibration pipeline")
    p.add_argument('--dataset', type=str, default=None, help='NPZ dataset with objpoints,imgpoints (optional)')
    p.add_argument('--images', type=str, default=None, help='Image folder (optional). If dataset omitted, images will be used and corners detected.')
    p.add_argument('--cols', type=int, default=9, help='checkerboard inner cols')
    p.add_argument('--rows', type=int, default=6, help='checkerboard inner rows')
    p.add_argument('--square', type=float, default=0.041, help='square size in meters')
    p.add_argument('--k', type=int, default=5, help='k folds')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min_blur', type=float, default=150.0, help='min Laplacian var to keep image')
    p.add_argument('--min_coverage', type=float, default=0.20, help='min coverage fraction to keep image')
    p.add_argument('--bootstrap', type=int, default=0, help='bootstrap trials (optional)')
    p.add_argument('--out', type=str, default='best_calib.yaml', help='output YAML for best intrinsics')
    p.add_argument('--no_refit', action='store_true', help="Don't refit final calibrateCamera on all images (default: refit)")
    p.add_argument('--fix_principal_point', action='store_true', help='Fix principal point during calibration')
    p.add_argument('--calib_flags', type=int, default=0, help='extra cv2 calibration flags (bitmask)')
    args = p.parse_args()

    # Load dataset
    if args.dataset:
        objpoints_raw, imgpoints_raw, shapes = load_npz_dataset(args.dataset)
        print(f"Loaded dataset from {args.dataset}: {len(imgpoints_raw)} frames")
        # shapes may be None; get first image size from shapes or ask user
        if shapes is not None and len(shapes) > 0:
            image_shape = shapes[0]
            image_shape = (int(image_shape[0]), int(image_shape[1]))
        else:
            # We can't get size from npz - require images or pass --images to get shape
            if args.images:
                imgf = sorted(glob.glob(os.path.join(args.images, '*')))
                if len(imgf) == 0:
                    raise RuntimeError("No images in folder to infer shape")
                timg = cv2.imread(imgf[0], cv2.IMREAD_COLOR)
                image_shape = (timg.shape[1], timg.shape[0])
            else:
                # default fallback
                image_shape = (1280, 960)
    else:
        # detect corners from images folder
        if not args.images:
            raise RuntimeError("Provide --dataset or --images")
        imgfiles = sorted(glob.glob(os.path.join(args.images, '*')))
        objpoints_raw, imgpoints_raw, shapes = detect_corners_from_images(imgfiles, args.cols, args.rows)
        if len(imgpoints_raw) == 0:
            raise RuntimeError("No corners found in any image")
        image_shape = (shapes[0][0], shapes[0][1])
        print(f"Detected corners in {len(imgpoints_raw)} images; image shape {image_shape}")

    # Normalize objpoints to actual square size
    obj_template_unit = make_objp(args.cols, args.rows, 1.0)
    objpoints = []
    for _ in imgpoints_raw:
        objpoints.append(obj_template_unit * args.square)
    # imgpoints already are lists of Nx2 arrays
    imgpoints = imgpoints_raw

    # Compute blur/coverage and filter
    keep_idxs = []
    blurs = []
    covs = []
    for i, ip in enumerate(imgpoints):
        # we need a grayscale image to compute blur; if dataset came from images, we can load; else skip blur filter
        blur_val = None
        cov_val = None
        # try to approximate blur by re-rendering hypothetical image: user should pass images if blur filter desired
        # If dataset came from images, we did compute shapes; otherwise skip blur check (or user should supply images)
        if args.dataset is None and args.images:
            # find corresponding image by index ordering - when created with detection, this matches
            # But if dataset from npz, skip blur
            imgpath = imgfiles[i]
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_val = compute_laplacian_var(gray)
            cov_val = compute_coverage(np.asarray(ip), img.shape)
        else:
            # we can compute coverage from points but not blur
            cov_val = compute_coverage(np.asarray(ip), (image_shape[0], image_shape[1]))
            blur_val = None

        blurs.append(blur_val if blur_val is not None else np.nan)
        covs.append(cov_val if cov_val is not None else np.nan)

        blur_ok = (blur_val is None) or (blur_val >= args.min_blur)
        cov_ok = (cov_val is None) or (cov_val >= args.min_coverage)

        if blur_ok and cov_ok:
            keep_idxs.append(i)
    print(f"Kept {len(keep_idxs)} / {len(imgpoints)} images after filtering (min_blur={args.min_blur}, min_coverage={args.min_coverage})")

    if len(keep_idxs) < args.k:
        raise RuntimeError(f"Too few images after filtering ({len(keep_idxs)}), need at least k={args.k}")

    # Build filtered lists
    objpoints_f = [objpoints[i] for i in keep_idxs]
    imgpoints_f = [imgpoints[i] for i in keep_idxs]

    # k-fold calibration
    print("Running k-fold calibration...")
    k_results = kfold_calibration(objpoints_f, imgpoints_f, image_shape, k=args.k, seed=args.seed, flags=args.calib_flags)

    # Evaluate each candidate globally on the full filtered dataset
    print("Evaluating candidates on the full filtered dataset...")
    global_results = []
    for i, res in enumerate(k_results):
        mean_err, per_img = reprojection_error_for_matrix(objpoints_f, imgpoints_f, res.K, res.D)
        global_results.append((mean_err, res.K, res.D, per_img))
        print(f"Candidate {i}: global mean reprojection {mean_err:.4f} px")

    if len(global_results) == 0:
        raise RuntimeError("No calibration candidates produced")

    # Pick best candidate
    best = min(global_results, key=lambda x: x[0])
    best_mean, best_K, best_D, best_perimg = best
    print(f"Selected best candidate: mean reproj = {best_mean:.4f} px")

    # Optionally bootstrap to estimate variability
    if args.bootstrap > 0:
        print(f"Running {args.bootstrap} bootstrap trials (this can take time)...")
        boot_stats = []
        rng = np.random.RandomState(args.seed)
        for b in range(args.bootstrap):
            # sample 70% for training
            n = len(imgpoints_f)
            sample_idx = rng.choice(n, size=int(0.7*n), replace=True)
            train_obj = [objpoints_f[i] for i in sample_idx]
            train_img = [imgpoints_f[i] for i in sample_idx]
            # calibrate
            ret, Kb, Db, rb, tb = cv2.calibrateCamera(train_obj, train_img, image_shape, None, None, flags=args.calib_flags)
            # eval on the left-out (approx) set
            test_idx = [i for i in range(n) if i not in sample_idx]
            test_obj = [objpoints_f[i] for i in test_idx]
            test_img = [imgpoints_f[i] for i in test_idx]
            if len(test_obj) == 0:
                continue
            mean_err_b, _ = reprojection_error_for_matrix(test_obj, test_img, Kb, Db)
            boot_stats.append((mean_err_b, Kb, Db))
        if len(boot_stats) > 0:
            errs = [x[0] for x in boot_stats]
            print(f"Bootstrap mean error: {np.mean(errs):.4f} px  std: {np.std(errs):.4f} px")

    # Refit final calibration on full filtered dataset using the best candidate as initial guess
    if not args.no_refit:
        print("Refitting final calibrateCamera on all filtered images with best candidate as initial guess...")
        flags = args.calib_flags | cv2.CALIB_USE_INTRINSIC_GUESS
        ret, K_final, D_final, rvecs_final, tvecs_final = cv2.calibrateCamera(objpoints_f, imgpoints_f, image_shape, best_K.copy(), best_D.copy(), flags=flags)
        print("Final calibrateCamera returned RMS reproj error:", ret)
    else:
        K_final = best_K
        D_final = best_D

    # Save final result
    save_camera_yaml(args.out, K_final, D_final, image_shape[0], image_shape[1])
    print("Done.")

if __name__ == '__main__':
    main()
