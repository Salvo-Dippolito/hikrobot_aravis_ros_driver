#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque

# -----------------------------
# Calibration matrix + distortion
# -----------------------------
# K = np.array([[1274.68638, 0, 660.635273],
#               [0, 1273.56819, 533.098699],
#               [0, 0, 1]], dtype=np.float32)

K = np.array([[1292.06779, 0, 656.907035],
              [0, 1295.77329, 508.768017],
              [0, 0, 1]], dtype=np.float32)

dist = np.array([-0.07804858,  0.13329546, -0.00138415, -0.00035261,  0], dtype=np.float32)
# dist = np.array([-1000,  1000, 1000, 1000,  0], dtype=np.float32)

# -----------------------------
# Chessboard parameters
# -----------------------------
CHESS_ROWS = 13
CHESS_COLS = 19
SQUARE_SIZE_M = 0.02  # meters
ROLLING_FRAMES = 10     # average over last N frames

# -----------------------------
# Globals
# -----------------------------
bridge = CvBridge()
rolling_estimates = deque(maxlen=ROLLING_FRAMES)

# -----------------------------
# Utilities
# -----------------------------
def estimate_square_size(corners):
    objp = np.zeros((CHESS_ROWS*CHESS_COLS,3), dtype=np.float32)
    objp[:,:2] = np.mgrid[0:CHESS_COLS,0:CHESS_ROWS].T.reshape(-1,2)
    objp *= SQUARE_SIZE_M

    success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None

    square_lengths = []
    for i in range(CHESS_ROWS-1):
        for j in range(CHESS_COLS-1):
            idx_tl = i*CHESS_COLS + j
            idx_tr = idx_tl + 1
            idx_bl = idx_tl + CHESS_COLS
            p_tl = objp[idx_tl]
            p_tr = objp[idx_tr]
            p_bl = objp[idx_bl]
            width = np.linalg.norm(p_tr - p_tl)
            height = np.linalg.norm(p_bl - p_tl)
            square_lengths.append((width+height)/2)
    mean_size_mm = np.mean(square_lengths)*1000
    return mean_size_mm

def process_frame(cv_image):
    h, w = cv_image.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1)
    undistorted = cv2.undistort(cv_image, K, dist, None, new_K)

    # Compute difference map
    diff = cv2.diff(cv_image, undistorted)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_color = cv2.applyColorMap(cv2.convertScaleAbs(diff_gray, alpha=5), cv2.COLORMAP_JET)

    # Overlay difference map semi-transparently
    overlay = cv2.addWeighted(cv_image, 0.7, diff_color, 0.3, 0)

    # Chessboard detection
    gray_raw = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray_undist = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    found_raw, corners_raw = cv2.findChessboardCorners(gray_raw, (CHESS_COLS, CHESS_ROWS), None)
    found_undist, corners_undist = cv2.findChessboardCorners(gray_undist, (CHESS_COLS, CHESS_ROWS), None)

    mean_raw = mean_undist = None
    if found_raw:
        mean_raw = estimate_square_size(corners_raw)
    if found_undist:
        mean_undist = estimate_square_size(corners_undist)
        if mean_undist is not None:
            rolling_estimates.append(mean_undist)

    # Rolling average
    if rolling_estimates:
        rolling_mean = np.mean(rolling_estimates)
        rolling_std = np.std(rolling_estimates)
    else:
        rolling_mean = rolling_std = None

    # Safe print
    raw_str = f"{mean_raw:.2f}" if mean_raw is not None else "N/A"
    undist_str = f"{mean_undist:.2f}" if mean_undist is not None else "N/A"
    roll_str = f"{rolling_mean:.2f} ± {rolling_std:.2f}" if rolling_mean is not None else "N/A"
    print(f"Raw: {raw_str} mm | Undist: {undist_str} mm | Rolling mean: {roll_str}")

    # Draw chessboard corners
    display_raw = cv2.drawChessboardCorners(cv_image.copy(), (CHESS_COLS, CHESS_ROWS), corners_raw, found_raw)
    display_undist = cv2.drawChessboardCorners(undistorted.copy(), (CHESS_COLS, CHESS_ROWS), corners_undist, found_undist)

    # Stack horizontally: raw | undistorted | difference overlay
    combined = np.hstack((display_raw, display_undist, overlay))
    cv2.imshow("Raw | Undistorted | Difference", combined)
    cv2.waitKey(1)

# -----------------------------
# ROS callback
# -----------------------------
def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        rospy.logerr(f"CvBridge error: {e}")
        return
    process_frame(cv_image)

# -----------------------------
# Main
# -----------------------------
def main():
    rospy.init_node('undistortion_tester', anonymous=True)
    rospy.Subscriber("/left_camera/image", Image, image_callback)
    rospy.loginfo("Undistortion tester node started.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down node.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
