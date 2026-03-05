# aravis_camera_ros_driver

## Overview

`aravis_camera_ros_driver` is a ROS package designed to interface with industrial cameras using the Aravis library and GenICam protocol. It provides ROS nodes and utilities for camera image acquisition, calibration, and integration with other ROS tools. The package is suitable for robotics, computer vision, and research applications requiring robust camera handling and calibration workflows.

So far only external trigger  mode has been implemented in the driver to work with an HIKRobot USB3 camera, but the general structure presented in this code is easily modifiable to work with any aravis or genicam compatible device.

The package also presents scripts to collect images for intrinsic camera calibration and scripts to calibrate camera intrinsics from the collected calibration images.

## Features

- **Camera Node**: Launch and manage camera streams using Aravis and GenICam.
- **Calibration Tools**: Scripts and utilities for camera calibration, including sample collection and undistortion.
- **Flexible Configuration**: YAML-based configuration for camera and calibration parameters.
- **Launch Files**: Predefined launch files for triggering cameras and running calibration workflows.
- **RViz Integration**: RViz configuration for visualizing camera data.

## Directory Structure

```
.
├── CMakeLists.txt                # Build configuration
├── package.xml                   # ROS package manifest
├── include/aravis_camera_ros_driver/
│   └── collect_calib_samples.h   # Header for calibration sample collection
├── src/
│   ├── aravis_camera_node.cpp    # Main camera node implementation
│   ├── collect_calib_samples.cpp # Calibration sample collection node
│   ├── snapshot.cpp              # Snapshot utility
│   └── ...
├── scripts/
│   ├── camera-calibration.py     # Camera calibration script
│   ├── dataset_camera_calibrator.py # Dataset calibration utility
│   ├── k_folds_camera_calibrator.py  # New calibration workflow
│   └── test_undistort.py         # Undistortion test script
├── config/
│   ├── calib_params.yaml         # Calibration parameters
│   └── camera_params.yaml        # Camera parameters
├── calib_images/                 # Calibration images and results
│   ├── corners.yaml
│   ├── corners_weird.yaml
│   └── ...
├── launch/
│   ├── aravis_camera_trigger.launch      # Camera trigger launch
│   ├── aravis_camera_trigger_old.launch  # Legacy trigger launch
│   └── test_calib.launch                # Calibration test launch
├── rviz/
│   └── rviz_cam.rviz             # RViz configuration
└── genicam_output.xml            # GenICam XML output
```

## Installation

### Dependencies
- ROS (tested on ROS Melodic/Noetic)
- Aravis library
- OpenCV (for calibration scripts)
- Python 3 (for scripts)

### Build Instructions
1. Clone this repository into your ROS workspace `src` directory:
   ```bash
   cd ~/ros_ws/src
   git clone <repo_url>
   ```
2. Install dependencies:
   ```bash
   sudo apt-get install libaravis-0.8-dev python3-opencv
   # Or use rosdep for ROS dependencies
   rosdep install --from-paths . --ignore-src -r -y
   ```
3. Build the workspace:
   ```bash
   cd ~/ros_ws
   catkin_make
   source devel/setup.bash
   ```

## Usage


### Launching the ROS Driver

To launch the main ROS camera driver node:

```bash
cd ~/ros_ws
source devel/setup.bash
roslaunch aravis_camera_ros_driver aravis_camera_trigger.launch
```

This will start the camera node with parameters specified in the `config/` directory. Make sure your camera is connected and powered on.

### Launching Calibration Scripts

This package provides as well scripts for intrinsic camera calibration. 
#### Guided Image Dataset Collection

To collect calibration images interactively, with region/pose guidance:

```bash
cd ~/ros_ws
source devel/setup.bash
rosrun aravis_camera_ros_driver collect_calib_samples
```

This launches the C++ node, which subscribes to the camera image topic and opens an OpenCV window for manual sample selection. Images and detected corners are saved to `calib_images/corners.yaml`.

#### Calibration Scripts

**Standard calibration with filtering and k-fold cross-validation:**

```bash
cd ~/ros_ws/src/aravis_camera_ros_driver/scripts
python3 camera-calibration.py
```
This script guides you through calibration using the collected dataset, performs filtering, k-fold cross-validation, and saves the best camera matrix to YAML.

**Calibration using a dataset (with region/pose info):**

```bash
python3 dataset_camera_calibrator.py
```
Uses the dataset in `calib_images/corners.yaml` and performs calibration and analysis, including per-region error reporting.

**K-fold cross-validation calibration (advanced, stable intrinsic matrix estimation):**

```bash
python3 k_folds_camera_calibrator.py
```
Performs k-fold cross-validation for robust intrinsic matrix estimation, useful for datasets with pose/region diversity.

**Undistortion test:**

```bash
python3 test_undistort.py
```
Visualizes and verifies calibration results.

Calibration parameters and results will be saved in the `calib_images/` or `config/` directories as YAML files.

### Configuration

- **Camera Parameters**: Edit `config/camera_params.yaml` to set camera-specific parameters (resolution, frame rate, gain, exposition etc.. )
- **Calibration Parameters**: Edit `config/calib_params.yaml` or use calibration scripts to generate/update these files.

### Visualization

- Use the provided RViz configuration:
  ```bash
  rviz -d $(rospack find aravis_camera_ros_driver)/rviz/rviz_cam.rviz
  ```

## Scripts

- `camera-calibration.py`: Interactive camera calibration tool.
- `dataset_camera_calibrator.py`: Calibrate using a dataset of images.
- `k_folds_camera_calibrator.py`: Calibrate camera matrix over image dataset using k-fold cross validation, provides a more stable estimation of the intrinsic camera matrix
- `test_undistort.py`: Test and visualize undistortion results.

## Launch Files

- `aravis_camera_trigger.launch`: Main launch file for camera node.
- `test_calib.launch`: Launch for calibration testing.



## Acknowledgements

- [Aravis Project](https://github.com/AravisProject/aravis)
- ROS Community
- OpenCV

