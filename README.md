# hikrobot_aravis_ros_driver

A ROS 2 (Jazzy) driver for **Hikrobot GigE/USB3 machine-vision cameras** using the [Aravis](https://github.com/AravisProject/aravis) library. Includes a companion snapshot node that saves frames to disk on demand via a ROS service.

---

## Features

- **Aravis-based acquisition** — uses the GenICam-compliant Aravis 0.8 library, supporting GigE Vision and USB3 Vision cameras
- **Configurable image pipeline** — pixel format (RGB8, BayerRG8, BayerRG12Packed, BayerGB12Packed), auto/manual exposure, gain, gamma, and optional image rescaling
- **Hardware trigger support** — optional external-trigger acquisition mode
- **Shared-memory timestamp** — exposes the capture timestamp via a memory-mapped file for synchronization with external sensors (e.g. LiDAR)
- **Snapshot service** — a separate node subscribes to the image topic and saves the next frame to disk when a `std_srvs/Trigger` service is called
- **Respawn-safe** — the camera node is configured to respawn automatically in the launch file

---

## Package Structure

```
aravis_camera_ros_driver/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   └── camera_params.yaml          # Camera parameters (name, topic, exposure, gain, …)
├── launch/
│   └── aravis_camera_trigger.launch.py   # Launches camera + snapshot nodes
├── rviz/
│   └── rviz_cam.rviz               # RViz config for quick image viewing
├── src/
│   ├── aravis_camera_node.cpp       # Main camera driver node
│   └── snapshot.cpp                 # On-demand frame-saving node
├── calib_images/                    # Stored checkerboard corner detections
└── trash/                           # Legacy / deprecated files
```

---

## Dependencies

### System

| Dependency | Install |
|------------|---------|
| Aravis 0.8 | `sudo apt install libaravis-dev` or build from source |
| OpenCV | `sudo apt install libopencv-dev` |
| GLib / GObject | Pulled in by Aravis |

### ROS 2

- `rclcpp`, `rclpy`
- `std_msgs`, `geometry_msgs`, `sensor_msgs`
- `std_srvs`
- `image_transport`, `cv_bridge`
- `tf2_ros`

---

## Configuration

All parameters are set in [config/camera_params.yaml](config/camera_params.yaml):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `CameraName` | string | `""` | Aravis camera identifier (serial or user-defined name). Empty = first camera found |
| `TopicName` | string | `left_camera/image` | Image topic name |
| `SharedFile` | string | `/home/percro_drone/timeshare` | Path to the shared-memory timestamp file |
| `TriggerEnable` | bool | `true` | Enable hardware trigger mode |
| `ExposureAutoMode` | int | `2` | `0` = manual, `1` = once, `2` = continuous |
| `ExposureTime` | int | `5000` | Manual exposure time (µs), used when `ExposureAutoMode=0` |
| `AutoExposureTimeLower` | int | `100` | Auto-exposure lower limit (µs) |
| `AutoExposureTimeUpper` | int | `20000` | Auto-exposure upper limit (µs) |
| `GainAuto` | int | `2` | `0` = manual, `1` = once, `2` = continuous |
| `Gain` | double | `15.0` | Manual gain value, used when `GainAuto=0` |
| `Gamma` | double | `0.7` | Gamma correction value |
| `GammaSelector` | int | `1` | Camera gamma selector register |
| `PixelFormat` | int | `0` | `0` = RGB8, `1` = BayerRG8, `2` = BayerRG12Packed, `3` = BayerGB12Packed |
| `ImageScale` | double | `1.0` | Output image scale factor (e.g. `0.5` for half resolution) |
| `AcquisitionTimeoutUs` | int | `100000` | Buffer acquisition timeout (µs) |

---

## Usage

### Build

```bash
cd ~/ros_ws
colcon build --packages-select hikrobot_aravis_ros_driver
source install/setup.bash
```

### Launch

```bash
ros2 launch hikrobot_aravis_ros_driver aravis_camera_trigger.launch.py
```

This starts two nodes:

| Node | Description |
|------|-------------|
| `aravis_camera_node` | Opens the camera, configures it from `camera_params.yaml`, and publishes images continuously |
| `snapshot_node` | Subscribes to the image topic and waits for service calls to save frames |

### View images

```bash
# RViz (using the included config)
rviz2 -d $(ros2 pkg prefix hikrobot_aravis_ros_driver)/share/hikrobot_aravis_ros_driver/rviz/rviz_cam.rviz

# Or quick check with rqt
ros2 run rqt_image_view rqt_image_view
```

### Save a snapshot

```bash
ros2 service call /save_frame std_srvs/srv/Trigger
```

The next received frame will be saved to `~/ros_ws/src/FAST-Calib/calib_data/` as a JPEG with a nanosecond timestamp filename.

---

## ROS 2 API

### aravis_camera_node

| | Name | Type | Description |
|---|------|------|-------------|
| **Pub** | `<TopicName>` | `sensor_msgs/msg/Image` | Camera images (via `image_transport`) |

Published images use `frame_id = "camera"` and are stamped with the ROS clock at reception time.

### snapshot_node

| | Name | Type | Description |
|---|------|------|-------------|
| **Sub** | `/left_camera/image` | `sensor_msgs/msg/Image` | Source image stream |
| **Srv** | `/save_frame` | `std_srvs/srv/Trigger` | Trigger saving the next frame to disk |

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_format` | `jpg` | Output file format (`jpg`, `png`, etc.) |

---

## Shared-Memory Timestamp

The camera node creates (or opens) a memory-mapped file at `SharedFile` containing a `time_stamp` struct:

```c
struct time_stamp {
  int64_t high;
  int64_t low;
};
```

This allows external processes (e.g. a LiDAR driver) to read the latest camera capture time for hardware synchronization. The file is created at startup and deleted on shutdown.

---

## Pixel Format Reference

| Value | Enum | Aravis Format | Notes |
|-------|------|---------------|-------|
| 0 | `PF_RGB8` | `ARV_PIXEL_FORMAT_RGB_8_PACKED` | 3-channel, no debayer needed |
| 1 | `PF_BAYER_RG8` | `ARV_PIXEL_FORMAT_BAYER_RG_8` | 8-bit Bayer, debayered to RGB via OpenCV |
| 2 | `PF_BAYER_RG12PACKED` | `ARV_PIXEL_FORMAT_BAYER_RG_12_PACKED` | 12-bit Bayer, converted to 8-bit then debayered |
| 3 | `PF_BAYER_GB12PACKED` | `ARV_PIXEL_FORMAT_BAYER_GB_12_PACKED` | 12-bit Bayer (GB pattern) |

