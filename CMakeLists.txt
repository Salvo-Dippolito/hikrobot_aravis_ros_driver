cmake_minimum_required(VERSION 3.8)
project(hikrobot_aravis_ros_driver)

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------------
# Find dependencies
# -----------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)


find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(ARAVIS REQUIRED IMPORTED_TARGET aravis-0.8)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

include_directories(
  /opt/ros/jazzy/include
)
# -----------------------------------------------------------------------------
# External include directories
# -----------------------------------------------------------------------------
set(EXTERNAL_INCLUDES
  ${OpenCV_INCLUDE_DIRS}
  ${ARAVIS_INCLUDE_DIRS}
)

# -----------------------------------------------------------------------------
# C++ executables
# -----------------------------------------------------------------------------
add_executable(aravis_camera_node src/aravis_camera_node.cpp)
add_executable(snapshot_node src/snapshot.cpp)

target_include_directories(aravis_camera_node PRIVATE ${EXTERNAL_INCLUDES})
target_include_directories(snapshot_node PRIVATE ${EXTERNAL_INCLUDES})

# Link external libraries
target_link_libraries(aravis_camera_node PkgConfig::ARAVIS ${OpenCV_LIBS})
target_link_libraries(snapshot_node PkgConfig::ARAVIS ${OpenCV_LIBS})

# ROS 2 dependencies (handles include directories and linking automatically)
ament_target_dependencies(aravis_camera_node
  rclcpp
  std_msgs
  image_transport
  cv_bridge
  sensor_msgs
  OpenCV
)

ament_target_dependencies(snapshot_node
  rclcpp
  std_msgs
  geometry_msgs
  cv_bridge
  sensor_msgs
  std_srvs
  tf2_ros
  OpenCV
)

# -----------------------------------------------------------------------------
# Install C++ executables
# -----------------------------------------------------------------------------
install(TARGETS
  aravis_camera_node
  snapshot_node
  DESTINATION lib/${PROJECT_NAME}
)

# -----------------------------------------------------------------------------
# Python hooks (for future Python nodes)
# -----------------------------------------------------------------------------
# ament_python_install_package(${PROJECT_NAME})

# -----------------------------------------------------------------------------
# Install config, rviz, and launch directories
# -----------------------------------------------------------------------------
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

install(
  DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}/
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
)

# -----------------------------------------------------------------------------
# Export package
# -----------------------------------------------------------------------------
ament_package()
