#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <std_srvs/Trigger.h>

bool save_next = false;
std::string image_format = "jpg";  // default

void save_frame_callback(const sensor_msgs::ImageConstPtr& msg) {
    if (!save_next) return;
    save_next = false;

    cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
    //give frame a unique name based on timestamp
    std::string filename = "/home/percro_drone/ros_ws/src/FAST-Calib/calib_data/frame_" + std::to_string(ros::Time::now().toNSec()) + "." + image_format;

    std::vector<int> params;
    if (image_format == "jpg" || image_format == "jpeg") {
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(95);  // good balance between quality and size
    }

    if (cv::imwrite(filename, img)) {
        ROS_INFO("Saved frame to %s", filename.c_str());
    } else {
        ROS_WARN("Failed to save frame to %s", filename.c_str());
    }
}

bool save_request(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res) {
    save_next = true;
    res.success = true;
    res.message = "Next frame will be saved.";
    return true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "snapshot_node");
    ros::NodeHandle nh;

    // get image format param
    nh.param<std::string>("image_format", image_format, "jpg");

    ros::Subscriber sub = nh.subscribe("/aravis_camera_node/left_camera/image", 1, save_frame_callback);
    ros::ServiceServer srv = nh.advertiseService("save_frame", save_request);

    ROS_INFO("Ready to save a frame on service call /save_frame");
    ros::spin();
}
