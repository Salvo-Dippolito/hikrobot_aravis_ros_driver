#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <string>
#include <vector>
#include <chrono>

class SnapshotNode : public rclcpp::Node
{
public:
    SnapshotNode()
        : Node("snapshot_node"), save_next_(false)
    {
        // Declare and get the image format parameter
        this->declare_parameter<std::string>("image_format", "jpg");
        this->get_parameter("image_format", image_format_);

        // Subscription to image topic
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/left_camera/image", 10,
            std::bind(&SnapshotNode::imageCallback, this, std::placeholders::_1));

        // Service to trigger saving
        save_service_ = this->create_service<std_srvs::srv::Trigger>(
            "save_frame",
            std::bind(&SnapshotNode::handleSaveRequest, this,
                      std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Snapshot node started. Waiting for /save_frame service call...");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        if (!save_next_)
            return;

        save_next_ = false;

        cv::Mat img;
        try
        {
            img = cv_bridge::toCvShare(msg, "bgr8")->image;
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Create a unique filename based on timestamp
        auto now = this->get_clock()->now();
        std::stringstream filename;
        filename << "/home/percro_drone/ros_ws/src/FAST-Calib/calib_data/frame_"
                 << now.nanoseconds() << "." << image_format_;

        std::vector<int> params;
        if (image_format_ == "jpg" || image_format_ == "jpeg")
        {
            params.push_back(cv::IMWRITE_JPEG_QUALITY);
            params.push_back(95);
        }

        if (cv::imwrite(filename.str(), img))
        {
            RCLCPP_INFO(this->get_logger(), "Saved frame to %s", filename.str().c_str());
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Failed to save frame to %s", filename.str().c_str());
        }
    }

    void handleSaveRequest(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        save_next_ = true;
        response->success = true;
        response->message = "Next frame will be saved.";
        RCLCPP_INFO(this->get_logger(), "Received save_frame request — will save next frame.");
    }

    // Members
    bool save_next_;
    std::string image_format_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SnapshotNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}





//==================================================================================================

// #include <ros/ros.h>
// #include <sensor_msgs/Image.h>
// #include <cv_bridge/cv_bridge.h>
// #include <opencv2/opencv.hpp>
// #include <std_srvs/Trigger.h>

// bool save_next = false;
// std::string image_format = "jpg";  // default

// void save_frame_callback(const sensor_msgs::ImageConstPtr& msg) {
//     if (!save_next) return;
//     save_next = false;

//     cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
//     //give frame a unique name based on timestamp
//     std::string filename = "/home/percro_drone/ros_ws/src/FAST-Calib/calib_data/frame_" + std::to_string(ros::Time::now().toNSec()) + "." + image_format;

//     std::vector<int> params;
//     if (image_format == "jpg" || image_format == "jpeg") {
//         params.push_back(cv::IMWRITE_JPEG_QUALITY);
//         params.push_back(95);  // good balance between quality and size
//     }

//     if (cv::imwrite(filename, img)) {
//         ROS_INFO("Saved frame to %s", filename.c_str());
//     } else {
//         ROS_WARN("Failed to save frame to %s", filename.c_str());
//     }
// }

// bool save_request(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res) {
//     save_next = true;
//     res.success = true;
//     res.message = "Next frame will be saved.";
//     return true;
// }

// int main(int argc, char** argv) {
//     ros::init(argc, argv, "snapshot_node");
//     ros::NodeHandle nh;

//     // get image format param
//     nh.param<std::string>("image_format", image_format, "jpg");

//     ros::Subscriber sub = nh.subscribe("/left_camera/image", 1, save_frame_callback);
//     ros::ServiceServer srv = nh.advertiseService("save_frame", save_request);

//     ROS_INFO("Ready to save a frame on service call /save_frame");
//     ros::spin();
// }
