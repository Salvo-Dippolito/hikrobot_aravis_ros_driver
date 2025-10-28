#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <arv.h>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string>
#include <map>

struct time_stamp {
  int64_t high;
  int64_t low;
};


struct CameraData {
    image_transport::Publisher pub;
    int pixel_format;
    double image_scale;
};

static volatile bool exit_flag = false;
static time_stamp* pointt = nullptr;

void sigint_handler(int) {
    exit_flag = true;
}

enum PixelFormat {
    PF_RGB8,
    PF_BAYER_RG8,
    PF_BAYER_RG12PACKED,
    PF_BAYER_GB12PACKED
};

// std::map<std::string, PixelFormat> string_to_pixelformat = {
//     // {"RGB8", PF_RGB8},
//     {"BayerRG8", PF_BAYER_RG8},
//     {"BayerRG12Packed", PF_BAYER_RG12PACKED},
//     {"BayerGB12Packed", PF_BAYER_GB12PACKED}
// };

void SignalHandler(int signal) {
  if (signal == SIGINT) {  // Catch SIGINT signal triggered by Ctrl + C
    fprintf(stderr, "\nReceived Ctrl+C, exiting...\n");
    exit_flag = true;    // Set exit flag
  }
}

void SetupSignalHandler() {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = SignalHandler; // Set handler function
  sigemptyset(&sigIntHandler.sa_mask);      // Clear signal mask
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);
}


class AravisCameraNode

{
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Publisher pub_;
    
    std::string camera_name_;

    std::string topic_name_, shared_file_;
    int AcquisitionTimeoutUs, AutoExposureTimeLower, AutoExposureTimeUpper, ExposureTime, ExposureAutoMode,
        GainAuto, GammaSelector, TriggerEnable, PixelFormat;
    float Gain, Gamma;
    double ImageScale;
    bool TriggerEnable_bool;
    // int PixelFormat;

    ArvCamera* camera_ = nullptr;
    ArvStream* stream_ = nullptr;
    ArvDevice* device_ = nullptr;
    std::thread capture_thread_;

public:
    // AravisCameraNode()
    //     : nh_("~"), it_(nh_) //this gives the node's topic a namespace
    AravisCameraNode()
        : nh_(), it_(nh_)
    {


        nh_.param<std::string>("CameraName", camera_name_, "");
        nh_.param<std::string>("TopicName", topic_name_, "left_camera/image");
        nh_.param<std::string>("SharedFile", shared_file_, "/tmp/timeshare");
        nh_.param<double>("ImageScale", ImageScale, 1.0);
        nh_.param<int>("AcquisitionTimeoutUs", AcquisitionTimeoutUs, 100000);
        nh_.param<bool>("TriggerEnable", TriggerEnable_bool, true);
        nh_.param<int>("AutoExposureTimeLower", AutoExposureTimeLower, 100);
        nh_.param<int>("AutoExposureTimeUpper", AutoExposureTimeUpper, 20000);
        nh_.param<int>("ExposureTime", ExposureTime, 5000);
        nh_.param<int>("ExposureAutoMode", ExposureAutoMode, 2);
        nh_.param<int>("GainAuto", GainAuto, 2);
        nh_.param<float>("Gain", Gain, 15.0);
        nh_.param<float>("Gamma", Gamma, 0.7);
        nh_.param<int>("GammaSelector", GammaSelector, 1);
        nh_.param<int>("TriggerEnable", TriggerEnable, 1);
        nh_.param<int>("PixelFormat", PixelFormat, 0); // 0: RGB8, 1: BayerRG8, 2: BayerRG12Packed, 3: BayerGB12Packed

        pub_ = it_.advertise(topic_name_, 1);
        ROS_INFO("Publishing images to topic: %s", topic_name_.c_str());

        // // Pixel format
        // if (string_to_pixelformat.find(pixel_format_str_) != string_to_pixelformat.end())
        //     pixel_format_ = string_to_pixelformat[pixel_format_str_];
        // else {
        //     ROS_WARN("Unknown pixel format '%s', defaulting to BayerRG8", pixel_format_str_.c_str());
        //     pixel_format_ = PF_BAYER_RG8;
        // }

        setup_shared_memory();
        SetupSignalHandler();
        start_camera();
    }

    ~AravisCameraNode() {
        stop_camera();
        cleanup_shared_memory();
    }

private:
    void setup_shared_memory() {

        // std::string path_for_time_stamp = "/home/" + std::string(user_name) + "/timeshare";

        int fd = open(shared_file_.c_str(), O_RDWR);
        if (fd == -1) {
            // If the file doesn't exist, create it and size it
            fd = open(shared_file_.c_str(), O_RDWR | O_CREAT, 0666);
            if (fd == -1) {
                ROS_WARN("Failed to open or create shared memory file %s: %s",
                        shared_file_.c_str(), strerror(errno));
                pointt = nullptr;
                return;
            }

            if (ftruncate(fd, sizeof(time_stamp)) == -1) {
                ROS_WARN("Failed to resize shared memory file %s: %s",
                        shared_file_.c_str(), strerror(errno));
                close(fd);
                pointt = nullptr;
                return;
            }
            ROS_INFO("Created new shared memory file: %s", shared_file_.c_str());
        }
        pointt = static_cast<time_stamp*>(mmap(nullptr, sizeof(time_stamp), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));

        if (pointt == MAP_FAILED) {
            ROS_WARN("Failed to mmap shared memory file %s: %s",
                    shared_file_.c_str(), strerror(errno));
            pointt = nullptr;
        } else {
            ROS_INFO("Mapped shared memory file: %s", shared_file_.c_str());
        }

        close(fd);  // safe to close after mmap
    }

    void cleanup_shared_memory() {
        if (pointt && pointt != MAP_FAILED) {
            munmap(pointt, sizeof(time_stamp));
            pointt = nullptr;
        }

        if (!shared_file_.empty()) {
            if (unlink(shared_file_.c_str()) == 0) {
                ROS_INFO("Deleted shared memory file: %s", shared_file_.c_str());
            } else {
                ROS_WARN("Failed to delete shared memory file: %s", shared_file_.c_str());
            }
        }
    
    }

    static void new_buffer_cb(ArvStream *stream, CameraData *data) {
        ArvBuffer *buffer = arv_stream_try_pop_buffer(stream);
        if (!buffer) {
            ROS_WARN("Failed to pop buffer from stream");
            return;
        }

        if (arv_buffer_get_status(buffer) == ARV_BUFFER_STATUS_SUCCESS) {
            const void* raw_data = arv_buffer_get_data(buffer, nullptr);
            int width = arv_buffer_get_image_width(buffer);
            int height = arv_buffer_get_image_height(buffer);
            cv::Mat raw, rgb;

            switch (data->pixel_format) {
                case 0: raw = cv::Mat(height, width, CV_8UC3, const_cast<void*>(raw_data)); rgb = raw; break;
                case 1: raw = cv::Mat(height, width, CV_8UC1, const_cast<void*>(raw_data)); cv::cvtColor(raw, rgb, cv::COLOR_BayerRG2RGB); break;
                case 2: raw = cv::Mat(height, width, CV_16UC1, const_cast<void*>(raw_data)); raw.convertTo(raw, CV_8UC1, 1.0/16.0); cv::cvtColor(raw, rgb, cv::COLOR_BayerRG2RGB); break;
                case 3: raw = cv::Mat(height, width, CV_16UC1, const_cast<void*>(raw_data)); raw.convertTo(raw, CV_8UC1, 1.0/16.0); cv::cvtColor(raw, rgb, cv::COLOR_BayerGB2RGB); break;
            }

            if (data->image_scale != 1.0)
                cv::resize(rgb, rgb, cv::Size(), data->image_scale, data->image_scale);

            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", rgb).toImageMsg();
            msg->header.stamp = ros::Time::now();
            msg->header.frame_id = "camera";
            data->pub.publish(msg);
        }

        arv_stream_push_buffer(stream, buffer); // return buffer to queue
    }

    bool start_camera() {
        camera_ = arv_camera_new(camera_name_.c_str());
        if (!camera_) {
            ROS_ERROR("Could not open camera: %s", camera_name_.c_str());
            return false;
        }

        ArvDevice *device = arv_camera_get_device(camera_);
        if (!device) {
            ROS_ERROR("Could not get device from camera: %s", camera_name_.c_str());
            return false;
        }

        arv_device_set_boolean_feature_value(device, "AcquisitionFrameRateEnable", false );


        // Set pixel format
        switch (PixelFormat) {
            case PF_RGB8: arv_camera_set_pixel_format(camera_, ARV_PIXEL_FORMAT_RGB_8_PACKED); break; 
            case PF_BAYER_RG8: arv_camera_set_pixel_format(camera_, ARV_PIXEL_FORMAT_BAYER_RG_8); break;
            case PF_BAYER_RG12PACKED: arv_camera_set_pixel_format(camera_, ARV_PIXEL_FORMAT_BAYER_RG_12_PACKED); break;
            case PF_BAYER_GB12PACKED: arv_camera_set_pixel_format(camera_, ARV_PIXEL_FORMAT_BAYER_GB_12_PACKED); break;
        }


        // Check if exposure time control is supported
        if (arv_camera_is_exposure_time_available(camera_)) {

            // If auto exposure is supported
            if (arv_camera_is_exposure_auto_available(camera_)) {
                // Disable or enable auto exposure
                if (ExposureAutoMode == 0) {
                    // Manual exposure mode
                    // Set ExposureAuto to "Off"
                    arv_device_set_string_feature_value(device, "ExposureAuto", "Off");
                    arv_camera_set_exposure_time_auto(camera_, ARV_AUTO_OFF);
                    ROS_INFO("Exposure auto disabled (manual mode).");

                    arv_camera_set_exposure_time(camera_, ExposureTime);
                    ROS_INFO("Exposure time set to %d us", ExposureTime);
                }
                else if (ExposureAutoMode == 1) {
                    // Set ExposureAuto to "Once"
                    arv_device_set_string_feature_value(device, "ExposureAuto", "Once");
                    arv_device_set_integer_feature_value(device, "AutoExposureTimeLowerLimit", AutoExposureTimeLower);
                    arv_device_set_integer_feature_value(device, "AutoExposureTimeUpperLimit", AutoExposureTimeUpper);
                    arv_camera_set_exposure_time_auto(camera_, ARV_AUTO_ONCE);
                    ROS_INFO("Exposure auto set to ONCE mode.");
                }
                else if (ExposureAutoMode == 2) {
                    // Set ExposureAuto to "Continuous"
                    arv_device_set_string_feature_value(device, "ExposureAuto", "Continuous");
                    arv_device_set_integer_feature_value(device, "AutoExposureTimeLowerLimit", AutoExposureTimeLower);
                    arv_device_set_integer_feature_value(device, "AutoExposureTimeUpperLimit", AutoExposureTimeUpper);
                    arv_camera_set_exposure_time_auto(camera_, ARV_AUTO_CONTINUOUS);
                    ROS_INFO("Exposure auto set to CONTINUOUS mode.");
                }
            } else {
                ROS_WARN("Exposure auto control not available, using manual exposure.");
                arv_camera_set_exposure_time_auto(camera_, ARV_AUTO_OFF);
                arv_camera_set_exposure_time(camera_, ExposureTime);
            }
        } else {
            ROS_WARN("Exposure time control not available on this camera.");
        }

        // Gain control
        if (arv_camera_is_gain_available(camera_)) {
            if (GainAuto == 0) {  // Manual
                arv_camera_set_gain_auto(camera_, ARV_AUTO_OFF);
                arv_camera_set_gain(camera_, Gain);  // float, e.g., 15.0
                ROS_INFO("Gain set to %.2f (manual)", Gain);
            } else if (GainAuto == 1) { // Once
                arv_camera_set_gain_auto(camera_, ARV_AUTO_ONCE);
                ROS_INFO("Gain auto set to ONCE");
            } else if (GainAuto == 2) { // Continuous
                arv_camera_set_gain_auto(camera_, ARV_AUTO_CONTINUOUS);
                ROS_INFO("Gain auto set to CONTINUOUS");
            }
        } else {
            ROS_WARN("Gain control not available");
        }

        // // Gamma control
        arv_device_set_integer_feature_value(device, "GammaSelector", GammaSelector);
        arv_device_set_float_feature_value(device, "Gamma", Gamma);

        // if (arv_camera_is_gamma_available(camera_)) {
        //     // Gamma selector, if supported
        //     // Some cameras only allow one gamma curve
        //     arv_camera_set_gamma(camera_, Gamma);  // e.g., 0.7
        //     ROS_INFO("Gamma set to %.2f", Gamma);
        // } else {
        //     ROS_WARN("Gamma control not available");
        // }

        guint n_sources = 0;
        const char **sources = arv_camera_get_available_trigger_sources(camera_, &n_sources);

        ROS_INFO("Available trigger sources (%d):", n_sources);
        for (guint i = 0; i < n_sources; ++i) {
            ROS_INFO("  %s", sources[i]);
        }
        g_free(sources);

        guint n_triggers = 0;
        const char **triggers = arv_camera_get_available_triggers(camera_, &n_triggers);

        ROS_INFO("Available triggers (%d):", n_triggers);
        for (guint i = 0; i < n_triggers; ++i) {
            ROS_INFO("  %s", triggers[i]);
        }
        g_free(triggers);
        arv_device_set_string_feature_value(device, "TriggerMode", "On");
        arv_device_set_string_feature_value(device, "TriggerSelector", "FrameStart"); // the trigger event
        arv_device_set_string_feature_value(device, "TriggerActivation", "RisingEdge");
        arv_device_set_string_feature_value(device, "TriggerSource", "Anyway"); 
        // arv_device_set_boolean_feature_value(device, "TriggerEnable", true );
        ROS_INFO("TriggerMode: %s", arv_device_get_string_feature_value(device, "TriggerMode"));
        ROS_INFO("AcquisitionBurstFrameCount: %ld", arv_device_get_integer_feature_value(device, "AcquisitionBurstFrameCount"));
        ROS_INFO("TriggerActivation: %s", arv_device_get_string_feature_value(device, "TriggerActivation"));
        ROS_INFO("TriggerSelector: %s", arv_device_get_string_feature_value(device, "TriggerSelector"));
        ROS_INFO("TriggerSource: %s", arv_device_get_string_feature_value(device, "TriggerSource"));

        const char *current_source = arv_camera_get_trigger_source(camera_);
        if (current_source)
            ROS_INFO("Current trigger source: %s", current_source);
        else
            ROS_WARN("No trigger source currently set.");


        stream_ = arv_camera_create_stream(camera_, nullptr, nullptr);
        if (!stream_) {
            ROS_ERROR("Failed to create Aravis stream");
            g_object_unref(camera_);
            camera_ = nullptr;
            return false;
        }

        size_t payload = arv_camera_get_payload(camera_);

        // Push buffers to the stream
        for (int i = 0; i < 50; ++i) {
            ROS_INFO("Pushing buffer %d to stream", i);
            arv_stream_push_buffer(stream_, arv_buffer_new(payload, nullptr));
        }

        // arv_camera_set_acquisition_mode(camera_, ARV_ACQUISITION_MODE_CONTINUOUS);
        // Connect signal-based callback
        CameraData* cam_data = new CameraData{pub_, PixelFormat, ImageScale};
        g_signal_connect(stream_, "new-buffer", G_CALLBACK(new_buffer_cb), cam_data);
        arv_stream_set_emit_signals(stream_, true);

        arv_camera_set_acquisition_mode(camera_, ARV_ACQUISITION_MODE_CONTINUOUS);

        arv_camera_start_acquisition(camera_);

        // capture_thread_ = std::thread(&AravisCameraNode::capture_loop, this);
        ROS_INFO("Camera acquisition started");
        return true;
    }

    void stop_camera() {
        exit_flag = true;
        // if (capture_thread_.joinable())
        //     capture_thread_.join();

        if (camera_) {
            arv_camera_stop_acquisition(camera_);
            g_object_unref(camera_);
            camera_ = nullptr;
        }

        if (stream_) {
            arv_stream_set_emit_signals(stream_, FALSE);
            g_object_unref(stream_);
            stream_ = nullptr;
        }

        ROS_INFO("Camera stopped");
    }

    void capture_loop() {
        while (ros::ok() && !exit_flag) {
            ArvBuffer* buffer = arv_stream_timeout_pop_buffer(stream_, AcquisitionTimeoutUs);
            if (!buffer) {
                ROS_WARN("Failed to retrieve buffer");
                continue;
            }

            if (arv_buffer_get_status(buffer) == ARV_BUFFER_STATUS_SUCCESS) {
                const void* data = arv_buffer_get_data(buffer, nullptr);
                int width = arv_buffer_get_image_width(buffer);
                int height = arv_buffer_get_image_height(buffer);
                cv::Mat raw, rgb;
                ROS_INFO("Received image: %dx%d", width, height);
                // Convert pixel format to RGB for ROS
                switch (PixelFormat) {
                    case PF_RGB8:
                        raw = cv::Mat(height, width, CV_8UC3, const_cast<void*>(data));
                        rgb = raw;
                        break;
                    case PF_BAYER_RG8:
                        raw = cv::Mat(height, width, CV_8UC1, const_cast<void*>(data));
                        cv::cvtColor(raw, rgb, cv::COLOR_BayerRG2RGB);
                        break;
                    case PF_BAYER_RG12PACKED:
                    case PF_BAYER_GB12PACKED:
                        raw = cv::Mat(height, width, CV_16UC1, const_cast<void*>(data));
                        raw.convertTo(raw, CV_8UC1, 1.0 / 16.0); // scale 12-bit to 8-bit
                        cv::cvtColor(raw, rgb, (PixelFormat == PF_BAYER_RG12PACKED) ? cv::COLOR_BayerRG2RGB : cv::COLOR_BayerGB2RGB);
                        break;
                }

                if (ImageScale != 1.0)
                    cv::resize(rgb, rgb, cv::Size(), ImageScale, ImageScale);

                ros::Time stamp;
                if (TriggerEnable && pointt && pointt != MAP_FAILED && pointt->low != 0)
                    stamp = ros::Time(static_cast<double>(pointt->low) / 1e9);
                else
                    stamp = ros::Time::now();

                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", rgb).toImageMsg();
                msg->header.stamp = stamp;
                msg->header.frame_id = "camera";
                pub_.publish(msg);
            }

            arv_stream_push_buffer(stream_, buffer);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "aravis_camera_node");
    signal(SIGINT, sigint_handler);

    AravisCameraNode node;
    // ros::spin();
    // return 0;
    ros::Rate rate(10);

    while (ros::ok() && !exit_flag) {
        ros::spinOnce();
        rate.sleep();
    }
}
