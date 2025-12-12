// aravis_camera_node.cpp
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
#include <vector>
#include <memory>
#include <cstring>
#include <errno.h>

struct time_stamp {
  int64_t high;
  int64_t low;
};

struct CameraData {
    image_transport::Publisher pub;
    int pixel_format;
    double image_scale;
    bool use_shared_timestamp; 
};

// static volatile bool exit_flag = false;
static time_stamp* pointt = nullptr;

// void sigint_handler(int) {
//     exit_flag = true;
// }

// void SignalHandler(int signal) {
//   if (signal == SIGINT) {
//     fprintf(stderr, "\nReceived Ctrl+C, exiting...\n");
//     exit_flag = true;
//   }
// }

// void SetupSignalHandler() {
//   struct sigaction sigIntHandler;
//   sigIntHandler.sa_handler = SignalHandler;
//   sigemptyset(&sigIntHandler.sa_mask);
//   sigIntHandler.sa_flags = 0;
//   sigaction(SIGINT, &sigIntHandler, NULL);
// }

enum PixelFormat {
    PF_RGB8 = 0,
    PF_BAYER_RG8 = 1,
    PF_BAYER_RG12PACKED = 2,
    PF_BAYER_GB12PACKED = 3
};

// static void unpack_bayer12_packed_to_u16(const uint8_t* src, size_t src_bytes, uint16_t* dst, size_t dst_pixels) {
//     // Standard 12-bit packed format: 3 bytes -> 2 pixels
//     // byte0 = low 8 bits of pix0
//     // byte1 = high 4 bits of pix0 | low 4 bits of pix1 << 4
//     // byte2 = high 8 bits of pix1
//     size_t src_idx = 0;
//     size_t dst_idx = 0;

//     while ((src_idx + 2) < src_bytes && (dst_idx + 1) < dst_pixels) {
//         uint8_t b0 = src[src_idx++];
//         uint8_t b1 = src[src_idx++];
//         uint8_t b2 = src[src_idx++];

//         uint16_t pix0 = (uint16_t)b0 | (uint16_t)(b1 & 0x0F) << 8;
//         uint16_t pix1 = (uint16_t)(b1 >> 4) | (uint16_t)b2 << 4;

//         dst[dst_idx++] = pix0;
//         dst[dst_idx++] = pix1;
//     }

//     // If pixel count is odd, the final pixel might not be set by the loop;
//     // caller should ensure image dimensions are correct and src_bytes matches expectation.
// }



class AravisCameraNode {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Publisher pub_;

    std::string camera_name_;
    std::string topic_name_, shared_file_;
    int AcquisitionTimeoutUs, AutoExposureTimeLower, AutoExposureTimeUpper, ExposureTime, ExposureAutoMode,
        GainAuto, GammaSelector, TriggerEnable, pixel_format_param_;
    float Gain, Gamma;
    double ImageScale;
    bool TriggerEnable_bool;
    bool UseSharedTimestamp;


    ArvCamera* camera_ = nullptr;
    ArvStream* stream_ = nullptr;
    ArvDevice* device_ = nullptr;

    std::unique_ptr<CameraData> cam_data_; // stored to keep lifetime while stream is active

public:
    AravisCameraNode()
        : nh_(), it_(nh_) {

        // parameters (keep your original names)
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
        nh_.param<float>("Gain", Gain, 15.0f);
        nh_.param<float>("Gamma", Gamma, 0.7f);
        nh_.param<int>("GammaSelector", GammaSelector, 1);
        nh_.param<int>("TriggerEnable", TriggerEnable, 1);
        nh_.param<int>("PixelFormat", pixel_format_param_, 0); // 0: RGB8, 1: BayerRG8, 2: BayerRG12Packed, 3: BayerGB12Packed
        nh_.param<bool>("UseSharedTimestamp", UseSharedTimestamp, false);

        pub_ = it_.advertise(topic_name_, 1);
        ROS_INFO("Publishing images to topic: %s", topic_name_.c_str());

        if (UseSharedTimestamp) {setup_shared_memory();} else {ROS_INFO("Shared timestamp disabled, using Ros::Time.now()");}
        // SetupSignalHandler();
        start_camera();
    }

    ~AravisCameraNode() {
        stop_camera();
        if (UseSharedTimestamp) { cleanup_shared_memory();}
    }

private:
    void setup_shared_memory() {
        int fd = open(shared_file_.c_str(), O_RDWR);
        if (fd == -1) {
            // try to create
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

        close(fd);
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
            const void* raw_data = arv_buffer_get_data(buffer, NULL);
            int width = arv_buffer_get_image_width(buffer);
            int height = arv_buffer_get_image_height(buffer);
    
            cv::Mat raw, rgb;
    
            switch (data->pixel_format) {
    
                case PF_RGB8:
                    raw = cv::Mat(height, width, CV_8UC3,
                                  const_cast<void*>(raw_data));
                    rgb = raw;
                    break;
    
                case PF_BAYER_RG8:
                    raw = cv::Mat(height, width, CV_8UC1,
                                  const_cast<void*>(raw_data));
                    cv::cvtColor(raw, rgb, cv::COLOR_BayerRG2RGB);
                    break;
    
                case PF_BAYER_RG12PACKED:
                case PF_BAYER_GB12PACKED: {
                    // Aravis 0.6 delivers 12-bit packed: 2 pixels = 3 bytes
                    // We unpack manually into 16-bit buffer
                    int num_pixels = width * height;
                    std::vector<uint16_t> unpacked(num_pixels);
    
                    const uint8_t* src = static_cast<const uint8_t*>(raw_data);
    
                    for (int i = 0, j = 0; i < num_pixels; i += 2, j += 3) {
                        uint8_t b0 = src[j + 0];
                        uint8_t b1 = src[j + 1];
                        uint8_t b2 = src[j + 2];
    
                        // Pixel 0 = 12 MSBs
                        unpacked[i + 0] = (b0 << 4) | (b2 & 0x0F);
    
                        // Pixel 1 = 12 LSBs
                        unpacked[i + 1] = (b1 << 4) | (b2 >> 4);
                    }
    
                    cv::Mat tmp(height, width, CV_16UC1, unpacked.data());
    
                    // Scale down to 8-bit (12-bit → 8-bit)
                    tmp.convertTo(raw, CV_8UC1, 1.0 / 16.0);
    
                    if (data->pixel_format == PF_BAYER_RG12PACKED)
                        cv::cvtColor(raw, rgb, cv::COLOR_BayerRG2RGB);
                    else
                        cv::cvtColor(raw, rgb, cv::COLOR_BayerGB2RGB);
                    }
                    break;
            }
    
            if (data->image_scale != 1.0)
                cv::resize(rgb, rgb, cv::Size(), data->image_scale, data->image_scale);
    
            sensor_msgs::ImagePtr msg =
                cv_bridge::CvImage(std_msgs::Header(), "rgb8", rgb).toImageMsg();
    
           // Timestamp logic
            if (data->use_shared_timestamp && pointt) {
                msg->header.stamp.fromNSec(static_cast<uint64_t>(pointt->low));
            } else {
                msg->header.stamp = ros::Time::now();
            }

            msg->header.frame_id = "camera";
    
            data->pub.publish(msg);
        }
    
        arv_stream_push_buffer(stream, buffer);
    }
    
    bool start_camera() {
        camera_ = arv_camera_new(camera_name_.c_str());
        if (!camera_) {
            ROS_ERROR("Could not open camera: %s", camera_name_.c_str());
            return false;
        }

        device_ = arv_camera_get_device(camera_);
        if (!device_) {
            ROS_ERROR("Could not get device from camera: %s", camera_name_.c_str());
            if (G_IS_OBJECT(camera_)) g_object_unref(camera_);
            camera_ = nullptr;
            return false;
        }

        // disable framerate control if desired
        arv_device_set_boolean_feature_value(device_, "AcquisitionFrameRateEnable", false );

        // set pixel format on camera (if supported)
        switch (pixel_format_param_) {
            case PF_RGB8:
                arv_camera_set_pixel_format(camera_, ARV_PIXEL_FORMAT_RGB_8_PACKED);
                break;
            case PF_BAYER_RG8:
                arv_camera_set_pixel_format(camera_, ARV_PIXEL_FORMAT_BAYER_RG_8);
                break;
            case PF_BAYER_RG12PACKED:
                arv_camera_set_pixel_format(camera_, ARV_PIXEL_FORMAT_BAYER_RG_12_PACKED);
                break;
            case PF_BAYER_GB12PACKED:
                arv_camera_set_pixel_format(camera_, ARV_PIXEL_FORMAT_BAYER_GB_12_PACKED);
                break;
            default:
                ROS_WARN("Unknown PixelFormat param %d, defaulting to BayerRG8", pixel_format_param_);
                arv_camera_set_pixel_format(camera_, ARV_PIXEL_FORMAT_BAYER_RG_8);
                pixel_format_param_ = PF_BAYER_RG8;
                break;
        }

        // Exposure control
        if (arv_camera_is_exposure_time_available(camera_)) {
            if (arv_camera_is_exposure_auto_available(camera_)) {
                if (ExposureAutoMode == 0) {
                    arv_device_set_string_feature_value(device_, "ExposureAuto", "Off");
                    arv_camera_set_exposure_time_auto(camera_, ARV_AUTO_OFF);
                    arv_camera_set_exposure_time(camera_, ExposureTime);
                    ROS_INFO("Exposure auto disabled; exposure set to %d us", ExposureTime);
                } else if (ExposureAutoMode == 1) {
                    arv_device_set_string_feature_value(device_, "ExposureAuto", "Once");
                    arv_device_set_integer_feature_value(device_, "AutoExposureTimeLowerLimit", AutoExposureTimeLower);
                    arv_device_set_integer_feature_value(device_, "AutoExposureTimeUpperLimit", AutoExposureTimeUpper);
                    arv_camera_set_exposure_time_auto(camera_, ARV_AUTO_ONCE);
                    ROS_INFO("Exposure auto set to ONCE");
                } else {
                    arv_device_set_string_feature_value(device_, "ExposureAuto", "Continuous");
                    arv_device_set_integer_feature_value(device_, "AutoExposureTimeLowerLimit", AutoExposureTimeLower);
                    arv_device_set_integer_feature_value(device_, "AutoExposureTimeUpperLimit", AutoExposureTimeUpper);
                    arv_camera_set_exposure_time_auto(camera_, ARV_AUTO_CONTINUOUS);
                    ROS_INFO("Exposure auto set to CONTINUOUS");
                }
            } else {
                ROS_WARN("Exposure auto not available; using manual exposure");
                arv_camera_set_exposure_time_auto(camera_, ARV_AUTO_OFF);
                arv_camera_set_exposure_time(camera_, ExposureTime);
            }

        } else {
            ROS_WARN("Exposure time control not available on this camera.");
        }

        // Gain control
        if (arv_camera_is_gain_available(camera_)) {
            if (GainAuto == 0) {
                arv_camera_set_gain_auto(camera_, ARV_AUTO_OFF);
                arv_camera_set_gain(camera_, Gain);
                ROS_INFO("Gain set to %.2f (manual)", Gain);
            } else if (GainAuto == 1) {
                arv_camera_set_gain_auto(camera_, ARV_AUTO_ONCE);
                ROS_INFO("Gain auto set to ONCE");
            } else {
                arv_camera_set_gain_auto(camera_, ARV_AUTO_CONTINUOUS);
                ROS_INFO("Gain auto set to CONTINUOUS");
            }
        } else {
            ROS_WARN("Gain control not available");
        }

        // Gamma
        // Some cameras expose gamma as device features; attempt to set if possible
        if (device_) {
            // Try to set selector then gamma value; ignore failure
            arv_device_set_integer_feature_value(device_, "GammaSelector", GammaSelector);
            arv_device_set_float_feature_value(device_, "Gamma", Gamma);
        }

        // Trigger capabilities info
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

        // Set trigger configuration (best-effort; ignore failures)
        arv_device_set_string_feature_value(device_, "TriggerMode", "On");
        arv_device_set_string_feature_value(device_, "TriggerSelector", "FrameStart"); // the trigger event
        arv_device_set_string_feature_value(device_, "TriggerActivation", "RisingEdge");
        arv_device_set_string_feature_value(device_, "TriggerSource", "Anyway");

        const char *current_source = arv_camera_get_trigger_source(camera_);
        if (current_source)
            ROS_INFO("Current trigger source: %s", current_source);
        else
            ROS_WARN("No trigger source currently set.");

        // Create stream
        stream_ = arv_camera_create_stream(camera_, nullptr, nullptr);
        if (!stream_) {
            ROS_ERROR("Failed to create Aravis stream");
            if (G_IS_OBJECT(camera_)) g_object_unref(camera_);
            camera_ = nullptr;
            return false;
        }

        size_t payload = arv_camera_get_payload(camera_);

        // Push buffers (same count as before)
        for (int i = 0; i < 50; ++i) {
            arv_stream_push_buffer(stream_, arv_buffer_new(payload, nullptr));
        }

        // Prepare camera data and attach to stream
        cam_data_.reset(new CameraData{pub_, pixel_format_param_, ImageScale, UseSharedTimestamp});
        // Connect callback
        g_signal_connect(stream_, "new-buffer", G_CALLBACK(new_buffer_cb), cam_data_.get());
        arv_stream_set_emit_signals(stream_, true);

        arv_camera_set_acquisition_mode(camera_, ARV_ACQUISITION_MODE_CONTINUOUS);
        arv_camera_start_acquisition(camera_);

        ROS_INFO("Camera acquisition started");
        return true;
    }

    void stop_camera() {
        // exit_flag = true;

        if (camera_) {
            // Stop acquisition first
            avr_stop_acquisition_safe();
        }

        // Cleanup cam_data_ after stopping emission & unref of stream to ensure callback won't be called
        if (stream_) {
            // Stop stream signals and unref
            arv_stream_set_emit_signals(stream_, FALSE);

            if (G_IS_OBJECT(stream_)) {
                g_object_unref(stream_);
            }
            stream_ = nullptr;
        }

        // free CameraData (unique_ptr destructor)
        cam_data_.reset();

        if (camera_) {
            if (G_IS_OBJECT(camera_)) {
                g_object_unref(camera_);
            }
            camera_ = nullptr;
        }

        ROS_INFO("Camera stopped");
    }

    // small helper to stop acquisition with safety checks, placed here to keep stop_camera short
    void avr_stop_acquisition_safe() {
        if (camera_) {
            // Attempt to stop acquisition; ignore errors
            arv_camera_stop_acquisition(camera_);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "aravis_camera_node");
    // signal(SIGINT, sigint_handler);

    {
        // Node lives inside this scope
        AravisCameraNode node;
        ros::Rate rate(10);

        while (ros::ok()) {
            ros::spinOnce();
            rate.sleep();
        }
    } // <--- Destructor is guaranteed here

    return 0;
}
