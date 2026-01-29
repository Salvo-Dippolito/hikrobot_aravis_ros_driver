#ifndef ARAVIS_CAMERA_ROS_DRIVER_COLLECT_CALIB_SAMPLES_H
#define ARAVIS_CAMERA_ROS_DRIVER_COLLECT_CALIB_SAMPLES_H

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <array>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <ros/package.h>

namespace fs = std::filesystem;

/**
 * @brief Interactive calibration image collector with quality checks and region tracking
 * 
 * This class provides an interactive interface for collecting calibration images with 
 * checkerboard patterns. It tracks pose diversity (straight, tilted horizontal/vertical),
 * scale diversity (small/medium/large coverage), and ensures coverage across multiple
 * image regions (4 quadrants + full frame).
 */
class CalibCollector
{
public:
    /**
     * @brief Construct a new CalibCollector object
     * 
     * @param nh ROS node handle for parameters and subscriber
     */
    CalibCollector(ros::NodeHandle& nh);
    
    /**
     * @brief Destructor - closes YAML file storage
     */
    ~CalibCollector();

    /**
     * @brief Main processing loop - handles image capture and display
     */
    void spin();

private:
    // ===== Enumerations =====
    
    /**
     * @brief Pose classification for calibration diversity
     */
    enum PoseClass {
        STRAIGHT,   ///< No significant tilt
        TILT_H,     ///< Horizontal tilt (yaw)
        TILT_V,     ///< Vertical tilt (pitch)
        TILT_HV,    ///< Combined horizontal and vertical tilt
        POSE_COUNT  ///< Total number of pose classes
    };

    /**
     * @brief Scale classification based on checkerboard coverage
     */
    enum ScaleClass {
        SCALE_SMALL,    ///< Far from camera (< 40% coverage)
        SCALE_MEDIUM,   ///< Normal distance (40-60% coverage)
        SCALE_LARGE,    ///< Close to camera (> 60% coverage)
        SCALE_COUNT     ///< Total number of scale classes
    };

    /**
     * @brief Region tracking structure for calibration image distribution
     */
    struct Region
    {
        cv::Rect rect;  ///< Image region bounds

        int pose_counts[POSE_COUNT];          ///< Current count per pose
        int target_pose_counts[POSE_COUNT];   ///< Target count per pose

        int scale_counts[SCALE_COUNT];         ///< Current count per scale
        int target_scale_counts[SCALE_COUNT];  ///< Target count per scale

        /**
         * @brief Check if region has all required pose and scale combinations
         * @return true if all targets met, false otherwise
         */
        bool isComplete() const;
    };

    // ===== ROS Components =====
    ros::NodeHandle nh_;
    ros::Subscriber sub_;

    // ===== Image Data =====
    cv::Mat current_image_;
    bool has_image_;
    int saved_count_;
    std::vector<cv::Point2f> last_corners_;

    // ===== UI State =====
    bool mouse_clicked_;
    bool mouse_hover_;
    bool mouse_pressed_;
    bool last_frame_valid_;
    cv::Rect save_button_rect_;

    // ===== UI Layout Constants =====
    const cv::Size save_button_size_;
    const cv::Point save_button_margin_;
    const int info_panel_width;
    const int line_height;
    const int panel_padding;

    // ===== Calibration Parameters =====
    cv::Size checkerboard_;
    double square_size_;
    double min_coverage_;
    double min_sharpness_;
    bool require_full_cornerset_;
    std::string save_dir_;
    std::string image_topic_;
    int rows, cols;
    int start_region_;

    // ===== File Storage =====
    cv::FileStorage fs_;

    // ===== Current Frame Analysis =====
    double last_sharpness_;
    double last_coverage_;
    double last_tilt_h_;
    double last_tilt_v_;
    int current_region_;
    PoseClass current_pose_;
    ScaleClass current_scale_;

    // ===== Region Tracking =====
    std::vector<Region> regions_;
    int scale_counts[SCALE_COUNT];
    int target_scale_counts[SCALE_COUNT];

    // ===== Static Constants =====
    static constexpr std::array<const char*, 4> pose_names = 
        { "STRAIGHT", "TILT_H", "TILT_V", "TILT_HV" };
    static constexpr std::array<const char*, 3> scale_names = 
        { "SCALE_SMALL", "SCALE_MEDIUM", "SCALE_LARGE" };

    // ===== Callbacks =====
    
    /**
     * @brief ROS image callback
     * @param msg Image message from camera
     */
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

    /**
     * @brief OpenCV mouse event callback
     */
    static void mouseCallback(int event, int x, int y, int flags, void* userdata);

    // ===== Initialization =====
    
    /**
     * @brief Initialize image regions for tracking
     * @param img_w Image width
     * @param img_h Image height
     */
    void initRegions(int img_w, int img_h);

    // ===== Image Analysis =====
    
    /**
     * @brief Compute image sharpness using Laplacian variance
     * @param gray Grayscale image
     * @return Sharpness score
     */
    double computeSharpness(const cv::Mat& gray);

    /**
     * @brief Compute checkerboard coverage in region
     * @param corners Detected checkerboard corners
     * @param region Region to check coverage within
     * @return Coverage ratio [0.0, 1.0]
     */
    double computeCoverageInRegion(const std::vector<cv::Point2f>& corners, 
                                    const cv::Rect& region);

    /**
     * @brief Compute horizontal and vertical tilt from checkerboard sides
     * @param corners Detected checkerboard corners
     * @return Pair of (horizontal_tilt, vertical_tilt)
     */
    std::pair<double, double> computeTiltFromSidesHV(const std::vector<cv::Point2f>& corners);

    // ===== Classification =====
    
    /**
     * @brief Classify pose based on tilt values
     * @param tilt_h Horizontal tilt
     * @param tilt_v Vertical tilt
     * @return Pose classification
     */
    PoseClass classifyPose(double tilt_h, double tilt_v);

    /**
     * @brief Classify scale based on coverage
     * @param coverage Coverage ratio
     * @return Scale classification
     */
    ScaleClass classifyScale(double coverage);

    // ===== UI Rendering =====
    
    /**
     * @brief Process current image and display with UI overlay
     */
    void processAndDisplay();

    /**
     * @brief Compute save button rectangle based on image size
     * @param img_size Image dimensions
     * @return Button rectangle
     */
    cv::Rect computeSaveButtonRect(const cv::Size& img_size) const;

    /**
     * @brief Draw information panel with text lines
     * @param img Image to draw on
     * @param top_left Panel top-left corner
     * @param width Panel width
     * @param lines Text lines to display
     * @param bg_color Background color
     * @param text_color Text color
     */
    void drawInfoPanel(cv::Mat& img, const cv::Point& top_left, int width,
                       const std::vector<std::string>& lines,
                       const cv::Scalar& bg_color,
                       const cv::Scalar& text_color);

    // ===== File Operations =====
    
    /**
     * @brief Generate timestamped filename for saved image
     * @return Full path to image file
     */
    std::string generateTimestampedFilename();

    /**
     * @brief Save current image and update tracking counters
     */
    void saveImage();
};

#endif // ARAVIS_CAMERA_ROS_DRIVER_COLLECT_CALIB_SAMPLES_H
