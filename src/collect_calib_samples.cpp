#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <ros/package.h>

#include <yaml-cpp/yaml.h>


namespace fs = std::filesystem;

class CalibCollector
{
public:
    CalibCollector(ros::NodeHandle& nh)
        : nh_(nh), has_image_(false), saved_count_(0), mouse_clicked_(false)
    {
        // --- Load parameters ---

        std::string package_path = ros::package::getPath("aravis_camera_ros_driver");
        std::string default_save_dir_ = package_path + "/calib_images";

        nh_.param<int>("Rows", rows, 5);
        nh_.param<int>("Cols", cols, 7);
        nh_.param<double>("SquareSize", square_size_, 0.046);
        nh_.param<double>("MinCoverage", min_coverage_, 0.2);
        nh_.param<double>("MinSharpness", min_sharpness_, 150.0);
        nh_.param<bool>("RequireFullCornerset", require_full_cornerset_, true);
        nh_.param<std::string>("SaveDir", save_dir_, default_save_dir_);
        nh_.param<std::string>("ImageTopic", image_topic_, std::string("/left_camera/image"));
        nh_.param<int>("StartRegion", start_region_, 0);
        nh_.param<bool>("ReadFromDataset", read_from_dataset_, false);

        checkerboard_ = cv::Size(cols, rows);

        // --- Ensure save directory exists ---
        if (!fs::exists(save_dir_))
            fs::create_directories(save_dir_);

        // --- Subscriber ---
        sub_ = nh_.subscribe(image_topic_, 1, &CalibCollector::imageCallback, this);

        // --- OpenCV window ---
        cv::namedWindow("calibration_selector", cv::WINDOW_NORMAL);
        cv::setMouseCallback("calibration_selector", &CalibCollector::mouseCallback, this);



        ROS_INFO("CalibCollector initialized: %dx%d checkerboard, square=%.3f",
                 cols, rows, square_size_);

        std::string yaml_path = save_dir_ + "/corners.yaml";

        // --- Check if YAML file exists and load ---
        if (fs::exists(yaml_path)) {
            // Load the YAML file
            YAML::Node fs_ = YAML::LoadFile(yaml_path);
            if (fs_["images"]) {
                // Get the last frame id from the last entry
                const auto& images = fs_["images"];
                last_frame_id_ = images.size();
                ROS_INFO("Last frame ID from YAML: %d", last_frame_id_);
            }
        }else{
            fs_.open(yaml_path, cv::FileStorage::WRITE);

            fs_ << "board" << "{";
            fs_ << "rows" << rows;
            fs_ << "cols" << cols;
            fs_ << "square_size" << square_size_;
            fs_ << "}";
            
            fs_ << "images" << "[";
        }

    }

    ~CalibCollector(){
        fs_ << "]"; // close images list
        fs_.release();
    }


    void spin()
    {
        ros::Rate rate(30);
        while (ros::ok())
        {
            ros::spinOnce();
            if (!has_image_)
            {
                rate.sleep();
                continue;
            }

            if (regions_.empty()) initRegions(current_image_.cols, current_image_.rows);
            if (!read_from_dataset_) {
                processAndDisplay();
                int key = cv::waitKey(1);
                if (key == 'q') break;

                if ((key == ' ' || mouse_clicked_) && last_frame_valid_)
                {
                    saveImage();
                }
                mouse_clicked_ = false;
                rate.sleep();
            }else{
                processDataset();
            }
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;

    cv::Mat current_image_;
    bool has_image_;
    int saved_count_;
    bool mouse_clicked_;
    bool mouse_hover_ = false;
    bool mouse_pressed_ = false;

    bool last_frame_valid_;
    cv::Rect save_button_rect_;
    // --- UI layout constants ---
    const cv::Size save_button_size_   = cv::Size(220, 50);
    const cv::Point save_button_margin_ = cv::Point(20, 20);

    const int info_panel_width  = 380;
    const int line_height       = 22;
    const int panel_padding     = 10;


    cv::Size checkerboard_;


    double square_size_;
    double min_coverage_;
    double min_sharpness_;
    bool require_full_cornerset_;
    std::string save_dir_;
    std::string image_topic_;
    int rows,cols;
    std::vector<cv::Point2f> last_corners_;



    cv::FileStorage fs_;
    double last_sharpness_ = 0.0;
    double last_coverage_  = 0.0;
    double last_tilt_h_ = 0.0;
    double last_tilt_v_ = 0.0;

    int start_region_ = 0;

    std::string yaml_path;
    int last_frame_id_ = -1;

    bool dataset_initialized_ = false;
    std::vector<fs::path> dataset_images_;
    size_t dataset_index_ = 0;
    bool read_from_dataset_ ;


    // Region tracking
    enum PoseClass {
        STRAIGHT,
        TILT_H,      // yaw
        TILT_V,      // pitch
        TILT_HV,     // combined
        POSE_COUNT
    };


    PoseClass classifyPose(double tilt_h, double tilt_v)
    {
        const double tilt_thresh = 0.05;

        bool h = tilt_h > tilt_thresh;
        bool v = tilt_v > tilt_thresh;

        if (h && v) return TILT_HV;
        if (h)      return TILT_H;
        if (v)      return TILT_V;

        return STRAIGHT;
    }
    inline static constexpr std::array<const char*, 4> pose_names = { "STRAIGHT", "TILT_H", "TILT_V", "TILT_HV" };

    enum ScaleClass {
        SCALE_SMALL,
        SCALE_MEDIUM,
        SCALE_LARGE,
        SCALE_COUNT
    };
    inline static constexpr std::array<const char*, 3> scale_names = { "SCALE_SMALL", "SCALE_MEDIUM", "SCALE_LARGE" };

    ScaleClass current_scale_ = SCALE_MEDIUM;

    int scale_counts[SCALE_COUNT] = {0};
    int target_scale_counts[SCALE_COUNT];

    ScaleClass classifyScale(double coverage)
    {   
        double scale_thresh_low  = 0.4;
        double scale_thresh_high = 0.6;     

        if (current_region_==4) // whole frame
        {
            scale_thresh_low  = scale_thresh_low/2;
            scale_thresh_high = scale_thresh_high/2;  
        }
        if (coverage < scale_thresh_low) return SCALE_SMALL;
        if (coverage > scale_thresh_high) return SCALE_LARGE;
        return SCALE_MEDIUM;
    }

    struct Region
    {
        cv::Rect rect;

        int pose_counts[POSE_COUNT] = {0};
        int target_pose_counts[POSE_COUNT] = {0};

        int scale_counts[SCALE_COUNT] = {0};
        int target_scale_counts[SCALE_COUNT] = {0};

        bool isComplete() const
        {
            for (int i = 0; i < POSE_COUNT; ++i)
                if (pose_counts[i] < target_pose_counts[i])
                    return false;

            for (int i = 0; i < SCALE_COUNT; ++i)
                if (scale_counts[i] < target_scale_counts[i])
                    return false;

            return true;
        }
    };

    std::vector<Region> regions_;

    void initRegions(int img_w, int img_h)
    {
        regions_.clear();

        int w2 = img_w / 2;
        int h2 = img_h / 2;

        auto make_region = [&](cv::Rect r, bool full = false)
        {
            Region reg;
            reg.rect = r;

            // scale counts increase only when pose is straight and pose countes increase only when scale is medium

            // Orientation targets, all at medium scale
            reg.target_pose_counts[STRAIGHT] = full ? 4 : 2;
            reg.target_pose_counts[TILT_H]   = full ? 4 : 2;
            reg.target_pose_counts[TILT_V]   = full ? 4 : 2;
            reg.target_pose_counts[TILT_HV]  = full ? 4 : 4;


            // Scale targets, all at straight orientation
            reg.target_scale_counts[SCALE_SMALL]  = full ? 3 : 2 ;
            reg.target_scale_counts[SCALE_MEDIUM] = full ? 3 : 2;
            reg.target_scale_counts[SCALE_LARGE]  = full ? 3 : 2;

            return reg;
        };

        // 4 quadrants
        regions_.push_back(make_region(cv::Rect(0,   0,   w2, h2)));
        regions_.push_back(make_region(cv::Rect(w2,  0,   w2, h2)));
        regions_.push_back(make_region(cv::Rect(0,   h2,  w2, h2)));
        regions_.push_back(make_region(cv::Rect(w2,  h2,  w2, h2)));

        // 5th region: whole frame
        Region full = make_region(cv::Rect(0, 0, img_w, img_h), true);

        // // Typically fewer straight shots needed globally
        // full.target_pose_counts[STRAIGHT] = 4;
        // full.target_pose_counts[TILT_H]   = 3;
        // full.target_pose_counts[TILT_V]   = 3;
        // full.target_pose_counts[TILT_HV]  = 4;


        regions_.push_back(full);

        // ---- Apply start region ----
        start_region_ = std::clamp(start_region_, 0, (int)regions_.size() - 1);

        for (int i = 0; i < start_region_; ++i)
        {
            // Force regions before start_region_ as complete
            for (int p = 0; p < POSE_COUNT; ++p)
                regions_[i].pose_counts[p] = regions_[i].target_pose_counts[p];

            for (int s = 0; s < SCALE_COUNT; ++s)
                regions_[i].scale_counts[s] = regions_[i].target_scale_counts[s];
        }

        ROS_INFO("Starting calibration from region %d", start_region_);
    }

    int current_region_ = -1;
    PoseClass current_pose_ = STRAIGHT;
    
    // ===== ROS Callback =====
    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        try
        {
            auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8"); // converting from rgb8 to bgr8
            current_image_ = cv_ptr->image.clone();
            cv::flip(current_image_, current_image_, +1);  // horizontal mirror

            has_image_ = true;
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    // ===== Mouse Callback =====
    static void mouseCallback(int event, int x, int y, int, void* userdata)
    {
        //opencv's mouse callback keeps track of mouse position x,y and button events
        CalibCollector* self = static_cast<CalibCollector*>(userdata);

        cv::Point pt(x, y);
        self->mouse_hover_ = self->save_button_rect_.contains(pt);

        switch(event)
        {
            case cv::EVENT_LBUTTONDOWN:
                if (self->mouse_hover_)
                    self->mouse_pressed_ = true;
                break;
            case cv::EVENT_LBUTTONUP:
                if (self->mouse_pressed_ && self->mouse_hover_)
                    self->mouse_clicked_ = true; // trigger save
                self->mouse_pressed_ = false;
                break;
        }
    }

    // ===== Utilities =====

    cv::Rect computeSaveButtonRect(const cv::Size& img_size) const
    {
        return cv::Rect(
            img_size.width  - save_button_size_.width  - save_button_margin_.x,
            img_size.height - save_button_size_.height - save_button_margin_.y,
            save_button_size_.width,
            save_button_size_.height
        );
    }

    void drawInfoPanel(cv::Mat& img, const cv::Point& top_left, int width, const std::vector<std::string>& lines,
        const cv::Scalar& bg_color,
        const cv::Scalar& text_color)
    {
        int height = lines.size() * 22 + 2 * 10;

        cv::Rect panel_rect(top_left.x, top_left.y, width, height);

        // Background
        cv::rectangle(img, panel_rect, bg_color, cv::FILLED);

        // Text
        int y = top_left.y + 25;
        for (const auto& line : lines)
        {
            cv::putText(
                img,
                line,
                cv::Point(top_left.x + 10, y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2
            );
            y += 22;
        }
    }

    // Computes horizontal and vertical tilt based on opposing sides of the checkerboard
    std::pair<double, double> computeTiltFromSidesHV(const std::vector<cv::Point2f>& corners)
    {
        if (corners.size() != checkerboard_.area())
            return {0.0, 0.0}; // fallback if corners are incomplete

        int w = checkerboard_.width;
        int h = checkerboard_.height;

        // --- Horizontal tilt: top vs bottom row ---
        cv::Point2f top_left     = corners[0];
        cv::Point2f top_right    = corners[w - 1];
        cv::Point2f bottom_left  = corners[w*(h-1)];
        cv::Point2f bottom_right = corners[w*h - 1];

        double width_top    = cv::norm(top_right - top_left);
        double width_bottom = cv::norm(bottom_right - bottom_left);

        double h_tilt = 1.0 - std::min(width_top, width_bottom) / std::max(width_top, width_bottom);

        // --- Vertical tilt: left vs right column ---
        double height_left  = cv::norm(bottom_left - top_left);
        double height_right = cv::norm(bottom_right - top_right);

        double v_tilt = 1.0 - std::min(height_left, height_right) / std::max(height_left, height_right);

        return {h_tilt, v_tilt};
    }


    double computeSharpness(const cv::Mat& gray)
    {
        cv::Mat lap;
        cv::Laplacian(gray, lap, CV_64F);
        cv::Scalar mu, sigma;
        cv::meanStdDev(lap, mu, sigma);
        return sigma[0] * sigma[0];
    }

    double computeCoverageInRegion(const std::vector<cv::Point2f>& corners, const cv::Rect& region)
    {

        if (corners.size() != checkerboard_.area())
        {
            ROS_INFO("computeCoverageInRegion: incomplete board (corners=%zu, expected=%d)", 
                      corners.size(), checkerboard_.area());
            return 0.0; // incomplete board, TODO: handle partial coverage?
        }

        int cols = checkerboard_.width;
        int rows = checkerboard_.height;

        ROS_INFO("computeCoverageInRegion: board=%dx%d, region=[%d,%d,%dx%d]",
                  cols, rows, region.x, region.y, region.width, region.height);

        // Grab the four extreme corners from detected corners array
        const cv::Point2f& top_left     = corners[0];
        const cv::Point2f& top_right    = corners[cols - 1];
        const cv::Point2f& bottom_left  = corners[cols * (rows - 1)];
        const cv::Point2f& bottom_right = corners[cols * rows - 1];

        ROS_INFO("  top_left=(%.2f,%.2f), top_right=(%.2f,%.2f)",
                  top_left.x, top_left.y, top_right.x, top_right.y);
        ROS_INFO("  bottom_left=(%.2f,%.2f), bottom_right=(%.2f,%.2f)",
                  bottom_left.x, bottom_left.y, bottom_right.x, bottom_right.y);

        std::vector<cv::Point2f> polygon = { top_left, top_right, bottom_right, bottom_left };

        // Compute checkerboard polygon area with shoelace formula
        double area = 0.0;
        size_t n = polygon.size();

        for (size_t i = 0; i < n; ++i)
        {
            const cv::Point2f& p1 = polygon[i];
            const cv::Point2f& p2 = polygon[(i+1) % n];
            area += (p1.x * p2.y) - (p2.x * p1.y);
        }
        area = std::abs(area) / 2.0;

        double region_area = region.width * region.height;
        double coverage = std::min(1.0, area / region_area);

        ROS_INFO("  checkerboard_area=%.2f, region_area=%.2f, coverage=%.4f",
                  area, region_area, coverage);

        return coverage;

    }

    std::string generateTimestampedFilename()
    {
        // Increment the frame_id for new images
        last_frame_id_++;

        std::ostringstream ss;
        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);
        ss << save_dir_ << "/calib_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_" << last_frame_id_<< ".png";
        return ss.str();
    }

    void saveImage()
    {
        if (last_corners_.empty() || current_region_ < 0)
            return;
        
        // Generate a unique filename based on the last frame id
        std::string filename = generateTimestampedFilename();
        cv::imwrite(filename, current_image_);
        saved_count_++;

        Region& reg = regions_[current_region_];
        
        if (current_pose_ == STRAIGHT &&
            reg.scale_counts[current_scale_] < reg.target_scale_counts[current_scale_])
        {
            reg.scale_counts[current_scale_]++;
        }

        if (current_scale_ == SCALE_MEDIUM &&
            reg.pose_counts[current_pose_] < reg.target_pose_counts[current_pose_])
        {
            reg.pose_counts[current_pose_]++;
        }

        ROS_INFO("Saved image #%d: %s", saved_count_, filename.c_str());

        if (regions_[current_region_].isComplete())
            ROS_INFO("Region %d target reached! Move to another region.", current_region_);


        fs_ << "{";
        fs_ << "filename" << fs::path(filename).filename().string();
        fs_ << "image_size" << "[" << current_image_.cols << current_image_.rows << "]";
        fs_ << "sharpness" << last_sharpness_;
        fs_ << "coverage" << last_coverage_;
        fs_ << "region_id" << current_region_;
        fs_ << "pose_class" << pose_names[current_pose_];
        fs_ << "scale_class" << current_scale_;
        fs_ << "tilt_h" << last_tilt_h_;
        fs_ << "tilt_v" << last_tilt_v_;

        fs_ << "corners_count" << (int)last_corners_.size();

        
        fs_ << "corners" << "[";
        for (const auto& p : last_corners_)
            fs_ << "[" << p.x << p.y << "]";
        fs_ << "]";
        
        fs_ << "}";
            
    }

    void processAndDisplay()
    {
        cv::Mat img = current_image_.clone();
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        save_button_rect_ = computeSaveButtonRect(img.size());

        cv::Point info_tl(
            img.cols - info_panel_width - 20,
            20
        );


        std::vector<cv::Point2f> corners;
        bool found = false;

        try
        {
            found = cv::findChessboardCornersSB(
                gray,
                checkerboard_,
                corners,
                cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE
            );
        }
        catch (const cv::Exception& e)
        {
            ROS_WARN_THROTTLE(1.0,
                "findChessboardCornersSB failed (caught): %s", e.what());
            found = false;
        }
        last_frame_valid_ = false;

        double sharpness = 0.0, coverage = 0.0;
        bool full = false;

        if (found)
        {
            // Filter out far-away corners
            cv::Rect bbox = cv::boundingRect(corners);
            if (bbox.width < 20 || bbox.height < 20 || bbox.width > img.cols || bbox.height > img.rows)
            {
                found = false; // reject if bounding box too small or too large
            }
        }

        if (found)
        {
            cv::Point2f center(0.f,0.f);
            for (const auto& c : corners) center += c;
            center *= (1.0f / corners.size());

            current_region_ = -1;

            for (size_t i = 0; i < regions_.size(); ++i) {
                if (regions_[i].isComplete()) continue; // skip completed regions
                if (regions_[i].rect.contains(center)) {
                    current_region_ = (int)i;
                    break;  // stop at the first matching region
                }
            }



            if (current_region_ >= 0)
            {

                ROS_INFO("Current region: %d", current_region_);
            // print region dimension data
                ROS_INFO("Region dimensions: x=%d, y=%d, width=%d, height=%d",
                        regions_[current_region_].rect.x,
                        regions_[current_region_].rect.y,
                        regions_[current_region_].rect.width,
                        regions_[current_region_].rect.height);
                cv::drawChessboardCorners(img, checkerboard_, corners, found);

                sharpness = computeSharpness(gray);
                coverage = computeCoverageInRegion(corners, regions_[current_region_].rect);
                full = (corners.size() == checkerboard_.area());
                cv::Rect board_bbox = cv::boundingRect(corners);


                auto [tilt_h, tilt_v] = computeTiltFromSidesHV(corners);
                last_tilt_h_ = tilt_h;
                last_tilt_v_ = tilt_v;
                current_pose_ = classifyPose(tilt_h, tilt_v);
                current_scale_ = classifyScale(coverage);

                Region& reg = regions_[current_region_];
                
                if(current_region_==4){
                    min_coverage_ = 0.1; // for full frame region, lower coverage requirement, otherwise image gets too blurry if calib target too close
                }
                bool sharp_ok = sharpness >= min_sharpness_;
                bool cover_ok = coverage >= min_coverage_;
                bool full_ok  = (!require_full_cornerset_ || full);
                bool pose_needed  = reg.pose_counts[current_pose_] < reg.target_pose_counts[current_pose_]; 
                bool scale_needed = reg.scale_counts[current_scale_] < reg.target_scale_counts[current_scale_];
                bool board_fully_inside_region = regions_[current_region_].rect.contains(board_bbox.tl()) &&
                                                regions_[current_region_].rect.contains(board_bbox.br());

                last_frame_valid_ = (sharp_ok && cover_ok && full_ok && board_fully_inside_region) && (pose_needed || scale_needed);

                last_corners_ = corners;
                last_sharpness_ = sharpness;
                last_coverage_ = coverage;

                // --- Draw semi-transparent box for top-left text ---
                cv::rectangle(img, cv::Point(0,0), cv::Point(250, 120), cv::Scalar(50,50,50,100), cv::FILLED);

                // --- Top-left parameters ---
                int y_top = 25;
                auto drawParam = [&](const std::string& label, double value, double threshold)
                {
                    cv::Scalar color = (value >= threshold) ? cv::Scalar(0,200,0) : cv::Scalar(0,0,255);
                    std::ostringstream ss;
                    ss << label << ": " << std::fixed << std::setprecision(2) << value;
                    cv::putText(img, ss.str(), cv::Point(10, y_top), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
                    y_top += 25;
                };
                drawParam("Sharpness", sharpness, min_sharpness_);
                drawParam("Coverage", coverage, min_coverage_);
                drawParam("Full board", full ? 1.0 : 0.0, 1.0);
                drawParam("Pose needed", pose_needed ? 1.0 : 0.0, 1.0);

                // --- Bottom line: pose + scale + missing ---
                std::vector<std::string> lines;

                // --- Current pose ---
                lines.push_back(
                    std::string("Current pose: ") + pose_names[current_pose_]
                );

                // --- Current scale ---    
                lines.push_back(
                    std::string("Current scale: ") + scale_names[current_scale_]
                );


                lines.push_back(""); // spacer

                // --- Scale progress ---
                lines.push_back("Scale coverage:");
                lines.push_back(
                    std::string("  close: ") +
                    std::to_string(reg.scale_counts[SCALE_LARGE]) + "/" +
                    std::to_string(reg.target_scale_counts[SCALE_LARGE])
                );
                lines.push_back(
                    std::string("  normal: ") +
                    std::to_string(reg.scale_counts[SCALE_MEDIUM]) + "/" +
                    std::to_string(reg.target_scale_counts[SCALE_MEDIUM])
                );
                lines.push_back(
                    std::string("  far: ") +
                    std::to_string(reg.scale_counts[SCALE_SMALL]) + "/" +
                    std::to_string(reg.target_scale_counts[SCALE_SMALL])
                );

                lines.push_back(""); // spacer

                // --- Orientation progress ---
                lines.push_back("Orientation:");
                lines.push_back(
                    std::string("  straight: ") +
                    std::to_string(reg.pose_counts[STRAIGHT]) + "/" +
                    std::to_string(reg.target_pose_counts[STRAIGHT])
                );
                lines.push_back(
                    std::string("  tilt_v: ") +
                    std::to_string(reg.pose_counts[TILT_V]) + "/" +
                    std::to_string(reg.target_pose_counts[TILT_V])
                );
                lines.push_back(
                    std::string("  tilt_h: ") +
                    std::to_string(reg.pose_counts[TILT_H]) + "/" +
                    std::to_string(reg.target_pose_counts[TILT_H])
                );
                lines.push_back(
                    std::string("  tilt_hv: ") +
                    std::to_string(reg.pose_counts[TILT_HV]) + "/" +
                    std::to_string(reg.target_pose_counts[TILT_HV])
                );

                drawInfoPanel(
                    img,
                    info_tl,
                    info_panel_width,
                    lines,
                    cv::Scalar(40, 40, 120),   // dark blue background
                    cv::Scalar(255, 200, 80)   // readable light-blue/yellow text
                );
            }
        }
        else
        {
            last_corners_.clear();
            last_frame_valid_ = false;
            cv::putText(img, "Checkerboard NOT found", cv::Point(20,40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
            // goto draw_ui;
        }

        // draw_ui:

        // Determine next region needing images
        int next_region = -1;

        for (size_t i = 0; i < regions_.size(); ++i)
        {
            if (!regions_[i].isComplete())
            {
                next_region = (int)i;
                break;
            }
        }

        // Highlight next region in yellow (if any)
        if (next_region >= 0)
        {
            cv::rectangle(img, regions_[next_region].rect, cv::Scalar(0,255,255), 3);

        }

        // --- Move button to bottom-right ---
        // cv::Rect button_rect(img.cols - 240, img.rows - 70, 220, 50);
        cv::Scalar btn_color = cv::Scalar(40,40,40);
        if (mouse_hover_) btn_color = cv::Scalar(80,80,80);
        if (mouse_pressed_) btn_color = cv::Scalar(0,150,0);
        cv::rectangle(img, save_button_rect_, btn_color, cv::FILLED);
        cv::putText(img, "SAVE IMAGE", save_button_rect_.tl() + cv::Point(10,35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);

        // Saved counter
        cv::putText(img, "Saved: " + std::to_string(saved_count_), cv::Point(20,180),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);

        cv::imshow("calibration_selector", img);
    }

    int assignRegionByBoardContainment(
        const std::vector<cv::Point2f>& corners) const
    {
        if (corners.size() != checkerboard_.area())
            return -1;

        const int w = checkerboard_.width;
        const int h = checkerboard_.height;

        const cv::Point2f& tl = corners[0];
        const cv::Point2f& tr = corners[w - 1];
        const cv::Point2f& bl = corners[w * (h - 1)];
        const cv::Point2f& br = corners[w * h - 1];

        int best_region = -1;
        int best_area   = std::numeric_limits<int>::max();

        for (size_t i = 0; i < regions_.size(); ++i)
        {
            const cv::Rect& r = regions_[i].rect;

            if (r.contains(tl) &&
                r.contains(tr) &&
                r.contains(bl) &&
                r.contains(br))
            {
                int area = r.width * r.height;
                if (area < best_area)
                {
                    best_area   = area;
                    best_region = static_cast<int>(i);
                }
            }
        }

        return best_region;
    }

    void processDataset()
    {
        if (!dataset_initialized_)
        {
            dataset_images_.clear();

            for (const auto& entry : fs::directory_iterator(save_dir_))
            {
                if (!entry.is_regular_file())
                    continue;

                const auto ext = entry.path().extension().string();
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
                    dataset_images_.push_back(entry.path());
            }

            std::sort(dataset_images_.begin(), dataset_images_.end());

            if (dataset_images_.empty())
            {
                ROS_ERROR("ReadFromDataset enabled, but no images found in %s",
                        save_dir_.c_str());
                ros::shutdown();
                return;
            }

            ROS_INFO("Processing %zu images from dataset", dataset_images_.size());
            dataset_initialized_ = true;
            dataset_index_ = 0;
        }

        if (dataset_index_ >= dataset_images_.size())
        {
            ROS_INFO("Dataset processing complete");
            ros::shutdown();
            return;
        }

        // ---- Load image ----
        const fs::path& img_path = dataset_images_[dataset_index_];
        cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);

        if (img.empty())
        {
            ROS_WARN("Failed to load image: %s", img_path.c_str());
            dataset_index_++;
            return;
        }

        current_image_ = img;
        has_image_ = true;

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = false;

        try
        {
            found = cv::findChessboardCornersSB(
                gray,
                checkerboard_,
                corners,
                cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE
            );
        }
        catch (...)
        {
            found = false;
        }

        if (!found || corners.size() != checkerboard_.area())
        {
            ROS_WARN("Skipping image (board not found or incomplete): %s",
                    img_path.filename().c_str());
            dataset_index_++;
            return;
        }

        if (regions_.empty())
            initRegions(img.cols, img.rows);

        // ---- Determine region ----
        current_region_ = assignRegionByBoardContainment(corners);

        if (current_region_ < 0)
        {
            ROS_WARN("Board not fully contained in any region: %s",
                    img_path.filename().c_str());
            dataset_index_++;
            return;
        }

        // ---- Metrics ----
        last_sharpness_ = computeSharpness(gray);
        last_coverage_  = computeCoverageInRegion(corners, regions_[current_region_].rect);

        auto [tilt_h, tilt_v] = computeTiltFromSidesHV(corners);
        last_tilt_h_ = tilt_h;
        last_tilt_v_ = tilt_v;

        current_pose_  = classifyPose(tilt_h, tilt_v);
        current_scale_ = classifyScale(last_coverage_);

        bool sharp_ok = last_sharpness_ >= min_sharpness_;
        bool cover_ok = last_coverage_  >= min_coverage_;

        if (!(sharp_ok && cover_ok))
        {
            ROS_WARN("Rejected image (quality): %s", img_path.filename().c_str());
            dataset_index_++;
            return;
        }

        if (regions_[current_region_].isComplete())
        {
            ROS_INFO("Skipping image (region %d already complete): %s",
                    current_region_,
                    img_path.filename().c_str());
            dataset_index_++;
            return;
        }

        // ---- Update region counters ----
        Region& reg = regions_[current_region_];

        if (current_pose_ == STRAIGHT &&
            reg.scale_counts[current_scale_] < reg.target_scale_counts[current_scale_])
            reg.scale_counts[current_scale_]++;

        if (current_scale_ == SCALE_MEDIUM &&
            reg.pose_counts[current_pose_] < reg.target_pose_counts[current_pose_])
            reg.pose_counts[current_pose_]++;

        // ---- Write YAML entry ----
        fs_ << "{";
        fs_ << "filename" << img_path.filename().string();
        fs_ << "image_size" << "[" << img.cols << img.rows << "]";
        fs_ << "sharpness" << last_sharpness_;
        fs_ << "coverage" << last_coverage_;
        fs_ << "region_id" << current_region_;
        fs_ << "pose_class" << pose_names[current_pose_];
        fs_ << "scale_class" << current_scale_;
        fs_ << "tilt_h" << last_tilt_h_;
        fs_ << "tilt_v" << last_tilt_v_;
        fs_ << "corners_count" << (int)corners.size();
        fs_ << "corners" << "[";

        for (const auto& p : corners)
            fs_ << "[" << p.x << p.y << "]";

        fs_ << "]";
        fs_ << "}";

        ROS_INFO("Accepted image %zu/%zu: %s",
                dataset_index_ + 1,
                dataset_images_.size(),
                img_path.filename().c_str());

        dataset_index_++;
    }


};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "calib_interactive_collector_node");
    ros::NodeHandle nh("~");

    CalibCollector collector(nh);
    collector.spin();

    return 0;
}

