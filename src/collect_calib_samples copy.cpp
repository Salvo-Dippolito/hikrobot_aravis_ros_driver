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


        checkerboard_ = cv::Size(cols, rows);

        // --- Ensure save directory exists ---
        if (!fs::exists(save_dir_))
            fs::create_directories(save_dir_);

        // --- Subscriber ---
        sub_ = nh_.subscribe(image_topic_, 1, &CalibCollector::imageCallback, this);

        // --- OpenCV window ---
        cv::namedWindow("calibration_selector", cv::WINDOW_NORMAL);
        cv::setMouseCallback("calibration_selector", &CalibCollector::mouseCallback, this);

        // --- Button rectangle ---
        save_button_rect_ = cv::Rect(20, 100, 220, 50);

        ROS_INFO("CalibCollector initialized: %dx%d checkerboard, square=%.3f",
                 cols, rows, square_size_);

        cv::FileStorage fs_;

        std::string yaml_path = save_dir_ + "/corners.yaml";
        fs_.open(yaml_path, cv::FileStorage::WRITE);

        fs_ << "board" << "{";
        fs_ << "rows" << rows;
        fs_ << "cols" << cols;
        fs_ << "square_size" << square_size_;
        fs_ << "}";
        
        fs_ << "images" << "[";

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

            if (quadrants_.empty()) initQuadrants(current_image_.cols, current_image_.rows);

            processAndDisplay();
            int key = cv::waitKey(1);
            if (key == 'q') break;

            if ((key == ' ' || mouse_clicked_) && last_frame_valid_)
            {
                saveImage();
            }
            mouse_clicked_ = false;
            rate.sleep();
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



    // Quadrant tracking
    struct Quadrant {
        cv::Rect rect;
        int saved_count = 0;
    };

    std::vector<Quadrant> quadrants_;

    void initQuadrants(int img_w, int img_h) {
        quadrants_.clear();
        int w2 = img_w / 2;
        int h2 = img_h / 2;
        quadrants_.push_back({cv::Rect(0, 0, w2, h2)});       // top-left
        quadrants_.push_back({cv::Rect(w2, 0, w2, h2)});      // top-right
        quadrants_.push_back({cv::Rect(0, h2, w2, h2)});      // bottom-left
        quadrants_.push_back({cv::Rect(w2, h2, w2, h2)});     // bottom-right
    }   

    int target_per_quadrant_ = 20; // images per quadrant

    // Tilt tracking
    enum TiltClass { LOW_TILT = 0, MID_TILT = 1, HIGH_TILT = 2, TILT_COUNT };

    int tilt_counts_[4][TILT_COUNT] = {{0}};
    int target_per_tilt_ = 3;
    
    TiltClass classifyTilt(double a)
    {
        if (a < 0.10) return LOW_TILT;
        if (a < 0.30) return MID_TILT;
        return HIGH_TILT;
    }
    



    // ===== ROS Callback =====
    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        try
        {
            auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8"); // converting from rgb8 to bgr8
            current_image_ = cv_ptr->image.clone();
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

    double computeTiltAnisotropy(const std::vector<cv::Point2f>& corners)
    {
        if (corners.size() < 4)
            return 0.0;

        double dx = 0.0, dy = 0.0;
        int dx_n = 0, dy_n = 0;

        for (size_t i = 1; i < corners.size(); ++i)
        {
            cv::Point2f d = corners[i] - corners[i - 1];
            if (std::abs(d.x) > std::abs(d.y))
            {
                dx += std::abs(d.x);
                dx_n++;
            }
            else
            {
                dy += std::abs(d.y);
                dy_n++;
            }
        }

        if (dx_n == 0 || dy_n == 0)
            return 0.0;

        double hx = dx / dx_n;
        double hy = dy / dy_n;

        return std::abs(hx - hy) / std::max(hx, hy);
    }

    double computeSharpness(const cv::Mat& gray)
    {
        cv::Mat lap;
        cv::Laplacian(gray, lap, CV_64F);
        cv::Scalar mu, sigma;
        cv::meanStdDev(lap, mu, sigma);
        return sigma[0] * sigma[0];
    }

    double computeCoverage(const std::vector<cv::Point2f>& corners, int w, int h)
    {
        double min_x = 1e9, max_x = -1e9;
        double min_y = 1e9, max_y = -1e9;
        for (const auto& p : corners)
        {
            min_x = std::min(min_x, (double)p.x);
            max_x = std::max(max_x, (double)p.x);
            min_y = std::min(min_y, (double)p.y);
            max_y = std::max(max_y, (double)p.y);
        }
        double cov_x = (max_x - min_x) / w;
        double cov_y = (max_y - min_y) / h;
        return std::min(cov_x, cov_y);
    }

    std::string generateTimestampedFilename()
    {
        std::ostringstream ss;
        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);
        ss << save_dir_ << "/calib_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".png";
        return ss.str();
    }

    void saveImage()
    {
        if (last_corners_.empty())
            return;

        // Determine which quadrant the checkerboard is mostly in
        cv::Point2f center(0.f, 0.f);
        for (const auto& c : last_corners_) {
            center += c;
        }
        center *= (1.0f / last_corners_.size());
        
        int q_idx = 0;
        for (size_t i=0; i<quadrants_.size(); ++i) {
            if (quadrants_[i].rect.contains(center)) {
                q_idx = i;
                break;
            }
        }

        std::string filename = generateTimestampedFilename();
        cv::imwrite(filename, current_image_);
        saved_count_++;
        quadrants_[q_idx].saved_count++;

        ROS_INFO("Saved image #%d: %s", saved_count_, filename.c_str());

        if (quadrants_[q_idx].saved_count >= target_per_quadrant_)
            ROS_INFO("Quadrant %zu target reached! Move to another quadrant.", q_idx);

        double tilt_aniso = computeTiltAnisotropy(last_corners_);
        TiltClass tilt = classifyTilt(tilt_aniso);
        tilt_counts_[q_idx][tilt]++;


        fs_ << "{";
        fs_ << "filename" << fs::path(filename).filename().string();
        fs_ << "image_size" << "[" << current_image_.cols << current_image_.rows << "]";
        fs_ << "sharpness" << last_sharpness_;
        fs_ << "coverage" << last_coverage_;
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
        cv::flip(img, img, +1);  // horizontal mirror

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        // bool found = cv::findChessboardCorners(gray, checkerboard_, corners);
        bool found = cv::findChessboardCornersSB(gray,checkerboard_,corners); // suposedly more stable
        last_frame_valid_ = false;

        if (found)
        {
            cv::cornerSubPix(gray, corners, cv::Size(5,5), cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

            cv::drawChessboardCorners(img, checkerboard_, corners, found);

            double sharpness = computeSharpness(gray);
            double coverage = computeCoverage(corners, img.cols, img.rows);
            bool full = (corners.size() == checkerboard_.area());




            // Determine quadrant
            cv::Point2f center(0.f, 0.f);
            for (const auto& c : corners)
                center += c;
            center *= (1.0f / corners.size());

            int q_idx = 0;
            for (size_t i = 0; i < quadrants_.size(); ++i)
            {
                if (quadrants_[i].rect.contains(center))
                {
                    q_idx = i;
                    break;
                }
            }

            // Tilt evaluation
            double tilt_aniso = computeTiltAnisotropy(corners);
            TiltClass tilt = classifyTilt(tilt_aniso);

            // Check if this tilt class is still needed
            bool tilt_needed = tilt_counts_[q_idx][tilt] < target_per_tilt_;

            // Final validity
            last_frame_valid_ =
                sharpness > min_sharpness_ &&
                coverage > min_coverage_ &&
                (!require_full_cornerset_ || full) &&
                tilt_needed;

            std::ostringstream status;
            status << "Corners: " << corners.size()
                   << " Sharpness: " << (int)sharpness
                   << " Coverage: " << (int)(coverage*100) << "%";
            cv::putText(img, status.str(), cv::Point(20,30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        last_frame_valid_ ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);

            last_corners_ = corners;
            last_sharpness_ = sharpness;
            last_coverage_  = coverage;

        }
        else
        {
            cv::putText(img, "Checkerboard NOT found", cv::Point(20,40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
        }

        // Draw button
        cv::Scalar btn_color = cv::Scalar(40,40,40); // default
        if (mouse_hover_) btn_color = cv::Scalar(80,80,80);  // hover
        if (mouse_pressed_) btn_color = cv::Scalar(0,150,0); // pressed

        cv::rectangle(img, save_button_rect_, btn_color, cv::FILLED);
        cv::putText(img, "SAVE IMAGE", save_button_rect_.tl() + cv::Point(10,35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);

        // Draw saved counter
        cv::putText(img, "Saved: " + std::to_string(saved_count_), cv::Point(20,180),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);


        // Determine next quadrant needing images
        int next_quadrant = -1;

        for (size_t i = 0; i < quadrants_.size(); ++i)
        {
            if (quadrants_[i].saved_count < target_per_quadrant_)
            {
                next_quadrant = (int)i;
                break;
            }
        }

        // Highlight next quadrant in yellow (if any)
        if (next_quadrant >= 0)
        {
            cv::rectangle(img, quadrants_[next_quadrant].rect, cv::Scalar(0,255,255), 3);

            cv::putText(img, "Next quadrant",
                        quadrants_[next_quadrant].rect.tl() + cv::Point(5,40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0,255,255), 2);
        }

        std::ostringstream tilt_msg;
tilt_msg << "Tilt: "
         << (tilt == LOW_TILT ? "LOW" :
             tilt == MID_TILT ? "MID" : "HIGH");

cv::putText(img, tilt_msg.str(),
            cv::Point(20, 60),
            cv::FONT_HERSHEY_SIMPLEX, 0.7,
            cv::Scalar(255, 255, 0), 2);

        cv::imshow("calibration_selector", img);


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
