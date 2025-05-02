#include "shield_mppi/opponent_detection_node.h"
#include "shield_mppi/frenet_conversion.h" // Include the FrenetConverter implementation
#include <chrono>
#include <fstream>
#include <sstream>
#include <limits>
#include <ctime>

using namespace std::chrono_literals;
using std::placeholders::_1;

// Obstacle implementation
Obstacle::Obstacle(double x, double y, double size, double theta)
    : center_x(x), center_y(y), size(size), id(-1), theta(theta) {}

// Helper function to read waypoints from CSV
std::vector<std::vector<double>> readWaypointsFromCSV(const std::string &filename)
{
    std::vector<std::vector<double>> waypoints;
    std::ifstream file(filename);
    std::string line;

    // Skip header
    // std::getline(file, line);

    while (std::getline(file, line))
    {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ','))
        {
            // Trim whitespace
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            if (!value.empty())
            {
                row.push_back(std::stod(value));
            }
        }

        if (!row.empty())
        {
            waypoints.push_back(row);
        }
    }

    return waypoints;
}

// Helper function to get column indices for waypoints
std::map<std::string, int> columnNumbersForWaypoints()
{
    std::map<std::string, int> columns;
    columns["x_ref_m"] = 0;
    columns["y_ref_m"] = 1;
    columns["width_right_m"] = 6;
    columns["width_left_m"] = 5;
    // columns["x_normvec_m"] = 4;
    // columns["y_normvec_m"] = 5;
    // columns["alpha_m"] = 6;
    columns["s_racetraj_m"] = 4;
    columns["psi_racetraj_rad"] = 3;
    columns["vx"] = 2;
    return columns;
}

// Helper function to convert psi angles
std::vector<double> convertPsi(const std::vector<double> &psi)
{
    std::vector<double> result;
    for (double angle : psi)
    {
        angle += M_PI / 2;
        // Convert to range [-π, π]
        while (angle > M_PI)
            angle -= 2 * M_PI;
        while (angle < -M_PI)
            angle += 2 * M_PI;
        result.push_back(angle);
    }
    return result;
}

OpponentDetection::OpponentDetection()
    : Node("opponent_detection_node_cpp")
{

    // Declare and get parameters
    this->declare_parameter<bool>("is_sim", true);
    this->declare_parameter<bool>("plot_debug", false);
    this->declare_parameter<bool>("print_debug", false);
    this->declare_parameter<std::string>("waypoint_file", "/home/vaithak/Downloads/UPenn/F1Tenth/sim_ws/src/f1tenth_icra_race/waypoints/levine-practise-lane-optimal.csv");
    this->declare_parameter<int>("rate", 50);
    this->declare_parameter<int>("lambda", 10);
    this->declare_parameter<double>("sigma", 0.03);
    this->declare_parameter<double>("min_2_points_dist", 0.04);
    this->declare_parameter<int>("min_obs_size", 10);
    this->declare_parameter<double>("max_obs_size", 0.7);
    this->declare_parameter<double>("max_viewing_distance", 7.0);

    is_sim_ = this->get_parameter("is_sim").as_bool();
    plot_debug_ = this->get_parameter("plot_debug").as_bool();
    print_debug_ = this->get_parameter("print_debug").as_bool();
    std::string waypoint_file = this->get_parameter("waypoint_file").as_string();
    rate_ = this->get_parameter("rate").as_int();
    lambda_angle_ = this->get_parameter("lambda").as_int() * M_PI / 180.0;
    sigma_ = this->get_parameter("sigma").as_double();
    min_2_points_dist_ = this->get_parameter("min_2_points_dist").as_double();
    min_obs_size_ = this->get_parameter("min_obs_size").as_int();
    max_obs_size_ = this->get_parameter("max_obs_size").as_double();
    max_viewing_distance_ = this->get_parameter("max_viewing_distance").as_double();

    // Set up QoS profile
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10))
                   .reliability(rclcpp::ReliabilityPolicy::Reliable)
                   .durability(rclcpp::DurabilityPolicy::Volatile);

    // Create subscribers
    laser_frame_ = is_sim_ ? "ego_racecar/laser" : "laser";
    if (is_sim_)
    {
        pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom",
            qos,
            std::bind(&OpponentDetection::poseCallback, this, _1));
    }
    else
    {
        pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/pf/pose/odom",
            qos,
            std::bind(&OpponentDetection::poseCallback, this, _1));
    }

    laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan",
        qos,
        std::bind(&OpponentDetection::laserCallback, this, _1));

    // Create publishers
    if (plot_debug_)
    {
        breakpoints_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/perception/breakpoints_markers", 5);
        obstacles_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/perception/obstacles_markers_new", 5);
    }

    obstacles_msg_pub_ = this->create_publisher<f1tenth_icra_race_msgs::msg::ObstacleArray>(
        "/perception/detection/raw_obstacles", 5);

    // Initialize car state
    car_s_ = 0.0;
    car_global_x_ = 0.0;
    car_global_y_ = 0.0;
    car_global_yaw_ = 0.0;

    // Read waypoints from CSV
    auto waypoints = readWaypointsFromCSV(waypoint_file);
    if (waypoints.empty())
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to read waypoints from file: %s", waypoint_file.c_str());
        return;
    }

    auto columns = columnNumbersForWaypoints();

    // Extract waypoint data
    std::vector<double> x_points, y_points, yaw_points;
    for (const auto &waypoint : waypoints)
    {
        if (waypoint.size() <= std::max({columns["x_m"], columns["y_m"], columns["psi_racetraj_rad"],
                                         columns["s_racetraj_m"], columns["width_right_m"], columns["width_left_m"]}))
        {
            continue;
        }
        x_points.push_back(waypoint[columns["x_ref_m"]]);
        y_points.push_back(waypoint[columns["y_ref_m"]]);
        yaw_points.push_back(waypoint[columns["psi_racetraj_rad"]]);
        s_array_.push_back(waypoint[columns["s_racetraj_m"]]);
        d_right_array_.push_back(waypoint[columns["width_right_m"]] * 0.7);
        d_left_array_.push_back(waypoint[columns["width_left_m"]] * 0.7);
    }

    // Convert yaws
    // auto converted_yaws = convertPsi(yaw_points);
    auto converted_yaws = yaw_points;

    // Find track dimensions
    smallest_d_ = std::min(*std::min_element(d_right_array_.begin(), d_right_array_.end()),
                           *std::min_element(d_left_array_.begin(), d_left_array_.end()));
    biggest_d_ = std::max(*std::max_element(d_right_array_.begin(), d_right_array_.end()),
                          *std::max_element(d_left_array_.begin(), d_left_array_.end()));

    track_length_ = s_array_.back();

    // Initialize FrenetConverter
    frenet_converter_ = std::make_unique<FrenetConverter>(x_points, y_points, converted_yaws);
    RCLCPP_INFO(this->get_logger(), "[Opponent Detection]: initialized FrenetConverter object");

    // Create timer for main loop
    main_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / rate_),
        std::bind(&OpponentDetection::loop, this));
}

OpponentDetection::~OpponentDetection()
{
    // Cleanup - nothing special needed here
}

double OpponentDetection::normalizeS(double x, double track_length)
{
    double s = std::fmod(x, track_length);
    if (s > track_length / 2)
    {
        s -= track_length;
    }
    return s;
}

void OpponentDetection::poseCallback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
{
    // Get the current x, y position of the vehicle
    car_global_x_ = pose_msg->pose.pose.position.x;
    car_global_y_ = pose_msg->pose.pose.position.y;

    // Extract yaw from quaternion
    tf2::Quaternion q;
    tf2::fromMsg(pose_msg->pose.pose.orientation, q);
    car_global_yaw_ = tf2::getYaw(q);

    if (print_debug_)
    {
        RCLCPP_INFO(this->get_logger(), "Pose: %f, %f, %f",
                    car_global_x_, car_global_y_, car_global_yaw_);
    }

    // Convert the global coordinates to Frenet coordinates
    std::vector<double> s_coords, d_coords;
    frenet_converter_->getFrenet({car_global_x_}, {car_global_y_}, s_coords, d_coords);
    car_s_ = normalizeS(s_coords[0], track_length_);
}

void OpponentDetection::laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    laser_scans_ = msg;
    angle_increment_ = msg->angle_increment;
    angle_min_ = msg->angle_min;
    front_view_start_index_ = angleToIndex(-M_PI / 2);
    front_view_end_index_ = angleToIndex(M_PI / 2);

    // Initialize angles vector if not already done
    if (angles_.empty())
    {
        angles_.clear();
        for (double angle = -M_PI / 2; angle < M_PI / 2; angle += angle_increment_)
        {
            angles_.push_back(angle);
        }
    }
}

int OpponentDetection::angleToIndex(double angle)
{
    int index = static_cast<int>((angle - angle_min_) / angle_increment_);
    return index;
}

visualization_msgs::msg::MarkerArray OpponentDetection::clearMarkers()
{
    visualization_msgs::msg::MarkerArray marker_array;
    visualization_msgs::msg::Marker marker;
    marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(marker);
    return marker_array;
}

bool OpponentDetection::laserPointOnTrack(double s, double d, double car_s)
{
    if (normalizeS(s - car_s, track_length_) > max_viewing_distance_)
    {
        return false;
    }
    if (std::abs(d) >= biggest_d_)
    {
        return false;
    }
    if (std::abs(d) <= smallest_d_)
    {
        return true;
    }

    // Find the closest waypoint
    auto it = std::lower_bound(s_array_.begin(), s_array_.end(), s);
    size_t idx = 0;
    if (it != s_array_.begin())
    {
        idx = std::distance(s_array_.begin(), it) - 1;
    }

    if (d <= -d_right_array_[idx] || d >= d_left_array_[idx])
    {
        return false;
    }
    return true;
}

std::vector<std::vector<OpponentDetection::Point2D>> OpponentDetection::scans2ObsPointCloud(
    double car_s,
    const sensor_msgs::msg::LaserScan::SharedPtr scans,
    double car_x,
    double car_y,
    double car_yaw)
{

    // Initialize utility parameters
    double l = lambda_angle_;
    double d_phi = scans->angle_increment;
    double sigma = sigma_;

    // Extract scan ranges (only consider angles from -90 to 90 degrees)
    std::vector<float> ranges(scans->ranges.begin() + front_view_start_index_,
                              scans->ranges.begin() + front_view_end_index_ + 1);

    // Transform scan ranges to cloud points
    std::vector<Eigen::Vector4d> xyz_laser_frame;
    for (size_t i = 0; i < ranges.size(); ++i)
    {
        double angle = angles_[i];
        double r = ranges[i];
        double x = r * std::cos(angle);
        double y = r * std::sin(angle);
        xyz_laser_frame.push_back(Eigen::Vector4d(x, y, 0.0, 1.0));
    }

    // Create transformation matrix from laser frame to map frame
    Eigen::Matrix4d H_l2m = Eigen::Matrix4d::Identity();
    H_l2m(0, 3) = car_x;
    H_l2m(1, 3) = car_y;
    H_l2m(0, 0) = std::cos(car_yaw);
    H_l2m(0, 1) = -std::sin(car_yaw);
    H_l2m(1, 0) = std::sin(car_yaw);
    H_l2m(1, 1) = std::cos(car_yaw);

    // Transform points to map frame
    std::vector<Point2D> cloudPoints_list;
    for (const auto &point : xyz_laser_frame)
    {
        Eigen::Vector4d p_map = H_l2m * point;
        cloudPoints_list.push_back({p_map(0), p_map(1)});
    }

    // Segment the cloud points into smaller point clouds representing potential objects
    std::vector<std::vector<Point2D>> objects_pointcloud_list;
    if (cloudPoints_list.empty())
    {
        return objects_pointcloud_list;
    }

    objects_pointcloud_list.push_back({cloudPoints_list[0]});
    double div_const = std::sin(d_phi) / std::sin(l - d_phi);

    for (size_t i = 1; i < cloudPoints_list.size(); ++i)
    {
        double curr_range = ranges[i];
        double d_max = curr_range * div_const + 3 * sigma;

        // Calculate distance to previous point in the laser frame
        double dist_to_next_point = (xyz_laser_frame[i].head<2>() - xyz_laser_frame[i - 1].head<2>()).norm();

        if (dist_to_next_point < d_max)
        {
            objects_pointcloud_list.back().push_back(cloudPoints_list[i]);
        }
        else
        {
            objects_pointcloud_list.push_back({cloudPoints_list[i]});
        }
    }

    // Filter objects based on size and location
    std::vector<std::vector<Point2D>> filtered_objects;

    for (const auto &obj : objects_pointcloud_list)
    {
        // Skip objects that are too small
        if (obj.size() < static_cast<size_t>(min_obs_size_))
        {
            continue;
        }

        // Calculate mean position
        double mean_x = 0.0, mean_y = 0.0;
        for (const auto &point : obj)
        {
            mean_x += point.first;
            mean_y += point.second;
        }
        mean_x /= obj.size();
        mean_y /= obj.size();

        // Convert to Frenet coordinates
        std::vector<double> s_points, d_points;
        frenet_converter_->getFrenet({mean_x}, {mean_y}, s_points, d_points);

        // Keep object if it's on the track
        if (laserPointOnTrack(s_points[0], d_points[0], car_s))
        {
            filtered_objects.push_back(obj);
        }
    }

    // Publish debug markers if requested
    if (plot_debug_)
    {
        visualization_msgs::msg::MarkerArray markers_array;

        for (size_t idx = 0; idx < filtered_objects.size(); ++idx)
        {
            const auto &object = filtered_objects[idx];

            // First point marker
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = this->now();
            marker.id = idx * 10;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.scale.x = marker.scale.y = marker.scale.z = 0.25;
            marker.color.a = 0.5;
            marker.color.g = 1.0;
            marker.color.r = 0.0;
            marker.color.b = static_cast<float>(idx) / filtered_objects.size();
            marker.pose.position.x = object[0].first;
            marker.pose.position.y = object[0].second;
            marker.pose.orientation.w = 1.0;
            markers_array.markers.push_back(marker);

            // Last point marker
            marker.id = idx * 10 + 2;
            marker.pose.position.x = object.back().first;
            marker.pose.position.y = object.back().second;
            markers_array.markers.push_back(marker);
        }

        // Publish markers
        breakpoints_markers_pub_->publish(clearMarkers());
        if (filtered_objects.size() > 0)
        {
            breakpoints_markers_pub_->publish(markers_array);
        }
        
    }

    return filtered_objects;
}

std::vector<Obstacle> OpponentDetection::obsPointClouds2obsArray(
    const std::vector<std::vector<Point2D>> &objects_pointcloud_list)
{

    std::vector<Obstacle> current_obstacle_array;
    double min_dist = min_2_points_dist_;

    for (const auto &obstacle_points : objects_pointcloud_list)
    {
        // Convert point pairs to Eigen vectors for easier processing
        std::vector<Eigen::Vector2d> points;
        for (const auto &point : obstacle_points)
        {
            points.push_back(Eigen::Vector2d(point.first, point.second));
        }

        // Fit a rectangle to the data points
        std::vector<double> theta_values;
        for (int i = 0; i < 90; ++i)
        {
            theta_values.push_back(i * M_PI / 180.0);
        }

        // Find optimal theta for minimum bounding box
        double theta_opt = 0.0;
        double max_sum_recip = 0.0;

        for (double theta : theta_values)
        {
            double cos_theta = std::cos(theta);
            double sin_theta = std::sin(theta);

            // Project points onto rotated axes
            std::vector<double> distances1, distances2;
            for (const auto &point : points)
            {
                distances1.push_back(point.x() * cos_theta + point.y() * sin_theta);
                distances2.push_back(-point.x() * sin_theta + point.y() * cos_theta);
            }

            // Find min/max extents in each direction
            double min_dist1 = *std::min_element(distances1.begin(), distances1.end());
            double max_dist1 = *std::max_element(distances1.begin(), distances1.end());
            double min_dist2 = *std::min_element(distances2.begin(), distances2.end());
            double max_dist2 = *std::max_element(distances2.begin(), distances2.end());

            // Calculate D values (distances to edges)
            std::vector<double> D10, D11, D20, D21;
            for (double d : distances1)
            {
                D10.push_back(-d + max_dist1);
                D11.push_back(d - min_dist1);
            }
            for (double d : distances2)
            {
                D20.push_back(-d + max_dist2);
                D21.push_back(d - min_dist2);
            }

            // Choose minimum distances
            std::vector<double> D;
            for (size_t i = 0; i < D10.size(); ++i)
            {
                double d1 = std::min(D10[i], D11[i]);
                double d2 = std::min(D20[i], D21[i]);
                D.push_back(std::min(d1, d2));
                if (D.back() < min_dist)
                {
                    D.back() = min_dist;
                }
            }

            // Calculate sum of reciprocals
            double sum_recip = 0.0;
            for (double d : D)
            {
                sum_recip += 1.0 / d;
            }

            if (sum_recip > max_sum_recip)
            {
                max_sum_recip = sum_recip;
                theta_opt = theta;
            }
        }

        // Calculate projections for optimal angle
        double cos_opt = std::cos(theta_opt);
        double sin_opt = std::sin(theta_opt);

        std::vector<double> distances1, distances2;
        for (const auto &point : points)
        {
            distances1.push_back(point.x() * cos_opt + point.y() * sin_opt);
            distances2.push_back(-point.x() * sin_opt + point.y() * cos_opt);
        }

        double max_dist1 = *std::max_element(distances1.begin(), distances1.end());
        double min_dist1 = *std::min_element(distances1.begin(), distances1.end());
        double max_dist2 = *std::max_element(distances2.begin(), distances2.end());
        double min_dist2 = *std::min_element(distances2.begin(), distances2.end());

        // Calculate variance to determine orientation
        double var1 = 0.0, var2 = 0.0;
        double mean1 = std::accumulate(distances1.begin(), distances1.end(), 0.0) / distances1.size();
        double mean2 = std::accumulate(distances2.begin(), distances2.end(), 0.0) / distances2.size();

        for (double d : distances1)
            var1 += std::pow(d - mean1, 2);
        for (double d : distances2)
            var2 += std::pow(d - mean2, 2);
        var1 /= distances1.size();
        var2 /= distances2.size();

        // Determine corners based on detection distribution
        Eigen::Vector2d corner1, corner2;

        // Calculate the norms for edge detection
        double norm_max1 = 0.0, norm_min1 = 0.0;
        for (double d : distances1)
        {
            norm_max1 += std::pow(-d + max_dist1, 2);
            norm_min1 += std::pow(d - min_dist1, 2);
        }
        norm_max1 = std::sqrt(norm_max1);
        norm_min1 = std::sqrt(norm_min1);

        double norm_max2 = 0.0, norm_min2 = 0.0;
        for (double d : distances2)
        {
            norm_max2 += std::pow(-d + max_dist2, 2);
            norm_min2 += std::pow(d - min_dist2, 2);
        }
        norm_max2 = std::sqrt(norm_max2);
        norm_min2 = std::sqrt(norm_min2);

        // The obstacle has more detection in the vertical direction
        if (var2 > var1)
        {
            if (norm_max1 < norm_min1)
            {
                // Detections are nearer to the right edge
                // Lower right corner
                corner1 = Eigen::Vector2d(
                    cos_opt * max_dist1 - sin_opt * min_dist2,
                    sin_opt * max_dist1 + cos_opt * min_dist2);
                // Upper right corner
                corner2 = Eigen::Vector2d(
                    cos_opt * max_dist1 - sin_opt * max_dist2,
                    sin_opt * max_dist1 + cos_opt * max_dist2);
            }
            else
            {
                // Detections are nearer to the left edge
                // Upper left corner
                corner1 = Eigen::Vector2d(
                    cos_opt * min_dist1 - sin_opt * max_dist2,
                    sin_opt * min_dist1 + cos_opt * max_dist2);
                // Lower left corner
                corner2 = Eigen::Vector2d(
                    cos_opt * min_dist1 - sin_opt * min_dist2,
                    sin_opt * min_dist1 + cos_opt * min_dist2);
            }
        }
        else
        { // The obstacle has more detection in the horizontal direction
            if (norm_max2 < norm_min2)
            {
                // Detections are nearer to the top edge
                // Upper right corner
                corner1 = Eigen::Vector2d(
                    cos_opt * max_dist1 - sin_opt * max_dist2,
                    sin_opt * max_dist1 + cos_opt * max_dist2);
                // Upper left corner
                corner2 = Eigen::Vector2d(
                    cos_opt * min_dist1 - sin_opt * max_dist2,
                    sin_opt * min_dist1 + cos_opt * max_dist2);
            }
            else
            {
                // Detections are nearer to the bottom edge
                // Lower left corner
                corner1 = Eigen::Vector2d(
                    cos_opt * min_dist1 - sin_opt * min_dist2,
                    sin_opt * min_dist1 + cos_opt * min_dist2);
                // Lower right corner
                corner2 = Eigen::Vector2d(
                    cos_opt * max_dist1 - sin_opt * min_dist2,
                    sin_opt * max_dist1 + cos_opt * min_dist2);
            }
        }

        // Vector from corner1 to corner2
        Eigen::Vector2d colVec = corner2 - corner1;
        // Orthogonal vector to colVec
        Eigen::Vector2d orthVec(-colVec.y(), colVec.x());
        // Center position
        Eigen::Vector2d center = corner1 + 0.5 * colVec + 0.5 * orthVec;

        current_obstacle_array.push_back(
            Obstacle(center.x(), center.y(), colVec.norm(), theta_opt));
    }

    RCLCPP_DEBUG(this->get_logger(),
                 "[Opponent Detection] detected %zu raw obstacles.",
                 current_obstacle_array.size());

    return current_obstacle_array;
}

void OpponentDetection::checkObstacles(std::vector<Obstacle> &current_obstacles)
{
    // Delete obstacles that are too big
    auto it = std::remove_if(current_obstacles.begin(), current_obstacles.end(),
                             [this](const Obstacle &obs)
                             {
                                 return obs.size > max_obs_size_;
                             });

    size_t removed_count = std::distance(it, current_obstacles.end());
    current_obstacles.erase(it, current_obstacles.end());

    RCLCPP_DEBUG(this->get_logger(),
                 "[Opponent Detection] removed %zu obstacles as they are too big.",
                 removed_count);

    // Clear tracked obstacles and assign IDs to current obstacles
    tracked_obstacles_.clear();

    for (size_t idx = 0; idx < current_obstacles.size(); ++idx)
    {
        current_obstacles[idx].id = idx;
        tracked_obstacles_.push_back(current_obstacles[idx]);
    }

    RCLCPP_DEBUG(this->get_logger(),
                 "[Opponent Detection] tracking %zu obstacles.",
                 tracked_obstacles_.size());
}

void OpponentDetection::publishObstaclesMessage()
{
    auto obstacles_array_message = std::make_unique<f1tenth_icra_race_msgs::msg::ObstacleArray>();
    obstacles_array_message->header.stamp = this->now();
    obstacles_array_message->header.frame_id = "map";

    std::vector<double> x_center, y_center;
    for (const auto &obstacle : tracked_obstacles_)
    {
        x_center.push_back(obstacle.center_x);
        y_center.push_back(obstacle.center_y);
    }

    // Convert to Frenet coordinates
    std::vector<double> s_points, d_points;
    frenet_converter_->getFrenet(x_center, y_center, s_points, d_points);

    for (size_t idx = 0; idx < tracked_obstacles_.size(); ++idx)
    {
        const auto &obstacle = tracked_obstacles_[idx];
        double s = s_points[idx];
        double d = d_points[idx];

        auto obs_msg = f1tenth_icra_race_msgs::msg::ObstacleMsg();
        obs_msg.id = obstacle.id;
        obs_msg.s_start = s - obstacle.size / 2;
        obs_msg.s_end = s + obstacle.size / 2;
        obs_msg.d_left = d + obstacle.size / 2;
        obs_msg.d_right = d - obstacle.size / 2;
        obs_msg.s_center = s;
        obs_msg.d_center = d;
        obs_msg.size = obstacle.size;

        obstacles_array_message->obstacles.push_back(obs_msg);
    }

    obstacles_msg_pub_->publish(std::move(obstacles_array_message));
}

void OpponentDetection::publishObstaclesMarkers()
{
    visualization_msgs::msg::MarkerArray markers_array;

    for (const auto &obs : tracked_obstacles_)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->now();
        marker.id = obs.id;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.scale.x = marker.scale.y = marker.scale.z = obs.size;
        marker.color.a = 0.5;
        marker.color.g = 1.0;
        marker.color.r = 0.0;
        marker.color.b = 1.0;
        marker.pose.position.x = obs.center_x;
        marker.pose.position.y = obs.center_y;

        // Convert Euler angle to quaternion
        tf2::Quaternion q;
        q.setRPY(0, 0, obs.theta);
        marker.pose.orientation = tf2::toMsg(q);

        markers_array.markers.push_back(marker);
    }

    // This causes the markers to flicker in RViz, but likely doesn't affect the underlying algorithm
    obstacles_marker_pub_->publish(clearMarkers());
    if (!tracked_obstacles_.empty())
    {
        obstacles_marker_pub_->publish(markers_array);
    }
    
}

void OpponentDetection::loop()
{
    // Check if we have laser scans and car position
    if (!laser_scans_ || car_s_ == 0.0)
    {
        return;
    }

    auto scans = laser_scans_;
    double car_x = car_global_x_;
    double car_y = car_global_y_;
    double car_yaw = car_global_yaw_;
    double car_s = car_s_;

    // Obstacle detection
    std::chrono::steady_clock::time_point start_time;
    if (print_debug_)
    {
        start_time = std::chrono::steady_clock::now();
    }

    // Process lidar data to detect obstacles
    auto objects_pointcloud_list = scans2ObsPointCloud(car_s, scans, car_x, car_y, car_yaw);
    auto current_obstacles = obsPointClouds2obsArray(objects_pointcloud_list);
    checkObstacles(current_obstacles);

    // Publish detected obstacles
    publishObstaclesMessage();
    if (plot_debug_)
    {
        publishObstaclesMarkers();
    }

    if (print_debug_)
    {
        auto end_time = std::chrono::steady_clock::now();
        double latency = std::chrono::duration<double>(end_time - start_time).count();
        RCLCPP_INFO(this->get_logger(), "Latency checkObstacles: %.4f seconds", latency);
    }
}

int main(int argc, char *argv[])
{
    // Initialize ROS
    rclcpp::init(argc, argv);

    // Create the opponent detection node
    auto node = std::make_shared<OpponentDetection>();

    // Create an executor to manage the node
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);

    // Spin the node
    try
    {
        executor.spin();
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(node->get_logger(), "Error: %s", e.what());
    }
    catch (...)
    {
        RCLCPP_ERROR(node->get_logger(), "Unknown error occurred");
    }

    // Shutdown ROS
    rclcpp::shutdown();

    return 0;
}