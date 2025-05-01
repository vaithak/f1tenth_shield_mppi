#ifndef OPPONENT_DETECTION_H
#define OPPONENT_DETECTION_H

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/float32.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <cmath>

// Assuming these message types exist in your project
#include "f1tenth_icra_race_msgs/msg/obstacle_array.hpp"
#include "f1tenth_icra_race_msgs/msg/obstacle_msg.hpp"

// Forward declaration of FrenetConverter
class FrenetConverter;

/**
 * @brief Class representing an obstacle with position and size information
 */
class Obstacle
{
public:
    /**
     * @brief Constructor for Obstacle
     * @param x X-coordinate of the center
     * @param y Y-coordinate of the center
     * @param size Size of the obstacle
     * @param theta Orientation of the obstacle
     */
    Obstacle(double x, double y, double size, double theta);

    /**
     * @brief Calculate squared distance to another obstacle
     * @param obstacle The other obstacle
     * @return Squared distance between the obstacles
     */
    double squaredDist(const Obstacle &obstacle) const;

    // Obstacle properties
    double center_x;
    double center_y;
    double size;
    int id;
    double theta;
};

/**
 * @brief Class for detecting obstacles on a race track using lidar data
 *
 * This class implements a ROS2 node that detects obstacles on the track by
 * processing lidar scans and publishes the detected obstacles and visualization markers.
 */
class OpponentDetection : public rclcpp::Node
{
public:
    /**
     * @brief Constructor for OpponentDetection
     */
    OpponentDetection();

    /**
     * @brief Destructor for OpponentDetection
     */
    ~OpponentDetection();

private:
    // Type definition for 2D point
    using Point2D = std::pair<double, double>;

    /**
     * @brief Callback for pose messages
     * @param pose_msg Odometry message containing pose information
     */
    void poseCallback(const nav_msgs::msg::Odometry::SharedPtr pose_msg);

    /**
     * @brief Callback for laser scan messages
     * @param msg Laser scan message
     */
    void laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);

    /**
     * @brief Main processing loop
     */
    void loop();

    /**
     * @brief Convert angle to index in the laser scan data
     * @param angle Angle in radians
     * @return Index in the laser scan data array
     */
    int angleToIndex(double angle);

    /**
     * @brief Create a marker array to clear previous markers
     * @return Marker array with clear action
     */
    visualization_msgs::msg::MarkerArray clearMarkers();

    /**
     * @brief Check if a laser point is on the track
     * @param s S-coordinate in Frenet frame
     * @param d D-coordinate in Frenet frame
     * @param car_s S-coordinate of the car
     * @return True if the point is on the track, false otherwise
     */
    bool laserPointOnTrack(double s, double d, double car_s);

    /**
     * @brief Normalize s-coordinate to track length
     * @param x S-coordinate
     * @param track_length Length of the track
     * @return Normalized s-coordinate
     */
    double normalizeS(double x, double track_length);

    /**
     * @brief Convert laser scans to point clouds representing potential obstacles
     * @param car_s S-coordinate of the car
     * @param scans Laser scan message
     * @param car_x X-coordinate of the car
     * @param car_y Y-coordinate of the car
     * @param car_yaw Yaw angle of the car
     * @return List of point clouds representing potential obstacles
     */
    std::vector<std::vector<Point2D>> scans2ObsPointCloud(
        double car_s,
        const sensor_msgs::msg::LaserScan::SharedPtr scans,
        double car_x,
        double car_y,
        double car_yaw);

    /**
     * @brief Convert point clouds to obstacle objects
     * @param objects_pointcloud_list List of point clouds
     * @return List of obstacle objects
     */
    std::vector<Obstacle> obsPointClouds2obsArray(
        const std::vector<std::vector<Point2D>> &objects_pointcloud_list);

    /**
     * @brief Filter obstacles based on size and other criteria
     * @param current_obstacles List of detected obstacles
     */
    void checkObstacles(std::vector<Obstacle> &current_obstacles);

    /**
     * @brief Publish obstacle messages
     */
    void publishObstaclesMessage();

    /**
     * @brief Publish visualization markers for obstacles
     */
    void publishObstaclesMarkers();

    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;

    // Publishers
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr breakpoints_markers_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacles_marker_pub_;
    rclcpp::Publisher<f1tenth_icra_race_msgs::msg::ObstacleArray>::SharedPtr obstacles_msg_pub_;

    // Timer
    rclcpp::TimerBase::SharedPtr main_timer_;

    // Parameters
    bool is_sim_;
    bool plot_debug_;
    bool print_debug_;
    double rate_;
    double lambda_angle_;
    double sigma_;
    double min_2_points_dist_;
    int min_obs_size_;
    double max_obs_size_;
    double max_viewing_distance_;

    // State variables
    double car_s_;
    double car_global_x_;
    double car_global_y_;
    double car_global_yaw_;

    // Laser scan data
    sensor_msgs::msg::LaserScan::SharedPtr laser_scans_;
    double angle_increment_;
    double angle_min_;
    int front_view_start_index_;
    int front_view_end_index_;
    std::vector<double> angles_;

    // Track information
    std::vector<double> d_right_array_;
    std::vector<double> d_left_array_;
    std::vector<double> s_array_;
    double smallest_d_;
    double biggest_d_;
    double track_length_;
    std::string laser_frame_;

    // Obstacles
    std::vector<Obstacle> tracked_obstacles_;

    // Frenet converter
    std::unique_ptr<FrenetConverter> frenet_converter_;
};

#endif // OPPONENT_DETECTION_HPP