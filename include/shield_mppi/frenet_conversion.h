
#ifndef FRENET_CONVERSION_H
#define FRENET_CONVERSION_H

#include <vector>
#include <cmath>
#include <algorithm>

/**
 * @brief Class that handles conversion between global (x,y) and Frenet (s,d) coordinates
 *
 * The Frenet coordinate system is commonly used in autonomous driving where:
 * - s represents the distance along the reference path
 * - d represents the lateral offset from the reference path
 */
class FrenetConverter
{
public:
    /**
     * @brief Constructor for FrenetConverter
     * @param x Vector of x-coordinates of the reference path
     * @param y Vector of y-coordinates of the reference path
     * @param yaw Vector of yaw angles of the reference path
     */
    FrenetConverter(const std::vector<double> &x, const std::vector<double> &y,
                    const std::vector<double> &yaw);

    /**
     * @brief Convert from Cartesian (x,y) to Frenet (s,d) coordinates
     * @param x Vector of x-coordinates
     * @param y Vector of y-coordinates
     * @param[out] s Vector of resulting s-coordinates
     * @param[out] d Vector of resulting d-coordinates
     */
    void getFrenet(const std::vector<double> &x, const std::vector<double> &y,
                   std::vector<double> &s, std::vector<double> &d);

    // /**
    //  * @brief Convert from Frenet (s,d) to Cartesian (x,y) coordinates
    //  * @param s Vector of s-coordinates
    //  * @param d Vector of d-coordinates
    //  * @param[out] x Vector of resulting x-coordinates
    //  * @param[out] y Vector of resulting y-coordinates
    //  */
    // void getXY(const std::vector<double> &s, const std::vector<double> &d,
    //            std::vector<double> &x, std::vector<double> &y);

private:
    /**
     * @brief Find the closest waypoint to a given (x,y) point
     * @param x X-coordinate of the point
     * @param y Y-coordinate of the point
     * @return Index of the closest waypoint
     */
    size_t findClosestWaypoint(double x, double y);

    // Reference path waypoints
    std::vector<double> x_waypoints_;
    std::vector<double> y_waypoints_;
    std::vector<double> yaw_waypoints_;

    // Cumulative distances along the path
    std::vector<double> s_waypoints_;
};

#endif // FRENET_CONVERSION_HPP
