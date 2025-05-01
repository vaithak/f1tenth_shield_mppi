#include "f1tenth_icra_race/frenet_conversion.h"
#include <cmath>
#include <limits>
#include <stdexcept>

FrenetConverter::FrenetConverter(const std::vector<double> &x, const std::vector<double> &y,
                                 const std::vector<double> &yaw)
    : x_waypoints_(x), y_waypoints_(y), yaw_waypoints_(yaw)
{

    // Check that all input vectors have the same size
    if (x.size() != y.size() || x.size() != yaw.size())
    {
        throw std::invalid_argument("Input vectors must have the same size");
    }

    // Compute cumulative distances along the path (s-coordinate)
    s_waypoints_.resize(x.size());
    s_waypoints_[0] = 0.0;

    for (size_t i = 1; i < x.size(); ++i)
    {
        double dx = x[i] - x[i - 1];
        double dy = y[i] - y[i - 1];
        s_waypoints_[i] = s_waypoints_[i - 1] + std::sqrt(dx * dx + dy * dy);
    }
}

size_t FrenetConverter::findClosestWaypoint(double x, double y)
{
    double min_dist = std::numeric_limits<double>::max();
    size_t closest_wp = 0;

    for (size_t i = 0; i < x_waypoints_.size(); ++i)
    {
        double dist = std::pow(x - x_waypoints_[i], 2) + std::pow(y - y_waypoints_[i], 2);
        if (dist < min_dist)
        {
            min_dist = dist;
            closest_wp = i;
        }
    }

    return closest_wp;
}

void FrenetConverter::getFrenet(const std::vector<double> &x, const std::vector<double> &y,
                                std::vector<double> &s, std::vector<double> &d)
{
    s.resize(x.size());
    d.resize(y.size());

    for (size_t i = 0; i < x.size(); ++i)
    {
        // Find the closest waypoint
        size_t prev_wp = findClosestWaypoint(x[i], y[i]);

        // Get the coordinates of the waypoint
        double wp_x = x_waypoints_[prev_wp];
        double wp_y = y_waypoints_[prev_wp];

        // Calculate heading from waypoint to point
        double tangent_x = std::cos(yaw_waypoints_[prev_wp]);
        double tangent_y = std::sin(yaw_waypoints_[prev_wp]);
        double perpendicular_x = -tangent_y;
        double perpendicular_y = tangent_x;

        // Calculate the s value
        s[i] = s_waypoints_[prev_wp];

        // Calculate frenet d value (lateral displacement)
        double seg_x = x[i] - wp_x;
        double seg_y = y[i] - wp_y;

        // Calculate d value
        d[i] = (seg_x * perpendicular_x + seg_y * perpendicular_y);

        // Adjust s value if the point is ahead of the waypoint
        s[i] += (seg_x * tangent_x + seg_y * tangent_y);
    }
}

// void FrenetConverter::getXY(const std::vector<double> &s, const std::vector<double> &d,
//                             std::vector<double> &x, std::vector<double> &y)
// {
//     x.resize(s.size());
//     y.resize(s.size());

//     for (size_t i = 0; i < s.size(); ++i)
//     {
//         // Find the previous waypoint
//         size_t prev_wp = 0;
//         while (prev_wp < s_waypoints_.size() - 1 && s_waypoints_[prev_wp] <= s[i])
//         {
//             ++prev_wp;
//         }
//         if (prev_wp > 0)
//         {
//             --prev_wp; // Ensure we get the waypoint just before s
//         }

//         // The next waypoint
//         size_t next_wp = (prev_wp + 1) % x_waypoints_.size();

//         // Calculate interpolation ratio
//         double heading = std::atan2(y_waypoints_[next_wp] - y_waypoints_[prev_wp],
//                                     x_waypoints_[next_wp] - x_waypoints_[prev_wp]);

//         // The x,y,s along the segment
//         double seg_s = s[i] - s_waypoints_[prev_wp];
//         double seg_x = x_waypoints_[prev_wp] + seg_s * std::cos(heading);
//         double seg_y = y_waypoints_[prev_wp] + seg_s * std::sin(heading);

//         double perp_heading = heading + M_PI / 2.0;
//         x[i] = seg_x + d[i] * std::cos(perp_heading);
//         y[i] = seg_y + d[i] * std::sin(perp_heading);
//     }
// }