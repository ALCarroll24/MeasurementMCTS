#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
// #include <grid_map_msgs/msg/grid_map.hpp>
// #include <grid_map_core/grid_map_core.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
// #include <grid_map_ros/GridMapRosConverter.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <cmath>

class PointCloudToGridMap : public rclcpp::Node
{
public:
    PointCloudToGridMap()
    : Node("pointcloud_to_gridmap")
    {
        // Declare parameters
        this->declare_parameter<double>("resolution", 0.1);
        this->declare_parameter<double>("grid_size_x", 50.0);
        this->declare_parameter<double>("grid_size_y", 50.0);
        this->declare_parameter<bool>("align_x_axis_to_vehicle", false);
        this->declare_parameter<double>("rotation_angle", 0.0); // In radians

        // Setup subscriber and publisher
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "velodyne_points", 10, std::bind(&PointCloudToGridMap::pointCloudCallback, this, std::placeholders::_1));
        gridmap_pub_ = this->create_publisher<grid_map_msgs::msg::GridMap>("gaussian_gridmap", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Get parameters
        double resolution = this->get_parameter("resolution").as_double();
        double grid_size_x = this->get_parameter("grid_size_x").as_double();
        double grid_size_y = this->get_parameter("grid_size_y").as_double();
        bool align_x_axis = this->get_parameter("align_x_axis_to_vehicle").as_bool();
        double rotation_angle = this->get_parameter("rotation_angle").as_double();

        // Convert PointCloud2 to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // Set up GridMap
        grid_map::GridMap grid_map({"elevation", "num_points", "mean_height", "std_dev_height"});
        grid_map.setFrameId(msg->header.frame_id);
        grid_map.setGeometry(grid_map::Length(grid_size_x, grid_size_y), resolution, grid_map::Position(0.0, 0.0));

        // Prepare matrices for number of points, mean height, and standard deviation
        Eigen::MatrixXd num_points = Eigen::MatrixXd::Zero(grid_map.getSize()(0), grid_map.getSize()(1));
        Eigen::MatrixXd mean_heights = Eigen::MatrixXd::Zero(grid_map.getSize()(0), grid_map.getSize()(1));
        Eigen::MatrixXd squared_heights = Eigen::MatrixXd::Zero(grid_map.getSize()(0), grid_map.getSize()(1));

        // Iterate over point cloud and accumulate data into the grid map
        for (const auto& point : cloud->points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                continue;
            }

            // Optionally rotate the point to align with the front of the vehicle
            double x = point.x;
            double y = point.y;
            if (align_x_axis) {
                double cos_angle = cos(rotation_angle);
                double sin_angle = sin(rotation_angle);
                x = cos_angle * point.x - sin_angle * point.y;
                y = sin_angle * point.x + cos_angle * point.y;
            }

            // Get the corresponding grid cell
            grid_map::Position position(x, y);
            if (!grid_map.isInside(position)) {
                continue;
            }

            grid_map::Index index;
            grid_map.getIndex(position, index);

            // Accumulate values for each cell
            num_points(index(0), index(1)) += 1;
            mean_heights(index(0), index(1)) += point.z;
            squared_heights(index(0), index(1)) += point.z * point.z;
        }

        // Compute mean and standard deviation for each cell
        for (grid_map::GridMapIterator it(grid_map); !it.isPastEnd(); ++it) {
            grid_map::Index index(*it);
            if (num_points(index(0), index(1)) > 0) {
                double n = num_points(index(0), index(1));
                double mean = mean_heights(index(0), index(1)) / n;
                double variance = (squared_heights(index(0), index(1)) / n) - (mean * mean);
                double stddev = std::sqrt(variance);

                grid_map.at("num_points", *it) = n;
                grid_map.at("mean_height", *it) = mean;
                grid_map.at("std_dev_height", *it) = stddev;
                grid_map.at("elevation", *it) = mean;
            } else {
                grid_map.at("num_points", *it) = 0;
                grid_map.at("mean_height", *it) = NAN;
                grid_map.at("std_dev_height", *it) = NAN;
                grid_map.at("elevation", *it) = NAN;
            }
        }


        // Get the grid map message as a unique pointer
        std::unique_ptr<grid_map_msgs::msg::GridMap> grid_map_msg = grid_map::GridMapRosConverter::toMessage(grid_map);

        // Publish the grid map message
        gridmap_pub_->publish(std::move(*grid_map_msg));  // Move the unique pointer into the message
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr gridmap_pub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudToGridMap>());
    rclcpp::shutdown();
    return 0;
}
