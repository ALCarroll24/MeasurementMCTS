#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pointcloud_array_msgs/msg/point_cloud2_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/voxel_grid.h>

class PointCloudPolygonNode : public rclcpp::Node
{
public:
    PointCloudPolygonNode() : Node("point_cloud_polygon_node")
    {
        // Parameters
        this->declare_parameter("height_threshold", -1.5);
        this->declare_parameter("cluster_tolerance", 1.0);
        this->declare_parameter("min_cluster_size", 30);
        this->declare_parameter("max_cluster_size", 10000);
        // Vehicle footprint parameters
        this->declare_parameter("vehicle_footprint_x_pos", 1.0);
        this->declare_parameter("vehicle_footprint_x_neg", -1.0);
        this->declare_parameter("vehicle_footprint_y_pos", 2.7);
        this->declare_parameter("vehicle_footprint_y_neg", -2.7);

        // Subscriber and publishers
        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10, std::bind(&PointCloudPolygonNode::pointCloudCallback, this, std::placeholders::_1));
        filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_cloud", 10);
        footprint_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/vehicle_footprint", 10);
        downsampled_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/downsampled_cloud", 10);
        clouds_pub_ = this->create_publisher<pointcloud_array_msgs::msg::PointCloud2Array>("/object_clouds", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Load parameters
        double height_threshold = this->get_parameter("height_threshold").as_double();
        double cluster_tolerance = this->get_parameter("cluster_tolerance").as_double();
        int min_cluster_size = this->get_parameter("min_cluster_size").as_int();
        int max_cluster_size = this->get_parameter("max_cluster_size").as_int();
        double x_pos = this->get_parameter("vehicle_footprint_x_pos").as_double();
        double x_neg = this->get_parameter("vehicle_footprint_x_neg").as_double();
        double y_pos = this->get_parameter("vehicle_footprint_y_pos").as_double();
        double y_neg = this->get_parameter("vehicle_footprint_y_neg").as_double();

        // Convert PointCloud2 to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        // Apply height filter
        // std::cout << "Before filtering: " << cloud->size() << " points" << std::endl;
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(height_threshold, std::numeric_limits<float>::max());
        pass.filter(*cloud);

        // Remove points within the vehicle footprint
        pcl::CropBox<pcl::PointXYZ> crop_box_filter;
        crop_box_filter.setInputCloud(cloud);
        crop_box_filter.setMin(Eigen::Vector4f(x_neg, y_neg, std::numeric_limits<float>::lowest(), 1.0));
        crop_box_filter.setMax(Eigen::Vector4f(x_pos, y_pos, std::numeric_limits<float>::max(), 1.0));
        crop_box_filter.setNegative(true);  // Remove points inside the box
        crop_box_filter.filter(*cloud);

        // Publish the filtered cloud
        sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*cloud, filtered_cloud_msg);
        filtered_cloud_msg.header = msg->header;
        filtered_cloud_pub_->publish(filtered_cloud_msg);

        // Publish vehicle footprint marker
        visualization_msgs::msg::Marker footprint_marker;
        footprint_marker.header.frame_id = msg->header.frame_id;
        footprint_marker.header.stamp = msg->header.stamp;
        footprint_marker.ns = "vehicle_footprint";
        footprint_marker.id = 0;
        footprint_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        footprint_marker.action = visualization_msgs::msg::Marker::ADD;
        footprint_marker.scale.x = 0.05;  // Line width

        // Set color for the footprint marker
        footprint_marker.color.r = 0.0;
        footprint_marker.color.g = 1.0;
        footprint_marker.color.b = 0.0;
        footprint_marker.color.a = 0.5;

        // Define the footprint points
        geometry_msgs::msg::Point p1, p2, p3, p4;
        p1.x = x_neg;
        p1.y = y_neg;
        p1.z = 0.0;
        p2.x = x_pos;
        p2.y = y_neg;
        p2.z = 0.0;
        p3.x = x_pos;
        p3.y = y_pos;
        p3.z = 0.0;
        p4.x = x_neg;
        p4.y = y_pos;
        p4.z = 0.0;

        footprint_marker.points.push_back(p1);
        footprint_marker.points.push_back(p2);
        footprint_marker.points.push_back(p3);
        footprint_marker.points.push_back(p4);
        footprint_marker.points.push_back(p1);  // Close the loop

        // Publish the footprint marker
        footprint_marker_pub_->publish(footprint_marker);

        // Downsample the point cloud with a VoxelGrid filter
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(0.1f, 0.1f, 0.1f);  // Adjust leaf size based on your data
        sor.filter(*cloud);
        // std::cout << "After downsampling: " << cloud->size() << " points" << std::endl;

        // Publish the downsampled cloud
        sensor_msgs::msg::PointCloud2 downsampled_cloud_msg;
        pcl::toROSMsg(*cloud, downsampled_cloud_msg);
        downsampled_cloud_msg.header = msg->header;
        downsampled_cloud_pub_->publish(downsampled_cloud_msg);

        // DBSCAN-based clustering (using Euclidean Cluster Extraction in PCL)
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(cluster_tolerance);  // Distance tolerance
        ec.setMinClusterSize(min_cluster_size);
        ec.setMaxClusterSize(max_cluster_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);
        // std::cout << "Number of clusters: " << cluster_indices.size() << std::endl;

        // Create output PointCloud2Array message
        pointcloud_array_msgs::msg::PointCloud2Array cloud_array_msg;

        // Iterate through each cluster and create a PointCloud2 message
        for (const auto &indices : cluster_indices)
        {
            // Extract the cluster points
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto &index : indices.indices)
            {
                cluster->points.push_back(cloud->points[index]);
            }

            // Convert the cluster to PointCloud2 message
            sensor_msgs::msg::PointCloud2 cluster_msg;
            pcl::toROSMsg(*cluster, cluster_msg);
            cluster_msg.header = msg->header;
            cloud_array_msg.clouds.push_back(cluster_msg);
        }
        // Publish the PointCloud2Array message
        clouds_pub_->publish(cloud_array_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<pointcloud_array_msgs::msg::PointCloud2Array>::SharedPtr clouds_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr downsampled_cloud_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr footprint_marker_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudPolygonNode>());
    rclcpp::shutdown();
    return 0;
}
