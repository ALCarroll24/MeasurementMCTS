#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_box.h>

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
        downsampled_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/downsampled_cloud", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/object_points_array", 10);
        footprint_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/vehicle_footprint", 10);
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

        // Publish vehicle footprint marker (same as before)
        // ... (Code for footprint marker) ...

        // Downsample the point cloud with a VoxelGrid filter
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(0.1f, 0.1f, 0.1f);  // Adjust leaf size based on your data
        sor.filter(*cloud);

        // Publish the downsampled cloud
        sensor_msgs::msg::PointCloud2 downsampled_cloud_msg;
        pcl::toROSMsg(*cloud, downsampled_cloud_msg);
        downsampled_cloud_msg.header = msg->header;
        downsampled_cloud_pub_->publish(downsampled_cloud_msg);

        // DBSCAN-based clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(cluster_tolerance);
        ec.setMinClusterSize(min_cluster_size);
        ec.setMaxClusterSize(max_cluster_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        // Create markers for each cluster
        visualization_msgs::msg::MarkerArray marker_array;
        int cluster_id = 0;

        for (const auto &indices : cluster_indices)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto &index : indices.indices)
            {
                cluster->points.push_back(cloud->points[index]);
            }

            // Apply RANSAC for rectangular prism fitting
            pcl::SampleConsensusModelBox<pcl::PointXYZ>::Ptr model_box(new pcl::SampleConsensusModelBox<pcl::PointXYZ>(cluster));
            pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_box);
            ransac.setDistanceThreshold(0.01);
            ransac.computeModel();
            
            Eigen::VectorXf coefficients;
            ransac.getModelCoefficients(coefficients);

            // Create a Marker to represent the box
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = msg->header.frame_id;
            marker.header.stamp = msg->header.stamp;
            marker.ns = "box";
            marker.id = cluster_id++;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = coefficients[0];
            marker.pose.position.y = coefficients[1];
            marker.pose.position.z = coefficients[2];
            marker.scale.x = coefficients[3];
            marker.scale.y = coefficients[4];
            marker.scale.z = coefficients[5];
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 0.6;

            marker_array.markers.push_back(marker);
        }

        // Publish the MarkerArray
        marker_pub_->publish(marker_array);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
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
