#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
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
        // this->declare_parameter("convex_hull_tolerance", 0.01);
        this->declare_parameter("cluster_tolerance", 1.);  // DBSCAN tolerance (how close points must be to form a cluster)
        this->declare_parameter("min_cluster_size", 30);
        this->declare_parameter("max_cluster_size", 10000);

        // Subscriber and publisher
        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10, std::bind(&PointCloudPolygonNode::pointCloudCallback, this, std::placeholders::_1));
        filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_cloud", 10);
        downsampled_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/downsampled_cloud", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/polygon_marker_array", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Load parameters
        double height_threshold = this->get_parameter("height_threshold").as_double();
        // double convex_hull_tolerance = this->get_parameter("convex_hull_tolerance").as_double();
        double cluster_tolerance = this->get_parameter("cluster_tolerance").as_double();
        int min_cluster_size = this->get_parameter("min_cluster_size").as_int();
        int max_cluster_size = this->get_parameter("max_cluster_size").as_int();

        // Convert PointCloud2 to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        // Apply height filter
        std::cout << "Before filtering: " << cloud->size() << " points" << std::endl;
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(height_threshold, std::numeric_limits<float>::max());
        pass.filter(*cloud);
        std::cout << "After filtering: " << cloud->size() << " points" << std::endl;

        // Publish the filtered cloud
        sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*cloud, filtered_cloud_msg);
        filtered_cloud_msg.header = msg->header;
        filtered_cloud_pub_->publish(filtered_cloud_msg);

        // Downsample the point cloud with a VoxelGrid filter
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(0.1f, 0.1f, 0.1f);  // Adjust leaf size based on your data
        sor.filter(*cloud);
        std::cout << "After downsampling: " << cloud->size() << " points" << std::endl;

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
        ec.setClusterTolerance(cluster_tolerance); // Distance tolerance
        ec.setMinClusterSize(min_cluster_size);
        ec.setMaxClusterSize(max_cluster_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);
        std::cout << "Number of clusters: " << cluster_indices.size() << std::endl;

        // Create markers for each cluster
        visualization_msgs::msg::MarkerArray marker_array;
        int cluster_id = 0;

        // Clear previous markers
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);
        
        for (const auto& indices : cluster_indices)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& index : indices.indices)
            {
                cluster->points.push_back(cloud->points[index]);
            }

            // Compute convex hull for the cluster
            pcl::ConvexHull<pcl::PointXYZ> hull;
            pcl::PointCloud<pcl::PointXYZ>::Ptr hull_points(new pcl::PointCloud<pcl::PointXYZ>());
            hull.setInputCloud(cluster);
            hull.setDimension(2);  // Assuming a 2D convex hull
            // hull.setAlpha(convex_hull_tolerance);
            hull.reconstruct(*hull_points);

            // Create a Marker to represent the polygon
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = msg->header.frame_id;
            marker.header.stamp = this->get_clock()->now();
            marker.ns = "polygons";
            marker.id = cluster_id++;
            marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = 0.05;  // Line width

            // Set color for the marker
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            // Add the points of the polygon to the marker
            for (const auto& point : hull_points->points)
            {
                geometry_msgs::msg::Point p;
                p.x = point.x;
                p.y = point.y;
                p.z = point.z;
                marker.points.push_back(p);
            }

            // Close the polygon
            if (!marker.points.empty())
            {
                marker.points.push_back(marker.points.front());
            }
            std::cout << "Cluster " << cluster_id << " has " << marker.points.size() << " points" << std::endl;
            marker_array.markers.push_back(marker);
        }

        // Publish the MarkerArray
        std::cout << "Publishing " << marker_array.markers.size() << " markers" << std::endl;
        marker_pub_->publish(marker_array);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr downsampled_cloud_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudPolygonNode>());
    rclcpp::shutdown();
    return 0;
}
