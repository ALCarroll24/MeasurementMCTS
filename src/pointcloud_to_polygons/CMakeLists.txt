cmake_minimum_required(VERSION 3.8)
project(pointcloud_to_polygons)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(pointcloud_array_msgs REQUIRED)

# add_executable(pointcloud_to_corners_ransac_node src/pointcloud_to_corners_ransac_node.cpp)
add_executable(pointcloud_clustering_node src/pointcloud_clustering_node.cpp)
add_executable(pointcloud_to_object_points_node src/pointcloud_to_object_points_node.cpp)
add_executable(pointcloud_segmentation_node src/pointcloud_segmentation_node.cpp)

# ament_target_dependencies(pointcloud_to_corners_ransac_node
#   rclcpp 
#   sensor_msgs 
#   pcl_conversions 
#   pcl_ros 
#   grid_map_ros
#   Eigen3
# )

ament_target_dependencies(pointcloud_clustering_node
  rclcpp 
  sensor_msgs 
  pcl_conversions 
  pcl_ros 
  grid_map_ros
  Eigen3
)

ament_target_dependencies(pointcloud_to_object_points_node
  rclcpp 
  sensor_msgs 
  pcl_conversions 
  pcl_ros 
  grid_map_ros
  Eigen3
)

ament_target_dependencies(pointcloud_segmentation_node
  rclcpp 
  sensor_msgs 
  pcl_conversions 
  pcl_ros 
  grid_map_ros
  Eigen3
  pointcloud_array_msgs
)

# Install targets
# install(TARGETS pointcloud_to_corners_ransac_node
#   DESTINATION lib/${PROJECT_NAME}
# )
install(TARGETS pointcloud_clustering_node
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS pointcloud_to_object_points_node
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS pointcloud_segmentation_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
