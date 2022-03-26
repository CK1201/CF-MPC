#ifndef _FLIGHT_CORRIDOT_H_
#define _FLIGHT_CORRIDOT_H_

#include <iostream>
#include <vector>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <nav_msgs/Odometry.h>
#include <octomap_ros/conversions.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/conversions.h>

#include <decomp_ros_utils/data_ros_utils.h>
#include <decomp_util/ellipsoid_decomp.h>

#include <plan_env/grid_map.h>

#include <path_searching/dyn_a_star.h>

using namespace std;
using namespace Eigen;

namespace flight_corridor
{
    class FLIGHTCORRIDOR{
    private:
        bool HaveOdom_, HaveMap_;
        /* parameters */
        vec_Vec3f Obstacle_, Path_;
        sensor_msgs::PointCloud ObstaclePointCloud_;
        sensor_msgs::PointCloud2 PointCloudRaw_, OctoMapCenter_;
        nav_msgs::Odometry QuadOdom_;
        octomap::OcTree* OctoMap_;
        EllipsoidDecomp3D decomp_util;
        GridMap::Ptr GridMap_;
        AStar::Ptr AStar_;

        /* ROS utils */
        ros::Timer ExecTimer_;
        ros::Subscriber OdomSub_, PointCloudRawSub_, OctoMapSub_, OctoMapCenterSub_;
        ros::Publisher PointCloudPub_, PathPub_, EllipsoidPub_, PolyhedronPub_;

        /* ROS functions */
        void execSFCCallback(const ros::TimerEvent &e);
        void QuadOdomCallback(const nav_msgs::Odometry::ConstPtr &msg);
        void PointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void OctoMapCallback(const octomap_msgs::Octomap::ConstPtr &msg);
        void OctoMapCenterCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

        /* Other functions */
        vec_Vec3f getPath();

    public:
        FLIGHTCORRIDOR(/* args */){}
        ~FLIGHTCORRIDOR(){}
        void init(ros::NodeHandle &nh);
    };

}

#endif