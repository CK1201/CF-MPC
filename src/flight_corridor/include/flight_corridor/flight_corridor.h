#ifndef _FLIGHT_CORRIDOT_H_
#define _FLIGHT_CORRIDOT_H_


#include <ros/ros.h>
#include <Eigen/Eigen>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <nav_msgs/Odometry.h>
#include <octomap_ros/conversions.h>
#include <octomap/octomap.h>
// #include <octomap/ColorOcTree.h>
#include <octomap_msgs/Octomap.h>
// #include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/conversions.h>

using namespace std;
using namespace Eigen;

namespace flight_corridor
{
    class FLIGHTCORRIDOR{
    private:
        bool HaveOdom_;
        /* parameters */
        sensor_msgs::PointCloud2 PointCloudRaw_, Map_;
        nav_msgs::Odometry QuadOdom_;
        octomap_msgs::Octomap OctoMap_;
        octomap::AbstractOcTree* AbstractOcTree_;
        octomap::OcTree* OcTree_;

        /* ROS utils */
        // ros::Timer exec_timer_;
        ros::Subscriber OdomSub_, PointCloudRawSub_, OctoMapSub_;
        ros::Publisher PointCloudPub_;
        // tf::TransformBroadcaster br_;
        // tf::Transform transform_map2base_, transform_base2livox_;

        /* ROS functions */
        void QuadOdomCallback(const nav_msgs::Odometry::ConstPtr &msg);
        void PointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void OctoMapCallback(const octomap_msgs::Octomap::ConstPtr &msg);

    public:
        FLIGHTCORRIDOR(/* args */){}
        ~FLIGHTCORRIDOR(){}
        void init(ros::NodeHandle &nh);
    };

}

#endif