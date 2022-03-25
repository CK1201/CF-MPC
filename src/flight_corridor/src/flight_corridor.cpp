#include <flight_corridor/flight_corridor.h>

namespace flight_corridor
{
    void FLIGHTCORRIDOR::init(ros::NodeHandle &nh){
        HaveOdom_ = false;
        OdomSub_ = nh.subscribe<nav_msgs::Odometry>("/hummingbird/ground_truth/odometry", 1, &FLIGHTCORRIDOR::QuadOdomCallback, this);
        PointCloudRawSub_ = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/color/points", 1, &FLIGHTCORRIDOR::PointCloudCallback, this);
        OctoMapSub_ = nh.subscribe<octomap_msgs::Octomap>("/octomap_full", 1, &FLIGHTCORRIDOR::OctoMapCallback, this);

        PointCloudPub_ = nh.advertise<sensor_msgs::PointCloud>("/obstacle", 1);
    }

    void FLIGHTCORRIDOR::QuadOdomCallback(const nav_msgs::Odometry::ConstPtr &msg){
        HaveOdom_ = true;
        QuadOdom_ = *msg;
    }

    void FLIGHTCORRIDOR::PointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
        if(!HaveOdom_)
            return;
        PointCloudRaw_ = *msg;
        sensor_msgs::PointCloud cloud;
        bool success = sensor_msgs::convertPointCloud2ToPointCloud(PointCloudRaw_, cloud);
        cloud.header.frame_id = "map";
        // PointCloudPub_.publish(cloud);

    }

    void FLIGHTCORRIDOR::OctoMapCallback(const octomap_msgs::Octomap::ConstPtr &msg){
        // OctoMap_ = *msg;
        // AbstractOcTree_ = octomap_msgs::msgToMap(*msg);
        octomap::OcTree *octree = new octomap::OcTree(msg->resolution);
        std::stringstream datastream;
        if (msg->data.size() > 0){
            datastream.write((const char*) &msg->data[0], msg->data.size());
            if (msg->binary)
                octree->readBinaryData(datastream);
            else
                octree->readData(datastream);
        }
        OcTree_ = octree;
        // cout << msg->id << endl;
        // cout << msg->binary << endl;
        // cout << msg->data.size() << endl;
        cout << OcTree_->isNodeOccupied << endl;
        
        // octomap_ros::pointsOctomapToPointCloud2()
        // OcTree_->
        // getOccupied()
        
    }
}
