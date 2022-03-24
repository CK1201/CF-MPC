#include <occupancy_map/occupancy_map.h>

namespace costmap
{
    void COSTMAP::init(ros::NodeHandle &nh){
        HaveOdom_ = false;
        OdomSub_ = nh.subscribe<nav_msgs::Odometry>("/hummingbird/ground_truth/odometry", 1, &COSTMAP::QuadOdomCallback, this);
        PointCloudRawSub_ = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/color/points", 1, &COSTMAP::PointCloudCallback, this);

        PointCloudPub_ = nh.advertise<sensor_msgs::PointCloud>("/obstacle", 1);
    }

    void COSTMAP::QuadOdomCallback(const nav_msgs::Odometry::ConstPtr &msg){
        HaveOdom_ = true;
        QuadOdom_ = *msg;
    }

    void COSTMAP::PointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
        if(!HaveOdom_)
            return;
        PointCloudRaw_ = *msg;
        sensor_msgs::PointCloud cloud;
        bool success = sensor_msgs::convertPointCloud2ToPointCloud(PointCloudRaw_, cloud);
        cloud.header.frame_id = "world";
        

        // PointCloudPub_.publish(cloud);

        // pcl::PointCloud<pcl::PointXYZ> PCLcloud;
        // pcl::fromROSMsg (*msg, PCLcloud);
        // sensor_msgs::PointCloud cloud;
        // bool success = sensor_msgs::convertPointCloud2ToPointCloud(*msg, cloud);
        // PCPub.publish(cloud);
        // printf ("Cloud: width = %d, height = %d\n", msg->width, msg->height);
        // BOOST_FOREACH (const pcl::PointXYZ& pt, msg->points)
        // printf ("\t(%f, %f, %f)\n", pt.x, pt.y, pt.z);
        // sensor_msgs::PointCloud2 cloud;
        // pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // pcl::PCLPointCloud2 pcl_pc2;
        // pcl_conversions::toPCL(*msg, pcl_pc2);
        // pcl::fromPCLPointCloud2(pcl_pc2,*out_cloud);

        // pcl::PointCloud<pcl::PointXYZ> cloud_pcl_xyzi;
        // pcl::fromROSMsg(msg, cloud_pcl_xyz);

        // sensor_msgs::PointCloud cloud;
        // cloud.points.resize(msg->width);
        // sensor_msgs::convertPointCloud2ToPointCloud
        // for(int i = 0; i < cloud_pcl_xyzi.points.size(); i++){
        //     std::cout << cloud_pcl_xyzi.points[i] << std::endl;
            // cloud.points[i].x = cloud_pcl_xyzi.points[i, 0];
            // cloud.points[i].y = cloud_pcl_xyzi.points[i](1);
            // cloud.points[i].z = cloud_pcl_xyzi.points[i](2);
    }
}
