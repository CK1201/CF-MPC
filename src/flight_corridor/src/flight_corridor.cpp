#include <flight_corridor/flight_corridor.h>

namespace flight_corridor
{
    void FLIGHTCORRIDOR::init(ros::NodeHandle &nh){
        HaveOdom_ = false;
        HaveMap_ = false;

        GridMap_.reset(new GridMap);
        GridMap_->initMap(nh);
        AStar_.reset(new AStar);
        AStar_->initGridMap(GridMap_, Eigen::Vector3i(100, 100, 100));

        ExecTimer_ = nh.createTimer(ros::Duration(0.1), &FLIGHTCORRIDOR::execSFCCallback, this);

        OdomSub_ = nh.subscribe<nav_msgs::Odometry>("/hummingbird/ground_truth/odometry", 1, &FLIGHTCORRIDOR::QuadOdomCallback, this);
        // PointCloudRawSub_ = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/color/points", 1, &FLIGHTCORRIDOR::PointCloudCallback, this);
        // OctoMapSub_ = nh.subscribe<octomap_msgs::Octomap>("/octomap_full", 1, &FLIGHTCORRIDOR::OctoMapCallback, this);
        OctoMapCenterSub_ = nh.subscribe<sensor_msgs::PointCloud2>("/octomap_point_cloud_centers", 1, &FLIGHTCORRIDOR::OctoMapCenterCallback, this);

        PointCloudPub_ = nh.advertise<sensor_msgs::PointCloud>("/obstacle", 1);
        PathPub_ = nh.advertise<nav_msgs::Path>("path", 1, true);
        EllipsoidPub_ = nh.advertise<decomp_ros_msgs::EllipsoidArray>("ellipsoid_array", 1, true);
        PolyhedronPub_ = nh.advertise<decomp_ros_msgs::PolyhedronArray>("polyhedron_array", 1, true);

        decomp_util.set_local_bbox(Vec3f(1, 1, 1)); // bounding box x, y, z(in meter)

    }

    void FLIGHTCORRIDOR::execSFCCallback(const ros::TimerEvent &e){
        if (!HaveOdom_ || !HaveMap_)
            return;

        Path_ = FLIGHTCORRIDOR::getPath();

        decomp_util.dilate(Path_);

        nav_msgs::Path PathMsg = DecompROS::vec_to_path(Path_);
        PathMsg.header.frame_id = "map";
        PathPub_.publish(PathMsg);

        // decomp_ros_msgs::EllipsoidArray EllipsoidMsg = DecompROS::ellipsoid_array_to_ros(decomp_util.get_ellipsoids());
        // EllipsoidMsg.header.frame_id = "map";
        // EllipsoidPub_.publish(EllipsoidMsg);

        // auto Polyhedron = decomp_util.get_polyhedrons();
        decomp_ros_msgs::PolyhedronArray PolyhedronMsg = DecompROS::polyhedron_array_to_ros(decomp_util.get_polyhedrons());
        PolyhedronMsg.header.frame_id = "map";
        PolyhedronPub_.publish(PolyhedronMsg);
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
        
        // sensor_msgs::PointCloud2 msg2;
        // pcl::toROSMsg(cloud, msg2);
    }

    void FLIGHTCORRIDOR::OctoMapCallback(const octomap_msgs::Octomap::ConstPtr &msg){
        OctoMap_ = (octomap::OcTree*)octomap_msgs::msgToMap(*msg);
        // OctoMap_->search()
        // cout << OcTree_->getResolution() << endl;
        // sensor_msgs::PointCloud2 cloud;
        // octomap_ros::pointsOctomapToPointCloud2(OcTree_->getOccupied(), cloud);
    }

    void FLIGHTCORRIDOR::OctoMapCenterCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
        HaveMap_ = true;
        OctoMapCenter_ = *msg;
        sensor_msgs::convertPointCloud2ToPointCloud(OctoMapCenter_, ObstaclePointCloud_);
        Obstacle_ = DecompROS::cloud_to_vec(ObstaclePointCloud_);
        decomp_util.set_obs(Obstacle_);
        

        // pcl::PointCloud<pcl::PointXYZ> PCLcloud;
        // pcl::fromROSMsg(OctoMapCenter_, PCLcloud);
    }

    vec_Vec3f FLIGHTCORRIDOR::getPath(){
        vec_Vec3f path;
        Eigen::Matrix<double, 3, 1> start, goal, temp;
        start << QuadOdom_.pose.pose.position.x, QuadOdom_.pose.pose.position.y, QuadOdom_.pose.pose.position.z;
        goal << 10, 0, 2;
        path.push_back(start);
        temp << 4.9, 0, 2.6;
        path.push_back(temp);
        temp << 5.2, 0, 2.6;
        path.push_back(temp);
        path.push_back(goal);
        return path;
    }
}
