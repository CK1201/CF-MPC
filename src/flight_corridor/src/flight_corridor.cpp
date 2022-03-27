#include <flight_corridor/flight_corridor.h>

namespace flight_corridor
{
    void FLIGHTCORRIDOR::init(ros::NodeHandle &nh){
        HaveOdom_ = false;
        HaveMap_ = false;

        GridMap_.reset(new GridMap);
        GridMap_->initMap(nh);
        // AStar_.reset(new AStar);
        // AStar_->initGridMap(GridMap_, Eigen::Vector3i(100, 100, 100));

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
        Eigen::Vector3d goal(10, 0, 3);
        Path_ = FLIGHTCORRIDOR::getPath(goal);

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
        // sensor_msgs::PointCloud cloud;
        // bool success = sensor_msgs::convertPointCloud2ToPointCloud(PointCloudRaw_, cloud);
        // cloud.header.frame_id = "map";
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
        // FLIGHTCORRIDOR::updateGradMap();

        // pcl::PointCloud<pcl::PointXYZ> PCLcloud;
        // pcl::fromROSMsg(OctoMapCenter_, PCLcloud);
    }

    vec_Vec3f FLIGHTCORRIDOR::getPath(Eigen::Vector3d goal){
        Eigen::Vector3d start(QuadOdom_.pose.pose.position.x, QuadOdom_.pose.pose.position.y, QuadOdom_.pose.pose.position.z);
        std::vector<Eigen::Vector3i> PathID = this->AStar(start, goal);
        // auto PathID = this->JPS(start, goal);

        vec_Vec3f Path;
        Eigen::Matrix<double, 3, 1> temp;
        Eigen::Vector3d tempPos;
        for (unsigned int i = 0; i < PathID.size(); i++){
            GridMap_->indexToPos(PathID[i], tempPos);
            temp << tempPos[0], tempPos[1], tempPos[2];
            Path.push_back(temp);
        }
            
        return Path;
    }

    std::vector<Eigen::Vector3i> FLIGHTCORRIDOR::AStar(Eigen::Vector3d Start, Eigen::Vector3d Goal){
        Eigen::Vector3i StartID, GoalID, tempID;
        Eigen::Vector3d tempPos;
        GridMap_->posToIndex(Start, StartID);
        GridMap_->posToIndex(Goal, GoalID);

        std::vector<Eigen::Vector3i> PathID;
        PathID.push_back(StartID);

        tempPos << 4.9, 0, 2.6;
        GridMap_->posToIndex(tempPos, tempID);
        PathID.push_back(tempID);
        tempPos << 5.2, 0, 2.6;
        GridMap_->posToIndex(tempPos, tempID);
        PathID.push_back(tempID);

        PathID.push_back(GoalID);
        return PathID;
    }
}
