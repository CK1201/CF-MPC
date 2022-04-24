#include <flight_corridor/flight_corridor.h>

namespace flight_corridor
{
    void FLIGHTCORRIDOR::init(ros::NodeHandle &nh){
        HaveOdom_ = false;
        HaveMap_ = false;
        HavePoly_ = false;
        HaveMapVis_ = false;
        Eigen::Vector3d bounding_box;
        last_poly_time_ = ros::Time::now().toSec() - 200;

        nh.param("flight_corridor/bounding_box_x", bounding_box[0], 1.0);
        nh.param("flight_corridor/bounding_box_y", bounding_box[1], 1.0);
        nh.param("flight_corridor/bounding_box_z", bounding_box[2], 1.0);
        nh.param("flight_corridor/goal_x", Goal_[0], 0.0); // 目标位置
        nh.param("flight_corridor/goal_y", Goal_[1], 0.0);
        nh.param("flight_corridor/goal_z", Goal_[2], 0.0);
        nh.param("flight_corridor/use_jps", UseJPS_, true);
        nh.param("flight_corridor/use_prior", UsePrior_, true);
        nh.param("flight_corridor/offset", offset_, 0.0);
        

        GridMap_.reset(new GridMap);
        GridMap_->initMap(nh);
        // AStar_.reset(new AStar);
        // AStar_->initGridMap(GridMap_, Eigen::Vector3i(100, 100, 100));

        ExecTimer_ = nh.createTimer(ros::Duration(0.05), &FLIGHTCORRIDOR::execSFCCallback, this);

        OdomSub_ = nh.subscribe<nav_msgs::Odometry>("/hummingbird/ground_truth/odometry", 1, &FLIGHTCORRIDOR::QuadOdomCallback, this);
        OctoMapCenterSub_ = nh.subscribe<sensor_msgs::PointCloud2>("/octomap_point_cloud_centers", 1, &FLIGHTCORRIDOR::OctoMapCenterCallback, this);
        // GridMapInfSub_ = nh.subscribe<sensor_msgs::PointCloud2>("/grid_map/occupancy_inflate", 1, &FLIGHTCORRIDOR::GridMapInfCallback, this);
        GridMapVisSub_ = nh.subscribe<sensor_msgs::PointCloud2>("/grid_map/vis", 1, &FLIGHTCORRIDOR::GridMapVisCallback, this);

        PointCloudPub_ = nh.advertise<sensor_msgs::PointCloud>("/obstacle", 1);
        PathPub_ = nh.advertise<nav_msgs::Path>("path", 1, true);
        EllipsoidPub_ = nh.advertise<decomp_ros_msgs::EllipsoidArray>("ellipsoid_array", 1, true);
        PolyhedronPub_ = nh.advertise<decomp_ros_msgs::PolyhedronArray>("polyhedron_array", 1, true);

        decomp_util.set_local_bbox(Vec3f(bounding_box[0], bounding_box[1], bounding_box[2])); // bounding box x, y, z(in meter)

    }

    void FLIGHTCORRIDOR::execSFCCallback(const ros::TimerEvent &e){
        if (!HaveOdom_ || !HaveMap_)
            return;
        bool need_new_poly = false;
        if(UsePrior_){
            need_new_poly = true;
        }else{
            if(ros::Time::now().toSec() - last_poly_time_ > 50)
            need_new_poly = true;
            else if (GridMapVis_.points.size() - pt_num_ > 20){
                pt_num_ = GridMapVis_.points.size();
                need_new_poly = true;
                // cout << "need: " << need_new_poly << endl;
            }
        }
        
        // if (GridMapVis_.points.size() < pt_num_)
        //     pt_num_ = GridMapVis_.points.size();
        // else if (checkCollision()){
        //     need_new_poly = true;
        //     // cout << "need: " << need_new_poly << endl;
        // }

        // cout << "need: " << need_new_poly << endl;
        // PathPub_.publish(PathMsg_);
        // PolyhedronPub_.publish(PolyhedronMsg_);
        // EllipsoidPub_.publish(EllipsoidMsg_);
        if (need_new_poly){
            last_poly_time_ = ros::Time::now().toSec();
            Eigen::Vector3d start(QuadOdom_.pose.pose.position.x, QuadOdom_.pose.pose.position.y, QuadOdom_.pose.pose.position.z);
            Eigen::Vector3i id1, id2;
            GridMap_->posToIndex(Goal_, id1);
            GridMap_->posToIndex(start, id2);
            if (GridMap_->isKnownOccupied(id1) || GridMap_->isKnownOccupied(id2)){
                if(GridMap_->isKnownOccupied(id1))
                    cout << "Goal is not collisionfree" << endl;
                else
                    cout << "Start is not collisionfree" << endl;
                return;
            }

            double time = ros::Time::now().toSec();
            Path_ = Plan(start, Goal_);
            // cout << "getPath: " << ros::Time::now().toSec() - time << endl;
            decomp_util.dilate(Path_);
            Polyhedrons_ = decomp_util.get_polyhedrons();
            // cout << "get_polyhedrons: " << ros::Time::now().toSec() - time << endl;
            for(unsigned char i = 0; i < Polyhedrons_.size() - 1; i++) {
                auto pt_inside = (Path_[i] + Path_[i+1]) / 2;
                for (unsigned int j = 0; j < Polyhedrons_[i].hyperplanes().size(); j++) {
                    auto n = Polyhedrons_[i].hyperplanes()[j].n_;
                    double b = Polyhedrons_[i].hyperplanes()[j].p_.dot(n);
                    if (n.dot(pt_inside) > b)
                        Polyhedrons_[i].hyperplanes()[j].n_ = -Polyhedrons_[i].hyperplanes()[j].n_;
                    Polyhedrons_[i].hyperplanes()[j].p_ = Polyhedrons_[i].hyperplanes()[j].p_ - Polyhedrons_[i].hyperplanes()[j].n_ / Polyhedrons_[i].hyperplanes()[j].n_.norm() * offset_;
                }
            }
            // cout << "redirect: " << ros::Time::now().toSec() - time << endl;

            PathMsg_ = DecompROS::vec_to_path(Path_);
            PathMsg_.header.frame_id = "map";
            PathPub_.publish(PathMsg_);

            PolyhedronMsg_ = DecompROS::polyhedron_array_to_ros(Polyhedrons_);
            PolyhedronMsg_.header.frame_id = "map";
            PolyhedronPub_.publish(PolyhedronMsg_);

            EllipsoidMsg_ = DecompROS::ellipsoid_array_to_ros(decomp_util.get_ellipsoids());
            EllipsoidMsg_.header.frame_id = "map";
            EllipsoidPub_.publish(EllipsoidMsg_);

            HavePoly_ = true;
        }
    }

    void FLIGHTCORRIDOR::QuadOdomCallback(const nav_msgs::Odometry::ConstPtr &msg){
        HaveOdom_ = true;
        QuadOdom_ = *msg;
    }

    void FLIGHTCORRIDOR::OctoMapCenterCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
        HaveMap_ = true;
        // OctoMapCenter_ = ;
        sensor_msgs::PointCloud ObstaclePointCloud;
        sensor_msgs::convertPointCloud2ToPointCloud(*msg, ObstaclePointCloud);
        decomp_util.set_obs(DecompROS::cloud_to_vec(ObstaclePointCloud));
    }

    void FLIGHTCORRIDOR::GridMapInfCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
        HaveMap_ = true;
        sensor_msgs::PointCloud ObstaclePointCloud;
        sensor_msgs::convertPointCloud2ToPointCloud(*msg, ObstaclePointCloud);
        decomp_util.set_obs(DecompROS::cloud_to_vec(ObstaclePointCloud));
    }

/**
 * This function is a callback function that is called when a new message is received on the topic
 * "/grid_map_vis". The message is converted to a point cloud and stored in the variable GridMapVis_
 * 
 * @param msg The message that contains the point cloud data.
 */
    void FLIGHTCORRIDOR::GridMapVisCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
        HaveMapVis_ = true;
        sensor_msgs::convertPointCloud2ToPointCloud(*msg, GridMapVis_);
    }

    vec_Vec3f FLIGHTCORRIDOR::Plan(Eigen::Vector3d Start, Eigen::Vector3d Goal){
        // Eigen::Vector3i StartIndex, GoalIndex, tempIndex;
        Vec3f start(Start);
        Vec3f goal(Goal);
        // GridMap_->posToIndex(Start, StartIndex);
        // GridMap_->posToIndex(Goal, GoalIndex);
        Eigen::Vector3d ori;
        Eigen::Vector3d size;
        GridMap_->getRegion(ori, size);
        Eigen::Matrix<double, 3, 1> ori1(ori);
        Eigen::Matrix<int, 3, 1> size1(GridMap_->mp_.map_voxel_num_);
        // cout << size << endl;
        std::shared_ptr<JPS::VoxelMapUtil> map_util = std::make_shared<JPS::VoxelMapUtil>();
        // vector<int> map;
        // double time = ros::Time::now().toSec();
        std::vector<signed char> map(GridMap_->md_.occupancy_buffer_inflate_.begin(), GridMap_->md_.occupancy_buffer_inflate_.end());
        // cout << "store map: " << ros::Time::now().toSec() - time<< endl;
        map_util->setMap(ori1, size1, map, GridMap_->mp_.resolution_);
        std::unique_ptr<JPSPlanner3D> planner_ptr(new JPSPlanner3D(true));
        planner_ptr->setMapUtil(map_util);
        planner_ptr->updateMap();
        // cout << "upadte map: " << ros::Time::now().toSec() - time<< endl;
        planner_ptr->plan(start, goal, 1, UseJPS_);
        // cout << "plan: " << ros::Time::now().toSec() - time<< endl;
        return planner_ptr->getRawPath();
    }

    bool FLIGHTCORRIDOR::checkCollision(){
        if(!HaveMapVis_)
            return false;
        int cnt = 0;
        // cout << GridMapVis_.points.size() << endl;
        for (unsigned int i = 0; i < GridMapVis_.points.size(); ++i){
            Vec3f pt(GridMapVis_.points[i].x, GridMapVis_.points[i].y, GridMapVis_.points[i].z);
            for(unsigned int k = 0; k < Polyhedrons_.size() - 1; k++) {
                bool pt_in = true;
                for (unsigned int j = 0; j < Polyhedrons_[k].hyperplanes().size(); j++) {
                    auto n = Polyhedrons_[k].hyperplanes()[j].n_;
                    double b = Polyhedrons_[k].hyperplanes()[j].p_.dot(n);
                    if (n.dot(pt) > b){
                        pt_in = false;
                        break;
                    }
                }
                if(pt_in){
                    cnt++;
                    if(cnt>5){
                        return true;
                    }
                    break;
                }
            }
        }
        // // cout << "do not cal new poly" << endl;
        return false;
    }
}
