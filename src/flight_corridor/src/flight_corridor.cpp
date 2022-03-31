#include <flight_corridor/flight_corridor.h>

namespace flight_corridor
{
    void FLIGHTCORRIDOR::init(ros::NodeHandle &nh){
        HaveOdom_ = false;
        HaveMap_ = false;

        nh.param("flight_corridor/goal_x", Goal_[0], 0.0); // 目标位置
        nh.param("flight_corridor/goal_y", Goal_[1], 0.0);
        nh.param("flight_corridor/goal_z", Goal_[2], 0.0);

        GridMap_.reset(new GridMap);
        GridMap_->initMap(nh);
        // AStar_.reset(new AStar);
        // AStar_->initGridMap(GridMap_, Eigen::Vector3i(100, 100, 100));

        ExecTimer_ = nh.createTimer(ros::Duration(0.01), &FLIGHTCORRIDOR::execSFCCallback, this);

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
        
        Path_ = FLIGHTCORRIDOR::getPath(Goal_);

        decomp_util.dilate(Path_);

        nav_msgs::Path PathMsg = DecompROS::vec_to_path(Path_);
        PathMsg.header.frame_id = "map";
        PathPub_.publish(PathMsg);

        // decomp_ros_msgs::EllipsoidArray EllipsoidMsg = DecompROS::ellipsoid_array_to_ros(decomp_util.get_ellipsoids());
        // EllipsoidMsg.header.frame_id = "map";
        // EllipsoidPub_.publish(EllipsoidMsg);

        auto Polyhedrons = decomp_util.get_polyhedrons();
        for(int i = 0; i < Polyhedrons.size() - 1; i++) {
            auto pt_inside = (Path_[i] + Path_[i+1]) / 2;
            for (unsigned int j = 0; j < Polyhedrons[i].hyperplanes().size(); j++) {
                auto n = Polyhedrons[i].hyperplanes()[j].n_;
                double b = Polyhedrons[i].hyperplanes()[j].p_.dot(n);
                if (n.dot(pt_inside) > b)
                    Polyhedrons[i].hyperplanes()[j].n_ = -Polyhedrons[i].hyperplanes()[j].n_;
            }
        }

        decomp_ros_msgs::PolyhedronArray PolyhedronMsg = DecompROS::polyhedron_array_to_ros(Polyhedrons);
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
        // std::vector<Eigen::Vector3i> PathID = this->AStarPlan(start, goal);
        

        // vec_Vec3f Path;
        // Eigen::Matrix<double, 3, 1> temp;
        // Eigen::Vector3d tempPos;
        // for (unsigned int i = 0; i < PathID.size(); i++){
        //     GridMap_->indexToPos(PathID[i], tempPos);
        //     temp << tempPos[0], tempPos[1], tempPos[2];
        //     Path.push_back(temp);
        // }
            
        return this->JPSPlan(start, goal);;
    }

    vec_Vec3f FLIGHTCORRIDOR::JPSPlan(Eigen::Vector3d Start, Eigen::Vector3d Goal){
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
        std::shared_ptr<JPS::VoxelMapUtil> map_util = std::make_shared<JPS::VoxelMapUtil>();
        // vector<int> map;
        std::vector<signed char> map(GridMap_->md_.occupancy_buffer_inflate_.begin(), GridMap_->md_.occupancy_buffer_inflate_.end());
        map_util->setMap(ori1, size1, map, GridMap_->mp_.resolution_);

        std::unique_ptr<JPSPlanner3D> planner_ptr(new JPSPlanner3D(true));
        planner_ptr->setMapUtil(map_util);
        planner_ptr->updateMap();
        bool valid_jps = planner_ptr->plan(start, goal, 1, true);
        return planner_ptr->getRawPath();
        // for(const auto& it: path_jps)
        //     std::cout << it.transpose() << std::endl;
    }

    std::vector<Eigen::Vector3i> FLIGHTCORRIDOR::AStarPlan(Eigen::Vector3d Start, Eigen::Vector3d Goal){
        Eigen::Vector3i StartIndex, GoalIndex, tempIndex;
        Eigen::Vector3d tempPos;
        // Vec3f start();
        // Vec3f goal();
        // std::unique_ptr<JPSPlanner3D> PlannerPtr(new JPSPlanner3D(true)); // Declare a planner
        // std::shared_ptr<JPS::VoxelMapUtil> MapUtil = std::make_shared<JPS::VoxelMapUtil>();
        // MapUtil->setMap(GridMap_->getOrigin(), GridMap_->getRegion, GridMap_->boundIndex ,GridMap_->getResolution)
        // PlannerPtr->setMapUtil(MapUtil); // Set collision checking function
        // PlannerPtr->updateMap();
        // bool ValidJPS = PlannerPtr->plan(start, goal, 1, true); // Plan from start to goal using JPS

        
        GridMap_->posToIndex(Start, StartIndex);
        GridMap_->posToIndex(Goal, GoalIndex);
        GridMap_->indexToPos(StartIndex, Start);
        GridMap_->indexToPos(GoalIndex, Goal);

        GridNodePtr startPtr = new GridNode(StartIndex, Start);
        GridNodePtr endPtr   = new GridNode(GoalIndex, Goal);

        std::multimap<double, GridNodePtr> openSet;
        openSet.clear();

        GridNodePtr currentPtr  = NULL;
        GridNodePtr neighborPtr = NULL;

        startPtr->gScore = 0;
        startPtr->fScore = getHeu(startPtr,endPtr) + startPtr->gScore;   
        startPtr->id = 1; 
        startPtr->coord = Start;
        startPtr->nodeMapIt = openSet.insert(make_pair(startPtr->fScore, startPtr));

        
        GridNodeMap_ = new GridNodePtr ** [GridMap_->mp_.map_voxel_num_[0]];
        for(int i = 0; i < GridMap_->mp_.map_voxel_num_[0]; i++){
            GridNodeMap_[i] = new GridNodePtr * [GridMap_->mp_.map_voxel_num_[1]];
            for(int j = 0; j < GridMap_->mp_.map_voxel_num_[1]; j++){
                GridNodeMap_[i][j] = new GridNodePtr [GridMap_->mp_.map_voxel_num_[2]];
                for( int k = 0; k < GridMap_->mp_.map_voxel_num_[2];k++){
                    Vector3i tmpIdx(i,j,k);
                    Vector3d pos;
                    GridMap_->indexToPos(tmpIdx, pos);
                    // Vector3d pos = gridIndex2coord(tmpIdx);
                    GridNodeMap_[i][j][k] = new GridNode(tmpIdx, pos);
                }
            }
        }


        int flag = 0;
        if(GridMap_->isKnownOccupied(GoalIndex))
            flag = 1;
        if (GoalIndex(0) < 0 || GoalIndex(0) > GridMap_->mp_.map_voxel_num_[0] || GoalIndex(1) < 0 || GoalIndex(1) > GridMap_->mp_.map_voxel_num_[1] || GoalIndex(2) < 0 || GoalIndex(2) > GridMap_->mp_.map_voxel_num_[2])
            flag = 1;
        vector<GridNodePtr> neighborPtrSets;
        vector<double> edgeCostSets;
        while (!openSet.empty())
        {
            if(flag)
                break;
            bool tie_breaker = false;
            std::multimap<double, GridNodePtr> tempSet;
            if (tie_breaker){
                tempSet.clear();
                GridNodePtr tempPtr = openSet.begin()->second;
                double fmin = openSet.begin()->first;
                auto it = openSet.begin();
                while (it != openSet.end()){
                    if(tempPtr->fScore != fmin) break;
                    tempSet.insert(make_pair(tempPtr->hScore, tempPtr));
                    it++;
                    tempPtr = it->second;
                }
                currentPtr = tempSet.begin()->second;
            }
            else{
                currentPtr = openSet.begin()->second;
            }

            openSet.erase(currentPtr->nodeMapIt);
            if (currentPtr->id == -1)
                continue;
            currentPtr->id = -1;
            if(currentPtr->index == GoalIndex){
                // ros::Time time_2 = ros::Time::now();
                // running_time = (time_2 - time_1).toSec()*1000.0;
                // terminatePtr = currentPtr;
                // ROS_WARN("[A*]{sucess}  Time in A*  is %f ms, path cost if %f m", (time_2 - time_1).toSec() * 1000.0, currentPtr->gScore * GridMap_->getResolution() );
                // double length = currentPtr->gScore * GridMap_->getResolution();

                std::vector<Eigen::Vector3i> path;
                std::vector<GridNodePtr> gridPath;
                /*
                *
                *
                STEP 8:  trace back from the curretnt nodePtr to get all nodes along the path
                please write your code below
                *      
                */
                // GridNodePtr currentPtr;
                Vector3i dir(0, 0, 0);
                Vector3i lastdir(-2, -2, -2);
                gridPath.push_back(currentPtr);
                path.push_back(currentPtr->index);
                // if(currentPtr->cameFrom==NULL)
                //     break;
                Vector3i indexCurr = currentPtr->index;
                Vector3i indexLast = currentPtr->cameFrom->index;
                // currentPtr = currentPtr->cameFrom;
                while (true){
                    dir = indexCurr - indexLast;
                    if(dir != lastdir){
                        lastdir = dir;
                        gridPath.push_back(currentPtr);
                        path.push_back(currentPtr->index);
                    }
                    currentPtr = currentPtr->cameFrom;
                    if(currentPtr->cameFrom==NULL){
                        gridPath.push_back(currentPtr);
                        path.push_back(currentPtr->index);
                        break;
                    }
                    indexCurr = currentPtr->index;
                    indexLast = currentPtr->cameFrom->index;
                }
                // reverse(gridPath.begin(),gridPath.end()); 
                reverse(path.begin(),path.end());

                return path;
                // return;
            }
            //get the succetion
            this->AstarGetSucc(currentPtr, neighborPtrSets, edgeCostSets);
            for (int i = 0; i < (int)neighborPtrSets.size(); i++){
                neighborPtr = neighborPtrSets[i];
                if (neighborPtr->id == 0){
                    neighborPtr->id = 1;
                    neighborPtr->gScore = currentPtr->gScore + edgeCostSets[i];
                    neighborPtr->hScore = getHeu(neighborPtr, endPtr);
                    neighborPtr->fScore = neighborPtr->hScore + neighborPtr->gScore;
                    neighborPtr->cameFrom = currentPtr;
                    neighborPtr->nodeMapIt = openSet.insert(make_pair(neighborPtr->fScore, neighborPtr));
                    continue;
                }
                else if(neighborPtr->id == 1 && neighborPtr->gScore > currentPtr->gScore+edgeCostSets[i]){
                    openSet.erase(neighborPtr->nodeMapIt);
                    neighborPtr->gScore = currentPtr->gScore + edgeCostSets[i];
                    neighborPtr->fScore = neighborPtr->hScore + neighborPtr->gScore;
                    neighborPtr->cameFrom = currentPtr;
                    neighborPtr->nodeMapIt = openSet.insert(make_pair(neighborPtr->fScore, neighborPtr));
                    continue;
                }
                else{//this node is in closed set
                    /*
                    *
                    please write your code below
                    *        
                    */
                    // cout << "closed list" << endl;
                    // cout << "id: " << neighborPtr->id << endl;
                    continue;
                }
            }
        }
        cout << "********failed********" << endl;
        
        std::vector<Eigen::Vector3i> PathIndex;
        return PathIndex;


        // 
        // PathID.push_back(StartIndex);

        // tempPos << 4.9, 0, 2.6;
        // GridMap_->posToIndex(tempPos, tempIndex);
        // PathID.push_back(tempIndex);
        // tempPos << 5.2, 0, 2.6;
        // GridMap_->posToIndex(tempPos, tempIndex);
        // PathID.push_back(tempIndex);

        // PathID.push_back(GoalIndex);
        
        
    }

    double FLIGHTCORRIDOR::getHeu(GridNodePtr node1, GridNodePtr node2){
        double dis;
        //Dijkstra
        dis = 0;

        //Manhattan
        dis = abs(node1->index(0) - node2->index(0)) +
            abs(node1->index(1) - node2->index(1)) +
            abs(node1->index(2) - node2->index(2));

        //Euclidean
        dis = sqrt(
            (node1->index(0) - node2->index(0)) * (node1->index(0) - node2->index(0)) +
            (node1->index(1) - node2->index(1)) * (node1->index(1) - node2->index(1)) +
            (node1->index(2) - node2->index(2)) * (node1->index(2) - node2->index(2)));

        //Diagonal
        double dx, dy, dz, dmin, d1, d2;
        dx = abs(node1->index(0) - node2->index(0));
        dy = abs(node1->index(1) - node2->index(1));
        dz = abs(node1->index(2) - node2->index(2));
        dmin = min(min(dx, dy), dz);
        if(dmin==dx){
            d1 = dy;
            d2 = dz;
        }
        else if (dmin == dy){
            d1 = dx;
            d2 = dz;
        }
        else{
            d1 = dx;
            d2 = dy;
        }
        // dis = sqrt(3) * dmin + sqrt(2) * min(abs(d1 - dmin), abs(d2 - dmin)) + abs(d1 - d2);

        return dis;
    }

    void FLIGHTCORRIDOR::AstarGetSucc(GridNodePtr currentPtr, vector<GridNodePtr> & neighborPtrSets, vector<double> & edgeCostSets){   
        neighborPtrSets.clear();
        edgeCostSets.clear();
        int count = 0;
        GridNodePtr neighborPtr;
        for (int i = -1; i < 2;i++){
            for (int j = -1; j < 2; j++){
                for (int k = -1; k < 2;k++){
                    if(i==0&&j==0&&k==0)
                        continue;
                    Vector3i tmpIdx(currentPtr->index(0) + i, currentPtr->index(1) + j, currentPtr->index(2) + k);
                    if (GridMap_->isKnownOccupied(tmpIdx)) continue;
                    if (tmpIdx(0) < 0 || tmpIdx(0) >= GridMap_->mp_.map_voxel_num_[0] || tmpIdx(1) < 0 || tmpIdx(1) >= GridMap_->mp_.map_voxel_num_[1] || tmpIdx(2) < 0 || tmpIdx(2) >= GridMap_->mp_.map_voxel_num_[2]) continue;
                    neighborPtr = GridNodeMap_[tmpIdx(0)][tmpIdx(1)][tmpIdx(2)];
                    // neighborPtr = 
                    if (neighborPtr->id == -1) continue;
                    neighborPtrSets.push_back(neighborPtr);
                    edgeCostSets.push_back(sqrt(
                        (tmpIdx(0) - currentPtr->index(0)) * (tmpIdx(0) - currentPtr->index(0)) +
                        (tmpIdx(1) - currentPtr->index(1)) * (tmpIdx(1) - currentPtr->index(1)) +
                        (tmpIdx(2) - currentPtr->index(2)) * (tmpIdx(2) - currentPtr->index(2))));
                    count++;
                }
            }
        }
    }
}
