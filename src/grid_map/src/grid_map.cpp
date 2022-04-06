#include "grid_map/grid_map.h"

// #define current_img_ md_.depth_image_[image_cnt_ & 1]
// #define last_img_ md_.depth_image_[!(image_cnt_ & 1)]

void GridMap::initMap(ros::NodeHandle &nh)
{
  node_ = nh;

  /* get parameter */
  double x_size, y_size, z_size;
  node_.param("grid_map/resolution", mp_.resolution_, -1.0); // 栅格大小
  node_.param("grid_map/map_size_x", x_size, -1.0); // 地图大小， 米
  node_.param("grid_map/map_size_y", y_size, -1.0);
  node_.param("grid_map/map_size_z", z_size, -1.0);
  node_.param("grid_map/local_update_range_x", mp_.local_update_range_(0), -1.0);
  node_.param("grid_map/local_update_range_y", mp_.local_update_range_(1), -1.0);
  node_.param("grid_map/local_update_range_z", mp_.local_update_range_(2), -1.0);
  node_.param("grid_map/obstacles_inflation", mp_.obstacles_inflation_, -1.0); // 栅格膨胀大小
  node_.param("grid_map/frame_id", mp_.frame_id_, string("world")); //地图坐标系
  node_.param("grid_map/local_map_margin", mp_.local_map_margin_, 1); //
  node_.param("grid_map/ground_height", mp_.ground_height_, 1.0); //地面高度

  node_.param("grid_map/fx", mp_.fx_, -1.0); //相机内参
  node_.param("grid_map/fy", mp_.fy_, -1.0);
  node_.param("grid_map/cx", mp_.cx_, -1.0);
  node_.param("grid_map/cy", mp_.cy_, -1.0);

  node_.param("grid_map/use_depth_filter", mp_.use_depth_filter_, true);
  node_.param("grid_map/depth_filter_tolerance", mp_.depth_filter_tolerance_, -1.0);
  node_.param("grid_map/depth_filter_maxdist", mp_.depth_filter_maxdist_, -1.0);
  node_.param("grid_map/depth_filter_mindist", mp_.depth_filter_mindist_, -1.0);
  node_.param("grid_map/depth_filter_margin", mp_.depth_filter_margin_, -1);
  node_.param("grid_map/k_depth_scaling_factor", mp_.k_depth_scaling_factor_, -1.0);
  node_.param("grid_map/skip_pixel", mp_.skip_pixel_, -1);

  node_.param("grid_map/p_hit", mp_.p_hit_, 0.70);
  node_.param("grid_map/p_miss", mp_.p_miss_, 0.35);
  node_.param("grid_map/p_min", mp_.p_min_, 0.12);
  node_.param("grid_map/p_max", mp_.p_max_, 0.97);
  node_.param("grid_map/p_occ", mp_.p_occ_, 0.80);
  node_.param("grid_map/min_ray_length", mp_.min_ray_length_, -0.1);
  node_.param("grid_map/max_ray_length", mp_.max_ray_length_, -0.1);

  node_.param("grid_map/visualization_truncate_height", mp_.visualization_truncate_height_, -0.1); // 大于此高度的点云不显示在rviz上（用于过滤屋顶，设置为虚拟屋顶-0.1）
  node_.param("grid_map/virtual_ceil_height", mp_.virtual_ceil_height_, -0.1); // 虚拟屋顶
  node_.param("grid_map/virtual_ceil_yp", mp_.virtual_ceil_yp_, -0.1);
  node_.param("grid_map/virtual_ceil_yn", mp_.virtual_ceil_yn_, -0.1);

  node_.param("grid_map/show_occ_time", mp_.show_occ_time_, false);
  node_.param("grid_map/pose_type", mp_.pose_type_, 1);

  node_.param("grid_map/odom_depth_timeout", mp_.odom_depth_timeout_, 1.0);

  mp_.resolution_inv_ = 1 / mp_.resolution_;
  mp_.map_origin_ = Eigen::Vector3d(-x_size / 2.0, -y_size / 2.0, mp_.ground_height_);
  mp_.map_size_ = Eigen::Vector3d(x_size, y_size, z_size);

  mp_.prob_hit_log_ = logit(mp_.p_hit_);
  mp_.prob_miss_log_ = logit(mp_.p_miss_);
  mp_.clamp_min_log_ = logit(mp_.p_min_);
  mp_.clamp_max_log_ = logit(mp_.p_max_);
  mp_.min_occupancy_log_ = logit(mp_.p_occ_);
  mp_.unknown_flag_ = 0.01;

  cout << "hit: " << mp_.prob_hit_log_ << endl;
  cout << "miss: " << mp_.prob_miss_log_ << endl;
  cout << "min log: " << mp_.clamp_min_log_ << endl;
  cout << "max: " << mp_.clamp_max_log_ << endl;
  cout << "thresh log: " << mp_.min_occupancy_log_ << endl;

  for (int i = 0; i < 3; ++i)
    mp_.map_voxel_num_(i) = ceil(mp_.map_size_(i) / mp_.resolution_);

  mp_.map_min_boundary_ = mp_.map_origin_;
  mp_.map_max_boundary_ = mp_.map_origin_ + mp_.map_size_;

  // initialize data buffers

  int buffer_size = mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2);

  md_.occupancy_buffer_ = vector<double>(buffer_size, mp_.clamp_min_log_ - mp_.unknown_flag_);
  md_.occupancy_buffer_inflate_ = vector<char>(buffer_size, 0);

  md_.count_hit_and_miss_ = vector<short>(buffer_size, 0);
  md_.count_hit_ = vector<short>(buffer_size, 0);
  md_.flag_rayend_ = vector<char>(buffer_size, -1);
  md_.flag_traverse_ = vector<char>(buffer_size, -1);

  md_.raycast_num_ = 0;

  md_.proj_points_.resize(640 * 480 / mp_.skip_pixel_ / mp_.skip_pixel_);
  md_.proj_points_cnt = 0;

  md_.cam2body_ << 0.0, 0.0, 1.0, 0.0,
                  -1.0, 0.0, 0.0, 0.0,
                  0.0, -1.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 1.0;

  resetBuffer();

  /* init callback */

  OctoMapCenterSub_ = nh.subscribe<sensor_msgs::PointCloud2>("/octomap_point_cloud_centers", 1, &GridMap::OctoMapCenterCallback, this);
  // OctoMapSub_ = nh.subscribe<octomap_msgs::Octomap>("/octomap_full", 1, &GridMap::OctoMapCallback, this);
  // vis_timer_ = node_.createTimer(ros::Duration(0.11), &GridMap::visCallback, this);

  // map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("grid_map/occupancy", 10);
  map_inf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/occupancy_inflate", 10);
  map_vis_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/vis", 10);

  md_.occ_need_update_ = false;
  md_.local_updated_ = false;
  md_.has_first_depth_ = false;
  md_.has_odom_ = false;
  md_.has_cloud_ = false;
  md_.image_cnt_ = 0;
  md_.last_occ_update_time_.fromSec(0);

  md_.fuse_time_ = 0.0;
  md_.update_num_ = 0;
  md_.max_fuse_time_ = 0.0;

  md_.flag_depth_odom_timeout_ = false;
  md_.flag_use_depth_fusion = false;

  
}

void GridMap::resetBuffer(){
  Eigen::Vector3d min_pos = mp_.map_min_boundary_;
  Eigen::Vector3d max_pos = mp_.map_max_boundary_;

  resetBuffer(min_pos, max_pos);

  md_.local_bound_min_ = Eigen::Vector3i::Zero();
  md_.local_bound_max_ = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();
}

void GridMap::resetBuffer(Eigen::Vector3d min_pos, Eigen::Vector3d max_pos){
  Eigen::Vector3i min_id, max_id;
  posToIndex(min_pos, min_id);
  posToIndex(max_pos, max_id);

  boundIndex(min_id);
  boundIndex(max_id);

  /* reset occ and dist buffer */
  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z)
      {
        md_.occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
      }
}

void GridMap::OctoMapCenterCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
  double time = ros::Time::now().toSec();
  md_.has_cloud_ = true;
  pcl::PointCloud<pcl::PointXYZ> latest_cloud, cloud;
  pcl::fromROSMsg(*msg, latest_cloud);
  if (latest_cloud.points.size() == 0)
    return;
  // resetBuffer();

  pcl::PointXYZ pt, pt2;
  Eigen::Vector3d p3d;

  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  int inf_step_z = 1;
  
  for (size_t i = 0; i < latest_cloud.points.size(); ++i)
  {
    pt = latest_cloud.points[i];
    for (int x = -inf_step; x <= inf_step; ++x)
      for (int y = -inf_step; y <= inf_step; ++y)
        for (int z = -inf_step_z; z <= inf_step_z; ++z){
          p3d(0) = pt.x + x * mp_.resolution_, p3d(1) = pt.y + y * mp_.resolution_, p3d(2) = pt.z + z * mp_.resolution_;
          setOccupied(p3d);
          // Eigen::Vector3i id;
          // posToIndex(p3d, id);
          // boundIndex(id);
          // indexToPos(id, p3d);
          pt2.x = p3d(0), pt2.y = p3d(1), pt2.z = p3d(2);
          cloud.push_back(pt2);
        }

    // setOccupancy(p3d, 1.0);
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  map_vis_pub_.publish(cloud_msg);
  // add virtual ceiling to limit flight height
  if (mp_.virtual_ceil_height_ > -0.5) {
    int ceil_id = floor((mp_.virtual_ceil_height_ - mp_.map_origin_(2)) * mp_.resolution_inv_);
    for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
      for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y){
        Eigen::Vector3i id(x, y, ceil_id);
        boundIndex(id);
        md_.occupancy_buffer_inflate_[toAddress(id)] = 1;
        indexToPos(id, p3d);
        pt2.x = p3d(0), pt2.y = p3d(1), pt2.z = p3d(2);
        cloud.push_back(pt2);
        // md_.occupancy_buffer_[toAddress(id)] = 1;
      }
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  // sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  map_inf_pub_.publish(cloud_msg);
  // map_vis_pub_.publish(cloud_msg);

  // ROS_INFO("process pointcloud");
  // cout << latest_cloud.points.size() << endl;
  // cout << "time for octo2grid: " << ros::Time::now().toSec() - time << endl << endl;
  
}

void GridMap::OctoMapCallback(const octomap_msgs::Octomap::ConstPtr &msg){
  OctoMap_ = (octomap::OcTree*)octomap_msgs::msgToMap(*msg);
  Eigen::Vector3d p3d;
  pcl::PointCloud<pcl::PointXYZI>::Ptr occupied_nodes(new pcl::PointCloud<pcl::PointXYZI>());
  // double time = ros::Time::now().toSec();
  // ROS_INFO("process pointcloud");
  for(octomap::OcTree::leaf_iterator it = OctoMap_->begin_leafs(), end = OctoMap_->end_leafs();it != end; ++it){
    pcl::PointXYZI cube_center;
    cube_center.x = it.getX();
    cube_center.y = it.getY();
    cube_center.z = it.getZ();
    cube_center.intensity = it.getDepth();

    p3d(0) = cube_center.x, p3d(1) = cube_center.y, p3d(2) = cube_center.z;
    // cout << p3d << endl;
    this->setOccupied(p3d);
    this->setOccupancy(p3d, 1);
  
    if(OctoMap_->isNodeOccupied(*it)){
      occupied_nodes->points.push_back(cube_center);
    }
  }
  occupied_nodes->header.frame_id = mp_.frame_id_;
  if (mp_.virtual_ceil_height_ > -0.5) {
    int ceil_id = floor((mp_.virtual_ceil_height_ - mp_.map_origin_(2)) * mp_.resolution_inv_);
    for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
      for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y){
        Eigen::Vector3i id(x, y, ceil_id);
        boundIndex(id);
        md_.occupancy_buffer_inflate_[toAddress(id)] = 1;
        md_.occupancy_buffer_[toAddress(id)] = 1;
      }
  }
  // ROS_INFO("process pointcloud");
  // cout << occupied_nodes->size() << endl;
  // cout << ros::Time::now().toSec() - time << endl << endl;
}

void GridMap::visCallback(const ros::TimerEvent & /*event*/){
  publishMapInflate(true);
  // publishMap();
  return;
}

void GridMap::publishMap()
{

  // if (map_pub_.getNumSubscribers() <= 0)
  //   return;
  // ROS_WARN("send map");
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  // Eigen::Vector3i min_cut = md_.local_bound_min_;
  // Eigen::Vector3i max_cut = md_.local_bound_max_;

  Eigen::Vector3i min_cut = Eigen::Vector3i::Zero();
  Eigen::Vector3i max_cut = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();

  int lmm = mp_.local_map_margin_ / 2;
  min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
  max_cut += Eigen::Vector3i(lmm, lmm, lmm);

  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        // if (md_.occupancy_buffer_[toAddress(x, y, z)] < mp_.min_occupancy_log_)
        // cout << md_.occupancy_buffer_[toAddress(x, y, z)] << endl;
        if (md_.occupancy_buffer_[toAddress(x, y, z)] < 0.1)
          continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > mp_.visualization_truncate_height_)
          continue;
        // cout << "Map" << pos << endl;
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_pub_.publish(cloud_msg);

  // ROS_INFO("pub map");
}

void GridMap::publishMapInflate(bool all_info)
{

  // if (map_inf_pub_.getNumSubscribers() <= 0)
  //   return;
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  // Eigen::Vector3i min_cut = md_.local_bound_min_;
  // Eigen::Vector3i max_cut = md_.local_bound_max_;

  Eigen::Vector3i min_cut = Eigen::Vector3i::Zero();
  Eigen::Vector3i max_cut = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();
  // cout << max_cut << endl;

  if (all_info)
  {
    int lmm = mp_.local_map_margin_;
    min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
    max_cut += Eigen::Vector3i(lmm, lmm, lmm);
  }

  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        // cout << x << y << z << endl;
        if (md_.occupancy_buffer_inflate_[toAddress(x, y, z)] == 0)
          continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > mp_.visualization_truncate_height_)
          continue;

        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  map_inf_pub_.publish(cloud_msg);
}

bool GridMap::odomValid(){return md_.has_odom_;}

bool GridMap::hasDepthObservation(){return md_.has_first_depth_;}

Eigen::Vector3d GridMap::getOrigin(){return mp_.map_origin_;}

int GridMap::getVoxelNum(){return mp_.map_voxel_num_[0] * mp_.map_voxel_num_[1] * mp_.map_voxel_num_[2];}

void GridMap::getRegion(Eigen::Vector3d &ori, Eigen::Vector3d &size){ ori = mp_.map_origin_, size = mp_.map_size_; }