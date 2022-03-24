#include <ros/ros.h>
#include <occupancy_map/occupancy_map.h>

using namespace costmap;

int main(int argc, char **argv){

    ros::init(argc, argv, "costmap_node");
    ros::NodeHandle nh("~");
    
    COSTMAP CostMap;
    CostMap.init(nh);

    // ros::Duration(1.0).sleep();
    ros::spin();

    return 0;
}
