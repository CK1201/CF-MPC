#include <ros/ros.h>
#include <flight_corridor/flight_corridor.h>

using namespace flight_corridor;

int main(int argc, char **argv){

    ros::init(argc, argv, "flight_corridor_node");
    ros::NodeHandle nh("~");
    
    FLIGHTCORRIDOR FlightCorridor;
    FlightCorridor.init(nh);

    // ros::Duration(1.0).sleep();
    ros::spin();

    return 0;
}
