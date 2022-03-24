import rospy
from sensor_msgs.msg import PointCloud, PointCloud2
import pcl


class FlightCorridorGenerator:
    def __init__(self) -> None:
        CameraName = rospy.get_param('~CameraName', default='camera')
        self.PointCloudRawSub = rospy.Subscriber('/' + CameraName + '/depth/color/points', PointCloud2, self.PointCloudRawCallback)
        self.PointCloudProcessedPub = rospy.Publisher('/' + CameraName + '/autopilot/start', PointCloud, queue_size=1, tcp_nodelay=True)
        
        msg = PointCloud2()
        # msg.data.

    def PointCloudRawCallback(self, msg):
        # msg.
        self.PointCloudProcessedPub.publish()



def main():
    FlightCorridorGenerator()

if __name__ == "__main__":
    main()