#include "laserMapping.hpp"
#include <csignal>

std::shared_ptr<LaserMapping> ltlio_node;

void SigHandle(int sig) {
    ROS_INFO("catch sig %d", sig);
    ltlio_node->saveMap();
    ros::shutdown();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    signal(SIGINT, SigHandle);

    ltlio_node = std::make_shared<LaserMapping>(nh);

    ros::Rate rate(1000);
    while (ros::ok()) {
        ros::spinOnce();
        ltlio_node->timer_callback();
        rate.sleep();
    }

    ros::spin();
    return 0;
}
