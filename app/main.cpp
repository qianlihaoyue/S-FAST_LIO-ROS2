#include "laserMapping.hpp"
#include <csignal>

std::shared_ptr<LaserMapping> fastlio_node;

void SigHandle(int sig) {
    std::cout << "catch sig " << sig << std::endl;
    fastlio_node->saveMap();
    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    fastlio_node = std::make_shared<LaserMapping>();

    rclcpp::spin(fastlio_node);
};