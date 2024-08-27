#pragma once

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>

enum LID_TYPE { AVIA = 1, VELO16, OUST64, MID360 };
enum TIME_UNIT { SEC = 0, MS = 1, US = 2, NS = 3 };

// clang-format off
namespace velodyne_ros {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (std::uint16_t, ring, ring)
)

namespace mid360_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      uint8_t tag;
      uint8_t line;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(mid360_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint8_t, tag, tag)
    (std::uint8_t, line, line)
)

namespace ouster_ros {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_ros

POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
)

// clang-format on

class Preprocess {
    typedef pcl::PointXYZINormal PointType;

public:
    Preprocess();
    ~Preprocess();

    void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, pcl::PointCloud<PointType>::Ptr &pcl_out);
    void process(const sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud<PointType>::Ptr &pcl_out);
    void set(int lid_type, double bld, int pfilt_num);

    pcl::PointCloud<PointType> pl_full, pl_surf;
    float time_unit_scale;
    int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;
    double blind, max_range;
    bool given_offset_time;

private:
    void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void mid360_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
};
