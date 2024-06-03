#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_broadcaster.h>
// #include <visualization_msgs/msg/marker.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <fstream>
#include <mutex>
#include <omp.h>

#include "IMU_Processing.hpp"
#include "preprocess.h"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define PUBFRAME_PERIOD (20)

class LaserMapping : public rclcpp::Node {
public:
    void readParameters();
    void initLIO();

    BoxPointType LocalMap_Points;       // ikd-tree地图立方体的2个角点
    bool Localmap_Initialized = false;  // 局部地图是否初始化
    void lasermap_fov_segment();
    void map_incremental();

    void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull);
    void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap);
    void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster>& tf_br);
    void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath);

    /*** Time Log Variables ***/
    int add_point_size = 0, kdtree_delete_counter = 0;
    bool time_sync_en = false, extrinsic_est_en = true, path_en = true, runtime_pos_log = false;
    /**************************/

    float DET_RANGE = 300.0f;
    const float MOV_THRESHOLD = 1.5f;
    double time_diff_lidar_to_imu = 0.0;

    mutex mtx_buffer;

    std::string lid_topic, imu_topic;

    double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
    double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
    double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0;
    int feats_down_size = 0, NUM_MAX_ITERATIONS = 0;

    bool lidar_pushed, flg_first_scan = true, flg_EKF_inited;
    bool scan_pub_en = false, dense_pub_en = false;

    std::vector<BoxPointType> cub_needrm;
    std::vector<PointVector> Nearest_Points;
    std::vector<double> extrinT{3, 0.0};
    std::vector<double> extrinR{9, 0.0};
    std::deque<double> time_buffer;
    std::deque<PointCloudXYZI::Ptr> lidar_buffer;
    std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

    PointCloudXYZI::Ptr featsFromMap{new PointCloudXYZI()};
    PointCloudXYZI::Ptr feats_undistort{new PointCloudXYZI()};
    PointCloudXYZI::Ptr feats_down_body{new PointCloudXYZI()};   // 畸变纠正后降采样的单帧点云，lidar系
    PointCloudXYZI::Ptr feats_down_world{new PointCloudXYZI()};  // 畸变纠正后降采样的单帧点云，W系

    // Loc
    bool loc_mode = false;
    PointCloudXYZI::Ptr prior_cloud{new PointCloudXYZI()};
    std::vector<double> init_guess{6, 0};
    std::string loadmap_dir;
    void initLoc();

    // SaveMap
    int pcd_save_en = 0;
    double filter_size_savemap = 0.2;
    pcl::VoxelGrid<PointType> downSizeFilterSaveMap;
    std::string savemap_dir;
    PointCloudXYZI::Ptr pcl_wait_save{new PointCloudXYZI()};
    void saveMap();
    void saveMap(const std::string& path);

    double filter_size_surf_min = 0, filter_size_map_min = 0;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;

    KD_TREE<PointType> ikdtree;

    V3D Lidar_T_wrt_IMU{Zero3d};
    M3D Lidar_R_wrt_IMU{Eye3d};

    /*** EKF inputs and output ***/
    MeasureGroup Measures;

    esekfom::esekf kf;

    state_ikfom state_point;
    Eigen::Vector3d pos_lid;  // 估计的W系下的位置

    nav_msgs::msg::Path path;
    nav_msgs::msg::Odometry odomAftMapped;
    geometry_msgs::msg::PoseStamped msg_body_pose;

    shared_ptr<Preprocess> p_pre{new Preprocess()};
    shared_ptr<ImuProcess> p_imu{new ImuProcess()};

    LaserMapping(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options) {
        readParameters();
        initLIO();
        initLoc();
    }

private:
    void timer_callback();

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull, pubLaserCloudFull_body;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect, pubLaserCloudMap;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    rclcpp::TimerBase::SharedPtr timer_;

    template <typename T>
    void declare_and_get_parameter(const std::string& name, T& variable, const T& default_value) {
        this->declare_parameter<T>(name, default_value);
        this->get_parameter(name, variable);
    }
    template <typename T>
    void set_posestamp(T& out) {
        out.pose.position.x = state_point.pos(0);
        out.pose.position.y = state_point.pos(1);
        out.pose.position.z = state_point.pos(2);

        auto q_ = Eigen::Quaterniond(state_point.rot.matrix());
        out.pose.orientation.x = q_.coeffs()[0];
        out.pose.orientation.y = q_.coeffs()[1];
        out.pose.orientation.z = q_.coeffs()[2];
        out.pose.orientation.w = q_.coeffs()[3];
    }

    template <typename T>
    void pointBodyToWorld(const Eigen::Matrix<T, 3, 1>& pi, Eigen::Matrix<T, 3, 1>& po) {
        V3D p_body(pi[0], pi[1], pi[2]);
        V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

        po[0] = p_global(0);
        po[1] = p_global(1);
        po[2] = p_global(2);
    }

    void pointBodyToWorld(PointType const* const pi, PointType* const po) {
        V3D p_body(pi->x, pi->y, pi->z);
        V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
    }

    void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg);
    double timediff_lidar_wrt_imu = 0.0;
    bool timediff_set_flg = false;
    void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg);
    void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in);
    double lidar_mean_scantime = 0.0;
    int scan_num = 0;
    bool sync_packages(MeasureGroup& meas);
};
