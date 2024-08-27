#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <fstream>
#include <mutex>
#include <omp.h>

#include "IMU_Processing.hpp"
#include "preprocess.h"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
// #define PUBFRAME_PERIOD (20)

class LaserMapping {
public:
    void readParameters();
    void initLIO();

    void map_incremental();

    void publish_frame_world();
    void publish_cloud(const ros::Publisher& pubLaserCloudMap, const CloudType::Ptr& cloud);
    void publish_odometry(const ros::Publisher& pubOdomAftMapped);
    void publish_path(const ros::Publisher& pubPath);

    /*** Time Log Variables ***/
    int add_point_size = 0, kdtree_delete_counter = 0;
    bool time_sync_en = false, path_en = true, runtime_pos_log = false;
    FILE* fp;
    void dump_lio_state_to_log(FILE* fp);
    std::ofstream fout_traj;
    /**************************/

    double time_diff_lidar_to_imu = 0.0;

    mutex mtx_buffer;

    std::string lid_topic, imu_topic;

    double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
    double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
    double lidar_end_time = 0, first_lidar_time = 0.0;
    int feats_down_size = 0;

    bool lidar_pushed, flg_first_scan = true, flg_EKF_inited;
    bool scan_pub_en = false, dense_pub_en = false;

    std::vector<PointVector> Nearest_Points;
    std::deque<double> time_buffer;
    std::deque<CloudType::Ptr> lidar_buffer;
    std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

    CloudType::Ptr feats_undistort{new CloudType()};
    CloudType::Ptr feats_down_body{new CloudType()};
    CloudType::Ptr feats_down_world{new CloudType()};

    // Dyna
    void dyna_incremental();

    // Loc
    bool loc_mode = false;
    CloudType::Ptr prior_cloud{new CloudType()};
    std::vector<double> init_guess{6, 0};
    std::string loadmap_dir;
    void initLoc();

    // SaveMap
    int pcd_save_en = 0;
    double filter_size_savemap = 0.2;
    pcl::VoxelGrid<PointType> downSizeFilterSaveMap;
    std::string savemap_dir;
    CloudType::Ptr pcl_wait_save{new CloudType()};
    CloudType::Ptr pcl_effect_save{new CloudType()};
    std::vector<CloudType::Ptr> pcl_save_block;
    void savemap_callback(const ros::TimerEvent& event);
    void saveMap();
    void saveMap(const std::string& path);

    double filter_size_surf_min = 0, filter_size_map_min = 0;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    // pcl::VoxelGrid<PointType> downSizeFilterMap;

    KD_TREE<PointType> ikdtree;

    V3D Lidar_T_wrt_IMU{Zero3d};
    M3D Lidar_R_wrt_IMU{Eye3d};

    ros::Publisher marker_pub_;
    void publish_pca(const Eigen::Matrix3d& covariance_matrix);

    /*** EKF inputs and output ***/
    MeasureGroup Measures;

    esekfom::esekf kf;

    state_ikfom state_point;
    Eigen::Vector3d pos_lid;  // 估计的W系下的位置

    nav_msgs::Path path;
    nav_msgs::Odometry odomAftMapped;
    geometry_msgs::PoseStamped msg_body_pose;

    shared_ptr<Preprocess> p_pre{new Preprocess()};
    shared_ptr<ImuProcess> p_imu{new ImuProcess()};

    LaserMapping(const ros::NodeHandle& nh_) : nh(nh_) {
        readParameters();
        initLIO();
        initLoc();
    }

    void timer_callback();

private:
    ros::NodeHandle nh;

    ros::Publisher pubLaserCloudFull, pubLaserCloudFull_body;
    ros::Publisher pubLaserCloudEffect, pubLaserCloudNoEffect, pubLaserCloudMap, pubLaserCloudFlash;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubPath;

    ros::Subscriber sub_imu, sub_pcl;

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

        po[0] = p_global(0), po[1] = p_global(1), po[2] = p_global(2);
    }

    void pointBodyToWorld(PointType const* const pi, PointType* const po) {
        V3D p_body(pi->x, pi->y, pi->z);
        V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

        po->x = p_global(0), po->y = p_global(1), po->z = p_global(2);
        po->intensity = pi->intensity;
        po->curvature = pi->curvature;  // 仅用于可视化
    }

    void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr& msg);
    double timediff_lidar_wrt_imu = 0.0;
    bool timediff_set_flg = false;
    void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr& msg);
    void imu_cbk(const sensor_msgs::Imu::ConstPtr& msg_in);
    double lidar_mean_scantime = 0.0;
    int scan_num = 0;
    bool sync_packages(MeasureGroup& meas);
};
