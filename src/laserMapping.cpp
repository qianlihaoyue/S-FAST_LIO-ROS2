#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/vector3.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>
#include <csignal>
#include <fstream>
#include <mutex>
#include <thread>
#include <omp.h>

#include "IMU_Processing.hpp"
#include "preprocess.h"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
int add_point_size = 0, kdtree_delete_counter = 0;
bool pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string map_file_path, lid_topic, imu_topic;

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0;
double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int feats_down_size = 0, NUM_MAX_ITERATIONS = 0, pcd_save_interval = -1, pcd_index = 0;

bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<BoxPointType> cub_needrm;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());   // 畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());  // 畸变纠正后降采样的单帧点云，W系

// Loc
bool loc_mode = false;
PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
vector<double> init_pos(3, 0.0);
vector<double> init_rot{0, 0, 0, 1};

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;

esekfom::esekf kf;

state_ikfom state_point;
Eigen::Vector3d pos_lid;  // 估计的W系下的位置

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) {
    mtx_buffer.lock();

    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    if (get_time_sec(msg->header.stamp) < last_timestamp_lidar) {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) {
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();

    if (get_time_sec(msg->header.stamp) < last_timestamp_lidar) {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    last_timestamp_lidar = get_time_sec(msg->header.stamp);

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty()) {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty()) {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in) {
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) {
        msg->header.stamp = rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu) {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
// 把当前要处理的LIDAR和IMU数据打包到meas
bool sync_packages(MeasureGroup& meas) {
    if (lidar_buffer.empty() || imu_buffer.empty()) return false;

    /*** push a lidar scan ***/
    if (!lidar_pushed) {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 5)  // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!" << std::endl;
        } else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime) {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        } else {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime +=
                (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;  // 注意curvature中存储的是相对第一个点的时间
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time) return false;

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if (imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void pointBodyToWorld(PointType const* const pi, PointType* const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Eigen::Matrix<T, 3, 1>& pi, Eigen::Matrix<T, 3, 1>& po) {
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

BoxPointType LocalMap_Points;       // ikd-tree地图立方体的2个角点
bool Localmap_Initialized = false;  // 局部地图是否初始化
void lasermap_fov_segment() {
    cub_needrm.clear();  // 清空需要移除的区域
    kdtree_delete_counter = 0;

    V3D pos_LiD = pos_lid;  // W系下位置
    // 初始化局部地图范围，以pos_LiD为中心,长宽高均为cube_len
    if (!Localmap_Initialized) {
        for (int i = 0; i < 3; i++) {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    // 各个方向上pos_LiD与局部地图边界的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++) {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离（1.5*300m）太小，标记需要移除need_move(FAST-LIO2论文Fig.3)
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;  // 如果不需要，直接返回，不更改局部地图

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    // 需要移动的距离
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++) {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);

    if (cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);  // 删除指定范围内的点
}

// 根据最新估计位姿  增量添加点云到map
void init_ikdtree() {
    // 加载读取点云数据到cloud中
    string all_points_dir(string(string(ROOT_DIR) + "PCD/") + "scans.pcd");
    if (pcl::io::loadPCDFile<PointType>(all_points_dir, *cloud) == -1) {
        PCL_ERROR("Read file fail!\n");
    }

    ikdtree.set_downsample_param(filter_size_map_min);
    ikdtree.Build(cloud->points);
    std::cout << "---- ikdtree size: " << ikdtree.size() << std::endl;
}

// 根据最新估计位姿  增量添加点云到map
void map_incremental() {
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++) {
        // 转换到世界坐标系
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        if (!Nearest_Points[i].empty() && flg_EKF_inited) {
            const PointVector& points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType mid_point;  // 点所在体素的中心
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);  // 如果距离最近的点都在体素外，则该点不需要Downsample
                continue;
            }
            for (int j = 0; j < NUM_MATCH_POINTS; j++) {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[j], mid_point) < dist)  // 如果近邻点距离 < 当前点距离，不添加该点
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        } else {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull) {
    if (scan_pub_en) {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++) {
            pointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull->publish(laserCloudmsg);
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en) {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++) {
            pointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i]);
        }

        static int scan_wait_num = 0;
        scan_wait_num++;

        if (scan_wait_num % 4 == 0) *pcl_wait_save += *laserCloudWorld;

        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval) {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap) {
    sensor_msgs::msg::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap->publish(laserCloudMap);
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

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster>& tf_br) {
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped->publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "camera_init";
    trans.child_frame_id = "body";
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_br->sendTransform(trans);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath) {
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = rclcpp::Time(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

shared_ptr<ImuProcess> p_imu(new ImuProcess());

class LaserMapping : public rclcpp::Node {
public:
    LaserMapping(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options) {
        declare_and_get_parameter<bool>("publish.path_en", path_en, true);
        declare_and_get_parameter<bool>("publish.scan_publish_en", scan_pub_en, true);
        declare_and_get_parameter<bool>("publish.dense_publish_en", dense_pub_en, true);
        declare_and_get_parameter<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);

        declare_and_get_parameter<string>("common.lid_topic", lid_topic, "/livox/lidar");
        declare_and_get_parameter<string>("common.imu_topic", imu_topic, "/livox/imu");
        declare_and_get_parameter<bool>("common.time_sync_en", time_sync_en, false);  // 时间同步
        declare_and_get_parameter<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);

        declare_and_get_parameter<double>("filter_size_corner", filter_size_corner_min, 0.5);
        declare_and_get_parameter<double>("filter_size_surf", filter_size_surf_min, 0.5);
        declare_and_get_parameter<double>("filter_size_map", filter_size_map_min, 0.5);
        declare_and_get_parameter<double>("cube_side_length", cube_len, 200);            // 地图的局部区域的长度（FastLio2论文中有解释）
        declare_and_get_parameter<int>("point_filter_num", p_pre->point_filter_num, 2);  // 采样间隔，即每隔point_filter_num个点取1个点
        declare_and_get_parameter<int>("max_iteration", NUM_MAX_ITERATIONS, 4);          // 卡尔曼滤波的最大迭代次数

        declare_and_get_parameter<float>("mapping.det_range", DET_RANGE, 300.f);  // 激光雷达的最大探测范围

        declare_and_get_parameter<double>("mapping.gyr_cov", gyr_cov, 0.1);         // IMU陀螺仪的协方差
        declare_and_get_parameter<double>("mapping.acc_cov", acc_cov, 0.1);         // IMU加速度计的协方差
        declare_and_get_parameter<double>("mapping.b_gyr_cov", b_gyr_cov, 0.0001);  // IMU陀螺仪偏置的协方差
        declare_and_get_parameter<double>("mapping.b_acc_cov", b_acc_cov, 0.0001);  // IMU加速度计偏置的协方差
        declare_and_get_parameter<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);

        declare_and_get_parameter<double>("preprocess.blind", p_pre->blind, 0.01);         // 最小距离阈值，即过滤掉0～blind范围内的点云
        declare_and_get_parameter<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);  // 激光雷达的类型
        declare_and_get_parameter<int>("preprocess.scan_line", p_pre->N_SCANS, 16);        // 激光雷达扫描的线数（livox avia为6线）
        declare_and_get_parameter<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        declare_and_get_parameter<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);

        declare_and_get_parameter<bool>("pcd_save.pcd_save_en", pcd_save_en, false);  // 是否将点云地图保存到PCD文件
        declare_and_get_parameter<int>("pcd_save.interval", pcd_save_interval, -1);

        declare_and_get_parameter<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());  // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
        declare_and_get_parameter<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());  // 雷达相对于IMU的外参R

        std::cout << "Lidar_type: " << p_pre->lidar_type << std::endl;

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = "camera_init";

        /*** ROS subscribe initialization ***/
        if (p_pre->lidar_type == AVIA) {
            sub_pcl_livox = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
        } else {
            sub_pcl_pc = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        }
        sub_imu = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 10, imu_cbk);
        pubLaserCloudFull = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudEffect = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubLaserCloudMap = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
        pubOdomAftMapped = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath = this->create_publisher<nav_msgs::msg::Path>("/path", 20);

        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        timer_ = rclcpp::create_timer(this, this->get_clock(), std::chrono::milliseconds(10), std::bind(&LaserMapping::timer_callback, this));

        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
        p_imu->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, acc_cov), V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov),
                         V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        declare_and_get_parameter<bool>("loc.en", loc_mode, false);
        if (loc_mode) {
            declare_and_get_parameter<string>("map_file_path", map_file_path, "");  // 地图保存路径
            init_ikdtree();                                                         // 读取点云文件 初始化ikdtree

            declare_and_get_parameter<vector<double>>("loc.init_pos", init_pos, vector<double>());  // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
            declare_and_get_parameter<vector<double>>("loc.init_rot", init_rot, vector<double>());  // 雷达相对于IMU的外参R
            state_point = kf.get_x();
            state_point.pos = Eigen::Vector3d(init_pos[0], init_pos[1], init_pos[2]);
            Eigen::Quaterniond q(init_rot[3], init_rot[0], init_rot[1], init_rot[2]);
            Sophus::SO3 SO3_q(q);
            state_point.rot = SO3_q;
            kf.change_x(state_point);
        }
    }

private:
    void timer_callback() {
        if (sync_packages(Measures)) {
            double t0 = omp_get_wtime();

            if (flg_first_scan) {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            p_imu->Process(Measures, kf, feats_undistort);

            // 如果feats_undistort为空 ROS_WARN
            if (feats_undistort->empty() || (feats_undistort == NULL)) {
                std::cout << "No point, skip this scan!" << std::endl;
                return;
            }

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            lasermap_fov_segment();  // 更新localmap边界，然后降采样当前帧点云

            // 点云下采样
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            feats_down_size = feats_down_body->points.size();

            // std::cout << "feats_down_size :" << feats_down_size << std::endl;
            if (feats_down_size < 5) {
                std::cout << "No point, skip this scan!" << std::endl;
                return;
            }

            // 初始化ikdtree(ikdtree为空时)
            if (ikdtree.Root_Node == nullptr) {
                ikdtree.set_downsample_param(filter_size_map_min);
                feats_down_world->resize(feats_down_size);
                for (int i = 0; i < feats_down_size; i++) {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));  // lidar坐标系转到世界坐标系
                }
                ikdtree.Build(feats_down_world->points);  // 根据世界坐标系下的点构建ikdtree
                return;
            }

            if (0)  // If you need to see map point, change to "if(1)"
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
                // std::cout << "ikdtree size: " << featsFromMap->points.size() << std::endl;
            }

            double t1 = omp_get_wtime();
            /*** iterated state estimation ***/
            Nearest_Points.resize(feats_down_size);  // 存储近邻点的vector
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);
            double t2 = omp_get_wtime();

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped, tf_broadcaster);

            /*** add the feature points to map kdtree ***/
            if (!loc_mode) {
                feats_down_world->resize(feats_down_size);
                map_incremental();
            }

            /******* Publish points *******/
            if (path_en) publish_path(pubPath);
            if (scan_pub_en || pcd_save_en) publish_frame_world(pubLaserCloudFull);

            double t3 = omp_get_wtime();

            // runtime_pos_log
            if (1) {
                // int cur_pts = (int)feats_down_world->size();
                // printf("ds: %d match: %d%% ", cur_pts, (int)((float)effct_feat_num / cur_pts * 100));
                printf("[tim] ICP: %0.2f total: %0.2f", (t2 - t1) * 1000.0, (t3 - t0) * 1000.0);
                std::cout << std::endl;
            }
        }
    }

private:
    template <typename T>
    void declare_and_get_parameter(const std::string& name, T& variable, const T& default_value) {
        this->declare_parameter<T>(name, default_value);
        this->get_parameter(name, variable);
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    rclcpp::TimerBase::SharedPtr timer_;
};

void SigHandle(int sig) {
    flg_exit = true;
    std::cout << "catch sig " << sig << std::endl;
    sig_buffer.notify_all();

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t i = 1; i <= pcd_index; i++) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(i) + string(".pcd"));
            pcl::PCDReader reader;
            reader.read(all_points_dir, *cloud_temp);
            *cloud = *cloud + *cloud_temp;
        }

        string file_name = string("GlobalMap.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *cloud);

        //////////////////////////////////////
        PointVector().swap(ikdtree.PCL_Storage);
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        featsFromMap->clear();
        featsFromMap->points = ikdtree.PCL_Storage;
        std::cout << "ikdtree size: " << featsFromMap->points.size() << std::endl;
        string file_name1 = string("GlobalMap_ikdtree.pcd");
        pcl::PCDWriter pcd_writer1;
        string all_points_dir1(string(string(ROOT_DIR) + "PCD/") + file_name1);
        cout << "current scan saved to /PCD/" << file_name1 << endl;
        pcd_writer1.writeBinary(all_points_dir1, *featsFromMap);
    }

    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    rclcpp::spin(std::make_shared<LaserMapping>());

    if (rclcpp::ok()) rclcpp::shutdown();
};