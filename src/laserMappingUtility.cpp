#include "laserMapping.hpp"

#include <sys/stat.h>  // mkdir, stat

inline bool createDirectoryIfNotExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory does not exist, create it
        if (mkdir(path.c_str(), 0755) == 0) {
            return true;
        } else {
            std::cerr << "Error creating directory: " << path << std::endl;
            return false;
        }
    } else if (info.st_mode & S_IFDIR) {
        return true;
    } else {
        std::cerr << "Path exists but is not a directory: " << path << std::endl;
        return false;
    }
}

void LaserMapping::readParameters() {
    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    // nh.param<bool>("publish/dense_publish_en", dense_pub_en, false);
    nh.param<bool>("publish/runtime_pos_log", runtime_pos_log, true);

    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);  // 时间同步
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);

    nh.param<double>("common/filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("common/filter_size_map", filter_size_map_min, 0.5);

    nh.param<int>("kf/maximum_iter", kf.maximum_iter, 4);
    nh.param<double>("kf/epsi", kf.epsi, 0.001);
    nh.param<int>("kf/NUM_MATCH_POINTS", kf.NUM_MATCH_POINTS, 5);
    nh.param<float>("kf/plane_thr", kf.plane_thr, 0.1);

    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);         // IMU陀螺仪的协方差
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);         // IMU加速度计的协方差
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);  // IMU陀螺仪偏置的协方差
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);  // IMU加速度计偏置的协方差

    std::vector<double> extrinT(3, 0.0);
    std::vector<double> extrinR(9, 0.0);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>(3, 0));  // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>(9, 0));  // 雷达相对于IMU的外参R
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);

    // Dyna
    int ivox_capacity;
    double ivox_resolution;
    nh.param<bool>("dyna/en", kf.USE_DYNA, false);
    nh.param<int>("dyna/ivox_capacity", ivox_capacity, 10000);
    nh.param<double>("dyna/ivox_resolution", ivox_resolution, 0.25);
    nh.param<bool>("dyna/use_chi", kf.use_chi, false);
    nh.param<double>("dyna/chi_square_critical", kf.chi_square_critical, 3.84);

    HVox<PointType>::Options ivox_options_;
    ivox_options_.nearby_type_ = HVox<PointType>::NearbyType::NEARBY6;
    ivox_options_.res_ = ivox_resolution;
    ivox_options_.capacity_ = ivox_capacity;
    kf.hvox_dyna = std::make_shared<HVox<PointType>>(ivox_options_);

    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);  // 最小距离阈值，即过滤掉0～blind范围内的点云
    nh.param<double>("preprocess/max_range", p_pre->max_range, 100.0);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);  // 激光雷达的类型
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);        // 激光雷达扫描的线数（livox avia为6线）
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 2);  // 采样间隔，即每隔point_filter_num个点取1个点

    nh.param<int>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<double>("pcd_save/filter_size_savemap", filter_size_savemap, 0.2);
    nh.param<string>("pcd_save/savemap_dir", savemap_dir, string(ROOT_DIR) + "PCD/");
    downSizeFilterSaveMap.setLeafSize(filter_size_savemap, filter_size_savemap, filter_size_savemap);

    createDirectoryIfNotExists(savemap_dir);
    fp = fopen((savemap_dir + "/pos_log.csv").c_str(), "w");
    if (fp == nullptr) std::cerr << "Failed to open file for writing" << std::endl;
    fout_traj.open(savemap_dir + "/traj_tum.txt", std::ios::out);
    fout_traj.setf(std::ios::fixed, std::ios::floatfield);
    fout_traj.precision(6);
}

void LaserMapping::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar) {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    CloudType::Ptr ptr(new CloudType());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
}

void LaserMapping::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr& msg) {
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar) {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty()) {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty()) {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    CloudType::Ptr ptr(new CloudType());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    mtx_buffer.unlock();
}

void LaserMapping::imu_cbk(const sensor_msgs::Imu::ConstPtr& msg_in) {
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) {
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu) {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    // sig_buffer.notify_all();
}

// 把当前要处理的LIDAR和IMU数据打包到meas
bool LaserMapping::sync_packages(MeasureGroup& meas) {
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
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void LaserMapping::publish_frame_world() {
    CloudType::Ptr laserCloudEffect(new CloudType());
    CloudType::Ptr laserCloudNonEffect(new CloudType());

    int slow = 0;
    for (int i = 0; i < feats_down_world->size(); i++) {
        auto& pt = feats_down_world->points[i];
        if (i == kf.effect_idx[slow]) {
            laserCloudEffect->push_back(pt);
            slow++;
        } else
            laserCloudNonEffect->push_back(pt);
    }

    publish_cloud(pubLaserCloudEffect, laserCloudEffect);
    publish_cloud(pubLaserCloudNoEffect, laserCloudNonEffect);
    publish_cloud(pubLaserCloudFull, feats_down_world);
}

void LaserMapping::publish_cloud(const ros::Publisher& pubLaserCloudMap, const CloudType::Ptr& cloud) {
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*cloud, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

void LaserMapping::publish_odometry(const ros::Publisher& pubOdomAftMapped) {
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
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

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));
}

void LaserMapping::publish_path(const ros::Publisher& pubPath) {
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void LaserMapping::publish_pca(const Eigen::Matrix3d& covariance_matrix) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covariance_matrix);
    Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
    Eigen::Matrix3d eigenvectors = eigensolver.eigenvectors();

    visualization_msgs::Marker marker;
    marker.header.frame_id = "camera_init";
    marker.header.stamp = ros::Time().fromSec(lidar_end_time);
    marker.ns = "pca";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    // Marker位置
    marker.pose.position.x = pos_lid.x();
    marker.pose.position.y = pos_lid.y();
    marker.pose.position.z = pos_lid.z();

    // 由于输入的是拟合平面法向量，所以主方向（2）是约束最多的方向
    // Marker方向 - 使用 PCA 的最小方向
    Eigen::Vector3d axis = eigenvectors.col(0).normalized();
    Eigen::Quaterniond quat;
    quat.setFromTwoVectors(Eigen::Vector3d::UnitX(), axis);

    marker.pose.orientation.x = quat.x();
    marker.pose.orientation.y = quat.y();
    marker.pose.orientation.z = quat.z();
    marker.pose.orientation.w = quat.w();

    // Marker尺寸（与特征值成比例）
    eigenvalues = eigenvalues.normalized();
    marker.scale.x = eigenvalues(2);  // 最大特征值
    marker.scale.y = eigenvalues(0);
    marker.scale.z = eigenvalues(0);  // 最小特征值

    // std::cout << eigenvalues << std::endl;

    // Marker颜色
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.5f;

    // marker.lifetime = rclcpp::Duration(0);

    marker_pub_.publish(marker);
}

void LaserMapping::dump_lio_state_to_log(FILE* fp) {
    auto x_ = kf.get_x();
    V3D rot_ang = x_.rot.matrix().eulerAngles(0, 1, 2);  // ZYX顺序

    fprintf(fp, "%lf ,", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf ,%lf ,%lf ,", rot_ang(0), rot_ang(1), rot_ang(2));  // Angle
    fprintf(fp, "%lf ,%lf ,%lf ,", x_.pos(0), x_.pos(1), x_.pos(2));     // Pos
    fprintf(fp, "%lf ,%lf ,%lf ,", x_.vel(0), x_.vel(1), x_.vel(2));     // Vel

    // Debug
    fprintf(fp, "%d , - ,", feats_down_size);
    auto& rt = kf.match_rate;
    fprintf(fp, "%.3f ,%.3f ,%.3f,", rt.all, rt.lio, rt.prior);
    auto& dt = kf.degen_rate;
    fprintf(fp, "%.2f ,%.2f ,", dt.rate_1, dt.rate_2);
    fprintf(fp, "%.2f ,%.2f ,%.2f,%.2f ,%.2f ,%.2f,", dt.pos_min, dt.pos_trace, dt.rot_min, dt.rot_trace, dt.all_min, dt.all_trace);

    // fprintf(fp, "%d ,%d , ,", feats_down_size, (int)(kf.get_match_ratio() * 100));
    // fprintf(fp, "%lf ,%lf ,%lf ,", x_.bg(0), x_.bg(1), x_.bg(2));       // Bias_g
    // fprintf(fp, "%lf ,%lf ,%lf ,", x_.ba(0), x_.ba(1), x_.ba(2));       // Bias_a
    // fprintf(fp, "%lf ,%lf ,%lf ", x_.grav(0), x_.grav(1), x_.grav(2));  // Bias_a
    fprintf(fp, "\r\n");

    fflush(fp);

    auto quad = x_.rot.unit_quaternion();
    fout_traj << Measures.lidar_beg_time << " " << x_.pos(0) << " " << x_.pos(1) << " " << x_.pos(2) << " " << quad.x() << " " << quad.y() << " " << quad.z()
              << " " << quad.w() << std::endl;
}

#include <pcl/io/pcd_io.h>

void LaserMapping::saveMap() { saveMap(savemap_dir); }
void LaserMapping::saveMap(const std::string& path) {
    if (pcd_save_en == 1) {
        for (auto& cloud : pcl_save_block) *pcl_wait_save += std::move(*cloud);
        try {
            std::cout << "ori points num: " << pcl_wait_save->points.size() << std::endl;
            downSizeFilterSaveMap.setInputCloud(pcl_wait_save);
            downSizeFilterSaveMap.filter(*pcl_wait_save);
            std::cout << "ds points num: " << pcl_wait_save->points.size() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception caught while filtering point cloud: " << e.what() << std::endl;
        }

        std::cout << " pcd save to: " << path << std::endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_wait_save_xyzi{new pcl::PointCloud<pcl::PointXYZI>()};
        pcl::copyPointCloud(*pcl_wait_save, *pcl_wait_save_xyzi);
        pcl::io::savePCDFileBinary(path + "scans.pcd", *pcl_wait_save_xyzi);
        // pcl::io::savePCDFileBinary(path + "effect.pcd", *pcl_effect_save);
    }
}