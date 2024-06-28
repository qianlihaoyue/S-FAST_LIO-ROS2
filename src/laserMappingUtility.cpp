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
    declare_and_get_parameter<bool>("publish.path_en", path_en, true);
    declare_and_get_parameter<bool>("publish.scan_publish_en", scan_pub_en, true);
    declare_and_get_parameter<bool>("publish.dense_publish_en", dense_pub_en, false);
    declare_and_get_parameter<bool>("publish.runtime_pos_log", runtime_pos_log, true);

    declare_and_get_parameter<string>("common.lid_topic", lid_topic, "/livox/lidar");
    declare_and_get_parameter<string>("common.imu_topic", imu_topic, "/livox/imu");
    declare_and_get_parameter<bool>("common.time_sync_en", time_sync_en, false);  // 时间同步
    declare_and_get_parameter<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);

    declare_and_get_parameter<double>("common.filter_size_surf", filter_size_surf_min, 0.5);
    declare_and_get_parameter<double>("common.filter_size_map", filter_size_map_min, 0.5);
    declare_and_get_parameter<double>("common.cube_len", cube_len, 200);  // 地图的局部区域的长度（FastLio2论文中有解释）

    declare_and_get_parameter<int>("kf.maximum_iter", kf.maximum_iter, 4);  // 卡尔曼滤波的最大迭代次数
    declare_and_get_parameter<double>("kf.epsi", kf.epsi, 0.001);
    declare_and_get_parameter<int>("kf.NUM_MATCH_POINTS", kf.NUM_MATCH_POINTS, 5);
    declare_and_get_parameter<float>("kf.plane_thr", kf.plane_thr, 0.1);
    declare_and_get_parameter<bool>("kf.extrinsic_est", kf.extrinsic_est, false);

    declare_and_get_parameter<float>("mapping.det_range", DET_RANGE, 300.f);    // 激光雷达的最大探测范围
    declare_and_get_parameter<double>("mapping.gyr_cov", gyr_cov, 0.1);         // IMU陀螺仪的协方差
    declare_and_get_parameter<double>("mapping.acc_cov", acc_cov, 0.1);         // IMU加速度计的协方差
    declare_and_get_parameter<double>("mapping.b_gyr_cov", b_gyr_cov, 0.0001);  // IMU陀螺仪偏置的协方差
    declare_and_get_parameter<double>("mapping.b_acc_cov", b_acc_cov, 0.0001);  // IMU加速度计偏置的协方差
    declare_and_get_parameter<bool>("mapping.extrinsic_est_en", extrinsic_est_en, false);

    declare_and_get_parameter<double>("preprocess.blind", p_pre->blind, 0.01);         // 最小距离阈值，即过滤掉0～blind范围内的点云
    declare_and_get_parameter<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);  // 激光雷达的类型
    declare_and_get_parameter<int>("preprocess.scan_line", p_pre->N_SCANS, 16);        // 激光雷达扫描的线数（livox avia为6线）
    declare_and_get_parameter<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
    declare_and_get_parameter<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
    declare_and_get_parameter<int>("preprocess.point_filter_num", p_pre->point_filter_num, 2);  // 采样间隔，即每隔point_filter_num个点取1个点

    declare_and_get_parameter<int>("pcd_save.pcd_save_en", pcd_save_en, false);
    declare_and_get_parameter<double>("pcd_save.filter_size_savemap", filter_size_savemap, 0.2);
    declare_and_get_parameter<string>("pcd_save.savemap_dir", savemap_dir, string(ROOT_DIR) + "PCD/");
    downSizeFilterSaveMap.setLeafSize(filter_size_savemap, filter_size_savemap, filter_size_savemap);

    createDirectoryIfNotExists(savemap_dir);
    fp = fopen((savemap_dir + "/pos_log.csv").c_str(), "w");
    if (fp == nullptr) std::cerr << "Failed to open file for writing" << std::endl;
    fout_traj.open(savemap_dir + "/traj_tum.txt", std::ios::out);
    fout_traj.setf(std::ios::fixed, std::ios::floatfield);
    fout_traj.precision(6);

    declare_and_get_parameter<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());  // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    declare_and_get_parameter<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());  // 雷达相对于IMU的外参R

    std::cout << "Lidar_type: " << p_pre->lidar_type << std::endl;
}

void LaserMapping::standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) {
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
    // sig_buffer.notify_all();
}

void LaserMapping::livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) {
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
    // sig_buffer.notify_all();
}

void LaserMapping::imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in) {
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

void LaserMapping::publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull) {
    PointCloudXYZI::Ptr laserCloudEffect(new PointCloudXYZI());
    for (auto& i : kf.effect_idx) laserCloudEffect->points.push_back(feats_down_world->points[i]);

    if (pcd_save_en == 1 && scan_num % 5 == 0) {
        PointCloudXYZI tmpcloud;
        downSizeFilterSaveMap.setInputCloud(feats_down_world);
        downSizeFilterSaveMap.filter(tmpcloud);
        *pcl_wait_save += tmpcloud;

        downSizeFilterSaveMap.setInputCloud(laserCloudEffect);
        downSizeFilterSaveMap.filter(tmpcloud);
        *pcl_effect_save += tmpcloud;
    }

    if (pcd_save_en == 1 && scan_num % 100 == 0) {
        downSizeFilterSaveMap.setInputCloud(pcl_wait_save);
        downSizeFilterSaveMap.filter(*pcl_wait_save);

        downSizeFilterSaveMap.setInputCloud(pcl_effect_save);
        downSizeFilterSaveMap.filter(*pcl_effect_save);
    }

    if (scan_pub_en) {
        *laserCloudEffect = *feats_down_world;
        for (auto& pt : laserCloudEffect->points) pt.intensity = 0;
        for (auto& i : kf.effect_idx) laserCloudEffect->points[i].intensity = 100;
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudEffect, laserCloudmsg);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        // pubLaserCloudEffect->publish(laserCloudmsg);

        // sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*feats_down_world, laserCloudmsg);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull->publish(laserCloudmsg);
    }
}

void LaserMapping::publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap) {
    sensor_msgs::msg::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap->publish(laserCloudMap);
}

void LaserMapping::publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped,
                                    std::unique_ptr<tf2_ros::TransformBroadcaster>& tf_br) {
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

void LaserMapping::publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath) {
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

void LaserMapping::dump_lio_state_to_log(FILE* fp) {
    auto x_ = kf.get_x();
    V3D rot_ang = x_.rot.matrix().eulerAngles(0, 1, 2);  // ZYX顺序

    fprintf(fp, "%lf ,", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf ,%lf ,%lf ,", rot_ang(0), rot_ang(1), rot_ang(2));  // Angle
    fprintf(fp, "%lf ,%lf ,%lf ,", x_.pos(0), x_.pos(1), x_.pos(2));     // Pos

    // Debug
    fprintf(fp, "%d ,%d ,", feats_down_size, (int)(kf.get_match_ratio() * 100));

    fprintf(fp, "%lf ,%lf ,%lf ,", x_.vel(0), x_.vel(1), x_.vel(2));    // Vel
    fprintf(fp, "%lf ,%lf ,%lf ,", x_.bg(0), x_.bg(1), x_.bg(2));       // Bias_g
    fprintf(fp, "%lf ,%lf ,%lf ,", x_.ba(0), x_.ba(1), x_.ba(2));       // Bias_a
    fprintf(fp, "%lf ,%lf ,%lf ", x_.grav(0), x_.grav(1), x_.grav(2));  // Bias_a
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
        try {
            std::cout << "ori points num: " << pcl_wait_save->points.size() << std::endl;
            downSizeFilterSaveMap.setInputCloud(pcl_wait_save);
            downSizeFilterSaveMap.filter(*pcl_wait_save);
            std::cout << "ds points num: " << pcl_wait_save->points.size() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception caught while filtering point cloud: " << e.what() << std::endl;
        }

        std::cout << " pcd save to: " << path << std::endl;
        pcl::io::savePCDFileBinary(path + "scans.pcd", *pcl_wait_save);
        pcl::io::savePCDFileBinary(path + "effect.pcd", *pcl_effect_save);
    }
}