#include "laserMapping.hpp"
#include "common_lib.h"

void LaserMapping::initLIO() {
    if (p_pre->lidar_type == AVIA) {
        sub_pcl_livox =
            this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, std::bind(&LaserMapping::livox_pcl_cbk, this, std::placeholders::_1));
    } else {
        sub_pcl_pc = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(),
                                                                              std::bind(&LaserMapping::standard_pcl_cbk, this, std::placeholders::_1));
    }
    sub_imu = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 10, std::bind(&LaserMapping::imu_cbk, this, std::placeholders::_1));
    pubLaserCloudFull = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
    pubLaserCloudFull_body = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
    pubLaserCloudEffect = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
    pubLaserCloudMap = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
    pubOdomAftMapped = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
    pubPath = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
    path.header.stamp = this->get_clock()->now();
    path.header.frame_id = "camera_init";

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    timer_ = rclcpp::create_timer(this, this->get_clock(), std::chrono::milliseconds(10), std::bind(&LaserMapping::timer_callback, this));

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    // downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, acc_cov), V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov),
                     V3D(b_acc_cov, b_acc_cov, b_acc_cov));
}

void LaserMapping::initLoc() {
    declare_and_get_parameter<bool>("loc.en", loc_mode, false);
    if (loc_mode) {
        declare_and_get_parameter<string>("loc.loadmap_dir", loadmap_dir, string(ROOT_DIR) + "PCD/");

        // 加载读取点云数据到cloud中
        string all_points_dir(loadmap_dir + "scans.pcd");
        if (pcl::io::loadPCDFile<PointType>(all_points_dir, *prior_cloud) == -1) std::cerr << "Read file fail! " << all_points_dir << std::endl;
        ikdtree.set_downsample_param(filter_size_map_min);
        ikdtree.Build(prior_cloud->points);
        std::cout << "---- ikdtree size: " << ikdtree.size() << std::endl;

        if (scan_pub_en) {
            PointVector().swap(ikdtree.PCL_Storage);
            ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
            featsFromMap->clear();
            featsFromMap->points = ikdtree.PCL_Storage;
            while (pubLaserCloudMap->get_subscription_count() == 0) std::this_thread::sleep_for(std::chrono::seconds(1));
            publish_map(pubLaserCloudMap);
        }

        declare_and_get_parameter<vector<double>>("loc.init_guess", init_guess, vector<double>());
        state_point = kf.get_x();
        // state_point.pos = Eigen::Vector3d(init_guess[3], init_guess[4], init_guess[5]);

        Eigen::Matrix4d tran_body_to_map =
            pcl::getTransformation(init_guess[3], init_guess[4], init_guess[5], init_guess[0], init_guess[1], init_guess[2]).cast<double>().matrix();
        state_point.rot = Sophus::SO3(tran_body_to_map.block<3, 3>(0, 0));
        state_point.pos = tran_body_to_map.block<3, 1>(0, 3);
        // 应用变换矩阵到当前状态
        // tmp_state.rot = tran_body_to_map.block<3, 3>(0, 0) * tmp_state.rot;
        // tmp_state.pos = tran_body_to_map.block<3, 3>(0, 0) * tmp_state.pos + tran_body_to_map.block<3, 1>(0, 3);

        kf.change_x(state_point);

        std::cout << "init pos: " << state_point.pos << std::endl << "init rot: " << state_point.rot << std::endl;
    }
}

void LaserMapping::lasermap_fov_segment() {
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
void LaserMapping::map_incremental() {
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++) {
        // 转换到世界坐标系
        // pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

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
            for (int j = 0; j < kf.NUM_MATCH_POINTS; j++) {
                if (points_near.size() < kf.NUM_MATCH_POINTS) break;
                // 如果近邻点距离 < 当前点距离，不添加该点
                if (calc_dist(points_near[j], mid_point) < dist) {
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

void LaserMapping::timer_callback() {
    if (sync_packages(Measures)) {
        double t0 = omp_get_wtime();

        if (flg_first_scan) {
            first_lidar_time = Measures.lidar_beg_time;
            p_imu->first_lidar_time = first_lidar_time;
            flg_first_scan = false;
            return;
        }

        p_imu->Process(Measures, kf, feats_undistort);

        if (feats_undistort->empty() || (feats_undistort == NULL)) {
            std::cerr << "No point, skip this scan!" << std::endl;
            return;
        }

        state_point = kf.get_x();
        // std::cout << "irot" << state_point.rot << std::endl;
        pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

        flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

        // FIXME:
        if (loc_mode == false) lasermap_fov_segment();  // 更新localmap边界，然后降采样当前帧点云

        PointCloudXYZI::Ptr cloud_de(new PointCloudXYZI), cloud_norm(new PointCloudXYZI);
        for (auto& pt : feats_undistort->points) {
            if (pt.intensity > 100)
                cloud_de->points.push_back(pt);
            else
                cloud_norm->points.push_back(pt);
        }
        *feats_undistort = std::move(*cloud_norm);
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);
        feats_down_size = feats_down_body->points.size();
        feats_down_world->resize(feats_down_size);

        if (feats_down_size < 5) {
            std::cerr << "No point, skip this scan!" << std::endl;
            return;
        }

        // 初始化ikdtree(ikdtree为空时)
        if (loc_mode == false && ikdtree.Root_Node == nullptr) {
            ikdtree.set_downsample_param(filter_size_map_min);
            for (int i = 0; i < feats_down_size; i++) pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            ikdtree.Build(feats_down_world->points);  // 根据世界坐标系下的点构建ikdtree
            return;
        }

        double t1 = omp_get_wtime();
        /*** iterated state estimation ***/
        PointCloudXYZI::Ptr cloud_ekf(new PointCloudXYZI);
        *cloud_ekf += *feats_down_body;
        *cloud_ekf += *cloud_de;
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, cloud_ekf, ikdtree, Nearest_Points);
        double t2 = omp_get_wtime();

        state_point = kf.get_x();
        // std::cout << "lrot" << state_point.rot << std::endl;
        pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

        /******* Publish odometry *******/
        publish_odometry(pubOdomAftMapped, tf_broadcaster);

        /*** add the feature points to map kdtree ***/
        for (int i = 0; i < feats_down_size; i++) pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        if (loc_mode == false) map_incremental();

        /******* Publish points *******/
        if (path_en) publish_path(pubPath);
        PointCloudXYZI::Ptr cloud_dew(new PointCloudXYZI(1, cloud_de->size()));
        for (int i = 0; i < cloud_de->size(); i++) pointBodyToWorld(&(cloud_de->points[i]), &(cloud_dew->points[i]));
        *feats_down_world += *cloud_dew;
        if (scan_pub_en || pcd_save_en) publish_frame_world(pubLaserCloudFull);

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*kf.cloudKeyPoses3D, laserCloudmsg);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudEffect->publish(laserCloudmsg);

        double t3 = omp_get_wtime();

        if (runtime_pos_log) {
            printf("norm: %d de: %d match: %d%% ", feats_down_size, (int)cloud_de->points.size(), (int)(kf.get_match_ratio() * 100));
            printf("[tim] ICP: %0.2f total: %0.2f", (t2 - t1) * 1000.0, (t3 - t0) * 1000.0);
            std::cout << std::endl;
            if (fp) dump_lio_state_to_log(fp);
        }
    }
}
