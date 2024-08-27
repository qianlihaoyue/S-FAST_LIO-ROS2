#include "laserMapping.hpp"
#include "common_lib.h"

void LaserMapping::initLIO() {
    cout << "lid_topic: " << lid_topic << endl;

    sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, &LaserMapping::livox_pcl_cbk, this)
                                        : nh.subscribe(lid_topic, 200000, &LaserMapping::standard_pcl_cbk, this);
    sub_imu = nh.subscribe(imu_topic, 200000, &LaserMapping::imu_cbk, this);

    pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    pubLaserCloudNoEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_noeffected", 100000);
    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    pubLaserCloudFlash = nh.advertise<sensor_msgs::PointCloud2>("/cloudFlash", 100000);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    marker_pub_ = nh.advertise<visualization_msgs::Marker>("pca_marker", 100000);

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    // downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    p_imu->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, acc_cov), V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov),
                     V3D(b_acc_cov, b_acc_cov, b_acc_cov));
}

void LaserMapping::initLoc() {
    nh.param<bool>("loc/en", loc_mode, false);
    if (loc_mode) {
        nh.param<string>("loc/loadmap_dir", loadmap_dir, string(ROOT_DIR) + "PCD/");

        // 加载读取点云数据到cloud中
        string all_points_dir(loadmap_dir + "scans.pcd");
        if (pcl::io::loadPCDFile<PointType>(all_points_dir, *prior_cloud) == -1) std::cerr << "Read file fail! " << all_points_dir << std::endl;
        ikdtree.set_downsample_param(filter_size_map_min);
        ikdtree.Build(prior_cloud->points);
        std::cout << "ori size: " << prior_cloud->points.size() << "---- ikdtree size: " << ikdtree.size() << std::endl;

        if (scan_pub_en) {
            PointVector().swap(ikdtree.PCL_Storage);
            ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
            CloudType::Ptr featsFromMap{new CloudType()};
            featsFromMap->points = ikdtree.PCL_Storage;
            std::cout << "---- PCL_Storage size: " << featsFromMap->points.size() << std::endl;
            downSizeFilterSurf.setLeafSize(1.0, 1.0, 1.0);
            downSizeFilterSaveMap.setInputCloud(featsFromMap);
            downSizeFilterSaveMap.filter(*featsFromMap);
            downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
            while (pubLaserCloudMap.getNumSubscribers() == 0) {
            };
            publish_cloud(pubLaserCloudMap, featsFromMap);
        }

        nh.param<vector<double>>("loc/init_guess", init_guess, vector<double>());
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

// 根据最新估计位姿  增量添加点云到map
void LaserMapping::map_incremental() {
    CloudType PointToAdd;
    CloudType PointNoNeedDownsample;
    CloudType PointSave;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++) {
        auto& p_world = feats_down_world->points[i];

        if (!Nearest_Points[i].empty() && flg_EKF_inited) {
            const PointVector& points_near = Nearest_Points[i];
            bool need_add = true;
            // BoxPointType Box_of_Point;
            PointType mid_point;  // 点所在体素的中心
            mid_point.x = floor(p_world.x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(p_world.y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(p_world.z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(p_world, mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
                PointNoNeedDownsample.push_back(p_world);  // 如果距离最近的点都在体素外，则该点不需要Downsample
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
            if (need_add)
                PointToAdd.push_back(p_world);
            else if (pcd_save_en) {
                // 只要最近点距离大于 filter_size_savemap, 或者高强度点（反光条）
                if (calc_dist(points_near[0], p_world) > filter_size_savemap || p_world.intensity > 100) PointSave.push_back(p_world);
            }
        } else {
            PointToAdd.push_back(p_world);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd.points, true);
    ikdtree.Add_Points(PointNoNeedDownsample.points, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();

    if (pcd_save_en) {
        *pcl_wait_save += PointToAdd;
        *pcl_wait_save += PointNoNeedDownsample;
        *pcl_wait_save += PointSave;
    }
}

std::mutex pcl_save_mutex_;
void LaserMapping::savemap_callback(const ros::TimerEvent& event) {
    if (pcd_save_en == false || pcl_wait_save->empty()) return;
    CloudType::Ptr pcl_buff_save{new CloudType()};
    {
        std::lock_guard<std::mutex> lock(pcl_save_mutex_);
        *pcl_buff_save = std::move(*pcl_wait_save);
    }
    downSizeFilterSaveMap.setInputCloud(pcl_buff_save);
    downSizeFilterSaveMap.filter(*pcl_buff_save);
    pcl_save_block.push_back(pcl_buff_save);
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
        pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

        flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);
        feats_down_size = feats_down_body->points.size();
        feats_down_world->resize(feats_down_size);

        if (feats_down_size < 5) {
            std::cerr << "No point, skip this scan!" << std::endl;
            return;
        }

        if (loc_mode == false && ikdtree.Root_Node == nullptr) {
            ikdtree.set_downsample_param(filter_size_map_min);
            for (int i = 0; i < feats_down_size; i++) pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            ikdtree.Build(feats_down_world->points);
            return;
        }

        double t1 = omp_get_wtime();
        /*** iterated state estimation ***/
        Nearest_Points.resize(feats_down_size);
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points);
        double t2 = omp_get_wtime();

        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

        /******* Publish odometry *******/
        publish_odometry(pubOdomAftMapped);

        /*** add the feature points to map kdtree ***/
        for (int i = 0; i < feats_down_size; i++) pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        if (loc_mode == false) map_incremental();

        /******* Publish points *******/
        if (path_en) publish_path(pubPath);
        if (scan_pub_en || pcd_save_en) {
            publish_frame_world();

            sensor_msgs::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*feats_undistort, laserCloudmsg);
            laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
            laserCloudmsg.header.frame_id = "body";
            pubLaserCloudFull_body.publish(laserCloudmsg);
        }
        publish_pca(kf.covariance_matrix);

        double t3 = omp_get_wtime();

        if (runtime_pos_log) {
            printf("ds: %d match: %d%% ", feats_down_size, (int)(kf.get_match_ratio() * 100));
            printf("[tim] ICP: %0.2f total: %0.2f", (t2 - t1) * 1000.0, (t3 - t0) * 1000.0);
            std::cout << std::endl;
            if (fp) dump_lio_state_to_log(fp);
        }
    }
}
