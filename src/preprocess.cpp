#include "preprocess.h"

Preprocess::Preprocess() : lidar_type(AVIA), blind(0.01), point_filter_num(1) {
    N_SCANS = 6;
    SCAN_RATE = 10;
    given_offset_time = false;
}

Preprocess::~Preprocess() {}

void Preprocess::set(int lid_type, double bld, int pfilt_num) {
    lidar_type = lid_type;
    blind = bld;
    point_filter_num = pfilt_num;
}

void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, pcl::PointCloud<PointType>::Ptr &pcl_out) {
    avia_handler(msg);
    *pcl_out = pl_surf;
}

void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud<PointType>::Ptr &pcl_out) {
    switch (time_unit) {
        case SEC:
            time_unit_scale = 1.e3f;
            break;
        case MS:
            time_unit_scale = 1.f;
            break;
        case US:
            time_unit_scale = 1.e-3f;
            break;
        case NS:
            time_unit_scale = 1.e-6f;
            break;
        default:
            time_unit_scale = 1.f;
            break;
    }

    switch (lidar_type) {
        case OUST64:
            oust64_handler(msg);
            break;

        case VELO16:
            velodyne_handler(msg);
            break;

        case MID360:
            mid360_handler(msg);
            break;

        default:
            printf("Error LiDAR Type");
            break;
    }
    *pcl_out = pl_surf;
}

void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    pl_surf.clear();
    pl_full.clear();

    int plsize = msg->point_num;
    pl_surf.reserve(plsize);
    pl_full.resize(plsize);

    uint valid_num = 0;

    for (uint i = 1; i < plsize; i++) {
        if ((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
            valid_num++;
            if (valid_num % point_filter_num == 0) {
                pl_full[i].x = msg->points[i].x;
                pl_full[i].y = msg->points[i].y;
                pl_full[i].z = msg->points[i].z;


                // if (msg->points[i].z > 4) continue;

                float range = pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z;
                if (range < (blind * blind)) continue;
                if (range > (max_range * max_range)) continue;

                // const double min_azimuth = M_PI / 4 * 3;
                // const double max_azimuth = M_PI / 3 + M_PI;
                double azimuth = atan2(pl_full[i].y, pl_full[i].x);
                // if (abs(azimuth) < min_azimuth) continue;

                const double min_azimuth = M_PI / 4;
                if (abs(azimuth) > min_azimuth) continue;

                // const double min_elevation = -M_PI / 6;
                // const double max_elevation = M_PI / 6;
                // double elevation = atan2(pl_full[i].z, sqrt(pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y));
                // if (elevation < min_elevation) continue;
                // if (elevation > max_elevation) continue;

                pl_full[i].intensity = msg->points[i].reflectivity;
                pl_full[i].curvature = msg->points[i].offset_time / float(1000000);  // use curvature as time of each laser points, curvature unit: ms

                if (((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) || (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) ||
                     (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7))) {
                    pl_surf.push_back(pl_full[i]);
                }
            }
        }
    }
}

void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    pl_surf.clear();
    pl_full.clear();
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    pl_surf.reserve(plsize);

    double time_stamp = msg->header.stamp.toSec();
    // cout << "===================================" << endl;
    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % point_filter_num != 0) continue;
        int line = pl_orig.points[i].ring;
        if (line % 8 != 0) continue;

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < (blind * blind)) continue;
        // if (range > (max_range * max_range)) continue;

        Eigen::Vector3d pt_vec;
        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;

        // const double min_azimuth = M_PI / 2;
        // double azimuth = atan2(added_pt.y, added_pt.x);
        // if (abs(azimuth) > min_azimuth) continue;

        added_pt.curvature = pl_orig.points[i].t * time_unit_scale;  // curvature unit: ms

        pl_surf.points.push_back(added_pt);
    }
}

void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    pl_surf.clear();
    pl_full.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * SCAN_RATE;  // scan angular velocity
    std::vector<bool> is_first(N_SCANS, true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);    // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0) {
        given_offset_time = true;
    } else {
        given_offset_time = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        double yaw_end = yaw_first;
        int layer_first = pl_orig.points[0].ring;
        for (uint i = plsize - 1; i > 0; i--) {
            if (pl_orig.points[i].ring == layer_first) {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    for (int i = 0; i < plsize; i++) {
        PointType added_pt;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

        if (!given_offset_time) {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer]) {
                // printf("layer: %d; is first: %d", layer, is_first[layer]);
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.curvature = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer]) {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
            } else {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.curvature < time_last[layer]) added_pt.curvature += 360.0 / omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
        }

        if (i % point_filter_num == 0) {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind)) {
                pl_surf.points.push_back(added_pt);
            }
        }
    }
}

void Preprocess::mid360_handler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    pcl::PointCloud<mid360_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    pl_surf.clear();
    pl_surf.reserve(plsize);

    for (uint i = 1; i < plsize; i++) {
        if (i % point_filter_num != 0) continue;
        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
        if (range < (blind * blind)) continue;
        if (range > (max_range * max_range)) continue;

        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;

        const double min_azimuth = M_PI / 4 * 3;
        // const double max_azimuth = M_PI / 3 + M_PI;
        double azimuth = atan2(added_pt.y, added_pt.x);
        if (abs(azimuth) < min_azimuth) continue;

        // const double min_elevation = -M_PI / 6;
        const double max_elevation = M_PI / 4;
        double elevation = atan2(added_pt.z, sqrt(added_pt.x * added_pt.x + added_pt.y * added_pt.y));
        // if (elevation < min_elevation) continue;
        if (elevation > max_elevation) continue;

        pl_surf.points.push_back(added_pt);
    }
}