#pragma once

#include <deque>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <lt_lio/Pose6D.h>

#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>

// #define PI_M (3.14159265358)
#define G_m_s2 (9.81)  // Gravaty const in GuangDong/China
// #define NUM_MATCH_POINTS (10)  // 5

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "Log/" + name))

typedef lt_lio::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> CloudType;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::Vector3f V3F;
typedef Eigen::Matrix3f M3F;

#define Eye3d M3D::Identity()
#define Eye3f M3F::Identity()
#define Zero3d V3D::Zero()
#define Zero3f V3F::Zero()

struct MeasureGroup  // Lidar data and imu dates for the current process
{
    MeasureGroup() {
        lidar_beg_time = 0.0;
        this->lidar.reset(new CloudType());
    };
    double lidar_beg_time;
    double lidar_end_time;
    CloudType::Ptr lidar;
    std::deque<sensor_msgs::Imu::ConstPtr> imu;
};

template <typename T>
auto set_pose6d(const double t, const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 1>& g, const Eigen::Matrix<T, 3, 1>& v,
                const Eigen::Matrix<T, 3, 1>& p, const Eigen::Matrix<T, 3, 3>& R) {
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++) {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++) rot_kp.rot[i * 3 + j] = R(i, j);
    }
    return std::move(rot_kp);
}

inline float calc_dist(PointType p1, PointType p2) {
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

template <typename T>
bool esti_plane(Eigen::Matrix<T, 4, 1>& pca_result, const PointVector& point, const T& threshold, int NUM_MATCH_POINTS) {
    // Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
    // Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;
    // 使用动态大小的矩阵
    Eigen::Matrix<T, Eigen::Dynamic, 3> A(NUM_MATCH_POINTS, 3);
    Eigen::Matrix<T, Eigen::Dynamic, 1> b(NUM_MATCH_POINTS);

    A.setZero();
    b.setOnes();
    b *= -1.0f;

    // 求A/Dx + B/Dy + C/Dz + 1 = 0 的参数
    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        A(j, 0) = point[j].x;
        A(j, 1) = point[j].y;
        A(j, 2) = point[j].z;
    }

    Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    // pca_result是平面方程的4个参数  /n是为了归一化
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    // 如果几个点中有距离该平面>threshold的点 认为是不好的平面 返回false
    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold) {
            return false;
        }
    }
    return true;
}
