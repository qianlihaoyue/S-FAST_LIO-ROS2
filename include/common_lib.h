#pragma once

#include <deque>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <fast_lio/msg/pose6_d.hpp>

#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/imu.hpp>

// #define PI_M (3.14159265358)
#define G_m_s2 (9.81)  // Gravaty const in GuangDong/China
// #define NUM_MATCH_POINTS (10)  // 5

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "Log/" + name))

typedef fast_lio::msg::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
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
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;
    double lidar_end_time;
    PointCloudXYZI::Ptr lidar;
    std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu;
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

#include <Eigen/Dense>

template <typename T>
bool esti_line(Eigen::Matrix<T, 6, 1>& line_result, const PointVector& points, const T& threshold, int NUM_MATCH_POINTS) {
    // 使用动态大小的矩阵
    Eigen::Matrix<T, Eigen::Dynamic, 3> A(NUM_MATCH_POINTS, 3);
    Eigen::Matrix<T, Eigen::Dynamic, 1> b(NUM_MATCH_POINTS);

    A.setZero();
    b.setZero();

    // 将点的坐标填入A矩阵
    for (int i = 0; i < NUM_MATCH_POINTS; ++i) {
        A(i, 0) = points[i].x;
        A(i, 1) = points[i].y;
        A(i, 2) = points[i].z;
    }

    // 计算中心点
    Eigen::Matrix<T, 3, 1> centroid = A.colwise().mean();

    // 将点减去中心点
    for (int i = 0; i < NUM_MATCH_POINTS; ++i) {
        A(i, 0) -= centroid(0);
        A(i, 1) -= centroid(1);
        A(i, 2) -= centroid(2);
    }

    // 计算协方差矩阵
    Eigen::Matrix<T, 3, 3> covariance = A.transpose() * A;

    // 计算特征值和特征向量
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> eigen_solver(covariance);
    Eigen::Matrix<T, 3, 1> eigen_values = eigen_solver.eigenvalues();
    Eigen::Matrix<T, 3, 3> eigen_vectors = eigen_solver.eigenvectors();

    // 直线方向为最大特征值对应的特征向量
    Eigen::Matrix<T, 3, 1> direction = eigen_vectors.col(2);

    // 结果为方向向量和中心点
    line_result.template head<3>() = direction;
    line_result.template tail<3>() = centroid;

    // 检查点与拟合直线的距离是否超过阈值
    for (int i = 0; i < NUM_MATCH_POINTS; ++i) {
        Eigen::Matrix<T, 3, 1> point(points[i].x, points[i].y, points[i].z);
        Eigen::Matrix<T, 3, 1> projected_point = centroid + direction * ((point - centroid).dot(direction));
        T distance = (point - projected_point).norm();
        if (distance > threshold) {
            return false;
        }
    }

    return true;
}

inline double get_time_sec(const builtin_interfaces::msg::Time& time) { return rclcpp::Time(time).seconds(); }

inline rclcpp::Time get_ros_time(double timestamp) {
    int32_t sec = std::floor(timestamp);
    auto nanosec_d = (timestamp - std::floor(timestamp)) * 1e9;
    uint32_t nanosec = nanosec_d;
    return rclcpp::Time(sec, nanosec);
}
