
#pragma once

#include <pcl/common/transforms.h>
#include "esekfom.hpp"

// IMU数据预处理：IMU初始化，IMU正向传播，反向传播补偿运动失真

#define MAX_INI_COUNT (10)  // 最大迭代次数
// 判断点的时间先后顺序(注意curvature中存储的是时间戳)
constexpr bool time_list(PointType& x, PointType& y) { return (x.curvature < y.curvature); };

class ImuProcess {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();
    ~ImuProcess();

    void Reset();
    void set_param(const V3D& transl, const M3D& rot, const V3D& gyr, const V3D& acc, const V3D& gyr_bias, const V3D& acc_bias);
    Eigen::Matrix<double, 12, 12> Q;  // 噪声协方差矩阵  对应论文式(8)中的Q
    void Process(MeasureGroup& meas, esekfom::esekf& kf_state, PointCloudXYZI::Ptr& pcl_un_);

    V3D cov_acc;              // 加速度协方差
    V3D cov_gyr;              // 角速度协方差
    V3D cov_acc_scale;        // 外部传入的 初始加速度协方差
    V3D cov_gyr_scale;        // 外部传入的 初始角速度协方差
    V3D cov_bias_gyr;         // 角速度bias的协方差
    V3D cov_bias_acc;         // 加速度bias的协方差
    double first_lidar_time;  // 当前帧第一个点云时间

    // wheel
    bool USE_WHEEL = false;
    M3D cov_wheel_nhc{Eye3d * 0.01};

    V3D pos_wheel{V3D::Zero()};
    Eigen::Vector3d wheel_velocity;
    void set_param_wheel(bool use_wheel, const M3D& cov, const V3D& init_pos) {
        USE_WHEEL = use_wheel;
        cov_wheel_nhc = cov;
        pos_wheel = init_pos;
    }

private:
    void IMU_init(const MeasureGroup& meas, esekfom::esekf& kf_state, int& N);
    void UndistortPcl(MeasureGroup& meas, esekfom::esekf& kf_state, PointCloudXYZI& pcl_in_out);
    // 噪声协方差Q的初始化(对应公式(8)的Q, 在IMU_Processing.hpp中使用)
    Eigen::Matrix<double, 12, 12> process_noise_cov() {
        Eigen::Matrix<double, 12, 12> Q = Eigen::MatrixXd::Zero(12, 12);
        Q.block<3, 3>(0, 0) = 0.0001 * Eigen::Matrix3d::Identity();
        Q.block<3, 3>(3, 3) = 0.0001 * Eigen::Matrix3d::Identity();
        Q.block<3, 3>(6, 6) = 0.00001 * Eigen::Matrix3d::Identity();
        Q.block<3, 3>(9, 9) = 0.00001 * Eigen::Matrix3d::Identity();
        return Q;
    }

    PointCloudXYZI::Ptr cur_pcl_un_;                  // 当前帧点云未去畸变
    sensor_msgs::msg::Imu::ConstSharedPtr last_imu_;  // 上一帧imu
    vector<Pose6D> IMUpose;                           // 存储imu位姿(反向传播用)
    M3D Lidar_R_wrt_IMU;                              // lidar到IMU的旋转外参
    V3D Lidar_T_wrt_IMU;                              // lidar到IMU的平移外参
    V3D mean_acc;                                     // 加速度均值,用于计算方差
    V3D mean_gyr;                                     // 角速度均值，用于计算方差
    V3D angvel_last;                                  // 上一帧角速度
    V3D acc_s_last;                                   // 上一帧加速度
    double start_timestamp_;                          // 开始时间戳
    double last_lidar_end_time_;                      // 上一帧结束时间戳
    int init_iter_num = 1;                            // 初始化迭代次数
    bool b_first_frame_ = true;                       // 是否是第一帧
    bool imu_need_init_ = true;                       // 是否需要初始化imu
};
