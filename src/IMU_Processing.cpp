#include "IMU_Processing.hpp"
#include "common_lib.h"
#include <Eigen/src/Core/Matrix.h>
#include <geometry_msgs/msg/detail/twist_with_covariance__struct.hpp>

ImuProcess::ImuProcess() : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1) {
    init_iter_num = 1;                           // 初始化迭代次数
    Q = process_noise_cov();                     // 调用use-ikfom.hpp里面的process_noise_cov初始化噪声协方差
    cov_acc = V3D(0.1, 0.1, 0.1);                // 加速度协方差初始化
    cov_gyr = V3D(0.1, 0.1, 0.1);                // 角速度协方差初始化
    cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001);  // 角速度bias协方差初始化
    cov_bias_acc = V3D(0.0001, 0.0001, 0.0001);  // 加速度bias协方差初始化
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;                          // 上一帧角速度初始化
    Lidar_T_wrt_IMU = Zero3d;                      // lidar到IMU的位置外参初始化
    Lidar_R_wrt_IMU = Eye3d;                       // lidar到IMU的旋转外参初始化
    last_imu_.reset(new sensor_msgs::msg::Imu());  // 上一帧imu初始化
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()  // 重置参数
{
    // std::cout << "Reset ImuProcess");
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;
    imu_need_init_ = true;                         // 是否需要初始化imu
    start_timestamp_ = -1;                         // 开始时间戳
    init_iter_num = 1;                             // 初始化迭代次数
    IMUpose.clear();                               // imu位姿清空
    last_imu_.reset(new sensor_msgs::msg::Imu());  // 上一帧imu初始化
    cur_pcl_un_.reset(new PointCloudXYZI());       // 当前帧点云未去畸变初始化
}

// 传入外部参数
void ImuProcess::set_param(const V3D& transl, const M3D& rot, const V3D& gyr, const V3D& acc, const V3D& gyr_bias, const V3D& acc_bias) {
    Lidar_T_wrt_IMU = transl;
    Lidar_R_wrt_IMU = rot;
    cov_gyr_scale = gyr;
    cov_acc_scale = acc;
    cov_bias_gyr = gyr_bias;
    cov_bias_acc = acc_bias;
}

// IMU初始化：利用开始的IMU帧的平均值初始化状态量x
void ImuProcess::IMU_init(const MeasureGroup& meas, esekfom::esekf& kf_state, int& N) {
    // MeasureGroup这个struct表示当前过程中正在处理的所有数据，包含IMU队列和一帧lidar的点云 以及lidar的起始和结束时间
    // 初始化重力、陀螺仪偏差、acc和陀螺仪协方差  将加速度测量值归一化为单位重力   **/
    V3D cur_acc, cur_gyr;

    if (b_first_frame_)  // 如果为第一帧IMU
    {
        Reset();  // 重置IMU参数
        N = 1;    // 将迭代次数置1
        b_first_frame_ = false;
        const auto& imu_acc = meas.imu.front()->linear_acceleration;  // IMU初始时刻的加速度
        const auto& gyr_acc = meas.imu.front()->angular_velocity;     // IMU初始时刻的角速度
        mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;                  // 第一帧加速度值作为初始化均值
        mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;                  // 第一帧角速度值作为初始化均值
        first_lidar_time = meas.lidar_beg_time;                       // 将当前IMU帧对应的lidar起始时间 作为初始时间
    }

    for (const auto& imu : meas.imu)  // 根据所有IMU数据，计算平均值和方差
    {
        const auto& imu_acc = imu->linear_acceleration;
        const auto& gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc += (cur_acc - mean_acc) / N;  // 根据当前帧和均值差作为均值的更新
        mean_gyr += (cur_gyr - mean_gyr) / N;

        cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) / N;
        cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) / N / N * (N - 1);

        N++;
    }

    state_ikfom init_state = kf_state.get_x();               // 在esekfom.hpp获得x_的状态
    init_state.grav = -mean_acc / mean_acc.norm() * G_m_s2;  // 得平均测量的单位方向向量 * 重力加速度预设值

    init_state.bg = mean_gyr;                   // 角速度测量作为陀螺仪偏差
    init_state.offset_T_L_I = Lidar_T_wrt_IMU;  // 将lidar和imu外参传入
    init_state.offset_R_L_I = Sophus::SO3(Lidar_R_wrt_IMU);

    /*** wheel velocity initialization ***/
    // if (USE_WHEEL && !meas.wheel.empty()) {
    //     double wheel_velocity = meas.wheel.back()->twist.twist.linear.x;
    //     V3D wheel_v_vec(wheel_velocity, 0.0, 0.0);
    //     init_state.vel = Wheel_R_wrt_IMU * (wheel_v_vec * wheel_s);  // initial rot is identity matrix
    // }

    kf_state.change_x(init_state);  // 将初始化后的状态传入esekfom.hpp中的x_

    Eigen::Matrix<double, 24, 24> init_P = Eigen::MatrixXd::Identity(24, 24);  // 在esekfom.hpp获得P_的协方差矩阵
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = init_P(23, 23) = 0.00001;
    kf_state.change_P(init_P);
    last_imu_ = meas.imu.back();

    // std::cout << "IMU init new -- init_state  " << init_state.pos  <<" " << init_state.bg <<" " << init_state.ba <<" " << init_state.grav << std::endl;
}

// 反向传播
void ImuProcess::UndistortPcl(MeasureGroup& meas, esekfom::esekf& kf_state, PointCloudXYZI& pcl_out) {
    /***将上一帧最后尾部的imu添加到当前帧头部的imu ***/
    auto v_imu = meas.imu;                                                  // 取出当前帧的IMU队列
    v_imu.push_front(last_imu_);                                            // 将上一帧最后尾部的imu添加到当前帧头部的imu
    const double& imu_end_time = get_time_sec(v_imu.back()->header.stamp);  // 拿到当前帧尾部的imu的时间
    const double& pcl_beg_time = meas.lidar_beg_time;                       // 点云开始和结束的时间戳
    const double& pcl_end_time = meas.lidar_end_time;

    // 根据点云中每个点的时间戳对点云进行重排序
    pcl_out = *(meas.lidar);
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);  // 这里curvature中存放了时间戳（在preprocess.cpp中）

    state_ikfom imu_state = kf_state.get_x();  // 获取上一次KF估计的后验状态作为本次IMU预测的初始状态
    IMUpose.clear();
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));
    // 将初始状态加入IMUpose中,包含有时间间隔，上一帧加速度，上一帧角速度，上一帧速度，上一帧位置，上一帧旋转矩阵

    /*** 前向传播 ***/
    V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;  // angvel_avr为平均角速度，acc_avr为平均加速度，acc_imu为imu加速度，vel_imu为imu速度，pos_imu为imu位置
    M3D R_imu;                                           // IMU旋转矩阵 消除运动失真的时候用

    double dt = 0;

    input_ikfom in;
    // 遍历本次估计的所有IMU测量并且进行积分，离散中值法 前向传播
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
        auto&& head = *(it_imu);      // 拿到当前帧的imu数据
        auto&& tail = *(it_imu + 1);  // 拿到下一帧的imu数据
        // 判断时间先后顺序：下一帧时间戳是否小于上一帧结束时间戳 不符合直接continue
        if (get_time_sec(tail->header.stamp) < last_lidar_end_time_) continue;

        angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),  // 中值积分
            0.5 * (head->angular_velocity.y + tail->angular_velocity.y), 0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x), 0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
            0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

        acc_avr = acc_avr * G_m_s2 / mean_acc.norm();  // 通过重力数值对加速度进行调整(除上初始化的IMU大小*9.8)

        // 如果IMU开始时刻早于上次雷达最晚时刻(因为将上次最后一个IMU插入到此次开头了，所以会出现一次这种情况)
        if (get_time_sec(head->header.stamp) < last_lidar_end_time_) {
            dt = get_time_sec(tail->header.stamp) - last_lidar_end_time_;  // 从上次雷达时刻末尾开始传播 计算与此次IMU结尾之间的时间差
        } else {
            dt = get_time_sec(tail->header.stamp) - get_time_sec(head->header.stamp);  // 两个IMU时刻之间的时间间隔
        }

        in.acc = acc_avr;  // 两帧IMU的中值作为输入in  用于前向传播
        in.gyro = angvel_avr;
        Q.block<3, 3>(0, 0).diagonal() = cov_gyr;  // 配置协方差矩阵
        Q.block<3, 3>(3, 3).diagonal() = cov_acc;
        Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
        Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;

        kf_state.predict(dt, Q, in);  // IMU前向传播，每次传播的时间间隔为dt

        // static Eigen::Vector3d pos_wheel = Wheel_R_wrt_IMU.transpose() * (-Wheel_T_wrt_IMU);

        /*** update by wheel meas ***/

        static double last_wheel_time = 0;
        if (!meas.wheel.empty()) {
            // 掐头
            while (get_time_sec(meas.wheel.front()->header.stamp) < get_time_sec(head->header.stamp)) {
                meas.wheel.pop_front();
            }

            // wheel 位于两个imu之间
            while (meas.wheel.size() && get_time_sec(meas.wheel.front()->header.stamp) < get_time_sec(tail->header.stamp)) {
#if 0
                double wheel_time = get_time_sec(meas.wheel.front()->header.stamp);
                if (last_wheel_time == 0) last_wheel_time = wheel_time;
                // pos 相对于车体，侧移的时候数据正常（y改变）
                auto& wpos = meas.wheel.front()->pose.pose.position;
                Eigen::Vector3d pos_delta(wpos.x, wpos.y, wpos.z);
                pos_delta = pos_delta * (wheel_time - last_wheel_time) / 0.05;  // 修正，当时间等于50ms时，pos_delta不变，防止丢包导致积分减少
                pos_delta = kf_state.get_x().rot * pos_delta;                   // 通过旋转修正
                // 自旋的时候仍然有问题
                pos_wheel += pos_delta;                                     // 累加
                // pos_w2imu = Wheel_R_wrt_IMU * pos_wheel + Wheel_T_wrt_IMU;  // 通过外参转移到imu坐标系

#endif
                auto&& wdata = meas.wheel.front()->twist.twist.linear;
                wheel_velocity << wdata.x, wdata.y, wdata.z;                              // x 0 0
                kf_state.update_iterated_dyn_share_wheel(cov_wheel_nhc, wheel_velocity);  // wheel更新

                meas.wheel.pop_front();

#if 0
                last_wheel_time = wheel_time;
#endif
            }
        }

        /*** update by gnss meas ***/
        if (!meas.gnss.empty()) {
            double gnss_time = get_time_sec(meas.gnss.front()->header.stamp);
            if (gnss_time < get_time_sec(head->header.stamp)) {
                meas.gnss.pop_front();
            } else {
                if (gnss_time < get_time_sec(tail->header.stamp)) {  // gnss位于两个imu之间

                    // FIXME:GNSS init
                    if (!gnss_heading_need_init_) {
                        auto&& pose = meas.gnss.front()->pose;

                        if (pose.covariance[0] < 200) {  // 航向初始化未完成或gnss水平噪声大于200不进行更新（遮挡区域）
                            // covariance
                            Eigen::Matrix3d gnss_cov = Eye3d;
                            gnss_cov(0, 0) = 0.01 * pose.covariance[0];
                            gnss_cov(1, 1) = 0.01 * pose.covariance[7];
                            gnss_cov(2, 2) = pose.covariance[14];
                            // pose
                            V3D gnss_pos(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
                            V3D gnss_pos_2 = GNSS_Heading * gnss_pos;
                            kf_state.update_iterated_dyn_share_gnss(gnss_cov, gnss_pos_2);

                            std::cout << "gnss update :" << gnss_pos.transpose() << " -> " << gnss_pos_2.transpose() << std::endl;
                        } else {
                            std::cerr << "gnss too noise !" << std::endl;
                        }
                    } else {
                        std::cerr << "gnss not initialized !" << std::endl;
                    }

                    meas.gnss.pop_front();
                }
            }
        }

        imu_state = kf_state.get_x();  // 更新IMU状态为积分后的状态
        // 更新上一帧角速度 = 后一帧角速度-bias
        angvel_last = V3D(tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z) - imu_state.bg;
        // 更新上一帧世界坐标系下的加速度 = R*(加速度-bias) - g
        acc_s_last = V3D(tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z) * G_m_s2 / mean_acc.norm();

        // std::cout << "acc_s_last: " << acc_s_last.transpose() << std::endl;
        // std::cout << "imu_state.ba: " << imu_state.ba.transpose() << std::endl;
        // std::cout << "imu_state.grav: " << imu_state.grav.transpose() << std::endl;
        acc_s_last = imu_state.rot * (acc_s_last - imu_state.ba) + imu_state.grav;
        // std::cout << "--acc_s_last: " << acc_s_last.transpose() << std::endl<< std::endl;

        double&& offs_t = get_time_sec(tail->header.stamp) - pcl_beg_time;  // 后一个IMU时刻距离此次雷达开始的时间间隔
        IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));
    }

    // 把最后一帧IMU测量也补上
    dt = abs(pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q, in);
    imu_state = kf_state.get_x();
    last_imu_ = meas.imu.back();          // 保存最后一个IMU测量，以便于下一帧使用
    last_lidar_end_time_ = pcl_end_time;  // 保存这一帧最后一个雷达测量的结束时间，以便于下一帧使用

    /***消除每个激光雷达点的失真（反向传播）***/
    if (pcl_out.points.begin() == pcl_out.points.end()) return;
    auto it_pcl = pcl_out.points.end() - 1;

    // 遍历每个IMU帧
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) {
        auto head = it_kp - 1;
        auto tail = it_kp;
        R_imu << MAT_FROM_ARRAY(head->rot);  // 拿到前一帧的IMU旋转矩阵
        // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
        vel_imu << VEC_FROM_ARRAY(head->vel);     // 拿到前一帧的IMU速度
        pos_imu << VEC_FROM_ARRAY(head->pos);     // 拿到前一帧的IMU位置
        acc_imu << VEC_FROM_ARRAY(tail->acc);     // 拿到后一帧的IMU加速度
        angvel_avr << VEC_FROM_ARRAY(tail->gyr);  // 拿到后一帧的IMU角速度

        // 之前点云按照时间从小到大排序过，IMUpose也同样是按照时间从小到大push进入的
        // 此时从IMUpose的末尾开始循环，也就是从时间最大处开始，因此只需要判断 点云时间需>IMU head时刻  即可   不需要判断 点云时间<IMU tail
        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
            dt = it_pcl->curvature / double(1000) - head->offset_time;  // 点到IMU开始时刻的时间间隔

            /*    P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)    */
            M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt).matrix());  // 点it_pcl所在时刻的旋转：前一帧的IMU旋转矩阵 * exp(后一帧角速度*dt)

            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);                                    // 点所在时刻的位置(雷达坐标系下)
            V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);  // 从点所在的世界位置-雷达末尾世界位置
            V3D P_compensate =
                imu_state.offset_R_L_I.matrix().transpose() *
                (imu_state.rot.matrix().transpose() * (R_i * (imu_state.offset_R_L_I.matrix() * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == pcl_out.points.begin()) break;
        }
    }
}

void ImuProcess::Process(MeasureGroup& meas, esekfom::esekf& kf_state, PointCloudXYZI::Ptr& cur_pcl_un_) {
    if (meas.imu.empty()) return;

    assert(meas.lidar != nullptr);

    if (imu_need_init_) {
        // The very first lidar frame
        IMU_init(meas, kf_state, init_iter_num);  // 如果开头几帧  需要初始化IMU参数

        imu_need_init_ = true;

        last_imu_ = meas.imu.back();

        state_ikfom imu_state = kf_state.get_x();

        if (init_iter_num > MAX_INI_COUNT) {
            cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
            imu_need_init_ = false;

            cov_acc = cov_acc_scale;
            cov_gyr = cov_gyr_scale;
            std::cout << "IMU Initial Done" << std::endl;
        }

        return;
    }

    // FIXME:GNSS init
    if (gnss_heading_need_init_) {
        if (!meas.gnss.empty()) {
            // Eigen::Vector3d gnss_pose(meas.gnss.back()->pose.pose.position.x, meas.gnss.back()->pose.pose.position.y, 0.0);
            // state_ikfom init_state = kf_state.get_x();  // 初始状态量
            // cout << "gnss norm " << gnss_pose.norm() << endl;
            // if (gnss_pose.norm() > 5) {  // GNSS 水平位移大于5m
            //     Eigen::Vector3d tmp_vec(init_state.pos.x(), init_state.pos.y(), 0.0);
            //     GNSS_Heading = Eigen::Quaterniond::FromTwoVectors(gnss_pose, tmp_vec).toRotationMatrix();
            //     // init_state.offset_R_G_I = GNSS_Heading;
            //     // kf_state.change_x(init_state);
            //     Eigen::Vector3d euler_angles = GNSS_Heading.eulerAngles(0, 1, 2);
            //     std::cout << "GNSS_Heading: " << euler_angles(2) * 180.0 / 3.1415926 << std::endl;
            // TODO:
            GNSS_Heading = Eigen::AngleAxisd(-36.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ()).matrix();
            gnss_heading_need_init_ = false;
            // }
        }
    }

    UndistortPcl(meas, kf_state, *cur_pcl_un_);
}
