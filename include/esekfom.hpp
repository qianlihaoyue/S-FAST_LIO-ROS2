#pragma once

#include "common_lib.h"
#include <cmath>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "ikd-Tree/ikd_Tree.h"
#include "use-ikfom.hpp"

// 该hpp主要包含：广义加减法，前向传播主函数，计算特征点残差及其雅可比，ESKF主函数

namespace esekfom {

struct dyn_share_datastruct {
    bool valid;                                                 // 有效特征点数量是否满足要求
    bool converge;                                              // 迭代时，是否已经收敛
    Eigen::Matrix<double, Eigen::Dynamic, 1> h;                 // 残差	(公式(14)中的z)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x;  // 雅可比矩阵H (公式(14)中的H)
};

class esekf {
public:
    typedef Eigen::Matrix<double, 24, 24> cov;              // 24X24的协方差矩阵
    typedef Eigen::Matrix<double, 24, 1> vectorized_state;  // 24X1的向量

    PointCloudXYZI::Ptr normvec{new PointCloudXYZI()};  // 特征点在地图中对应的平面参数(平面的单位法向量,以及当前点到平面距离)
    PointCloudXYZI::Ptr laserCloudOri{new PointCloudXYZI()};  // 有效特征点
    PointCloudXYZI::Ptr corr_normvect{new PointCloudXYZI()};  // 有效特征点对应点法相量
    std::vector<int> point_selected_surf;                     // 判断是否是有效特征点

    /////////////////////////// config
    double epsi = 0.001;   // ESKF迭代时，如果dx<epsi 认为收敛
    int maximum_iter = 3;  // 最大迭代次数

    bool extrinsic_est = false;

    int NUM_MATCH_POINTS = 5;  // 用多少个点拟合平面，一般取 5，提高该值，能提高拟合精度，提高定位稳定性，但耗时
    float plane_thr = 0.1;     // the threshold for plane criteria, the smaller, the flatter a plane

    ///////////////////////////
    PointCloudXYZI::Ptr cloudKeyPoses3D{new PointCloudXYZI};
    pcl::KdTreeFLANN<PointType>::Ptr priorTree{new pcl::KdTreeFLANN<PointType>};

    esekf() {
        for (int i = 0; i < 10; i++) {
            PointType pt;
            pt.x = i * 4, pt.y = 1.9, pt.z = 2, pt.intensity = i * 2;
            cloudKeyPoses3D->emplace_back(pt);
            pt.x = i * 4, pt.y = -1.9, pt.z = 2, pt.intensity = i * 2 + 1;
            cloudKeyPoses3D->emplace_back(pt);
        }

        priorTree->setInputCloud(cloudKeyPoses3D);
    };
    ~esekf(){};

    state_ikfom get_x() { return x_; }

    cov get_P() { return P_; }

    void change_x(state_ikfom& input_state) { x_ = input_state; }

    void change_P(cov& input_cov) { P_ = input_cov; }

    // 广义加法  公式(4)
    state_ikfom boxplus(state_ikfom x, Eigen::Matrix<double, 24, 1> f_) {
        state_ikfom x_r;
        x_r.pos = x.pos + f_.block<3, 1>(0, 0);

        x_r.rot = x.rot * Sophus::SO3::exp(f_.block<3, 1>(3, 0));
        x_r.offset_R_L_I = x.offset_R_L_I * Sophus::SO3::exp(f_.block<3, 1>(6, 0));

        x_r.offset_T_L_I = x.offset_T_L_I + f_.block<3, 1>(9, 0);
        x_r.vel = x.vel + f_.block<3, 1>(12, 0);
        x_r.bg = x.bg + f_.block<3, 1>(15, 0);
        x_r.ba = x.ba + f_.block<3, 1>(18, 0);
        x_r.grav = x.grav + f_.block<3, 1>(21, 0);

        return x_r;
    }

    // 对应公式(2) 中的f
    Eigen::Matrix<double, 24, 1> get_f(state_ikfom s, input_ikfom in) {
        // 对应顺序为速度(3)，角速度(3),外参T(3),外参旋转R(3)，加速度(3),角速度偏置(3),加速度偏置(3),位置(3)，与论文公式顺序不一致
        Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
        V3D omega = in.gyro - s.bg;                         // 输入的imu的角速度(也就是实际测量值) - 估计的bias值(对应公式的第1行)
        V3D a_inertial = s.rot.matrix() * (in.acc - s.ba);  //  输入的imu的加速度，先转到世界坐标系（对应公式的第3行）

        for (int i = 0; i < 3; i++) {
            res(i) = s.vel[i];                        // 速度（对应公式第2行）
            res(i + 3) = omega[i];                    // 角速度（对应公式第1行）
            res(i + 12) = a_inertial[i] + s.grav[i];  // 加速度（对应公式第3行）
        }

        return res;
    }

    // 对应公式(7)的Fx  注意该矩阵没乘dt，没加单位阵
    Eigen::Matrix<double, 24, 24> df_dx(state_ikfom s, input_ikfom in) {
        Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
        cov.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();  // 对应公式(7)第2行第3列   I
        V3D acc_ = in.acc - s.ba;                              // 测量加速度 = a_m - bias

        cov.block<3, 3>(12, 3) = -s.rot.matrix() * Sophus::SO3::hat(acc_);  // 对应公式(7)第3行第1列
        cov.block<3, 3>(12, 18) = -s.rot.matrix();                          // 对应公式(7)第3行第5列

        cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity();  // 对应公式(7)第3行第6列   I
        cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();  // 对应公式(7)第1行第4列 (简化为-I)
        return cov;
    }

    // 对应公式(7)的Fw  注意该矩阵没乘dt
    Eigen::Matrix<double, 24, 12> df_dw(state_ikfom s, input_ikfom in) {
        Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
        cov.block<3, 3>(12, 3) = -s.rot.matrix();              // 对应公式(7)第3行第2列  -R
        cov.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();  // 对应公式(7)第1行第1列  -A(w dt)简化为-I
        cov.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();  // 对应公式(7)第4行第3列  I
        cov.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();  // 对应公式(7)第5行第4列  I
        return cov;
    }

    // 前向传播  公式(4-8)
    void predict(double& dt, Eigen::Matrix<double, 12, 12>& Q, const input_ikfom& i_in) {
        Eigen::Matrix<double, 24, 1> f_ = get_f(x_, i_in);     // 公式(3)的f
        Eigen::Matrix<double, 24, 24> f_x_ = df_dx(x_, i_in);  // 公式(7)的df/dx
        Eigen::Matrix<double, 24, 12> f_w_ = df_dw(x_, i_in);  // 公式(7)的df/dw

        x_ = boxplus(x_, f_ * dt);  // 前向传播 公式(4)

        f_x_ = Eigen::Matrix<double, 24, 24>::Identity() + f_x_ * dt;  // 之前Fx矩阵里的项没加单位阵，没乘dt   这里补上

        P_ = (f_x_)*P_ * (f_x_).transpose() + (dt * f_w_) * Q * (dt * f_w_).transpose();  // 传播协方差矩阵，即公式(8)
    }

    double ratio_all = 0.;
    void get_match_ratio(double& all) { all = ratio_all; }
    double get_match_ratio() { return ratio_all; }

    // 计算每个特征点的残差及H矩阵
    void h_share_model(dyn_share_datastruct& ekfom_data, PointCloudXYZI::Ptr& feats_down_body, KD_TREE<PointType>& ikdtree, vector<PointVector>& Nearest_Points,
                       bool extrinsic_est) {
        static bool first_frame = true;
        int feats_down_size = feats_down_body->points.size();
        laserCloudOri->clear();
        corr_normvect->clear();
        laserCloudOri->resize(feats_down_size);
        corr_normvect->resize(feats_down_size);

#ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
        for (int i = 0; i < feats_down_size; i++) {
            PointType& point_body = feats_down_body->points[i];
            PointType point_world;

            V3D p_body(point_body.x, point_body.y, point_body.z);
            // 把Lidar坐标系的点先转到IMU坐标系，再根据前向传播估计的位姿x，转到世界坐标系
            V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);
            point_world.x = p_global(0);
            point_world.y = p_global(1);
            point_world.z = p_global(2);
            point_world.intensity = point_body.intensity;

            vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
            auto& points_near = Nearest_Points[i];  // Nearest_Points[i]打印出来发现是按照离point_world距离，从小到大的顺序的vector

            bool pt_line = (point_body.intensity > 100);

            std::vector<int> pointIdx;
            std::vector<float> pointDist;

            // double ta = omp_get_wtime();
            if (ekfom_data.converge) {
                // 寻找point_world的最近邻的平面点
                if (pt_line && first_frame == false) {
                    priorTree->nearestKSearch(point_world, 1, pointIdx, pointDist);
                    points_near.clear();
                    points_near.push_back(cloudKeyPoses3D->points[pointIdx[0]]);

                    point_selected_surf[i] = (pointDist[0] > 1) ? 0 : pt_line + 1;
                } else {
                    ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
                    // 判断是否是有效匹配点，与loam系列类似，要求特征点最近邻的地图点数量>阈值，距离<阈值  满足条件的才置为true
                    point_selected_surf[i] = (points_near.size() < NUM_MATCH_POINTS || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) ? 0 : pt_line + 1;
                }
            }
            if (!point_selected_surf[i]) continue;  // 如果该点不满足条件  不进行下面步骤

            if (point_selected_surf[i] == 1) {
                Eigen::Matrix<float, 4, 1> pabcd;  // 平面点信息
                // point_selected_surf[i] = false;    // 将该点设置为无效点，用来判断是否满足条件
                // 拟合平面方程ax+by+cz+d=0并求解点到平面距离
                if (esti_plane(pabcd, points_near, plane_thr, NUM_MATCH_POINTS)) {
                    float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);  // 当前点到平面的距离
                    // 如果残差大于经验阈值，则认为该点是有效点  简言之，距离原点越近的lidar点  要求点到平面的距离越苛刻
                    if (pd2 * pd2 < p_body.norm() / 81.0) {
                        point_selected_surf[i] = 1;
                        normvec->points[i].x = pabcd(0);  // 存储平面的单位法向量  以及当前点到平面距离
                        normvec->points[i].y = pabcd(1);
                        normvec->points[i].z = pabcd(2);
                        normvec->points[i].intensity = pd2;
                    } else
                        point_selected_surf[i] = 0;
                } else
                    point_selected_surf[i] = 0;
            } else {
                // Eigen::Matrix<float, 6, 1> pabcd;
                // // Eigen::Matrix<float, 6, 1> pabcd = prior_line[points_near[0].intensity];
                // if (esti_line(pabcd, points_near, plane_thr, points_near.size())) {
                //     V3D dir = pabcd.template head<3>().normalized().cast<double>();
                //     V3D centroid = pabcd.template tail<3>().cast<double>();
                //     V3D pab = (p_global - centroid).cross(dir);
                //     float pd2 = pab.norm();

                //     point_selected_surf[i] = 2;
                //     // normvec->points[i].x = (-pab(1) * dir(2) + pab(2) * dir(1)) / pd2;
                //     // normvec->points[i].y = (pab(0) * dir(2) - pab(2) * dir(0)) / pd2;
                //     // normvec->points[i].z = (-pab(0) * dir(1) + pab(1) * dir(0)) / pd2;
                //     normvec->points[i].x = (-pab(1) * dir(2)) / pd2;
                //     normvec->points[i].y = (pab(0) * dir(2)) / pd2;
                //     normvec->points[i].z = 0;
                //     normvec->points[i].intensity = pd2;

                // } else
                //     point_selected_surf[i] = 0;

                //////////////////////////////
                V3D pab = (p_global - points_near[0].getVector3fMap().cast<double>());
                pab(2) = 0;

                float pd2 = pab.norm();
                normvec->points[i].x = pab(0) / pd2;
                normvec->points[i].y = pab(1) / pd2;
                // normvec->points[i].z = pab(2) / pd2;
                normvec->points[i].z = 0;
                normvec->points[i].intensity = pd2;
            }
        }

        int effct_feat_num = 0;  // 有效特征点的数量
        for (int i = 0; i < feats_down_size; i++) {
            if (point_selected_surf[i]) {
                laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];  // 把这些点重新存到laserCloudOri中
                corr_normvect->points[effct_feat_num] = normvec->points[i];          // 存储这些点对应的法向量和到平面的距离
                effct_feat_num++;
            }
        }
        ratio_all = (float)effct_feat_num / feats_down_size;

        if (effct_feat_num < 1) {
            ekfom_data.valid = false;
            std::cout << "No Effective Points! " << std::endl;
            return;
        }

        if (first_frame) first_frame = false;

        // 雅可比矩阵H和残差向量的计算
        ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, 12);
        ekfom_data.h.resize(effct_feat_num);

        for (int i = 0; i < effct_feat_num; i++) {
            V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_);
            V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
            M3D point_I_crossmat;
            point_I_crossmat << SKEW_SYM_MATRX(point_I_);

            // 得到对应的平面的法向量
            const PointType& norm_p = corr_normvect->points[i];
            V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

            // 计算雅可比矩阵H
            V3D C(x_.rot.matrix().transpose() * norm_vec);
            V3D A(point_I_crossmat * C);
            if (extrinsic_est) {
                V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
                ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            } else {
                ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            }

            // 残差：点面距离
            ekfom_data.h(i) = -norm_p.intensity;
        }
    }

    // 广义减法
    vectorized_state boxminus(state_ikfom x1, state_ikfom x2) {
        vectorized_state x_r = vectorized_state::Zero();

        x_r.block<3, 1>(0, 0) = x1.pos - x2.pos;

        x_r.block<3, 1>(3, 0) = Sophus::SO3(x2.rot.matrix().transpose() * x1.rot.matrix()).log();
        x_r.block<3, 1>(6, 0) = Sophus::SO3(x2.offset_R_L_I.matrix().transpose() * x1.offset_R_L_I.matrix()).log();

        x_r.block<3, 1>(9, 0) = x1.offset_T_L_I - x2.offset_T_L_I;
        x_r.block<3, 1>(12, 0) = x1.vel - x2.vel;
        x_r.block<3, 1>(15, 0) = x1.bg - x2.bg;
        x_r.block<3, 1>(18, 0) = x1.ba - x2.ba;
        x_r.block<3, 1>(21, 0) = x1.grav - x2.grav;

        return x_r;
    }

    // ESKF
    void update_iterated_dyn_share_modified(double R, PointCloudXYZI::Ptr& feats_down_body, KD_TREE<PointType>& ikdtree, vector<PointVector>& Nearest_Points) {
        int feats_down_size = feats_down_body->points.size();
        Nearest_Points.resize(feats_down_size);  // 存储近邻点的vector
        normvec->resize(feats_down_size);
        point_selected_surf.resize(feats_down_size);
        std::fill(point_selected_surf.begin(), point_selected_surf.end(), false);

        dyn_share_datastruct dyn_share;
        dyn_share.valid = true;
        dyn_share.converge = true;
        int t = 0;
        state_ikfom x_propagated = x_;  // 这里的x_和P_分别是经过正向传播后的状态量和协方差矩阵，因为会先调用predict函数再调用这个函数
        cov P_propagated = P_;

        vectorized_state dx_new = vectorized_state::Zero();  // 24X1的向量

        // maximum_iter: kalman最大迭代次数
        for (int i = -1; i < maximum_iter; i++) {
            dyn_share.valid = true;
            // 计算雅克比，也就是点面残差的导数 H(代码里是h_x)
            h_share_model(dyn_share, feats_down_body, ikdtree, Nearest_Points, extrinsic_est);

            if (!dyn_share.valid) continue;

            vectorized_state dx;
            dx_new = boxminus(x_, x_propagated);  // 公式(18)中的 x^k - x^

            // 由于H矩阵是稀疏的，只有前12列有非零元素，后12列是零 因此这里采用分块矩阵的形式计算 减少计算量
            auto H = dyn_share.h_x;                                                     // m X 12 的矩阵
            Eigen::Matrix<double, 24, 24> HTH = Eigen::Matrix<double, 24, 24>::Zero();  // 矩阵 H^T * H
            HTH.block<12, 12>(0, 0) = H.transpose() * H;

            auto K_front = (HTH / R + P_.inverse()).inverse();
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;
            K = K_front.block<24, 12>(0, 0) * H.transpose() / R;  // 卡尔曼增益  这里R视为常数

            Eigen::Matrix<double, 24, 24> KH = Eigen::Matrix<double, 24, 24>::Zero();  // 矩阵 K * H
            KH.block<24, 12>(0, 0) = K * H;                                            // K:24 x effectnum  H: effectnum x 12
            Eigen::Matrix<double, 24, 1> dx_ = K * dyn_share.h + (KH - Eigen::Matrix<double, 24, 24>::Identity()) * dx_new;  // 公式(18)
            // std::cout << "dx_: " << dx_.transpose() << std::endl;
            x_ = boxplus(x_, dx_);  // 公式(18)

            dyn_share.converge = true;
            for (int j = 0; j < 24; j++) {
                if (std::fabs(dx_[j]) > epsi) {  // 如果dx>epsi 认为没有收敛
                    dyn_share.converge = false;
                    break;
                }
            }

            if (dyn_share.converge) t++;

            if (!t && i == maximum_iter - 2)  // 如果迭代了3次还没收敛 强制令成true，h_share_model函数中会重新寻找近邻点
            {
                dyn_share.converge = true;
            }

            // TODO:t>1改成t>0，对性能有较大提升
            // 原始代码，收敛后仍然会进行两次
            if (t > 0 || i == maximum_iter - 1) {
                P_ = (Eigen::Matrix<double, 24, 24>::Identity() - KH) * P_;  // 公式(19)
                return;
            }
        }
    }

private:
    state_ikfom x_;
    cov P_ = cov::Identity();
};

}  // namespace esekfom
