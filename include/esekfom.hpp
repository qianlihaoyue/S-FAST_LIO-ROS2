#pragma once

// #include "common_lib.h"
#include "HVox/hvox.h"
#include "ikd-Tree/ikd_Tree.h"
#include "use-ikfom.hpp"
#include <Eigen/Dense>
// #include <cmath>
// #include <iomanip>
#include <pcl/kdtree/kdtree_flann.h>

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
    typedef Eigen::Matrix<double, STATE_DIM, STATE_DIM> cov;       // STATE_DIMXSTATE_DIM的协方差矩阵
    typedef Eigen::Matrix<double, STATE_DIM, 1> vectorized_state;  // STATE_DIMX1的向量

    CloudType::Ptr normvec{new CloudType()};        // 特征点在地图中对应的平面参数(平面的单位法向量,以及当前点到平面距离)
    CloudType::Ptr laserCloudOri{new CloudType()};  // 有效特征点
    CloudType::Ptr corr_normvect{new CloudType()};  // 有效特征点对应点法相量
    std::vector<EstiPlane> point_selected_surf;     // 判断是否是有效特征点
    std::vector<int> effect_idx;

    /////////////////////////// config
    double epsi = 0.001;   // ESKF迭代时，如果dx<epsi 认为收敛
    int maximum_iter = 3;  // 最大迭代次数

    int NUM_MATCH_POINTS = 5;  // 用多少个点拟合平面，一般取 5，提高该值，能提高拟合精度，提高定位稳定性，但耗时
    float plane_thr = 0.1;     // the threshold for plane criteria, the smaller, the flatter a plane

    ///////////////////////////

    bool USE_DYNA = false;
    std::shared_ptr<HVox<PointType>> hvox_dyna = nullptr;

    esekf(){};
    ~esekf(){};

    state_ikfom get_x() { return x_; }

    cov get_P() { return P_; }

    void change_x(state_ikfom& input_state) { x_ = input_state; }

    void change_P(cov& input_cov) { P_ = input_cov; }

    // 广义加法  公式(4)
    state_ikfom boxplus(state_ikfom x, Eigen::Matrix<double, STATE_DIM, 1> f_) {
        state_ikfom x_r = x;
        x_r.pos += f_.block<3, 1>(0, 0);

        x_r.rot *= Sophus::SO3::exp(f_.block<3, 1>(3, 0));
        x_r.offset_R_L_I *= Sophus::SO3::exp(f_.block<3, 1>(6, 0));

        x_r.offset_T_L_I += f_.block<3, 1>(9, 0);
        x_r.vel += f_.block<3, 1>(12, 0);
        x_r.bg += f_.block<3, 1>(15, 0);
        x_r.ba += f_.block<3, 1>(18, 0);
        x_r.grav += f_.block<3, 1>(21, 0);

        return x_r;
    }

    // 对应公式(2) 中的f
    Eigen::Matrix<double, STATE_DIM, 1> get_f(state_ikfom s, input_ikfom in) {
        // 对应顺序为速度(3)，角速度(3),外参T(3),外参旋转R(3)，加速度(3),角速度偏置(3),加速度偏置(3),位置(3)，与论文公式顺序不一致
        Eigen::Matrix<double, STATE_DIM, 1> res = Eigen::Matrix<double, STATE_DIM, 1>::Zero();
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
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> df_dx(state_ikfom s, input_ikfom in) {
        Eigen::Matrix<double, STATE_DIM, STATE_DIM> cov = Eigen::Matrix<double, STATE_DIM, STATE_DIM>::Zero();
        cov.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();  // 对应公式(7)第2行第3列   I
        V3D acc_ = in.acc - s.ba;                              // 测量加速度 = a_m - bias

        cov.block<3, 3>(12, 3) = -s.rot.matrix() * Sophus::SO3::hat(acc_);  // 对应公式(7)第3行第1列
        cov.block<3, 3>(12, 18) = -s.rot.matrix();                          // 对应公式(7)第3行第5列

        cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity();  // 对应公式(7)第3行第6列   I
        cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();  // 对应公式(7)第1行第4列 (简化为-I)
        return cov;
    }

    // 对应公式(7)的Fw  注意该矩阵没乘dt
    Eigen::Matrix<double, STATE_DIM, 12> df_dw(state_ikfom s, input_ikfom in) {
        Eigen::Matrix<double, STATE_DIM, 12> cov = Eigen::Matrix<double, STATE_DIM, 12>::Zero();
        cov.block<3, 3>(12, 3) = -s.rot.matrix();              // 对应公式(7)第3行第2列  -R
        cov.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();  // 对应公式(7)第1行第1列  -A(w dt)简化为-I
        cov.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();  // 对应公式(7)第4行第3列  I
        cov.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();  // 对应公式(7)第5行第4列  I
        return cov;
    }

    void predict(double& dt, Eigen::Matrix<double, 12, 12>& Q, const input_ikfom& i_in) {
        Eigen::Matrix<double, STATE_DIM, 1> f_ = get_f(x_, i_in);            // 公式(3)的f
        Eigen::Matrix<double, STATE_DIM, STATE_DIM> f_x_ = df_dx(x_, i_in);  // 公式(7)的df/dx
        Eigen::Matrix<double, STATE_DIM, 12> f_w_ = df_dw(x_, i_in);         // 公式(7)的df/dw

        x_ = boxplus(x_, f_ * dt);  // 前向传播 公式(4)

        f_x_ = Eigen::Matrix<double, STATE_DIM, STATE_DIM>::Identity() + f_x_ * dt;  // 之前Fx矩阵里的项没加单位阵，没乘dt   这里补上

        P_ = (f_x_)*P_ * (f_x_).transpose() + (dt * f_w_) * Q * (dt * f_w_).transpose();  // 传播协方差矩阵，即公式(8)
    }

    struct MatchRate {
        float all = 0, prior = 0, lio = 0;
    };
    MatchRate match_rate;
    bool use_chi = false;
    double chi_square_critical = 3.84;

    Eigen::Matrix3d covariance_matrix;

    void pointBodyToWorld(PointType const* const pi, PointType* const po) {
        V3D p_body(pi->x, pi->y, pi->z);
        V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);

        po->x = p_global(0), po->y = p_global(1), po->z = p_global(2);
        po->intensity = pi->intensity;
    }

    // 计算每个特征点的残差及H矩阵
    void h_share_model(dyn_share_datastruct& ekfom_data, CloudType::Ptr& feats_down_body, KD_TREE<PointType>& ikdtree, vector<PointVector>& Nearest_Points) {
        int feats_down_size = feats_down_body->points.size();
        laserCloudOri->clear();
        corr_normvect->clear();
        laserCloudOri->resize(feats_down_size);
        corr_normvect->resize(feats_down_size);
        effect_idx.clear();
        std::vector<double> dis_val(feats_down_size);

        CloudType::Ptr feats_down_world(new CloudType(feats_down_size, 1));
        for (int i = 0; i < feats_down_size; i++) pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

#ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
        for (int i = 0; i < feats_down_size; i++) {
            PointType& point_body = feats_down_body->points[i];
            PointType& point_world = feats_down_world->points[i];

            vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
            auto& points_near = Nearest_Points[i];

            if (ekfom_data.converge) {
                ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);

                point_selected_surf[i] =
                    (points_near.size() < NUM_MATCH_POINTS || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) ? EstiPlane::None : EstiPlane::Prior;
            }
            if (point_selected_surf[i] == EstiPlane::None) continue;

            Eigen::Matrix<float, 4, 1> pabcd;
            // point_selected_surf[i] = EstiPlane::None;

            if (esti_plane(pabcd, points_near, plane_thr, NUM_MATCH_POINTS)) {
                float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);

                if (pd2 * pd2 < point_body.getVector3fMap().norm() / 81.0) {
                    // point_selected_surf[i] = EstiPlane::Prior;
                    normvec->points[i].x = pabcd(0), normvec->points[i].y = pabcd(1), normvec->points[i].z = pabcd(2);
                    normvec->points[i].intensity = pd2;

                    dis_val[i] = pd2;
                    point_body.curvature = abs(pd2) * 100.0;
                } else
                    point_selected_surf[i] = EstiPlane::None;
            } else
                point_selected_surf[i] = EstiPlane::None;
        }

        /////////////////////////////////////
        double stddev, mean, lower_bound, upper_bound;

        if (use_chi) {
            double sum_dis = 0, variance = 0.0;
            int cnt_dis = 0;
            for (int i = 0; i < feats_down_size; i++) {
                if (point_selected_surf[i] == EstiPlane::Prior) {
                    variance += dis_val[i] * dis_val[i];
                    sum_dis += dis_val[i];
                    cnt_dis++;
                }
            }
            variance /= cnt_dis;
            mean = sum_dis / cnt_dis;

            stddev = sqrt(variance);
            lower_bound = mean - chi_square_critical * stddev;
            upper_bound = mean + chi_square_critical * stddev;
            for (int i = 0; i < feats_down_size; i++) {
                if (point_selected_surf[i] == EstiPlane::None) continue;
                if (dis_val[i] < lower_bound || dis_val[i] > upper_bound) {
                    point_selected_surf[i] = EstiPlane::None;
                }
            }
            // if (ekfom_data.converge) {
            //     std::cout << std::setprecision(3) << "Mean: " << mean << " MeanChi: " << sum_chi / cnt_chi << " cntChi:" << cnt_chi
            //               << " rateChi: " << (double)cnt_chi * 100.0 / feats_down_size << "%" << std::endl;
            // }
        }

        if (USE_DYNA) {
#ifdef MP_EN
#pragma omp parallel for
#endif
            for (int i = 0; i < feats_down_size; i++) {
                if (point_selected_surf[i] == EstiPlane::Prior) continue;

                PointType& point_body = feats_down_body->points[i];
                PointType& point_world = feats_down_world->points[i];

                auto& points_near = Nearest_Points[i];

                if (ekfom_data.converge) {
                    hvox_dyna->GetClosestPoint(point_world, points_near, NUM_MATCH_POINTS);
                    point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? EstiPlane::None : EstiPlane::Cur;
                }
                if (point_selected_surf[i] == EstiPlane::None) continue;

                Eigen::Matrix<float, 4, 1> pabcd;
                if (esti_plane(pabcd, points_near, plane_thr, NUM_MATCH_POINTS)) {
                    float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                    if (pd2 * pd2 < point_body.getVector3fMap().norm() / 81.0) {
                        normvec->points[i].x = pabcd(0), normvec->points[i].y = pabcd(1), normvec->points[i].z = pabcd(2);
                        normvec->points[i].intensity = pd2;

                        dis_val[i] = pd2;
                        point_body.curvature = abs(pd2) * 100.0;
                    } else
                        point_selected_surf[i] = EstiPlane::None;
                } else
                    point_selected_surf[i] = EstiPlane::None;
            }

            if (use_chi) {
                // double stddev = sqrt(variance);
                // double lower_bound = -chi_square_critical * stddev;
                // double upper_bound = +chi_square_critical * stddev;
                for (int i = 0; i < feats_down_size; i++) {
                    if (point_selected_surf[i] != EstiPlane::Cur) continue;
                    if (dis_val[i] < lower_bound || dis_val[i] > upper_bound) {
                        point_selected_surf[i] = EstiPlane::None;
                    }
                }
            }
        }

        int effct_feat_num = 0, effect_num_prior = 0;
        for (int i = 0; i < feats_down_size; i++) {
            if (point_selected_surf[i] != EstiPlane::None) {
                laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
                corr_normvect->points[effct_feat_num] = normvec->points[i];
                effect_idx.push_back(i);
                effct_feat_num++;
                if (point_selected_surf[i] == EstiPlane::Prior) effect_num_prior++;
            }
        }

        match_rate.prior = (float)effect_num_prior / feats_down_size * 100.0;
        match_rate.all = (float)effct_feat_num / feats_down_size * 100.0;
        match_rate.lio = match_rate.all - match_rate.prior;

        if (effct_feat_num < 1) {
            ekfom_data.valid = false;
            std::cout << "No Effective Points! " << std::endl;
            return;
        }

        Eigen::Matrix3Xd vectors = Eigen::Matrix3Xd::Zero(3, effct_feat_num);
        for (int i = 0; i < effct_feat_num; i++) vectors.col(i) = corr_normvect->points[i].getVector3fMap().cast<double>();
        Eigen::MatrixXd centered = vectors.colwise() - vectors.rowwise().mean();
        covariance_matrix = (centered * centered.transpose());

        ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, 12);
        ekfom_data.h.resize(effct_feat_num);

        for (int i = 0; i < effct_feat_num; i++) {
            V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_);
            V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
            M3D point_I_crossmat;
            point_I_crossmat << SKEW_SYM_MATRX(point_I_);

            const PointType& norm_p = corr_normvect->points[i];
            V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

            V3D C(x_.rot.matrix().transpose() * norm_vec);
            V3D A(point_I_crossmat * C);
            ekfom_data.h_x.block<1, 12>(i, 0) << VEC_FROM_ARRAY(norm_vec), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

            ekfom_data.h(i) = -norm_p.intensity;
        }
    }

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

    struct DegenRate {
        double rate_1 = 0, rate_2 = 0;
        double pos_min = 0, pos_trace = 0;
        double rot_min = 0, rot_trace = 0;
        double all_min = 0, all_trace = 0;
    };
    DegenRate degen_rate;

    void calculateMinEigenvalueAndTrace(const Eigen::MatrixXd& matrix, double& min_eigenvalue, double& trace_value) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(matrix);
        Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
        min_eigenvalue = eigenvalues.minCoeff();
        trace_value = matrix.trace();
    }

    void update_iterated_dyn_share_modified(double R, CloudType::Ptr& feats_down_body, KD_TREE<PointType>& ikdtree, vector<PointVector>& Nearest_Points) {
        normvec->resize(int(feats_down_body->points.size()));
        point_selected_surf.resize(int(feats_down_body->points.size()));
        std::fill(point_selected_surf.begin(), point_selected_surf.end(), EstiPlane::None);

        dyn_share_datastruct dyn_share;
        dyn_share.valid = dyn_share.converge = true;
        int t = 0;
        state_ikfom x_propagated = x_;

        vectorized_state dx_new = vectorized_state::Zero();
        auto I = Eigen::Matrix<double, STATE_DIM, STATE_DIM>::Identity();

        for (int i = -1; i < maximum_iter; i++) {
            dyn_share.valid = true;
            h_share_model(dyn_share, feats_down_body, ikdtree, Nearest_Points);

            if (!dyn_share.valid) continue;

            dx_new = boxminus(x_, x_propagated);

            auto& H = dyn_share.h_x;
            Eigen::Matrix<double, STATE_DIM, STATE_DIM> HTH = Eigen::Matrix<double, STATE_DIM, STATE_DIM>::Zero();
            HTH.block<12, 12>(0, 0) = H.transpose() * H;

            auto K_front = (HTH / R + P_.inverse()).inverse();  // STATE_DIM x STATE_DIM
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;
            K = K_front.block<STATE_DIM, 12>(0, 0) * H.transpose() / R;  // 卡尔曼增益  这里R视为常数

            Eigen::Matrix<double, STATE_DIM, STATE_DIM> KH = Eigen::Matrix<double, STATE_DIM, STATE_DIM>::Zero();  // 矩阵 K * H
            KH.block<STATE_DIM, 12>(0, 0) = K * H;                                                                 // K:STATE_DIM x effectnum  H: effectnum x 12
            Eigen::Matrix<double, STATE_DIM, 1> dx_ = K * dyn_share.h + (KH - I) * dx_new;                         // 公式(18)
            // std::cout << "dx_: " << dx_.transpose() << std::endl;
            x_ = boxplus(x_, dx_);

            dyn_share.converge = true;
            for (int j = 0; j < STATE_DIM; j++) {
                if (std::fabs(dx_[j]) > epsi) {
                    dyn_share.converge = false;
                    break;
                }
            }

            if (dyn_share.converge) t++;
            if (!t && i == maximum_iter - 2) dyn_share.converge = true;

            if (t > 0 || i == maximum_iter - 1) {
                P_ = (I - KH) * P_;

                ////////////////////////////////
                Eigen::Matrix<double, 24, 24> fisher_matrix = HTH;  // 12*12 的 Fisher 信息矩阵

                calculateMinEigenvalueAndTrace(fisher_matrix.block<3, 3>(0, 0), degen_rate.pos_min, degen_rate.pos_trace);
                calculateMinEigenvalueAndTrace(fisher_matrix.block<3, 3>(3, 3), degen_rate.rot_min, degen_rate.rot_trace);

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver_p(fisher_matrix.block<3, 3>(0, 0));
                Eigen::Vector3d eigenvalues_p = eigensolver_p.eigenvalues();
                degen_rate.all_min = eigenvalues_p(0) / eigenvalues_p(2);

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver_r(fisher_matrix.block<3, 3>(3, 3));
                Eigen::Vector3d eigenvalues_r = eigensolver_r.eigenvalues();
                degen_rate.all_trace = eigenvalues_r(0) / eigenvalues_r(2);

                ////////////////////////////////

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covariance_matrix);
                Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
                // Eigen::Matrix3d eigenvectors = eigensolver.eigenvectors();
                degen_rate.rate_1 = eigenvalues(0) / eigenvalues(1);
                degen_rate.rate_2 = eigenvalues(0) / eigenvalues(2);

                // eigenvalues(0) 是最小的特征值
                return;
            }
        }
    }

private:
    state_ikfom x_;
    cov P_ = cov::Identity();
};

}  // namespace esekfom
