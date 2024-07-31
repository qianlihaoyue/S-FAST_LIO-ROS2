#pragma once

#include "common_lib.h"
#include "sophus/so3.h"

// 该hpp主要包含：状态变量x，输入量u的定义，以及正向传播中相关矩阵的函数

#define STATE_DIM 27  // 24+3

// 24维的状态量x
struct state_ikfom {
    V3D pos = V3D(0, 0, 0);
    Sophus::SO3 rot = Sophus::SO3(Eigen::Matrix3d::Identity());
    Sophus::SO3 offset_R_L_I = Sophus::SO3(Eigen::Matrix3d::Identity());
    V3D offset_T_L_I = V3D(0, 0, 0);
    V3D vel = V3D(0, 0, 0);
    V3D bg = V3D(0, 0, 0);
    V3D ba = V3D(0, 0, 0);
    V3D grav = V3D(0, 0, -G_m_s2);
    Sophus::SO3 offset_R_G_I = Sophus::SO3(Eigen::Matrix3d::Identity());
};

// 输入u
struct input_ikfom {
    V3D acc = V3D(0, 0, 0);
    V3D gyro = V3D(0, 0, 0);
};
