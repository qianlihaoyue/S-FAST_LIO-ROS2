# import numpy as np
# from scipy.spatial.transform import Rotation as R

# # 相机到LiDAR的平移和旋转
# T_cam1_os_translation = np.array([0.12944592, 0.04299934, -0.1137434])
# T_cam1_os_rotation = R.from_quat(
#     [-0.6116725, 0.39292797, -0.3567415, 0.58668551])

# # IMU到相机的平移和旋转
# T_imu1_cam1_translation = np.array([-0.0286307, -0.0031187, -0.0472054])
# T_imu1_cam1_rotation = R.from_quat(
#     [-0.4987278, 0.50105310, 0.50066704, 0.49954869])

# # 计算IMU到LiDAR的旋转
# T_imu1_os_rotation = T_cam1_os_rotation * T_imu1_cam1_rotation

# # 计算IMU到LiDAR的平移
# T_imu1_os_translation = T_cam1_os_rotation.apply(
#     T_imu1_cam1_translation) + T_cam1_os_translation

# # 输出结果
# print("IMU to LiDAR translation:")
# print("px: ", T_imu1_os_translation[0])
# print("py: ", T_imu1_os_translation[1])
# print("pz: ", T_imu1_os_translation[2])


# print("\nIMU to LiDAR rotation (quaternion):")
# print("mat: ", T_imu1_os_rotation.as_matrix())
# print("qx: ", T_imu1_os_rotation.as_quat()[0])
# print("qy: ", T_imu1_os_rotation.as_quat()[1])
# print("qz: ", T_imu1_os_rotation.as_quat()[2])
# print("qw: ", T_imu1_os_rotation.as_quat()[3])

import numpy as np
from scipy.spatial.transform import Rotation as R

# Helper function to create a transformation matrix from translation and quaternion
def create_transformation_matrix(px, py, pz, qx, qy, qz, qw):
    translation = np.array([px, py, pz])
    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    return transformation_matrix

# Left Camera extrinsics w.r.t. the OS128 Lidar
T_cam1_os = create_transformation_matrix(0.12944592, 0.04299934, -0.1137434,
                                         -0.6116725, 0.39292797, -0.3567415, 0.58668551)

# Left Camera IMU extrinsics w.r.t. Left Camera
T_imu1_cam1 = create_transformation_matrix(-0.0286307, -0.0031187, -0.0472054,
                                           -0.4987278, 0.50105310, 0.50066704, 0.49954869)

# Compute the IMU extrinsics w.r.t. the Lidar
T_imu1_os = np.dot(T_cam1_os, T_imu1_cam1)

# Extract the translation and quaternion from the resulting transformation matrix
translation_imu_os = T_imu1_os[:3, 3]
rotation_imu_os = T_imu1_os[:3, :3]

# Print the results
print("IMU to Lidar Translation:",rotation_imu_os , translation_imu_os)
 

# # Left Camera extrinsics w.r.t. the OS128 Lidar (origin of the sensor rack)

# T_cam1_os.px: 0.12944592
# T_cam1_os.py: 0.04299934
# T_cam1_os.pz: -0.1137434

# T_cam1_os.qx: -0.6116725
# T_cam1_os.qy: 0.39292797
# T_cam1_os.qz: -0.3567415
# T_cam1_os.qw: 0.58668551

# # Left Camera IMU extrinsics w.r.t. Left Camera

# T_imu1_cam1.px: -0.0286307
# T_imu1_cam1.py: -0.0031187
# T_imu1_cam1.pz: -0.0472054

# T_imu1_cam1.qx: -0.4987278
# T_imu1_cam1.qy: 0.50105310
# T_imu1_cam1.qz: 0.50066704
# T_imu1_cam1.qw: 0.49954869
