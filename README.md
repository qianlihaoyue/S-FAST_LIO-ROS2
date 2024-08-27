# S-FAST_LIO

## Update

- 1.0 - 6.21 - 格式化代码
- 1.1 - 8.20 - clang-format

- 2.0 - 5.31 - to ros2
- 2.1 - 5.31 - fix sophus path
- 2.2 - 6.3 - 封装 & 测试

- 2.3 - 6.23 - 引出kf参数，添加log输出
- 2.4 - 7.11 - 添加PCA法向分析
- 2.5 - 7.17 - 取消submap移动，优化地图保存逻辑

- branch - ros1

- b1.0 - 8.27 - 转ros1


## Simplified Implementation of FAST_LIO

S-FAST_LIO is a simplified implementation of FAST_LIO (Xu, Wei, et al. "Fast-lio2: Fast direct lidar-inertial odometry."), which is modified from [FAST_LIO](https://github.com/hku-mars/FAST_LIO). This code is clean and accessible. It is a reference material for SLAM beginners.The main modifications are as follows:
* The [Sophus](https://github.com/strasdat/Sophus) is used to define the state variables, instead of the complicated [IKFOM](https://github.com/hku-mars/IKFoM)
* The gravity component is directly defined by a Vector3d, thus the complicated calculation of two-dimensional manifold can be omitted
* The code structure has been optimized, and the unnecessary codes have been deleted
* Detailed Chinese notes are added to the code
* Add relocation function in established maps
* Support for Robosense LiDAR has been added

 In addition, the following links are also my previous works. I strongly recommend reading them, since they are the interpretation and detailed equation derivation of the FAST-LIO paper:

[FAST-LIO论文解读与详细公式推导(知乎)](https://zhuanlan.zhihu.com/p/587500859)

[FAST-LIO论文解读与详细公式推导(CSDN)](https://blog.csdn.net/weixin_44923488/article/details/128103159)



## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu >= 16.04.

### 1.2. **PCL && Eigen**
PCL >= 1.8, Eigen >= 3.3.4.

### 1.3. **livox_ros_driver2**
Follow [livox_ros_driver2 Installation](https://github.com/Livox-SDK/livox_ros_driver2).

### 1.4. **Sophus**
We use the old version of Sophus

```
cd thirdparty
chmod +x build.sh
./build.sh

```
or

```
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
mkdir build
cd build
cmake ../ -DUSE_BASIC_LOGGING=ON
make
sudo make install
```


## 7. Acknowledgements
Thanks for the authors of [FAST-LIO](https://github.com/hku-mars/FAST_LIO).
