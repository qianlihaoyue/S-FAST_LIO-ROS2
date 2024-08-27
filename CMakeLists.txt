cmake_minimum_required(VERSION 3.5)
project(fast_lio)

SET(CMAKE_BUILD_TYPE "Release") # Debug

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++0x -fexceptions")

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
add_definitions(-DMP_EN)
add_definitions(-DMP_PROC_NUM=3)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

set(Sophus_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/sophus/include")
set(Sophus_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/sophus/lib/libSophus.so")

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  include
)

add_message_files(
  FILES
  Pose6D.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

add_library(fastlio_lib
  src/laserMapping.cpp
  src/laserMappingUtility.cpp
  src/IMU_Processing.cpp
  src/preprocess.cpp
  include/ikd-Tree/ikd_Tree.cpp
)
target_link_libraries(fastlio_lib ${PCL_LIBRARIES} ${Sophus_LIBRARIES} ${catkin_LIBRARIES})

# ---------------- app --------------- #
add_executable(fastlio_main app/main.cpp)
target_link_libraries(fastlio_main fastlio_lib)