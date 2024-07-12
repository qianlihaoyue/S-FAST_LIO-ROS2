cmake_minimum_required(VERSION 3.5)
project(fast_lio)

SET(CMAKE_BUILD_TYPE "Release") # Debug

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++0x -fexceptions")

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
add_definitions(-DMP_EN)
add_definitions(-DMP_PROC_NUM=3)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

set(Sophus_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/sophus/include")
set(Sophus_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/sophus/lib/libSophus.so")

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  include
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Pose6D.msg"
)
ament_export_dependencies(rosidl_default_runtime)

set(dependencies
  rclcpp
  rclcpp_components
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  pcl_ros
  pcl_conversions
  livox_ros_driver2
)

add_library(fastlio_lib
  src/laserMapping.cpp
  src/laserMappingUtility.cpp
  src/IMU_Processing.cpp
  src/preprocess.cpp
  include/ikd-Tree/ikd_Tree.cpp
  include/sc/Scancontext.cpp
)
target_link_libraries(fastlio_lib ${PCL_LIBRARIES} ${Sophus_LIBRARIES})
ament_target_dependencies(fastlio_lib ${dependencies})
rosidl_target_interfaces(fastlio_lib ${PROJECT_NAME} "rosidl_typesupport_cpp")
install(TARGETS fastlio_lib DESTINATION lib/${PROJECT_NAME})

# ---------------- app --------------- #
add_executable(fastlio_mapping app/main.cpp)
target_link_libraries(fastlio_mapping fastlio_lib)

# ---------------- Install --------------- #
install(TARGETS fastlio_mapping DESTINATION lib/${PROJECT_NAME})

install(FILES ${Sophus_LIBRARIES} DESTINATION lib/${PROJECT_NAME})
set_target_properties(fastlio_mapping PROPERTIES LINK_FLAGS "-Wl,-rpath=${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}")

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()