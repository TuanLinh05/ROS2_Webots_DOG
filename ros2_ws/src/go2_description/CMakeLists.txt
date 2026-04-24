cmake_minimum_required(VERSION 3.8)
project(go2_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Tìm kiếm các gói phụ thuộc của ROS 2
find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

# Cài đặt (copy) các thư mục chứa dữ liệu của robot vào không gian làm việc
install(
  DIRECTORY urdf meshes xacro launch config dae worlds protos
  DESTINATION share/${PROJECT_NAME}
)

ament_package()