cmake_minimum_required(VERSION 3.8)
project(shield_mppi)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(f1tenth_icra_race_msgs REQUIRED)

# Include Cpp "include" directory
include_directories(include)

# Create cpp library for frenet conversion
add_library(frenet_conversion
  src/frenet_conversion.cpp
)

# Create Cpp executable
add_executable(opponent_detection_node_cpp
  src/opponent_detection_node.cpp
)
# Link Cpp executable to libraries
target_link_libraries(opponent_detection_node_cpp
  frenet_conversion
)

ament_target_dependencies(
  opponent_detection_node_cpp
  rclcpp
  sensor_msgs
  std_msgs
  nav_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
  f1tenth_icra_race_msgs
)
# Install Cpp header files
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)
# Install Cpp executables
install(TARGETS
  opponent_detection_node_cpp
  DESTINATION lib/${PROJECT_NAME}
)
# Install Cpp library
install(TARGETS
  frenet_conversion
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION bin
)


# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  scripts/shield_mppi_node.py
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
