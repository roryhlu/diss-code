cmake_minimum_required(VERSION 3.8)
project(repair_simulation)

# Find ament_cmake
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python package installation
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/repair_simulation"
)

# Install ROS2 launch files (if added later)
# install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

# Install Python scripts as ROS2 nodes
install(PROGRAMS
  repair_simulation/grasp_executor.py
  DESTINATION lib/${PROJECT_NAME}
)

# Standard ament package finalisation
ament_package()
