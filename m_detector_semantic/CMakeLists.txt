cmake_minimum_required(VERSION 3.0.2)
project(m_detector_semantic)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -pthread -fPIC")

# Find Python 3
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "🐍 Using Conda Python: $ENV{CONDA_PREFIX}")
    set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}")
    set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    
    # 自动检测 Python 版本
    execute_process(
        COMMAND $ENV{CONDA_PREFIX}/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys. version_info.minor}')"
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    set(PYTHON_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include/python${PYTHON_VERSION}")
    set(PYTHON_LIBRARIES "$ENV{CONDA_PREFIX}/lib/libpython${PYTHON_VERSION}.so")
    
    message(STATUS "  Python version: ${PYTHON_VERSION}")
    message(STATUS "  Include: ${PYTHON_INCLUDE_DIRS}")
    message(STATUS "  Library: ${PYTHON_LIBRARIES}")
else()
    message(STATUS "Using system Python")
    find_package(PythonLibs 3 REQUIRED)
endif()

# Get NumPy include directory
execute_process(
    COMMAND python3 -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT NUMPY_INCLUDE_DIR)
    message(WARNING "NumPy not found. Install with: pip3 install numpy")
    set(NUMPY_INCLUDE_DIR "/usr/local/lib/python3.8/dist-packages/numpy/core/include")
endif()

message(STATUS "Python include: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "NumPy include: ${NUMPY_INCLUDE_DIR}")

# Find catkin
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    pcl_ros
    pcl_conversions
    tf
    eigen_conversions
)

# Find other dependencies
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Catkin package
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS 
        roscpp 
        rospy
        std_msgs 
        sensor_msgs 
        geometry_msgs
        nav_msgs
        pcl_ros
        pcl_conversions
        tf
        eigen_conversions
    DEPENDS 
        PCL
        EIGEN3
        OpenCV
)

# Include directories
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)

# Link directories
link_directories(
    ${PCL_LIBRARY_DIRS}
    ${OpenCV_LIBRARY_DIRS}
)

# Add definitions
add_definitions(${PCL_DEFINITIONS})

###########
## Build ##
###########

# Semantic Segmentor Library
add_library(semantic_segmentor
    src/SemanticSegmentor.cpp
)
target_include_directories(semantic_segmentor PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)
target_link_libraries(semantic_segmentor
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${PYTHON_LIBRARIES}
)

# DynObjFilter Library
add_library(dyn_obj_filter
    src/DynObjFilter.cpp
)
target_link_libraries(dyn_obj_filter
    semantic_segmentor
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${PYTHON_LIBRARIES}
)

# DynObjCluster Library
add_library(dyn_obj_cluster
    src/DynObjCluster.cpp
)
target_link_libraries(dyn_obj_cluster
    semantic_segmentor
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
)

# Main executable
add_executable(${PROJECT_NAME}_node
    src/dynfilter_with_odom_semantic.cpp
)
target_link_libraries(${PROJECT_NAME}_node
    dyn_obj_filter
    dyn_obj_cluster
    semantic_segmentor
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${PYTHON_LIBRARIES}
)

# Visualization node
add_executable(visualization_semantic_node
    src/visualization_semantic.cpp
)
# ← 在这里把 semantic_segmentor（和其他需要的库）加入链接列表
target_link_libraries(visualization_semantic_node
    semantic_segmentor
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${PYTHON_LIBRARIES}
)

#############
## Install ##
#############

install(TARGETS 
    semantic_segmentor
    dyn_obj_filter
    dyn_obj_cluster
    ${PROJECT_NAME}_node
    visualization_semantic_node
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY rviz/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
)

install(PROGRAMS
    scripts/cenet_inference.py
    scripts/setup_cenet.sh
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)