cmake_minimum_required(VERSION 3.5)
project(opencapi)

set(CMAKE_CXX_STANDARD 14)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(opencapi main.cpp GLCM.h GLCM.cpp)
target_link_libraries(opencapi ${OpenCV_LIBS})