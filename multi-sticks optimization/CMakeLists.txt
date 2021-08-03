cmake_minimum_required(VERSION 2.8.12)
project(NCS_optimization)

#使用C++11
set(CMAKE_CXX_STANDARD 11)

#设置编译方式
set(CMAKE_BUILD_TYPE Release)

#输出二进制文件路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/output)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/lib)
set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/lib)
set (LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build/lib)

#添加外部文件
add_subdirectory(common/format_reader)


#设置DIR变量，包含opencvconfig.cmake
#set(OpenCV_DIR /opt/intel/openvino/opencv/cmake)
#查找opencv
find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)


#添加opencv链接库
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${InferenceEngine_INCLUDE_DIRS})


#添加可执行程序
add_executable(multi_stick_and_thread_async multi_stick_and_thread_async.cpp)
#2multi_stick_and_thread_sync
#3multi_stick_and_thread_async
#4multi_stick_and_thread_3req
#4multi_stick_and_thread_4req


#库链接到可执行程序上
target_include_directories(multi_stick_and_thread_async PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")
target_link_libraries(multi_stick_and_thread_async PRIVATE ${OpenCV_LIBS} ${InferenceEngine_LIBRARIES} format_reader IE::ie_cpu_extension pthread)


