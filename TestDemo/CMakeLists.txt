#set min cmake version
cmake_minimum_required(VERSION 3.16) 


#set project name
set(PROJECTNAME demo)
project(${PROJECTNAME})   

#set compilier
set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 11)
#set compilier language
#enable_language(C)
#enable_language(CXX)
#enable_language(CUDA)

#set build type
set(CMAKE_BUILD_TYPE Release)

#set cxx build flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#set include dir && add include dir to cmake
#opencv
set(OPENCV_INCLUDE_DIR /usr/local/include)
include_directories(${OPENCV_INCLUDE_DIR})
message("opencv include path :${OPENCV_INCLUDE_DIR}")
#boost
set(BOOST_INCLUDE_DIR /usr/include)
include_directories(${BOOST_INCLUDE_DIR})
message("boost include path :${BOOST_INCLUDE_DIR}")
#personmotordetection
set(PEROSONMOTOR_INCLUDE_DIR ../Export/include)
include_directories(${PEROSONMOTOR_INCLUDE_DIR})
message("personmotor include path :${PEROSONMOTOR_INCLUDE_DIR}")


#set lid dir && set dependece link lib 
#opencv
set(OPENCV_LIB_DIR /usr/local/lib)
link_libraries(${OPENCV_LIB_DIR}/libopencv_core.so
                ${OPENCV_LIB_DIR}/libopencv_imgproc.so
                ${OPENCV_LIB_DIR}/libopencv_highgui.so
                ${OPENCV_LIB_DIR}/libopencv_imgcodecs.so
                ${OPENCV_LIB_DIR}/libopencv_dnn.so
                ${OPENCV_LIB_DIR}/libopencv_tracking.so
                ${OPENCV_LIB_DIR}/libopencv_video.so
                ${OPENCV_LIB_DIR}/libopencv_videoio.so)
#boost
set(BOOST_LIB_DIR /usr/lib/x86_64-linux-gnu)
link_libraries(${BOOST_LIB_DIR}/libboost_filesystem.so
                ${BOOST_LIB_DIR}/libboost_system.so)
#personmotordetection
set(PEROSONMOTOR_LIB_DIR ../Export/lib)
#/root/project/personmotordetection/Export/lib
set(PEROSONMOTOR_LIB_DIR /root/project/personmotordetection/Export/lib)
link_libraries(${PEROSONMOTOR_LIB_DIR}/libpersonmotor_detection.so)

#add local source path 

#We do not recommend using GLOB to collect a list of source files from your source tree. 
#If no CMakeLists.txt file changes when a source is added or removed then the 
#generated build system cannot know when to ask CMake to regenerate.
#file(GLOB sources./*.cpp)

# *.h path
set(H_FILES 
../Export/include/PersonDetectionSdk.h
./Hungarian.h
./KalmanTracker.h
./PlateID.h
./TH_ErrorDef.h
./TH_PlateID.h
)
#*.cpp /*.c  path
set(CPP_FILES 
./TestYoloV4.cpp
./Hungarian.cpp
./KalmanTracker.cpp
./PlateID.cpp
)

message("${H_FILES}")
message("${CPP_FILES}")


#set exe path
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
add_executable(${PROJECTNAME} ${H_FILES} ${CPP_FILES})




