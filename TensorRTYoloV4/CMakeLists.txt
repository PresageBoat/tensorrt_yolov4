# Tesla V100
# ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]

# GeForce RTX 2080 Ti, RTX 2080, RTX 2070, Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000, Tesla T4, XNOR Tensor Cores
# ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]

# Jetson XAVIER
# ARCH= -gencode arch=compute_72,code=[sm_72,compute_72]

# GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

# GP100/Tesla P100 - DGX-1
# ARCH= -gencode arch=compute_60,code=sm_60

# For Jetson TX1, Tegra X1, DRIVE CX, DRIVE PX - uncomment:
# ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

# For Jetson Tx2 or Drive-PX2 uncomment:
# ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]


#set min cmake version
cmake_minimum_required(VERSION 3.10) 

#set version information
set(PERSONDETECTION_MAJOR_VERSION 1)
set(PERSONDETECTION_MINOR_VERSION 0)
set(PERSONDETECTION_PATCH_VERSION 0)
set(PERSONDETECTION_VERSION ${PERSONDETECTION_MAJOR_VERSION}.${PERSONDETECTION_MINOR_VERSION}.${PERSONDETECTION_PATCH_VERSION})

#set project name
set(PROJECTNAME personmotor_detection)
project($(PROJECTNAME))   

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

#set nvcc build flag
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75;)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
message("CUDA_NVCC_FLAGS  :${CUDA_NVCC_FLAGS}")


#set include dir && add include dir to cmake

#cuda
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")
include_directories(${CUDA_INCLUDE_DIR})
message("cuda include path :${CUDA_INCLUDE_DIR}")
#cudnn
set(CUDNN_INCLUDE_DIR /usr/local/cuda/include)
include_directories(${CUDNN_INCLUDE_DIR})
message("cudnn include path :${CUDNN_INCLUDE_DIR}")
#tensorrt
set(TENSORRT_INCLUDE_DIR /usr/local/TensorRT-7.0.0.11/include)
include_directories(${TENSORRT_INCLUDE_DIR})
message("tensorrt include path :${TENSORRT_INCLUDE_DIR}")




#set lid dir && set dependece link lib 
#cuda lib
set(CUDA_LIB_DIR /usr/local/cuda/lib64)
link_libraries(${CUDA_LIB_DIR}/libcublas.so
                ${CUDA_LIB_DIR}/libcudart.so
                ${CUDA_LIB_DIR}/libcurand.so
                ${CUDA_LIB_DIR}/stubs/libcuda.so)
#cudnn lib
set(CUDNN_LIB_DIR /usr/local/cuda/lib64)
link_libraries(${CUDNN_LIB_DIR}/libcudnn.so)
#tensorrt lib
set(TENSORRT_LIB_DIR /usr/local/TensorRT-7.0.0.11/lib)
link_libraries(${TENSORRT_LIB_DIR}/libnvinfer.so
                ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
                ${TENSORRT_LIB_DIR}/libmyelin.so
                ${TENSORRT_LIB_DIR}/libnvonnxparser.so)

#copy 3rdparty lib



#add local source path 

#We do not recommend using GLOB to collect a list of source files from your source tree. 
#If no CMakeLists.txt file changes when a source is added or removed then the 
#generated build system cannot know when to ask CMake to regenerate.
#file(GLOB sources./*.cpp)

# *.h path
set(H_FILES 
../Export/include/PersonDetectionSdk.h
./logging.h
./mish.h
./utils.h
./yololayer.h
./yolov4.h
)
#*.cpp /*.c  path
set(CPP_FILES 
./mish.cpp
./run.cpp
./yololayer.cpp
./yolov4.cpp
)
#*.cu
set(CU_FILES 
./mish.cu
./yololayer.cu
)
message("${H_FILES}")
message("${CPP_FILES}")
message("${CU_FILES}")

#set cuda devp
find_package(CUDA REQUIRED)

#set exe path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ../Export/lib/)
CUDA_ADD_LIBRARY(${PROJECTNAME} SHARED  ${H_FILES} ${CPP_FILES} ${CU_FILES})


#set  symbolic link
set_target_properties(${PROJECTNAME} PROPERTIES VERSION ${PERSONDETECTION_VERSION} SOVERSION 1)


#copy cuda so files
file(COPY ${CUDA_LIB_DIR}/libcublas.so DESTINATION ../Export/lib FOLLOW_SYMLINK_CHAIN)
file(COPY ${CUDA_LIB_DIR}/libcudart.so DESTINATION ../Export/lib FOLLOW_SYMLINK_CHAIN)
file(COPY ${CUDA_LIB_DIR}/libcurand.so DESTINATION ../Export/lib FOLLOW_SYMLINK_CHAIN)
file(COPY ${CUDA_LIB_DIR}/stubs/libcuda.so DESTINATION ../Export/lib FOLLOW_SYMLINK_CHAIN)

#copy cudnn so files
file(COPY ${CUDNN_LIB_DIR}/libcudnn.so DESTINATION ../Export/lib FOLLOW_SYMLINK_CHAIN)

#copy tensorrt so files
file(COPY ${TENSORRT_LIB_DIR}/libnvinfer.so DESTINATION ../Export/lib FOLLOW_SYMLINK_CHAIN)




