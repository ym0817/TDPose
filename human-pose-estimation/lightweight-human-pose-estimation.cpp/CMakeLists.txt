cmake_minimum_required(VERSION 3.10.0)
project(rtmpose_ort)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (ONNXRUNTIME_DIR "/home/ymm/ThirdPart/onnxruntime-linux-x64-1.15.0")
 set (OPENCV_DIR "/home/ymm/ThirdPart/opencv-4.8.0")



include_directories(${ONNXRUNTIME_DIR}/include
                    ${OPENCV_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                 #   ${CMAKE_CURRENT_SOURCE_DIR}/detect/x/include
                   
                     )

link_directories(
                 ${ONNXRUNTIME_DIR}/lib
                 ${OPENCV_DIR}/lib
                )



# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_LIST)
add_executable(rtmpose_ort
       # ${SRC_LIST}
       main.cpp
          )


target_compile_features(rtmpose_ort PRIVATE cxx_std_17)
target_link_libraries(rtmpose_ort 
        onnxruntime
        opencv_dnn
        opencv_highgui 
        opencv_imgproc 
        opencv_imgcodecs
        opencv_imgcodecs.so.408
        opencv_video
        opencv_core
        opencv_features2d
        opencv_videoio
)
