message(STATUS "text decode alone")
cmake_minimum_required(VERSION 3.10)
project(text_decode_project)

set(
    PYTHON_LIBS
    -lpython3
    -lboost_python-mt 
    -lboost_numpy-mt 
    -lboost_thread-mt
    -lboost_system-mt
)

set(OpenCV_LIBS
    -lopencv_gapi 
    -lopencv_core 
    -lopencv_text 
    -lopencv_ml 
    -lopencv_ts 
    -lopencv_objdetect
)

set(OpenCV_INCLUDE_DIRS 
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/opencv/include/ 
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Work/EdgeApp/License_Plate_Recognition/SOLID-Project-Framework/app/include
)
set(OpenCV_LIBS_DIR /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/opencv/build/lib)
set(Python_LIBS_DIR /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/Python-3.6.1)
set(Python_LIBS_DIR home/aswin/anaconda3/envs/enscalo_test/lib)
set(Boost_Python_LIBS_DIR /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/boost_1_65_1/stage/lib5)
set(TBB_SOURCE_DIR /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/tbb/src/tbb)
set(TBB_MALLOC_DIR /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/tbbmalloc/src)
set(TBB_INCLUDE_DIR /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/tbb/include)
set(Boost_Python_INCLUDE_DIR /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/boost_1_65_1)
set(Python_INCLUDE_DIRS 
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/numpy/numpy/core/include/ 
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/Python-3.6.1/Include
    /home/aswin/Documents/Courses/Udacity/Intel-Edge
)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

link_directories(${OpenCV_LIBS_DIR})
link_directories(${Python_LIBS_DIR})
link_directories(${Boost_Python_LIBS_DIR})

set(INTEL_ITT_INCLUDE_DIRS /opt/intel/ittapi/include)

# option(INTEL_SAMPLE_USE_OPENCV ON)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Python_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIR})
set(LIBS ${LIBS} ${PYTHON_LIBS})
set(LIBS ${LIBS} ${OpenCV_LIBS})

include( ${CMAKE_SOURCE_DIR}/../cmake_utils/CMakeFuncs.txt )
FIND_SOURCES()
FIND_COMMON_SOURCES()

set(COMMON_SOURCES
    ${TBB_SOURCE_DIR}/*.cpp
    ${TBB_MALLOC_DIR}/*.cpp
)

add_library(main SHARED
    ${PROJECT_SOURCE_DIR}/libmain.cpp
    #${PROJECT_SOURCE_DIR}/text_detection.cpp
    #${PROJECT_SOURCE_DIR}/text_recognition.cpp
    ${SOURCE_FILES})

target_link_libraries(main ${LIBS})
set_target_properties(main  PROPERTIES PUBLIC_HEADER  ${PROJECT_SOURCE_DIR}/list_file.h)# ${PROJECT_SOURCE_DIR}/text_decode.h ${PROJECT_SOURCE_DIR}/text_recognition.h) 
# set_target_properties(main  PROPERTIES LINK_FLAGS "-Wl,-soname,libmain.so")
