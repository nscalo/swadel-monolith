message(STATUS "text decode alone")
cmake_minimum_required(VERSION 3.10)
project(text_decode_project)

set(
    LIBS
    -lpython3
    -lboost_python-mt
    -lboost_numpy-mt
    -lboost_thread-mt
    -lboost_system-mt
    -lopencv_gapi
    -lopencv_core
    -lopencv_text
    -lopencv_ml
    -lopencv_ts
    -lopencv_objdetect
)

set(INCLUDE_DIRS 
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/tbb/include
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/boost_1_65_1
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/numpy/numpy/core/include/
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/opencv/include/
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Work/EdgeApp/License_Plate_Recognition/SOLID-Project-Framework/app/include
    /home/aswin/Documents/Courses/Udacity/Intel-Edge
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/Python-3.6.1/Doc/includes
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/Python-3.6.1/Include
)

set(LIB_DIRS
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/opencv/build/lib
    /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/boost_1_65_1/stage/lib5
    /home/aswin/anaconda3/envs/enscalo_test/lib
)

link_directories(${LIBS_DIR})

include_directories(${INCLUDE_DIRS})
set(LIBS ${LIBS})

include( ${CMAKE_SOURCE_DIR}/../cmake_utils/CMakeFuncs.txt )
FIND_SOURCES()

add_library(main SHARED
    ${PROJECT_SOURCE_DIR}/libmain.cpp
    #${PROJECT_SOURCE_DIR}/text_detection.cpp
    #${PROJECT_SOURCE_DIR}/text_recognition.cpp
    ${SOURCE_FILES})

target_link_libraries(main ${LIBS})
set_target_properties(main  PROPERTIES PUBLIC_HEADER  ${PROJECT_SOURCE_DIR}/list_file.h)
set_target_properties(main  PROPERTIES LINK_FLAGS "-Wl,-soname,libmain.so")
