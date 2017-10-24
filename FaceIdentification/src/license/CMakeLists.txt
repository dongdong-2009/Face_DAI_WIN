aux_source_directory (. SRC_LIST)
message(${SRC_LIST})
link_directories(${PROJECT_BINARY_DIR})
find_package(OpenCV REQUIRED)

ADD_LIBRARY (abc SHARED libabc.cpp)
 
add_executable(demo_for_lib_license demo_for_lib_license.cpp)
target_link_libraries(demo_for_lib_license abc)

