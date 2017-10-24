aux_source_directory (. SRC_LIST)
message(${SRC_LIST})
link_directories(${PROJECT_BINARY_DIR})
find_package(OpenCV REQUIRED)

#enable_testing ()
ADD_LIBRARY (face_rec SHARED face_rec.cpp)
target_link_libraries(face_rec facev_lib faced_lib facea_lib) 

#add_executable(demo demo.cpp)
#target_link_libraries(demo facev_lib face_rec faced_lib facea_lib ${OpenCV_LIBS}) 

add_executable(demo2 demo2.cpp)
target_link_libraries(demo2 facev_lib face_rec faced_lib facea_lib ${OpenCV_LIBS}) 
