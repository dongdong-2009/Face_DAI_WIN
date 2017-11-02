aux_source_directory (. SRC_LIST)
message(${SRC_LIST})
link_directories(${PROJECT_BINARY_DIR})
find_package(OpenCV REQUIRED)
#enable_testing ()



if(DUSE_SEETA)
ADD_LIBRARY (face_rec SHARED face_rec.cpp)
target_link_libraries(face_rec facev_lib faced_lib facea_lib libopencv_core.2413.dll.a libopencv_highgui.2413.dll.a libopencv_imgproc.2413.dll.a) 
#add_executable(demo demo.cpp)
#target_link_libraries(demo facev_lib face_rec faced_lib facea_lib libopencv_core.2413.dll.a libopencv_highgui.2413.dll.a libopencv_imgproc.2413.dll.a)
add_executable(demo2 demo2.cpp)
target_link_libraries(demo2 facev_lib face_rec faced_lib facea_lib libopencv_core.2413.dll.a libopencv_highgui.2413.dll.a libopencv_imgproc.2413.dll.a)
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDLIB_JPEG_SUPPORT -DDUSE_DLIB")
ADD_LIBRARY (face_rec SHARED dlib_face_rec.cpp)
target_link_libraries(face_rec libdlib.a faced_lib libopencv_core.2413.dll.a libopencv_highgui.2413.dll.a libopencv_imgproc.2413.dll.a)
add_executable(demo2 demo2.cpp)
target_link_libraries(demo2 face_rec libopencv_core.2413.dll.a libopencv_highgui.2413.dll.a libopencv_imgproc.2413.dll.a)
add_executable(verify_face_rec verify_face_rec.cpp)
target_link_libraries(verify_face_rec face_rec libopencv_core.2413.dll.a libopencv_highgui.2413.dll.a libopencv_imgproc.2413.dll.a)
#add_executable(dnn_face_recognition dnn_face_recognition.cpp)
#target_link_libraries(dnn_face_recognition libdlib.a faced_lib libopencv_core.2413.dll.a libopencv_highgui.2413.dll.a libopencv_imgproc.2413.dll.a)
endif()








