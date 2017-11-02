#将之前生成的libseeta_facedet_lib,libseeta_fa_lib复制到build文件夹中（1）
#`src/test/CMakeLists.txt`修改之后如下
aux_source_directory (. SRC_LIST)
message(${SRC_LIST})
link_directories(${PROJECT_BINARY_DIR})
#link_directories("~/SeetaFaceEngine-master/FaceIdentification/build")
#link_directories("~/facerecognize")
# add external libraries
find_package(OpenCV REQUIRED)

enable_testing ()
foreach (f ${SRC_LIST})
  string(REGEX REPLACE "[.]cpp" "" BIN ${f})
  add_executable(${BIN} ${f})
  target_link_libraries(${BIN} viplnet ${OpenCV_LIBS}  faced_lib facea_lib) 
endforeach ()
