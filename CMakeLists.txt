cmake_minimum_required(VERSION 2.8)

project(lens_undistort)

find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED)
add_subdirectory(deps/opencvhdfs)


include_directories(
	include
	deps/opencvhdfs/include/
	${CERES_INCLUDE_DIRS}
	${CMAKE_CURRENT_BINARY_DIR}/cmake
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


# configure VERSION to contain git information
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake/version.h" @ONLY)

file(GLOB lens_undistort_SRC
    src/distort.cpp
    src/fitUndistorsionModel.cpp
    src/lines.cpp
    src/prepare_unwrap.cpp
    src/undistort.cpp
)

add_executable(lens_undistort src/main_lens_undistort.cpp ${lens_undistort_SRC})
target_link_libraries(lens_undistort
	${CERES_LIBRARIES}
	gflags
	${OpenCV_LIBS}
	opencvhdfs_lib
)


add_executable(unwrap src/main_unwrap)
target_link_libraries(unwrap
	${CERES_LIBRARIES}
	gflags
	${OpenCV_LIBS}
	opencvhdfs_lib
)


add_executable(stereo_checkerboard_extractor src/main_stereo_checkerboard_extractor)
target_link_libraries(stereo_checkerboard_extractor
	${CERES_LIBRARIES}
	gflags
	${OpenCV_LIBS}
)

add_executable(stereo_calibration src/main_stereo_calibration.cpp ${lens_undistort_SRC})
target_link_libraries(stereo_calibration
	${CERES_LIBRARIES}
	gflags
	${OpenCV_LIBS}
	opencvhdfs_lib
)