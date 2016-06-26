#include "gflags/gflags.h"
#include "glog/logging.h"

#include <glob.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <string>

#include <opencv2/opencv.hpp>

#include "opencvhdfs.h"

#include "lines.h"
#include "undistort.h"

#include "version.h"

#define USAGE_MESSAGE "extract checkerboard images from time synchronized stereo video streams."

DEFINE_string(input, "", "Directory where the frames are stored");
DEFINE_string(output_hdf5, "", "Path to the hdf5 file storing the rectification maps.");

DEFINE_string(left_unwrap, "", "Left unwrap hdf5 matrix");
DEFINE_string(left_xml, "", "Left unwrap parameters in xml");

DEFINE_string(right_unwrap, "", "Right unwrap hdf5 matrix");
DEFINE_string(right_xml, "", "Right unwrap parameters in xml");

DEFINE_int64(board_width, 10, "Checkerboard width");
DEFINE_int64(board_height, 7, "Checkerboard height");
DEFINE_double(square_size, 24., "Checkerboard cell size in mm");

DEFINE_int64(max_frame_count, 10, "Max number of frames used for calibration.");

DEFINE_double(unwrap_factor, 1.0, "How big the unwrapped image should be? 0=only valid pixels  1=whole frame unwrapped. Has to be the same as used for the hdf5 maps.");


bool extract_corners(cv::Mat &frame, cv::Size boardSize, std::vector<cv::Point2f> &pointbuf)
{
	cv::Mat gray;
	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

	if( !cv::findChessboardCorners( frame, boardSize, pointbuf,
			cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE) )
	{
		return false;
	}

	cv::cornerSubPix( gray, pointbuf, cv::Size(11,11),
		cv::Size(-1,-1), cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1 ));

	return true;
}

void read_undistorsion_factors(std::string filename, double undistorsion_factors[], cv::Size &size)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	fs["cx"] >> undistorsion_factors[0];
	fs["cy"] >> undistorsion_factors[1];
	fs["k1"] >> undistorsion_factors[2];
	fs["k2"] >> undistorsion_factors[3];

	fs["width"] >> size.width;
	fs["height"] >> size.height;
}


int main(int argc, char** argv )
{
	gflags::SetUsageMessage(USAGE_MESSAGE);
	gflags::SetVersionString(VERSION);

	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);


	double  undistorsion_factors[2][MODEL_SIZE];
	cv::Size original_image_size[2];

	read_undistorsion_factors( FLAGS_left_xml, undistorsion_factors[0], original_image_size[0] );
	read_undistorsion_factors( FLAGS_right_xml, undistorsion_factors[1], original_image_size[1] );

	cv::Mat left_unwrap_map;
	CVHDFS::read( FLAGS_left_unwrap, "map", left_unwrap_map);

	cv::Mat right_unwrap_map;
	CVHDFS::read( FLAGS_right_unwrap, "map", right_unwrap_map);

	std::vector<std::vector<cv::Point2f> > left_imagePoints, right_imagePoints;

	cv::Size boardSize( FLAGS_board_width, FLAGS_board_height );
	cv::Size left_imageSize, right_imageSize, imageSize;

	std::string left_pattern = FLAGS_input+"/*_l.png";

	glob_t results;
	CHECK( !glob(left_pattern.c_str(), 0, NULL, &results) ) << "For some reason can't glob the input pattern";
	for (int pattern_idx = 0; pattern_idx < results.gl_pathc; pattern_idx++)
	{
		std::string left_path( results.gl_pathv[pattern_idx] );

		std::string right_path( left_path );
		right_path.replace( left_path.size() - 5, 1, "r");

		std::cout << left_path << " " << right_path << std::endl;

		cv::Mat left_frame = cv::imread( left_path );
		cv::Mat right_frame = cv::imread( right_path );

		CHECK( !left_frame.empty() );
		CHECK( !right_frame.empty() );

		cv::remap(left_frame, left_frame, left_unwrap_map, cv::Mat(), cv::INTER_LANCZOS4);
		cv::remap(right_frame, right_frame, right_unwrap_map, cv::Mat(), cv::INTER_LANCZOS4);

		left_imageSize = left_frame.size();
		right_imageSize = right_frame.size();

		std::vector<cv::Point2f> left_pointbuf, right_pointbuf;

		bool left_found = extract_corners( left_frame, boardSize, left_pointbuf);
		bool right_found = extract_corners( right_frame, boardSize, right_pointbuf);

		if( left_found && right_found ) {
			std::cout << "    found." << std::endl;
			left_imagePoints.push_back( left_pointbuf );
			right_imagePoints.push_back( right_pointbuf );
		}
	}
	globfree( &results);

	imageSize.width = std::max( left_imageSize.width, right_imageSize.width );
	imageSize.height = std::max( left_imageSize.height, right_imageSize.height );

	CHECK_GE( left_imagePoints.size(), 5 );

	if( left_imagePoints.size() > FLAGS_max_frame_count )
	{
		left_imagePoints.resize( FLAGS_max_frame_count );
		right_imagePoints.resize( FLAGS_max_frame_count );
	}

	std::vector<std::vector<cv::Point3f> > objectPoints(left_imagePoints.size());

	for( int i = 0; i < left_imagePoints.size(); i++ )
	{
		for( int j = 0; j < boardSize.height; j++ )
			for( int k = 0; k < boardSize.width; k++ )
				objectPoints[i].push_back(cv::Point3f(k*FLAGS_square_size, j*FLAGS_square_size, 0));
	}

	std::cout << "Running stereo calibration ..." << std::endl;

	cv::Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = cv::initCameraMatrix2D(objectPoints,left_imagePoints,left_imageSize,0);
	cameraMatrix[1] = cv::initCameraMatrix2D(objectPoints,right_imagePoints,right_imageSize,0);
	cv::Mat R, T, E, F;

	double rms = cv::stereoCalibrate(objectPoints, left_imagePoints, right_imagePoints,
					cameraMatrix[0], distCoeffs[0],
					cameraMatrix[1], distCoeffs[1],
					imageSize, R, T, E, F,
					cv::CALIB_FIX_K1 + 
					cv::CALIB_FIX_K2 + 
					cv::CALIB_FIX_K3 + 
					cv::CALIB_FIX_K4 + 
					cv::CALIB_FIX_K5 + 
					cv::CALIB_FIX_K6 + 
					cv::CALIB_FIX_S1_S2_S3_S4 + 
					cv::CALIB_ZERO_TANGENT_DIST,
					cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 100, 1e-5) );

	std::cout << "done with RMS error=" << rms << std::endl;

	double err = 0;
	int npoints = 0;
	std::vector<cv::Vec3f> lines[2];
	for( int i = 0; i < left_imagePoints.size(); i++ )
	{
		int npt = (int)left_imagePoints[i].size();
		cv::Mat imgpt[2];
		
		imgpt[0] = cv::Mat(left_imagePoints[i]);
		undistortPoints(imgpt[0], imgpt[0], cameraMatrix[0], distCoeffs[0], cv::Mat(), cameraMatrix[0]);
		computeCorrespondEpilines(imgpt[0], 1, F, lines[0]);

		imgpt[1] = cv::Mat(right_imagePoints[i]);
		undistortPoints(imgpt[1], imgpt[1], cameraMatrix[1], distCoeffs[1], cv::Mat(), cameraMatrix[1]);
		computeCorrespondEpilines(imgpt[1], 2, F, lines[1]);
		
		for( int j = 0; j < npt; j++ )
		{
			double errij = fabs(left_imagePoints[i][j].x*lines[1][j][0] +
								left_imagePoints[i][j].y*lines[1][j][1] + lines[1][j][2]) +
						   fabs(right_imagePoints[i][j].x*lines[0][j][0] +
								right_imagePoints[i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	std::cout << "average epipolar err = " <<  err/npoints << std::endl;

	cv::Mat left_R, right_R;
	cv::Mat left_P, right_P;
	cv::Mat Q;

	cv::stereoRectify(
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize,
		R, T, 
		left_R, right_R, left_P, right_P, Q
	);

	cv::Mat left_map_1, left_map_2, right_map_1, right_map_2;

	cv::initUndistortRectifyMap(
		cameraMatrix[0], distCoeffs[0],
		left_R, left_P,
		imageSize, CV_32FC2, left_map_1, left_map_2
	);

	cv::initUndistortRectifyMap(
		cameraMatrix[1], distCoeffs[1],
		right_R, right_P,
		imageSize, CV_32FC2, right_map_1, right_map_2
	);


	cv::Mat full_rectification_map[2];
	cv::Mat full_rectification_mask[2];

	std::cout << "concatenating rectification and unwrap map, left" << std::endl;
	concatenate_rectification_map_and_unwrap(
		undistorsion_factors[0],
		left_map_1,
		original_image_size[0],
		FLAGS_unwrap_factor,
		full_rectification_map[0],
		full_rectification_mask[0]
	);

	std::cout << "concatenating rectification and unwrap map, right" << std::endl;
	concatenate_rectification_map_and_unwrap(
		undistorsion_factors[1],
		right_map_1,
		original_image_size[1],
		FLAGS_unwrap_factor,
		full_rectification_map[1],
		full_rectification_mask[1]
	);

	CVHDFS::write( FLAGS_output_hdf5, "map_left", full_rectification_map[0]);
	CVHDFS::write( FLAGS_output_hdf5, "mask_left", full_rectification_mask[0]);
	CVHDFS::write( FLAGS_output_hdf5, "R_left", left_R);
	CVHDFS::write( FLAGS_output_hdf5, "P_left", left_P);
	CVHDFS::write( FLAGS_output_hdf5, "camera_left", cameraMatrix[0]);
	

	CVHDFS::write( FLAGS_output_hdf5, "map_right", full_rectification_map[1]);
	CVHDFS::write( FLAGS_output_hdf5, "mask_right", full_rectification_mask[1]);
	CVHDFS::write( FLAGS_output_hdf5, "R_right", right_R);
	CVHDFS::write( FLAGS_output_hdf5, "P_right", right_P);
	CVHDFS::write( FLAGS_output_hdf5, "camera_right", cameraMatrix[1]);

	CVHDFS::write( FLAGS_output_hdf5, "R", R);
	CVHDFS::write( FLAGS_output_hdf5, "T", T);
	CVHDFS::write( FLAGS_output_hdf5, "E", E);
	CVHDFS::write( FLAGS_output_hdf5, "F", F);
	CVHDFS::write( FLAGS_output_hdf5, "Q", Q);

}