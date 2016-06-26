#include "gflags/gflags.h"
#include "glog/logging.h"

#include <glob.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "version.h"

#define USAGE_MESSAGE "extract checkerboard images from time synchronized stereo video streams."

DEFINE_string(input_left, "", "Path for left video");
DEFINE_string(input_right, "", "Path for right video");
DEFINE_int64(frame_diff, 0, "Time sync variable. the x. frame on the left video and the (x+frame_diff). frame on the right should belong to the same timestamp.");
DEFINE_int64(first_frame, 0, "First frame id (on the left video) which will be processed.");
DEFINE_int64(last_frame, -1, "Last frame id (on the left video) which will be processed.");
DEFINE_int64(skipp_nframe, 1, "Skip n frames, and only process every nth.");
DEFINE_string(output, "", "Path for the directory, where the extracted frames will be deposited");

DEFINE_int64(board_width, 10, "Checkerboard width");
DEFINE_int64(board_height, 7, "Checkerboard height");

int main(int argc, char** argv )
{
	gflags::SetUsageMessage(USAGE_MESSAGE);
	gflags::SetVersionString(VERSION);

	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

	cv::VideoCapture cap_left;
	int left_frame_idx = 0;
	CHECK( cap_left.open( FLAGS_input_left ) ) << "can't open left video";


	cv::VideoCapture cap_right;
	int right_frame_idx = 0;
	CHECK( cap_right.open( FLAGS_input_right ) ) << "can't open right video";

	left_frame_idx = std::max( FLAGS_first_frame, std::abs(FLAGS_frame_diff)+1 );
	right_frame_idx = left_frame_idx + FLAGS_frame_diff;
	CHECK( cap_left.set(cv::CAP_PROP_POS_FRAMES, left_frame_idx) );
	CHECK( cap_right.set(cv::CAP_PROP_POS_FRAMES, right_frame_idx) );

	cv::Mat left_frame;
	cv::Mat right_frame;

	cv::Size boardSize( FLAGS_board_width, FLAGS_board_height );

	while(true)
	{
		left_frame_idx += FLAGS_skipp_nframe;
		right_frame_idx += FLAGS_skipp_nframe;

		if( FLAGS_last_frame>0 && left_frame_idx>FLAGS_last_frame )
		{
			break;
		}
		if( !cap_left.set(cv::CAP_PROP_POS_FRAMES, left_frame_idx)
			|| !cap_right.set(cv::CAP_PROP_POS_FRAMES, right_frame_idx))
		{
			break;
		}

		// consume frames until one or the other stream runs dry
		if( !cap_left.read(left_frame) || !cap_right.read(right_frame) )
		{
			break;
		}
		left_frame_idx++;
		right_frame_idx++;

		std::cout << left_frame_idx << " vs. " << right_frame_idx << std::endl;

		std::vector<cv::Point2f> left_pointbuf, right_pointbuf;

		bool left_found = cv::findChessboardCorners( left_frame, boardSize, left_pointbuf,
			cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

		bool right_found = cv::findChessboardCorners( right_frame, boardSize, right_pointbuf,
			cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

		if(left_found && right_found)
		{
			std::stringstream left_ss;
			left_ss << FLAGS_output << "/";
			left_ss << std::setfill('0') << std::setw(10) << left_frame_idx;
			left_ss << "_l.png";
			imwrite( left_ss.str(), left_frame);

			std::stringstream right_ss;
			right_ss << FLAGS_output << "/";
			right_ss << std::setfill('0') << std::setw(10) << left_frame_idx;
			right_ss << "_r.png";
			imwrite( right_ss.str(), right_frame);

            /*d

            */
        }
        else if(left_found || right_found)
        {
        	std::cout << "found half" << std::endl;

        	drawChessboardCorners( left_frame, boardSize, cv::Mat(left_pointbuf), left_found );
            drawChessboardCorners( right_frame, boardSize, cv::Mat(right_pointbuf), right_found );

        	imshow("left",left_frame);
			imshow("right",right_frame);
			cv::waitKey(1);
        }
	}
}