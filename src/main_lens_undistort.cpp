#include "gflags/gflags.h"
#include "glog/logging.h"

#include <glob.h>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "opencvhdfs.h"

#include "lines.h"
#include "undistort.h"

#include "version.h"

#define USAGE_MESSAGE "lets you calibrate lens distortion of your camera."

DEFINE_string(input, "", "Glob pattern for input videos or frames");
DEFINE_int64(max_line_count, 500, "Line extraction will be terminated after this much lines. Zero means all lines will be extracted.");
DEFINE_bool(visual_confirm, false, "Should every line be confirmed throught a ui?");
DEFINE_string(output_xml, "", "Path for xml or yaml output.");
DEFINE_string(output_hdf5, "", "Path for unwrapping matrix.");
DEFINE_double(unwrap_factor, 1.0, "How big the unwrapped image should be? 0=only valid pixels  1=whole frame unwrapped");


void process_frame( cv::Mat frame, Lines &lines )
{

	if( FLAGS_visual_confirm )
	{
		// every extracted line needs visual confirm

		cv::Mat vis_collect = frame.clone(); // this collects the frames already extracted

		Lines lines_extracted;
		extract_lines(frame, lines_extracted );

		for( Line &line : lines_extracted )
		{
			cv::Mat vis_temp = vis_collect.clone();

			bool even = true;
			while(true) {
				if( even )
				{
					draw_line(vis_temp, line, cv::Scalar(0,0,255));
				}
				else
				{
					draw_line(vis_temp, line, cv::Scalar(0,255,255));
				}
				even = !even;
				cv::imshow("frame", vis_temp );
				int c = cv::waitKey(100);

				if( c == 'a' )
				{
					draw_line(vis_collect, line, cv::Scalar(0,0,255));
					lines.push_back( line );
					break;
				}
				else if ( c == 'd' )
				{
					break;
				}
			}
		}
	}
	else
	{
		// no need for confirm, everything can go directly to the soup
		extract_lines(frame, lines );
	}
	
	
	
}

int main(int argc, char** argv )
{
	gflags::SetUsageMessage(USAGE_MESSAGE);
	gflags::SetVersionString(VERSION);

	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

	if( FLAGS_visual_confirm )
	{
		// display some description
		std::cout << "Hit 'a' to accept the flashing line, or 'd' to deny the proposal." << std::endl;
	}

	// this is the container holding all the lines to be straightened
	Lines lines;

	// frame size
	cv::Size frame_size;

	// enumerate through paths matched by glob pattern
	glob_t results;
	CHECK( !glob(FLAGS_input.c_str(), 0, NULL, &results) ) << "For some reason can't glob the input pattern";
	for (int pattern_idx = 0; pattern_idx < results.gl_pathc; pattern_idx++)
	{
		// do we have enough lines already?
		if ( FLAGS_max_line_count!=0 && lines.size()>FLAGS_max_line_count ) {
			// we most definietly have
			break;
		}
		std::string input_path( results.gl_pathv[pattern_idx] );

		std::cout << input_path << std::endl;

		// try to load as image
		cv::Mat frame = cv::imread(input_path);
		if( frame.data!=NULL )
		{
			frame_size = frame.size(); // TODO we should check if they are all the same size
			// managed to read
			process_frame( frame, lines );
			continue;
		}
		// couldn't read, maybe it's a video then?

		cv::VideoCapture cap;

		if( cap.open(input_path) )
		{
			// it's a video!
			while(true)
			{
				cv::Mat frame;
				if( !cap.read( frame ) )
				{
					break;
				}

				frame_size = frame.size(); // TODO we should check if they are all the same size (there might be multiple videos, or videos and frames!)
				process_frame( frame, lines );
			}

			continue;
		}

		// nop, not even a video. write out an error message then
		std::cout << "couldn't read " << input_path << std::endl;
	}

	globfree( &results );


	// okay we have our lines, we should fit the model now
	std::cout << "calibrating, might take a few minutes." << std::endl;
	double  undistorsion_factors[MODEL_SIZE];
	fitUndistorsionModel( lines, undistorsion_factors, frame_size );

	std::cout << "calibrated." << std::endl;
	std::cout << "cx: " << undistorsion_factors[0] << std::endl;
	std::cout << "cy: " << undistorsion_factors[1] << std::endl;
	std::cout << "k1: " << undistorsion_factors[2] << std::endl;
	std::cout << "k2: " << undistorsion_factors[3] << std::endl;

	// calibration is done
	// calculate unwrapping if requested

	cv::Mat unwrap_map, unwrap_mask;

	if( FLAGS_output_hdf5.size()>0 )
	{
		std::cout << "unwrapping, might take a few more minutes" << std::endl;
		prepare_unwrap( undistorsion_factors, frame_size, FLAGS_unwrap_factor, unwrap_map, unwrap_mask );
		std::cout << "unwrapping done" << std::endl;

		CVHDFS::write( FLAGS_output_hdf5, "map", unwrap_map);
		CVHDFS::write( FLAGS_output_hdf5, "mask", unwrap_mask);
	}

	// save the parameters into xml or yaml if requested

	if( FLAGS_output_xml.size()>0 )
	{
		// FLAGS_output_xml is not empty save then
		cv::FileStorage fs(FLAGS_output_xml, cv::FileStorage::WRITE);
		fs << "cx" << undistorsion_factors[0];
		fs << "cy" << undistorsion_factors[1];
		fs << "k1" << undistorsion_factors[2];
		fs << "k2" << undistorsion_factors[3];

		fs << "width" << frame_size.width;
		fs << "height" << frame_size.height;
		fs.release();
	}


}