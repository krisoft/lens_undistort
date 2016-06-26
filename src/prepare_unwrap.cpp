#include "undistort.h"

#include "glog/logging.h"

#include <iostream>

#include <limits>
#include <algorithm>


double interpolate( double a, double b, double factor )
{
	return a * factor + b * (1.0 - factor );
}


void prepare_unwrap(
	const double undistorsion_factors[MODEL_SIZE],
	cv::Size frame_size,
	double unwrap_factor,
	cv::Mat &unwrap_map,
	cv::Mat &unwrap_mask
)
{
	CHECK( std::numeric_limits<double>::has_infinity ) << "double doesn't have infinity on this system? wow!";

	cv::Point2d original_center(undistorsion_factors[0], undistorsion_factors[1]);
	cv::Point2d undistorted_center = undistort(undistorsion_factors, original_center );

	// first calculate the min and max rectangle
	// min is the rectangle where all the pixels are valid.
	// max is the rectangle where all the input pixels has been unwrapped. some pixels will be thus invalid
	double min_top = -1. * std::numeric_limits<double>::infinity();
	double min_bottom = std::numeric_limits<double>::infinity();
	double min_left = -1. * std::numeric_limits<double>::infinity();
	double min_right = std::numeric_limits<double>::infinity();

	double max_top = undistorted_center.y;
	double max_bottom = undistorted_center.y;
	double max_left = undistorted_center.x;
	double max_right = undistorted_center.x;

	for(int x=0; x<=frame_size.width; x++)
	{
		cv::Point2d edge_point =  undistort(undistorsion_factors, cv::Point2d( x, 0.0 ) );
		min_top = std::max( min_top, edge_point.y );
		max_top = std::min( max_top, edge_point.y );

		edge_point =  undistort(undistorsion_factors, cv::Point2d( x, frame_size.height ) );
		min_bottom = std::min( min_bottom, edge_point.y );
		max_bottom = std::max( max_bottom, edge_point.y );
	}

	for(int y=0; y<=frame_size.height; y++)
	{
		cv::Point2d edge_point =  undistort(undistorsion_factors, cv::Point2d( 0.0, y ) );
		min_left = std::max( min_left, edge_point.x );
		max_left = std::min( max_left, edge_point.x );

		edge_point =  undistort(undistorsion_factors, cv::Point2d( frame_size.width, y ) );
		min_right= std::min( min_right, edge_point.x );
		max_right = std::max( max_right, edge_point.x );
	}

	// safety check on unwrap_factor, don't want to terminate just because it's out of bounds, but don't want to be silent about it either
	if( unwrap_factor<0 || unwrap_factor>1.0 )
	{
		std::cerr << "unwrap_factor is out of bounds. will be clipped" << std::endl;
		unwrap_factor = std::min( 1.0, std::max( 0.0, unwrap_factor ));
	}

	// calculate result rectangle (interpolation between min and max rectangles by unwrap_factor )
	double unwrapped_top = interpolate( max_top, min_top, unwrap_factor );
	double unwrapped_bottom = interpolate( max_bottom, min_bottom, unwrap_factor );
	double unwrapped_left = interpolate( max_left, min_left, unwrap_factor );
	double unwrapped_right = interpolate( max_right, min_right, unwrap_factor );
	int unwrapped_width = (int)( unwrapped_right - unwrapped_left );
	int unwrapped_height = (int)( unwrapped_bottom - unwrapped_top );


	/*
	std::cout << "min rectangle" << std::endl;
	std::cout << "    top:" << min_top << std::endl;
	std::cout << "    bottom:" << min_bottom << std::endl;
	std::cout << "    left:" << min_left << std::endl;
	std::cout << "    right:" << min_right << std::endl;
	std::cout << "    width:" << (min_right-min_left) << std::endl;
	std::cout << "    height:" << (min_bottom-min_top) << std::endl;

	std::cout << "max rectangle" << std::endl;
	std::cout << "    top:" << max_top << std::endl;
	std::cout << "    bottom:" << max_bottom << std::endl;
	std::cout << "    left:" << max_left << std::endl;
	std::cout << "    right:" << max_right << std::endl;
	std::cout << "    width:" << (max_right-max_left) << std::endl;
	std::cout << "    height:" << (max_bottom-max_top) << std::endl << std::endl;


	std::cout << "original center: " << original_center << std::endl;
	std::cout << "frame_size: " << frame_size << std::endl;

	std::cout << "unwrapped rectangle" << std::endl;
	std::cout << "    top:" << unwrapped_top << std::endl;
	std::cout << "    bottom:" << unwrapped_bottom << std::endl;
	std::cout << "    left:" << unwrapped_left << std::endl;
	std::cout << "    right:" << unwrapped_right << std::endl;
	std::cout << "    width:" << unwrapped_width << std::endl;
	std::cout << "    height:" << unwrapped_height << std::endl << std::endl;
	*/

	// ensuring output matrixes has the correct type and size
	unwrap_map.create( unwrapped_height, unwrapped_width, CV_32FC2 );
	unwrap_mask.create( unwrapped_height, unwrapped_width, CV_8UC1 );

	int next_percentage_to_report = 5;
	int report_every = 5;

	for(int x=0; x<unwrapped_width; x++ )
	{
		for(int y=0; y<unwrapped_height; y++ )
		{
			cv::Point2d unwrapped_point(
				x + unwrapped_left,
				y + unwrapped_top
			);

			cv::Point2d original_point = distort(undistorsion_factors, unwrapped_point );
			unwrap_map.at<cv::Vec2f>( y, x ) = cv::Vec2f(
				original_point.x,
				original_point.y
			);

			bool valid = 
				original_point.x>=0 && original_point.x<frame_size.width
				&& original_point.y>=0 && original_point.x<frame_size.height;

			if( valid )
			{
				unwrap_mask.at<uchar>( y, x ) = 255.;
			}
			else
			{
				unwrap_mask.at<uchar>( y, x ) = 0.;
			}
		}

		double percentage = (double)x / (double)unwrapped_width;
		if( percentage*100 >= next_percentage_to_report )
		{
			std::cout << next_percentage_to_report << "%" << std::endl;
			next_percentage_to_report += report_every;
		} 
	}

}



void concatenate_rectification_map_and_unwrap(
	const double undistorsion_factors[MODEL_SIZE],
	cv::Mat &rectification_map,
	cv::Size frame_size,
	double unwrap_factor,
	cv::Mat &unwrap_map,
	cv::Mat &unwrap_mask
)
{
	CHECK( std::numeric_limits<double>::has_infinity ) << "double doesn't have infinity on this system? wow!";


	cv::Point2d original_center(undistorsion_factors[0], undistorsion_factors[1]);
	cv::Point2d undistorted_center = undistort(undistorsion_factors, original_center );

	// first calculate the min and max rectangle
	// min is the rectangle where all the pixels are valid.
	// max is the rectangle where all the input pixels has been unwrapped. some pixels will be thus invalid
	double min_top = -1. * std::numeric_limits<double>::infinity();
	double min_bottom = std::numeric_limits<double>::infinity();
	double min_left = -1. * std::numeric_limits<double>::infinity();
	double min_right = std::numeric_limits<double>::infinity();

	double max_top = undistorted_center.y;
	double max_bottom = undistorted_center.y;
	double max_left = undistorted_center.x;
	double max_right = undistorted_center.x;

	for(int x=0; x<=frame_size.width; x++)
	{
		cv::Point2d edge_point =  undistort(undistorsion_factors, cv::Point2d( x, 0.0 ) );
		min_top = std::max( min_top, edge_point.y );
		max_top = std::min( max_top, edge_point.y );

		edge_point =  undistort(undistorsion_factors, cv::Point2d( x, frame_size.height ) );
		min_bottom = std::min( min_bottom, edge_point.y );
		max_bottom = std::max( max_bottom, edge_point.y );
	}

	for(int y=0; y<=frame_size.height; y++)
	{
		cv::Point2d edge_point =  undistort(undistorsion_factors, cv::Point2d( 0.0, y ) );
		min_left = std::max( min_left, edge_point.x );
		max_left = std::min( max_left, edge_point.x );

		edge_point =  undistort(undistorsion_factors, cv::Point2d( frame_size.width, y ) );
		min_right= std::min( min_right, edge_point.x );
		max_right = std::max( max_right, edge_point.x );
	}

	// safety check on unwrap_factor, don't want to terminate just because it's out of bounds, but don't want to be silent about it either
	if( unwrap_factor<0 || unwrap_factor>1.0 )
	{
		std::cerr << "unwrap_factor is out of bounds. will be clipped" << std::endl;
		unwrap_factor = std::min( 1.0, std::max( 0.0, unwrap_factor ));
	}

	// calculate result rectangle (interpolation between min and max rectangles by unwrap_factor )
	double unwrapped_top = interpolate( max_top, min_top, unwrap_factor );
	double unwrapped_bottom = interpolate( max_bottom, min_bottom, unwrap_factor );
	double unwrapped_left = interpolate( max_left, min_left, unwrap_factor );
	double unwrapped_right = interpolate( max_right, min_right, unwrap_factor );

	// ensuring output matrixes has the correct type and size
	cv::Size rectification_size = rectification_map.size();
	unwrap_map.create( rectification_size.height, rectification_size.width, CV_32FC2 );
	unwrap_mask.create( rectification_size.height, rectification_size.width, CV_8UC1 );

	int next_percentage_to_report = 5;
	int report_every = 5;

	for(int x=0; x<rectification_size.width; x++ )
	{
		for(int y=0; y<rectification_size.height; y++ )
		{
			cv::Vec2f rectification_pix = rectification_map.at<cv::Vec2f>(y,x);

			cv::Point2d unwrapped_point(
				rectification_pix[0] + unwrapped_left,
				rectification_pix[1] + unwrapped_top
			);

			cv::Point2d original_point = distort(undistorsion_factors, unwrapped_point );
			unwrap_map.at<cv::Vec2f>( y, x ) = cv::Vec2f(
				original_point.x,
				original_point.y
			);

			bool valid = 
				original_point.x>=0 && original_point.x<frame_size.width
				&& original_point.y>=0 && original_point.x<frame_size.height;

			if( valid )
			{
				unwrap_mask.at<uchar>( y, x ) = 255.;
			}
			else
			{
				unwrap_mask.at<uchar>( y, x ) = 0.;
			}
		}

		double percentage = (double)x / (double)rectification_size.width;
		if( percentage*100 >= next_percentage_to_report )
		{
			std::cout << next_percentage_to_report << "%" << std::endl;
			next_percentage_to_report += report_every;
		} 
	}

}