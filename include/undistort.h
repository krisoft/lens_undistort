#ifndef UNDISTORT_H
#define UNDISTORT_H

#include <opencv2/opencv.hpp>

#include "lines.h"

#define MODEL_SIZE 4

// this one is fast
cv::Point2d undistort(const double undistorsion_factors[MODEL_SIZE], cv::Point2d pointToUnDistort );

// this one is slow, on the order of 0.1 second or so
cv::Point2d distort(const double undistorsion_factors[MODEL_SIZE], cv::Point2d pointToDistort );


// this one is very slow (about 2 minutes for 500 lines on my machine)
void fitUndistorsionModel( const Lines &lines, double undistorsion_factors[MODEL_SIZE], cv::Size frame_size );


// this one is very slow, on the order of distort running time times undistorted image area
void prepare_unwrap(
	const double undistorsion_factors[MODEL_SIZE],
	cv::Size frame_size,
	double unwrap_factor,
	cv::Mat &unwrap_map,
	cv::Mat &unwrap_mask
);


#endif // UNDISTORT_H