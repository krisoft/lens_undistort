#include "undistort.h"
#include "undistort_internal.hpp"

cv::Point2d undistort(const double undistorsion_factors[MODEL_SIZE], cv::Point2d pointToUnDistort )
{
	double out_x, out_y;
	undistort_internal<double>( pointToUnDistort.x, pointToUnDistort.y, undistorsion_factors, out_x, out_y );
	return cv::Point2d( out_x, out_y );
}
