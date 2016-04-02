#include "undistort.h"
#include "undistort_internal.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "ceres/ceres.h"

DEFINE_bool(details_calibration, true, "Should we print optimization info during calibration?");

struct LineStraigthnessError {
	LineStraigthnessError( const Line &line )
		: line(line) {};


	template <typename T>
	bool operator()( const T* const undistorsion_factors, // fx, fy, k1, k2,
	                 T* residuals) const {

		T err = T(0.0);

		cv::Point first = line.at( 0 );
		cv::Point last = line.at( line.size() - 1 );

		T firstx, firsty;
		T lastx, lasty;

		undistort_internal<T>( T(first.x), T(first.y), undistorsion_factors, firstx, firsty);
		undistort_internal<T>( T(last.x), T(last.y), undistorsion_factors, lastx, lasty);

		T a = lasty - firsty;
	    T b = firstx - lastx;
		T c = lastx*firsty - lasty*firstx;
		T d = ceres::sqrt(a*a + b*b);

		for(const cv::Point &point : line)
		{
			T px, py;
			undistort_internal<T>( T(point.x), T(point.y), undistorsion_factors, px, py);

			err += ceres::abs(a*px + b*py + c) / d;
		}

		residuals[0] = err / T(line.size());

		return true;
	}


	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const Line line) {
		return (new ceres::AutoDiffCostFunction<LineStraigthnessError, 1, 4>(
			new LineStraigthnessError(line)));
	}

	const Line line;
};


void fitUndistorsionModel( const Lines &lines, double undistorsion_factors[MODEL_SIZE], cv::Size frame_size )
{
	undistorsion_factors[0] = ((double)frame_size.width) / 2.0;
	undistorsion_factors[1] = ((double)frame_size.height) / 2.0;
	undistorsion_factors[2] = 0.0;
	undistorsion_factors[3] = 0.0;

	ceres::Problem problem;
	for(Line line : lines )
	{
		ceres::CostFunction* cost_function = LineStraigthnessError::Create( line );
		problem.AddResidualBlock( cost_function, nullptr, undistorsion_factors);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = FLAGS_details_calibration;

	ceres::Solver::Summary summary;
	Solve(options, &problem, &summary);
	if( FLAGS_details_calibration )
	{
		std::cout << summary.FullReport() << "\n";
	}
}