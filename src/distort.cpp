#include "undistort.h"
#include "undistort_internal.hpp"

#include "ceres/ceres.h"


struct UnDistorsionError {
	UnDistorsionError( const double x_after, const double y_after, const double distorsion_factors[4] )
		: x_after(x_after), y_after(y_after) {
			for(int i=0; i<4; i++){
				this->distorsion_factors[i] = distorsion_factors[i];
			}
		};


	template <typename T>
	bool operator()( const T* const point, // x, y
	                 T* residuals) const {

		T out_x, out_y;

		T df[4];
		for(int i=0; i<4; i++){
			df[i] = T( distorsion_factors[i] );
		}

		undistort_internal<T>( T(point[0]), T(point[1]), df, out_x, out_y );

		residuals[0] = out_x - T(x_after);
		residuals[1] = out_y - T(y_after);

		return true;
	}


	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double x_after, const double y_after, const double distorsion_factors[4] ) {
		return (new ceres::AutoDiffCostFunction<UnDistorsionError, 2, 2>(
			new UnDistorsionError(x_after, y_after, distorsion_factors)));
	}

	const double x_after;
	const double y_after;
	double distorsion_factors[4];
};

cv::Point2d distort(const double undistorsion_factors[MODEL_SIZE], cv::Point2d pointToDistort )
{
	double point[2];
	point[0] = 0.0;
	point[1] = 0.0;

	ceres::Problem problem;
	ceres::CostFunction* cost_function = UnDistorsionError::Create( pointToDistort.x, pointToDistort.y, undistorsion_factors );
	problem.AddResidualBlock( cost_function, nullptr, point);

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;

	ceres::Solver::Summary summary;
	Solve(options, &problem, &summary);

	return cv::Point2d( point[0], point[1] );
}