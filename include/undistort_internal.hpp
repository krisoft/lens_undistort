#ifndef UNDISTORT_INTERNAL_H
#define UNDISTORT_INTERNAL_H

template <typename T>
void undistort_internal( const T in_x, const T in_y, const T* const undistorsion_factors, T &out_x, T &out_y )
{
	T cx = undistorsion_factors[0];
	T cy = undistorsion_factors[1];
	T k1 = undistorsion_factors[2];
	T k2 = undistorsion_factors[3];

	T dx = in_x - cx;
	T dy = in_y - cy;

	T r2 = dx*dx + dy*dy;

	T scale = (1.0 + k1 * r2 + k2 * r2 * r2 );

	out_x = dx * scale  + cx;
	out_y = dy * scale + cy;
}


#endif // UNDISTORT_INTERNAL_H